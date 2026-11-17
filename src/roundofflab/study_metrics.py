"""Row-batched error measurements; no whole-cohort output arrays are retained."""
from __future__ import annotations
import numpy as np
from .analysis import _rms
from .quantize import validate_matrix


class _Squares:
    """Per-column scaled sum of squares, merged without squaring original data."""
    def __init__(self, columns: int):
        self.scale = np.zeros(columns)
        self.total = np.zeros(columns)
        self.count = 0

    def add(self, values: np.ndarray) -> None:
        if not np.isfinite(values).all():
            raise ValueError("nonfinite output; rescale matrices")
        # Preserve reduction order across batch boundaries, including very
        # ill-conditioned inputs. Batching controls storage, not the arithmetic.
        for row in values:
            peak = np.maximum(self.scale, np.abs(row))
            ratio = np.divide(self.scale, peak, out=np.zeros_like(peak), where=peak != 0)
            normalized = np.divide(row, peak, out=np.zeros_like(row), where=peak != 0)
            self.total = self.total * ratio**2 + normalized**2
            self.scale = peak
            self.count += 1

    def rms(self) -> np.ndarray:
        result = self.scale * np.sqrt(self.total / self.count)
        if np.any((self.scale != 0) & (result == 0)):
            raise ValueError("RMS underflow; rescale matrices")
        return result

    def mse(self) -> np.ndarray:
        rms = self.rms()
        result = rms**2
        if np.any((rms != 0) & (result == 0)):
            raise ValueError("squared-error underflow; rescale matrices")
        return result


def _mean_square(rms: np.ndarray) -> float:
    root = float(_rms(rms))
    result = root * root
    if root != 0 and result == 0:
        raise ValueError("squared-error underflow; rescale matrices")
    if not np.isfinite(result):
        raise ValueError("squared-error overflow; rescale matrices")
    return result


def _row_products(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Use the same matrix-vector contraction regardless of row batch size."""
    result = np.empty((len(inputs), len(weights)), dtype=np.float64)
    for i, row in enumerate(inputs):
        result[i] = weights @ row
    return result


def evaluate(weights: np.ndarray, candidate: np.ndarray, inputs: np.ndarray, *, batch_rows: int = 256) -> dict:
    """Measure both rounded-forward differences and direct error propagation.

    Lower-level API: callers managing many candidates should apply aggregate
    work budgets first. Float64 candidates/reference are retained; outputs are
    limited to row batches. No exact-arithmetic or GPU behavior is implied.
    """
    if type(batch_rows) is not int or not 1 <= batch_rows <= 4096:
        raise ValueError("batch_rows must be an integer from 1 to 4096")
    for name, array in (("weights", weights), ("candidate", candidate), ("inputs", inputs)):
        validate_matrix(array, name)
    if weights.shape != candidate.shape or weights.shape[1] != inputs.shape[1]:
        raise ValueError("candidate shape must equal weights; input channels must match")
    w, q = np.ascontiguousarray(weights, dtype=np.float64), np.ascontiguousarray(candidate, dtype=np.float64)
    nout = w.shape[0]
    ref_stats, observed_stats, propagation_stats = (_Squares(nout) for _ in range(3))
    worst = []
    max_observed = max_propagated = 0.0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            difference = q - w
            for start in range(0, len(inputs), batch_rows):
                x = np.ascontiguousarray(inputs[start:start+batch_rows], dtype=np.float64)
                reference = _row_products(x, w)
                output = _row_products(x, q)
                observed = output - reference
                propagated = _row_products(x, difference)
                ref_stats.add(reference); observed_stats.add(observed); propagation_stats.add(propagated)
                max_observed = max(max_observed, float(np.max(np.abs(observed))))
                max_propagated = max(max_propagated, float(np.max(np.abs(propagated))))
                magnitudes = np.abs(propagated)
                channels = np.argmax(magnitudes, axis=1)
                maxima = magnitudes[np.arange(len(x)), channels]
                # At most five worst samples, with deterministic row-index ties.
                order = np.lexsort((np.arange(len(x)), -maxima))[:5]
                worst.extend({"sample_index": start + int(i), "output_channel": int(channels[i]),
                              "max_abs_error": float(maxima[i])} for i in order if maxima[i] > 0)
                worst = sorted(worst, key=lambda item: (-item['max_abs_error'], item['sample_index']))[:5]
            ref_rms = ref_stats.rms()
            observed_rms, propagated_rms = observed_stats.rms(), propagation_stats.rms()
            observed_mse, propagated_mse = _mean_square(observed_rms), _mean_square(propagated_rms)
            denom = float(_rms(ref_rms))
            relative = float(_rms(observed_rms)) / denom if denom else None
            relative_propagated = float(_rms(propagated_rms)) / denom if denom else None
            om, pm = observed_stats.mse(), propagation_stats.mse()
            if any(v is not None and not np.isfinite(v) for v in (relative, relative_propagated)):
                raise ValueError("relative-error overflow; rescale matrices")
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("study arithmetic overflow; rescale matrices") from exc
    return {"metrics": {"output_mse": observed_mse, "propagated_mse": propagated_mse,
                        "output_max_abs_error": max_observed, "propagated_max_abs_error": max_propagated,
                        "relative_output_l2": relative, "relative_propagated_l2": relative_propagated},
            "channels": [{"index": i, "output_mse": float(om[i]), "propagated_mse": float(pm[i]),
                          "reference_rms": float(ref_rms[i])} for i in range(nout)],
            "worst_samples": worst,
            "rounding_mask_detected": observed_mse == 0 and propagated_mse > 0,
            "execution": {"maximum_batch_rows": min(len(inputs), batch_rows),
                          "batches": (len(inputs) + batch_rows - 1) // batch_rows}}
