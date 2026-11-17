"""Compare weight reconstruction and observed linear-layer output errors."""
from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations

import numpy as np

from .quantize import Config, default_configs, quantize, validate_matrix

POLICY = "float64; symmetric narrow-range; ties-to-even; linear percentile"


def fingerprint(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([value.dtype.str, list(value.shape)], separators=(",", ":")).encode())
    # Logical C-order bytes, independent of strides and file location.
    for row in value:
        digest.update(row.tobytes(order="C"))
    return digest.hexdigest()


def _bounds(size: int, maximum: int) -> list[list[int]]:
    edges = np.linspace(0, size, min(size, maximum) + 1, dtype=int)
    return [[int(a), int(b)] for a, b in zip(edges[:-1], edges[1:])]


def _heatmap(error: np.ndarray) -> dict:
    rows, cols = _bounds(error.shape[0], 24), _bounds(error.shape[1], 64)
    return {"metric": "mean_absolute_weight_error", "row_bounds": rows, "col_bounds": cols,
            "values": [[float(error[a:b, c:d].mean()) for c, d in cols] for a, b in rows]}


def _rms(value: np.ndarray, axis=None):
    # Scaled RMS avoids squaring large numbers before averaging.
    maximum = np.max(np.abs(value), axis=axis, keepdims=True)
    scaled = np.divide(value, maximum, out=np.zeros_like(value), where=maximum != 0)
    peak = np.squeeze(maximum, axis=axis)
    result = peak * np.sqrt(np.mean(scaled * scaled, axis=axis))
    if np.any((peak != 0) & (result == 0)):
        raise ValueError("RMS underflow; rescale your matrices")
    return result


def _mse(value: np.ndarray, axis=None):
    rms = _rms(value, axis=axis)
    result = rms * rms
    if np.any((rms != 0) & (result == 0)):
        raise ValueError("squared-error metric underflow; rescale your matrices")
    return result


def analyze(weights: np.ndarray, inputs: np.ndarray, configs=None, *,
            title: str = "RoundoffLab report", sample_label: str = "user-provided",
            max_elements: int = 4_000_000, max_output_elements: int = 4_000_000) -> dict:
    """Produce JSON-compatible evidence, not a model accuracy or speed claim.

    Work-size limits are not a peak-RSS limit. Outputs and weights are evaluated
    in float64, independently for every configuration. Inputs remain unchanged.
    """
    for name, limit in (("max_elements", max_elements), ("max_output_elements", max_output_elements)):
        if type(limit) is not int or limit < 1:
            raise ValueError(f"{name} must be a positive integer")
    # Cheap shape checks precede finite scans, float64 copies and matrix products.
    for name, value in (("weights", weights), ("inputs", inputs)):
        if not isinstance(value, np.ndarray) or value.ndim != 2 or min(value.shape) < 1:
            raise ValueError(f"{name} must be a nonempty 2D NumPy array")
    if weights.shape[1] != inputs.shape[1]:
        raise ValueError("weights and inputs must have the same input-channel count")
    if weights.size + inputs.size > max_elements:
        raise ValueError(f"input element limit exceeded ({max_elements}); use smaller matrices")
    if weights.shape[0] * inputs.shape[0] > max_output_elements:
        raise ValueError(f"output element limit exceeded ({max_output_elements}); use fewer samples")
    if weights.shape[0] > 4096:
        raise ValueError("report supports at most 4096 output channels; select a layer slice")
    validate_matrix(weights, "weights")
    validate_matrix(inputs, "inputs")
    if not isinstance(title, str) or not isinstance(sample_label, str):
        raise ValueError("title and sample_label must be strings")
    configs = default_configs() if configs is None else list(configs)
    if not 1 <= len(configs) <= 24 or any(not isinstance(c, Config) for c in configs):
        raise ValueError("supply between 1 and 24 Config objects")
    if len({c.id for c in configs}) != len(configs):
        raise ValueError("configuration IDs must be unique")
    provenance = {"weight_sha256": fingerprint(weights), "input_sha256": fingerprint(inputs),
                  "numpy_version": np.__version__, "numerical_policy": POLICY}
    w, x = np.asarray(weights, dtype=np.float64), np.asarray(inputs, dtype=np.float64)
    entries = []
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            reference = x @ w.T
            if not np.isfinite(reference).all():
                raise ValueError("reference output is nonfinite; rescale your matrices")
            reference_rms = float(_rms(reference))
            channel_rms = _rms(reference, axis=0)
            for config in configs:
                quantized = quantize(w, config)
                dw = quantized.weights - w
                output = x @ quantized.weights.T
                dy = output - reference
                weight_mse, output_mse = float(_mse(dw)), float(_mse(dy))
                relative = float(_rms(dy)) / reference_rms if reference_rms else None
                channel_w, channel_y = _mse(dw, axis=1), _mse(dy, axis=0)
                entries.append({
                    "id": config.id, "bits": config.bits, "mode": config.mode,
                    "group_size": config.group_size if config.mode == "group" else None,
                    "percentile": float(config.percentile),
                    "metrics": {"weight_mse": weight_mse, "weight_max_abs_error": float(np.max(np.abs(dw))),
                                "output_mse": output_mse, "output_max_abs_error": float(np.max(np.abs(dy))),
                                "relative_output_l2": relative, "clipped_count": quantized.clipped_count,
                                "scale_count": quantized.scale_count},
                    "channels": [{"index": i, "weight_mse": float(channel_w[i]),
                                  "output_mse": float(channel_y[i]), "reference_rms": float(channel_rms[i])}
                                 for i in range(w.shape[0])],
                    "heatmap": _heatmap(np.abs(dw)),
                })
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("analysis arithmetic overflow; rescale your matrices") from exc
    reversals = []
    for first, second in combinations(entries, 2):
        a, b = sorted((first, second), key=lambda c: c["metrics"]["weight_mse"])
        if (a["metrics"]["weight_mse"] < b["metrics"]["weight_mse"] and
                a["metrics"]["output_mse"] > b["metrics"]["output_mse"]):
            reversals.append({"lower_weight_error": a["id"], "lower_output_error": b["id"]})
    report = {"schema_version": 1, "title": title, "sample_label": sample_label,
              "shapes": {"weights": list(w.shape), "inputs": list(x.shape),
                         "output": [x.shape[0], w.shape[0]]},
              "provenance": provenance, "configurations": entries, "ranking_reversals": reversals,
              "warnings": ["Single-layer fake quantization is not a model-accuracy or performance benchmark.",
                           "Results depend on the supplied inputs; weight clipping does not use input calibration.",
                           "Scale counts and bit widths differ; this report does not estimate deployment costs.",
                           "Heatmaps show tile means, not individual weights when dimensions exceed 24 by 64."]}
    try:
        json.dumps(report, allow_nan=False)
    except ValueError as exc:
        raise ValueError("nonfinite report metric; rescale your matrices") from exc
    return report
