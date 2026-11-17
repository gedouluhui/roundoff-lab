"""Discovery-frozen selection with separately reported validation evidence."""
from __future__ import annotations
from collections.abc import Mapping
from itertools import combinations
import json
import re

import numpy as np
from .analysis import POLICY, _heatmap, _mse, fingerprint
from .quantize import Config, default_configs, quantize, validate_matrix
from .study_metrics import evaluate

WARNINGS = [
    "Single-layer numerical evidence, not model accuracy, significance or deployment performance.",
    "Validation independence and absence of leakage are not verified; separate files do not establish independence.",
    "External matrices and labels are user-declared; shape equality does not establish coordinate alignment or quantizer identity.",
    "Observed forward error and direct float64 propagated weight error are different numerical measurements.",
    "Repeatedly tuning after viewing validation reuses that data; use fresh inputs before broader conclusions.",
]


def _matrix_shape(array, name):
    if not isinstance(array, np.ndarray) or array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name} must be a nonempty 2D float32/64 matrix")


def study(weights, discovery, validation, *, candidates=None, configs=None, selected=None,
          selection_metric="propagated_mse", batch_rows=256, max_elements=4_000_000,
          max_output_elements=4_000_000, title="Candidate validation study") -> dict:
    """Freeze each candidate, select on discovery only, and report validation.

    candidates maps names to ALREADY-dequantized matrices; nothing is inferred
    about the exporter. configs are generated once per study from base weights.
    """
    if candidates is None:
        candidates = {}
    if not isinstance(candidates, Mapping):
        raise ValueError("candidates must map names to dequantized arrays")
    if not isinstance(title, str) or len(title) > 500:
        raise ValueError("title must be a string of at most 500 characters")
    if selection_metric not in ("propagated_mse", "output_mse"):
        raise ValueError("selection_metric must be propagated_mse or output_mse")
    if type(batch_rows) is not int or not 1 <= batch_rows <= 4096:
        raise ValueError("batch_rows must be an integer from 1 to 4096")
    for name, limit in (("max_elements", max_elements), ("max_output_elements", max_output_elements)):
        if type(limit) is not int or limit < 1:
            raise ValueError(f"{name} must be a positive integer")
    if configs is None:
        configs = [] if candidates else default_configs()
    if not isinstance(configs, (list, tuple)) or any(not isinstance(c, Config) for c in configs):
        raise ValueError("configs must be a list or tuple of Config objects")
    if not 1 <= len(candidates) + len(configs) <= 24:
        raise ValueError("supply between 1 and 24 total candidates/configurations")
    if len({c.id for c in configs}) != len(configs):
        raise ValueError("configuration IDs must be unique")
    for name in candidates:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", name):
            raise ValueError("candidate names must use 1-64 ASCII letters, digits, dot, underscore or hyphen")
    arrays = [("weights", weights), ("discovery", discovery), ("validation", validation)] + list(candidates.items())
    for name, array in arrays:
        _matrix_shape(array, name)
    if sum(a.size for _, a in arrays) > max_elements:
        raise ValueError(f"aggregate input element limit exceeded ({max_elements})")
    if weights.shape[0] > 4096:
        raise ValueError("study supports at most 4096 output channels")
    for name, array in (("discovery", discovery), ("validation", validation)):
        if array.shape[1] != weights.shape[1]:
            raise ValueError(f"{name} input channels differ from weights")
        if array.shape[0] * weights.shape[0] > max_output_elements:
            raise ValueError(f"{name} logical output element limit exceeded ({max_output_elements})")
    for name, array in candidates.items():
        if array.shape != weights.shape:
            raise ValueError(f"candidate {name} shape differs from weights")
    for name, array in arrays:
        validate_matrix(array, name)
    all_ids = [c.id for c in configs] + ['external:' + name for name in candidates]
    if selected is not None and selected not in all_ids:
        raise ValueError("selected must be a known candidate ID")
    w = np.asarray(weights, dtype=np.float64)
    base_hash = fingerprint(weights)
    cohorts = {}
    for name, inputs in (("discovery", discovery), ("validation", validation)):
        cohorts[name] = {"schema_version": 1, "title": title, "sample_label": f"user-declared {name}",
                         "shapes": {"weights": list(w.shape), "inputs": list(inputs.shape),
                                    "output": [len(inputs), w.shape[0]]},
                         "provenance": {"weight_sha256": base_hash, "input_sha256": fingerprint(inputs),
                                        "numpy_version": np.__version__, "numerical_policy": "float64 rowwise matvec; ordered scaled sum of squares; separate forward and propagated errors",
                                        "cohort": name, "batch_rows": batch_rows},
                         "configurations": [], "ranking_reversals": [], "warnings": list(WARNINGS)}
    sources = [(c.id, c, None) for c in configs] + [('external:' + name, None, arr) for name, arr in candidates.items()]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for candidate_id, config, external in sources:
                quantized = quantize(w, config) if config is not None else None
                q = quantized.weights if quantized is not None else np.asarray(external, dtype=np.float64)
                difference = q - w
                weight_mse = float(_mse(difference))
                channel_weight_mse = _mse(difference, axis=1)
                heatmap = _heatmap(np.abs(difference))
                q_hash = fingerprint(q if quantized is not None else external)
                metadata = {"id": candidate_id, "origin": "roundoff_simulation" if config is not None else "external_dequantized",
                            "candidate_sha256": q_hash, "quantization_policy": POLICY if config else None,
                            "bits": config.bits if config else None,
                            "mode": config.mode if config else "external",
                            "group_size": config.group_size if config and config.mode == "group" else None,
                            "percentile": float(config.percentile) if config else None}
                for cohort, inputs in (("discovery", discovery), ("validation", validation)):
                    evaluation = evaluate(w, q, inputs, batch_rows=batch_rows)
                    metrics = {"weight_mse": weight_mse, "weight_max_abs_error": float(np.max(np.abs(difference))),
                               "scale_count": quantized.scale_count if quantized else None,
                               "clipped_count": quantized.clipped_count if quantized else None, **evaluation['metrics']}
                    channels = [dict(c, weight_mse=float(channel_weight_mse[c['index']])) for c in evaluation['channels']]
                    cohorts[cohort]['configurations'].append(dict(metadata, metrics=metrics, channels=channels,
                        heatmap=heatmap, worst_samples=evaluation['worst_samples'],
                        rounding_mask_detected=evaluation['rounding_mask_detected'], execution=evaluation['execution']))
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("study arithmetic overflow; rescale matrices") from exc
    for view in cohorts.values():
        for first, second in combinations(view['configurations'], 2):
            a, b = sorted((first, second), key=lambda c: c['metrics']['weight_mse'])
            if a['metrics']['weight_mse'] < b['metrics']['weight_mse'] and a['metrics']['output_mse'] > b['metrics']['output_mse']:
                view['ranking_reversals'].append({'lower_weight_error': a['id'], 'lower_output_error': b['id']})
    discovery_entries = cohorts['discovery']['configurations']
    selected_id = selected if selected is not None else min(discovery_entries,
        key=lambda c: (c['metrics'][selection_metric], c['id']))['id']
    discovery_value = next(c['metrics'][selection_metric] for c in discovery_entries if c['id'] == selected_id)
    validation_entries = cohorts['validation']['configurations']
    validation_value = next(c['metrics'][selection_metric] for c in validation_entries if c['id'] == selected_id)
    best_value = min(c['metrics'][selection_metric] for c in validation_entries)
    identical = discovery.shape == validation.shape and np.array_equal(discovery, validation)
    warnings = list(WARNINGS)
    if identical:
        warnings.insert(0, "Discovery and validation contain identical numeric values; this is not a holdout check.")
    result = {"schema_version": 2, "report_type": "validation_study", "title": title,
              "cohorts": cohorts, "selection": {"method": "user_declared" if selected is not None else "discovery_minimum",
              "candidate_id": selected_id, "metric": selection_metric, "discovery_value": discovery_value,
              "validation_value": validation_value, "validation_best_ids": sorted(c['id'] for c in validation_entries
              if c['metrics'][selection_metric] == best_value), "excess_validation_metric": validation_value - best_value},
              "cohort_checks": {"identical_values": bool(identical), "independence_verified": False},
              "warnings": warnings}
    try:
        json.dumps(result, allow_nan=False)
    except ValueError as exc:
        raise ValueError("nonfinite study metric; rescale matrices") from exc
    return result
