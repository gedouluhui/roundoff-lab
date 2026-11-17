"""Synthetic, reproducible inputs; no pretrained models or private data."""
from __future__ import annotations
import numpy as np


def make_demo(seed: int = 7, scenario: str = "quiet-outlier") -> tuple[np.ndarray, np.ndarray]:
    if scenario not in ("quiet-outlier", "active-outlier"):
        raise ValueError("scenario must be quiet-outlier or active-outlier")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 0.15, size=(12, 64))
    weights[:, 0] = np.linspace(5, 8, 12)
    inputs = rng.normal(size=(96, 64))
    # An explicitly designed counterexample, not a sampled model benchmark.
    # The same outlier weights matter only when this input direction is active.
    if scenario == "quiet-outlier":
        inputs[:, 0] = 0
    return weights, inputs
