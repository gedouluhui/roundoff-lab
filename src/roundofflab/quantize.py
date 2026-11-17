"""Explicit, symmetric narrow-range, weight-only fake quantization."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class Config:
    bits: int = 4
    mode: str = "channel"
    group_size: int = 32
    percentile: float = 100.0

    def __post_init__(self) -> None:
        if type(self.bits) is not int or not 2 <= self.bits <= 8:
            raise ValueError("bits must be an integer from 2 to 8")
        if self.mode not in ("tensor", "channel", "group"):
            raise ValueError("mode must be tensor, channel, or group")
        if type(self.group_size) is not int or self.group_size < 1:
            raise ValueError("group_size must be a positive integer")
        if isinstance(self.percentile, bool) or not isinstance(self.percentile, Real):
            raise ValueError("percentile must be a finite number in (0, 100]")
        if not math.isfinite(self.percentile) or not 0 < self.percentile <= 100:
            raise ValueError("percentile must be a finite number in (0, 100]")

    @property
    def id(self) -> str:
        value = f"int{self.bits}-{self.mode}"
        if self.mode == "group":
            value += f"-g{self.group_size}"
        if self.percentile != 100:
            value += f"-p{self.percentile:g}"
        return value


@dataclass(frozen=True)
class Quantized:
    weights: np.ndarray
    scales: np.ndarray
    clipped_count: int

    @property
    def scale_count(self) -> int:
        return int(self.scales.size)


def validate_matrix(value: np.ndarray, name: str) -> None:
    """Inspect before making float64 copies. Only ordinary real float matrices."""
    if not isinstance(value, np.ndarray) or value.ndim != 2 or min(value.shape) < 1:
        raise ValueError(f"{name} must be a nonempty 2D NumPy array")
    if value.dtype.kind != "f" or value.dtype.itemsize not in (4, 8):
        raise ValueError(f"{name} must contain float32 or float64 values")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Infinity")


def quantize(weights: np.ndarray, config: Config) -> Quantized:
    """Return dequantized float64 weights; never mutate the caller's array.

    Scales are flattened in output-row / input-group order. Tensor mode has
    one scale. Percentiles describe weight clipping, not input calibration.
    """
    validate_matrix(weights, "weights")
    if not isinstance(config, Config):
        raise ValueError("config must be a Config")
    source = np.asarray(weights, dtype=np.float64)
    result = np.empty_like(source)
    qmax = 2 ** (config.bits - 1) - 1
    scales: list[float] = []
    clipped_count = 0
    rows, cols = source.shape
    if config.mode == "tensor":
        units = [(slice(None), slice(None))]
    else:
        width = config.group_size if config.mode == "group" else cols
        units = ((slice(r, r + 1), slice(c, min(c + width, cols)))
                 for r in range(rows) for c in range(0, cols, width))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for unit in units:
                block = source[unit]
                magnitude = np.abs(block)
                threshold = float(np.max(magnitude) if config.percentile == 100 else
                                  np.percentile(magnitude, config.percentile, method="linear"))
                scale = threshold / qmax if threshold > 0 else 1.0
                if threshold > 0 and scale == 0:
                    raise ValueError("quantization scale underflow; rescale the input weights")
                clipped_count += int(np.count_nonzero(magnitude > threshold))
                codes = np.rint(np.clip(block, -threshold, threshold) / scale)
                result[unit] = np.clip(codes, -qmax, qmax) * scale
                scales.append(scale)
    except FloatingPointError as exc:
        raise ValueError("quantization arithmetic overflow; rescale the input weights") from exc
    if not np.isfinite(result).all():
        raise ValueError("quantization produced nonfinite weights")
    return Quantized(result, np.asarray(scales, dtype=np.float64), clipped_count)


def default_configs() -> list[Config]:
    return [Config(mode="tensor"), Config(), Config(mode="group"), Config(bits=8),
            Config(percentile=99), Config(percentile=95)]
