"""Time interpolation helpers for validation caches."""

from __future__ import annotations

import numpy as np

from ..cache import ValidationInputs
from .types import Array


def interpolation_index_weight(times: Array, t: float) -> tuple[int, float]:
    """Return bracketing time index and linear interpolation weight for ``t``."""
    index = int(np.searchsorted(times, t, side="right") - 1)
    index = max(0, min(index, times.size - 2))
    t0 = float(times[index])
    t1 = float(times[index + 1])
    weight = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    return index, weight


def interpolate_time_series(values: Array, times: Array, t: float) -> Array:
    """Linearly interpolate a time-indexed array at physical time ``t``."""
    index, weight = interpolation_index_weight(times, t)
    return (1.0 - weight) * values[index] + weight * values[index + 1]


def interpolated_fields(inputs: ValidationInputs, t: float) -> tuple[Array, Array, Array]:
    """Return interpolated cached rho, P, and Q fields at physical time ``t``."""
    rho, p, q, _, _ = interpolated_cached_fields(inputs, t)
    return rho, p, q


def interpolated_cached_fields(inputs: ValidationInputs, t: float) -> tuple[Array, Array, Array, Array, Array]:
    """Return interpolated cached rho, P, Q, A, and Y_P fields at physical time ``t``."""
    rho = interpolate_time_series(inputs.rho, inputs.times, t)
    p = interpolate_time_series(inputs.p, inputs.times, t)
    q = interpolate_time_series(inputs.q, inputs.times, t)
    a = interpolate_time_series(inputs.a, inputs.times, t)
    y_p = interpolate_time_series(inputs.y_p, inputs.times, t)
    return rho, p, q, a, y_p


def interpolated_p_q(inputs: ValidationInputs, t: float) -> tuple[Array, Array]:
    """Return interpolated cached P and Q fields at physical time ``t``."""
    _, p, q = interpolated_fields(inputs, t)
    return p, q
