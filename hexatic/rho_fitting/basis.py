"""Chebyshev time basis helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class ChebyshevTimeResult:
    """Container for filtered time series, time derivatives, and diagnostic coefficients."""

    filtered: np.ndarray
    derivative: np.ndarray
    coefficients: np.ndarray
    times: np.ndarray
    scaled_times: np.ndarray
    cutoff: int


def temporal_power_spectrum(*coefficients: np.ndarray) -> np.ndarray:
    """Compute Chebyshev-mode power summed over all non-time axes.

    Parameters:
        coefficients: One or more coefficient arrays with time/mode as axis 0.

    Returns:
        One-dimensional power array with length equal to the shared coefficient count.
    """
    assert coefficients, "at least one coefficient array is required"
    power = np.zeros(coefficients[0].shape[0], dtype=np.float64)
    for coeff in coefficients:
        axes = tuple(range(1, coeff.ndim))
        power += np.sum(np.abs(coeff) ** 2, axis=axes)
    return power


def validate_cheb_cutoff(cutoff: int, frame_count: int) -> int:
    """Clamp a requested Chebyshev cutoff to the available frame count."""
    assert cutoff >= 1, "cheb_cutoff must be positive"
    return min(cutoff, frame_count)


def physical_times(steps: np.ndarray, timestep: float) -> np.ndarray:
    """Convert integer simulation steps to elapsed physical times."""
    steps = np.asarray(steps, dtype=np.float64)
    assert steps.ndim == 1 and steps.size > 0, "steps must be a non-empty 1D array"
    assert timestep > 0.0, "timestep must be positive"
    return (steps - steps[0]) * timestep


def chebyshev_filter_and_derivative(
    values: np.ndarray,
    steps: np.ndarray,
    timestep: float,
    cutoff: int,
) -> ChebyshevTimeResult:
    """Smooth a time series with a Chebyshev fit and evaluate its time derivative.

    Parameters:
        values: Time-indexed array whose axis 0 is frames; remaining axes are fitted
            independently with shared time coordinates.
        steps: Simulation step numbers aligned with ``values`` axis 0.
        timestep: Physical time represented by one simulation step.
        cutoff: Number of low-order Chebyshev modes retained in the filtered signal.

    Returns:
        ``ChebyshevTimeResult`` containing filtered values, physical-time derivatives,
        diagnostic full-spectrum coefficients, and the scaled time coordinates.

    Examples:
        ``result = chebyshev_filter_and_derivative(rho, steps, timestep, cutoff=10)``

    Edge cases:
        A single frame returns a zero derivative; repeated or decreasing steps are rejected.
    """
    values = np.asarray(values, dtype=np.float64)
    frame_count = values.shape[0]
    cutoff = validate_cheb_cutoff(cutoff, frame_count)
    times = physical_times(steps, timestep)
    assert times.size == frame_count, "steps length must match values time axis"
    values = _fill_nonfinite_time_values(values, times)

    scaled, half_span = _scaled_times(times)
    fit_coeffs = _fit_coefficients(values, scaled, cutoff - 1)
    diagnostic_coeffs = _diagnostic_coefficients(values, scaled)

    flat_coeffs = fit_coeffs.reshape((cutoff, -1))
    filtered = cheb.chebval(scaled, flat_coeffs).T.reshape(values.shape)
    if frame_count == 1:
        derivative = np.zeros_like(filtered)
    else:
        derivative_coeffs = cheb.chebder(flat_coeffs, axis=0)
        derivative = cheb.chebval(scaled, derivative_coeffs).T.reshape(values.shape) / half_span

    return ChebyshevTimeResult(
        filtered=np.ascontiguousarray(filtered),
        derivative=np.ascontiguousarray(derivative),
        coefficients=diagnostic_coeffs,
        times=times,
        scaled_times=scaled,
        cutoff=cutoff,
    )


def _scaled_times(times: np.ndarray) -> tuple[np.ndarray, float]:
    """Map physical times onto ``[-1, 1]`` and return the physical half-span."""
    if times.size == 1:
        return np.zeros_like(times), 1.0
    span = times[-1] - times[0]
    assert span > 0.0, "steps must increase over time"
    center = 0.5 * (times[0] + times[-1])
    half_span = 0.5 * span
    return (times - center) / half_span, half_span


def _fit_coefficients(values: np.ndarray, scaled_times: np.ndarray, degree: int) -> np.ndarray:
    """Fit Chebyshev coefficients independently for each non-time array element."""
    frame_count = values.shape[0]
    flat = values.reshape((frame_count, -1))
    vandermonde = cheb.chebvander(scaled_times, degree)
    coeffs, *_ = np.linalg.lstsq(vandermonde, flat, rcond=None)
    return coeffs.reshape((degree + 1, *values.shape[1:]))


def _fill_nonfinite_time_values(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Replace non-finite samples by temporal interpolation before spectral fitting."""
    if np.all(np.isfinite(values)):
        return values
    frame_count = values.shape[0]
    flat = values.reshape((frame_count, -1)).copy()
    for column in range(flat.shape[1]):
        series = flat[:, column]
        finite = np.isfinite(series)
        if finite.all():
            continue
        if not finite.any():
            series[:] = 0.0
        elif finite.sum() == 1:
            series[~finite] = series[finite][0]
        else:
            series[~finite] = np.interp(times[~finite], times[finite], series[finite])
    return flat.reshape(values.shape)


def _diagnostic_coefficients(values: np.ndarray, scaled_times: np.ndarray) -> np.ndarray:
    """Compute full-order coefficients after resampling onto Chebyshev-Lobatto nodes."""
    if values.shape[0] == 1:
        return values.copy()
    nodes, node_values = _values_at_chebyshev_lobatto_nodes(values, scaled_times)
    return _fit_coefficients(node_values, nodes, node_values.shape[0] - 1)


def _values_at_chebyshev_lobatto_nodes(
    values: np.ndarray,
    scaled_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate time-series values onto Chebyshev-Lobatto nodes in scaled time."""
    frame_count = values.shape[0]
    nodes = np.cos(np.pi * np.arange(frame_count) / (frame_count - 1))[::-1]
    flat = values.reshape((frame_count, -1))
    sampled = CubicSpline(scaled_times, flat, axis=0)(nodes)
    return nodes, sampled.reshape(values.shape)
