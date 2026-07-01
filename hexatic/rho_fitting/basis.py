"""Chebyshev time basis helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class ChebyshevTimeResult:
    filtered: np.ndarray
    derivative: np.ndarray
    coefficients: np.ndarray
    times: np.ndarray
    scaled_times: np.ndarray
    cutoff: int


def temporal_power_spectrum(*coefficients: np.ndarray) -> np.ndarray:
    assert coefficients, "at least one coefficient array is required"
    power = np.zeros(coefficients[0].shape[0], dtype=np.float64)
    for coeff in coefficients:
        axes = tuple(range(1, coeff.ndim))
        power += np.sum(np.abs(coeff) ** 2, axis=axes)
    return power


def validate_cheb_cutoff(cutoff: int, frame_count: int) -> int:
    assert cutoff >= 1, "cheb_cutoff must be positive"
    return min(cutoff, frame_count)


def physical_times(steps: np.ndarray, timestep: float) -> np.ndarray:
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
    values = np.asarray(values, dtype=np.float64)
    frame_count = values.shape[0]
    cutoff = validate_cheb_cutoff(cutoff, frame_count)
    times = physical_times(steps, timestep)
    assert times.size == frame_count, "steps length must match values time axis"

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
    if times.size == 1:
        return np.zeros_like(times), 1.0
    span = times[-1] - times[0]
    assert span > 0.0, "steps must increase over time"
    center = 0.5 * (times[0] + times[-1])
    half_span = 0.5 * span
    return (times - center) / half_span, half_span


def _fit_coefficients(values: np.ndarray, scaled_times: np.ndarray, degree: int) -> np.ndarray:
    frame_count = values.shape[0]
    flat = values.reshape((frame_count, -1))
    vandermonde = cheb.chebvander(scaled_times, degree)
    coeffs, *_ = np.linalg.lstsq(vandermonde, flat, rcond=None)
    return coeffs.reshape((degree + 1, *values.shape[1:]))


def _diagnostic_coefficients(values: np.ndarray, scaled_times: np.ndarray) -> np.ndarray:
    if values.shape[0] == 1:
        return values.copy()
    nodes, node_values = _values_at_chebyshev_lobatto_nodes(values, scaled_times)
    return _fit_coefficients(node_values, nodes, node_values.shape[0] - 1)


def _values_at_chebyshev_lobatto_nodes(
    values: np.ndarray,
    scaled_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = values.shape[0]
    nodes = np.cos(np.pi * np.arange(frame_count) / (frame_count - 1))[::-1]
    flat = values.reshape((frame_count, -1))
    sampled = CubicSpline(scaled_times, flat, axis=0)(nodes)
    return nodes, sampled.reshape(values.shape)
