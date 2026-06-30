"""Chebyshev time basis helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import chebyshev as cheb


@dataclass(frozen=True)
class ChebyshevTimeResult:
    filtered: np.ndarray
    derivative: np.ndarray
    coefficients: np.ndarray
    times: np.ndarray
    scaled_times: np.ndarray
    cutoff: int


def temporal_power_spectrum(rho_hat: np.ndarray, px_hat: np.ndarray, py_hat: np.ndarray) -> np.ndarray:
    return (
        np.sum(np.abs(rho_hat) ** 2, axis=(1, 2))
        + np.sum(np.abs(px_hat) ** 2, axis=(1, 2))
        + np.sum(np.abs(py_hat) ** 2, axis=(1, 2))
    )


def validate_cheb_cutoff(cutoff: int, frame_count: int) -> int:
    if cutoff < 1:
        raise ValueError("cheb_cutoff must be positive")
    return min(cutoff, frame_count)


def physical_times(steps: np.ndarray, timestep: float) -> np.ndarray:
    steps = np.asarray(steps, dtype=np.float64)
    if steps.ndim != 1 or steps.size == 0:
        raise ValueError("steps must be a non-empty 1D array")
    if timestep <= 0.0:
        raise ValueError("timestep must be positive")
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
    if times.size != frame_count:
        raise ValueError("steps length must match values time axis")

    scaled, half_span = _scaled_times(times)
    coeffs = _fit_coefficients(values, scaled)
    filtered_coeffs = np.zeros_like(coeffs)
    filtered_coeffs[:cutoff] = coeffs[:cutoff]

    flat_coeffs = filtered_coeffs.reshape((frame_count, -1))
    filtered = cheb.chebval(scaled, flat_coeffs).T.reshape(values.shape)
    if frame_count == 1:
        derivative = np.zeros_like(filtered)
    else:
        derivative_coeffs = cheb.chebder(flat_coeffs, axis=0)
        derivative = cheb.chebval(scaled, derivative_coeffs).T.reshape(values.shape) / half_span

    return ChebyshevTimeResult(
        filtered=np.ascontiguousarray(filtered),
        derivative=np.ascontiguousarray(derivative),
        coefficients=coeffs,
        times=times,
        scaled_times=scaled,
        cutoff=cutoff,
    )


def _scaled_times(times: np.ndarray) -> tuple[np.ndarray, float]:
    if times.size == 1:
        return np.zeros_like(times), 1.0
    span = times[-1] - times[0]
    if span <= 0.0:
        raise ValueError("steps must increase over time")
    center = 0.5 * (times[0] + times[-1])
    half_span = 0.5 * span
    return (times - center) / half_span, half_span


def _fit_coefficients(values: np.ndarray, scaled_times: np.ndarray) -> np.ndarray:
    frame_count = values.shape[0]
    flat = values.reshape((frame_count, -1))
    vandermonde = cheb.chebvander(scaled_times, frame_count - 1)
    coeffs, *_ = np.linalg.lstsq(vandermonde, flat, rcond=None)
    return coeffs.reshape((frame_count, *values.shape[1:]))
