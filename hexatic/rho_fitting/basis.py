"""Chebyshev time basis helpers backed by the Rust temporal operator module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _rho_fitting_core, _rho_fitting_core_import_error


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
    """Compute Chebyshev-mode power in Rust over all non-time axes."""
    assert coefficients, "at least one coefficient array is required"
    if _rho_fitting_core is None:
        raise ImportError(f"rho-fitting Rust core is unavailable: {_rho_fitting_core_import_error}")
    return np.asarray(
        _rho_fitting_core.temporal_power_spectrum(
            [np.ascontiguousarray(value, dtype=np.float64) for value in coefficients]
        )
    )


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
    """Apply shared Rust Chebyshev operators to every flattened spatial column.

    The Rust operator construction performs the dense least-squares factorization once,
    repairs non-finite temporal samples in parallel, and evaluates both the retained fit
    and its physical-time derivative. SciPy remains only for the diagnostic cubic-spline
    resampling path.
    """
    values = np.ascontiguousarray(values, dtype=np.float64)
    assert values.ndim >= 1, "values must have a temporal axis"
    frame_count = values.shape[0]
    cutoff = validate_cheb_cutoff(cutoff, frame_count)
    if _rho_fitting_core is None:
        raise ImportError(f"rho-fitting Rust core is unavailable: {_rho_fitting_core_import_error}")
    operators = _rho_fitting_core.TemporalOperators(
        np.ascontiguousarray(steps, dtype=np.int64),
        float(timestep),
        int(cutoff),
    )
    result = operators.apply(values)
    scaled_times = np.asarray(operators.scaled_times())
    return ChebyshevTimeResult(
        filtered=np.ascontiguousarray(np.asarray(result["filtered"])),
        derivative=np.ascontiguousarray(np.asarray(result["derivative"])),
        coefficients=np.ascontiguousarray(np.asarray(result["coefficients"])),
        times=np.asarray(operators.times()),
        scaled_times=scaled_times,
        cutoff=cutoff,
    )


def chebyshev_filter_fields(
    values: tuple[np.ndarray, ...],
    steps: np.ndarray,
    timestep: float,
    cutoff: int,
) -> tuple[ChebyshevTimeResult, ...]:
    """Filter multiple fields with one precomputed temporal operator set."""
    assert values, "at least one field is required"
    arrays = tuple(np.ascontiguousarray(value, dtype=np.float64) for value in values)
    frame_count = arrays[0].shape[0]
    assert all(value.ndim >= 1 and value.shape[0] == frame_count for value in arrays)
    cutoff = validate_cheb_cutoff(cutoff, frame_count)
    if _rho_fitting_core is None:
        raise ImportError(f"rho-fitting Rust core is unavailable: {_rho_fitting_core_import_error}")
    operators = _rho_fitting_core.TemporalOperators(
        np.ascontiguousarray(steps, dtype=np.int64),
        float(timestep),
        int(cutoff),
    )
    payloads = operators.apply_many(list(arrays))
    scaled_times = np.asarray(operators.scaled_times())
    results: list[ChebyshevTimeResult] = []
    for array, payload in zip(arrays, payloads, strict=True):
        results.append(
            ChebyshevTimeResult(
                filtered=np.ascontiguousarray(np.asarray(payload["filtered"])),
                derivative=np.ascontiguousarray(np.asarray(payload["derivative"])),
                coefficients=np.ascontiguousarray(np.asarray(payload["coefficients"])),
                times=np.asarray(operators.times()),
                scaled_times=scaled_times,
                cutoff=cutoff,
            )
        )
    return tuple(results)
