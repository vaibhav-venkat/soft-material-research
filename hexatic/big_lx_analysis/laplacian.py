"""Python binding for native Laplace-transform analysis and damped-cosine fitting."""

from __future__ import annotations

import ctypes
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._native import (
    F64Buffer,
    UsizeBuffer,
    copy_f64,
    copy_usize,
    encode_paths,
    load_library,
    raise_for_status,
)
from .dynamics import DynamicsResult


@dataclass(frozen=True)
class LaplacianOptions:
    frame_start: int = 0
    frame_stop: int | None = None
    timestep: float = 1.0
    max_lag: int | None = None
    device_ordinal: int = 0
    r_min: float | None = None
    r_max: float = 0.0
    r_points: int = 161
    omega_min: float | None = None
    omega_max: float | None = None
    omega_points: int = 241
    preferred_r_min: float | None = None
    preferred_r_max: float = 0.0
    preferred_r_points: int = 161
    preferred_omega_max: float | None = None
    preferred_omega_points: int = 241
    soft_l1_scale: float = 0.05
    tolerance: float = 1.0e-8
    maximum_evaluations: int = 20_000
    rank_tolerance: float = 1.0e-12


@dataclass(frozen=True)
class PreferredEstimate:
    coordinate: float
    coordinate_std: float
    log10_magnitude: float
    at_lower_boundary: bool
    at_upper_boundary: bool
    replicate_count: int


@dataclass(frozen=True)
class DampedCosineFit:
    amplitude: float
    rate: float
    omega: float
    phase: float
    offset: float
    r_squared: float
    evaluations: int
    converged: bool
    prediction: NDArray[np.float64]


@dataclass(frozen=True)
class LaplacianResult:
    dynamics: DynamicsResult
    r: NDArray[np.float64]
    omega: NDArray[np.float64]
    values: NDArray[np.complex128]
    preferred_r: PreferredEstimate
    preferred_omega: PreferredEstimate
    fit: DampedCosineFit


class _NativeOptions(ctypes.Structure):
    _fields_ = [
        ("frame_start", ctypes.c_size_t), ("frame_stop", ctypes.c_size_t),
        ("has_frame_stop", ctypes.c_bool), ("timestep", ctypes.c_double),
        ("max_lag", ctypes.c_size_t), ("has_max_lag", ctypes.c_bool),
        ("device_ordinal", ctypes.c_uint32), ("r_min", ctypes.c_double),
        ("has_r_min", ctypes.c_bool), ("r_max", ctypes.c_double),
        ("r_points", ctypes.c_size_t), ("omega_min", ctypes.c_double),
        ("has_omega_min", ctypes.c_bool), ("omega_max", ctypes.c_double),
        ("has_omega_max", ctypes.c_bool), ("omega_points", ctypes.c_size_t),
        ("preferred_r_min", ctypes.c_double), ("has_preferred_r_min", ctypes.c_bool),
        ("preferred_r_max", ctypes.c_double), ("preferred_r_points", ctypes.c_size_t),
        ("preferred_omega_max", ctypes.c_double),
        ("has_preferred_omega_max", ctypes.c_bool),
        ("preferred_omega_points", ctypes.c_size_t),
        ("soft_l1_scale", ctypes.c_double), ("tolerance", ctypes.c_double),
        ("maximum_evaluations", ctypes.c_size_t), ("rank_tolerance", ctypes.c_double),
    ]


class _NativePreferred(ctypes.Structure):
    _fields_ = [
        ("axis", ctypes.c_uint8), ("coordinate", ctypes.c_double),
        ("coordinate_std", ctypes.c_double), ("log10_magnitude", ctypes.c_double),
        ("at_lower_boundary", ctypes.c_bool), ("at_upper_boundary", ctypes.c_bool),
        ("replicate_count", ctypes.c_size_t),
    ]


class _NativeFit(ctypes.Structure):
    _fields_ = [
        ("amplitude", ctypes.c_double), ("rate", ctypes.c_double),
        ("omega", ctypes.c_double), ("phase", ctypes.c_double),
        ("offset", ctypes.c_double), ("r_squared", ctypes.c_double),
        ("evaluations", ctypes.c_size_t), ("converged", ctypes.c_bool),
        ("rate_at_lower_boundary", ctypes.c_bool),
        ("rate_at_upper_boundary", ctypes.c_bool),
        ("amplitude_at_upper_boundary", ctypes.c_bool), ("prediction", F64Buffer),
    ]


class _NativeResult(ctypes.Structure):
    _fields_ = [
        ("elapsed_time", F64Buffer), ("center", F64Buffer),
        ("velocity", F64Buffer), ("lag_indices", UsizeBuffer),
        ("lag_times", F64Buffer), ("pearson", F64Buffer),
        ("origin_counts", UsizeBuffer),
        ("r", F64Buffer), ("omega", F64Buffer),
        ("values_real", F64Buffer), ("values_imag", F64Buffer),
        ("shape", ctypes.c_size_t * 2),
        ("preferred_r", _NativePreferred), ("preferred_omega", _NativePreferred),
        ("fit", _NativeFit), ("owner", ctypes.c_void_p),
    ]


def _preferred(value: _NativePreferred) -> PreferredEstimate:
    return PreferredEstimate(
        value.coordinate, value.coordinate_std, value.log10_magnitude,
        bool(value.at_lower_boundary), bool(value.at_upper_boundary),
        value.replicate_count,
    )


def analyze_laplacian(
    static_file: str | Path,
    safetensor_files: str | Path | Sequence[str | Path],
    options: LaplacianOptions = LaplacianOptions(),
) -> LaplacianResult:
    """Run the complete native transform, preferred-coordinate, and fit analysis."""
    library = load_library("laplacian_analysis")
    version = library.laplacian_analysis_api_version
    version.argtypes = []
    version.restype = ctypes.c_uint32
    if version() != 2:
        raise RuntimeError("laplacian_analysis native library has an incompatible ABI")
    run = library.laplacian_analysis_run
    run.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t,
                    ctypes.POINTER(_NativeOptions), ctypes.POINTER(_NativeResult)]
    run.restype = ctypes.c_int
    release = library.laplacian_analysis_release
    release.argtypes = [ctypes.POINTER(_NativeResult)]
    release.restype = None

    paths, _owners = encode_paths(safetensor_files)
    o = options
    native_options = _NativeOptions(
        o.frame_start, o.frame_stop or 0, o.frame_stop is not None, o.timestep,
        o.max_lag or 0, o.max_lag is not None, o.device_ordinal,
        o.r_min or 0.0, o.r_min is not None, o.r_max, o.r_points,
        o.omega_min or 0.0, o.omega_min is not None,
        o.omega_max or 0.0, o.omega_max is not None, o.omega_points,
        o.preferred_r_min or 0.0, o.preferred_r_min is not None,
        o.preferred_r_max, o.preferred_r_points,
        o.preferred_omega_max or 0.0, o.preferred_omega_max is not None,
        o.preferred_omega_points, o.soft_l1_scale, o.tolerance,
        o.maximum_evaluations, o.rank_tolerance,
    )
    native_result = _NativeResult()
    status = run(os.fsencode(Path(static_file)), paths, len(paths),
                 ctypes.byref(native_options), ctypes.byref(native_result))
    raise_for_status(status, "laplacian_analysis")
    try:
        shape = tuple(native_result.shape)
        real = copy_f64(native_result.values_real).reshape(shape)
        imaginary = copy_f64(native_result.values_imag).reshape(shape)
        fit = native_result.fit
        return LaplacianResult(
            dynamics=DynamicsResult(
                elapsed_time=copy_f64(native_result.elapsed_time),
                center=copy_f64(native_result.center),
                velocity=copy_f64(native_result.velocity),
                lag_indices=copy_usize(native_result.lag_indices),
                lag_time=copy_f64(native_result.lag_times),
                pearson=copy_f64(native_result.pearson),
                origin_counts=copy_usize(native_result.origin_counts),
            ),
            r=copy_f64(native_result.r), omega=copy_f64(native_result.omega),
            values=real + 1j * imaginary,
            preferred_r=_preferred(native_result.preferred_r),
            preferred_omega=_preferred(native_result.preferred_omega),
            fit=DampedCosineFit(
                fit.amplitude, fit.rate, fit.omega, fit.phase, fit.offset,
                fit.r_squared, fit.evaluations, bool(fit.converged),
                copy_f64(fit.prediction),
            ),
        )
    finally:
        release(ctypes.byref(native_result))
