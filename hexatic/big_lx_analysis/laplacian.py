"""Python binding for Laplace-transform analysis and fitting."""

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._native import F64Buffer, copy_f64, encode_paths, load_library, raise_for_status


@dataclass(frozen=True)
class LaplacianOptions:
    r_points: int = 128
    omega_points: int = 256
    r_min: float | None = None
    r_max: float = 0.0
    omega_min: float | None = None
    omega_max: float | None = None


@dataclass(frozen=True)
class LaplacianResult:
    r: NDArray[np.float64]
    omega: NDArray[np.float64]
    values: NDArray[np.complex128]
    preferred_r: float
    preferred_omega: float
    fit_parameters: NDArray[np.float64]


def analyze_laplacian(
    safetensor_files: str | Path | list[str | Path],
    options: LaplacianOptions = LaplacianOptions(),
) -> LaplacianResult:
    """Validate correlation safetensors in Zig and return placeholder arrays."""
    library = load_library("laplacian_analysis")

    class NativeOptions(ctypes.Structure):
        _fields_ = [
            ("r_points", ctypes.c_size_t),
            ("omega_points", ctypes.c_size_t),
            ("r_min", ctypes.c_double),
            ("has_r_min", ctypes.c_bool),
            ("r_max", ctypes.c_double),
            ("omega_min", ctypes.c_double),
            ("has_omega_min", ctypes.c_bool),
            ("omega_max", ctypes.c_double),
            ("has_omega_max", ctypes.c_bool),
        ]

    class NativeFit(ctypes.Structure):
        _fields_ = [(name, ctypes.c_double) for name in (
            "amplitude", "rate", "omega", "phase", "offset", "r_squared"
        )]

    class NativeResult(ctypes.Structure):
        _fields_ = [
            ("r", F64Buffer),
            ("omega", F64Buffer),
            ("values_real", F64Buffer),
            ("values_imag", F64Buffer),
            ("preferred_r", ctypes.c_double),
            ("preferred_omega", ctypes.c_double),
            ("fit", NativeFit),
        ]

    run = library.laplacian_analysis_run
    run.argtypes = [
        ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t,
        ctypes.POINTER(NativeOptions), ctypes.POINTER(NativeResult),
    ]
    run.restype = ctypes.c_int
    release = library.laplacian_analysis_release
    release.argtypes = [ctypes.POINTER(NativeResult)]
    release.restype = None

    paths, _owners = encode_paths(safetensor_files)
    native_options = NativeOptions(
        options.r_points,
        options.omega_points,
        options.r_min or 0.0,
        options.r_min is not None,
        options.r_max,
        options.omega_min or 0.0,
        options.omega_min is not None,
        options.omega_max or 0.0,
        options.omega_max is not None,
    )
    native_result = NativeResult()
    status = run(paths, len(paths), ctypes.byref(native_options), ctypes.byref(native_result))
    raise_for_status(status, "laplacian_analysis")
    try:
        real = copy_f64(native_result.values_real)
        imaginary = copy_f64(native_result.values_imag)
        return LaplacianResult(
            r=copy_f64(native_result.r),
            omega=copy_f64(native_result.omega),
            values=real + 1j * imaginary,
            preferred_r=native_result.preferred_r,
            preferred_omega=native_result.preferred_omega,
            fit_parameters=np.array([
                native_result.fit.amplitude,
                native_result.fit.rate,
                native_result.fit.omega,
                native_result.fit.phase,
                native_result.fit.offset,
                native_result.fit.r_squared,
            ], dtype=np.float64),
        )
    finally:
        release(ctypes.byref(native_result))
