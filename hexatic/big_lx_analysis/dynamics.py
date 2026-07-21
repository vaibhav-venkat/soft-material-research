"""Python binding for center-of-mass dynamics analysis."""

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._native import F64Buffer, copy_f64, encode_paths, load_library, raise_for_status


@dataclass(frozen=True)
class DynamicsOptions:
    frame_start: int = 0
    frame_stop: int | None = None
    timestep: float = 1.0
    max_lag: int | None = None


@dataclass(frozen=True)
class DynamicsResult:
    elapsed_time: NDArray[np.float64]
    com_velocity: NDArray[np.float64]
    lag_time: NDArray[np.float64]
    pearson: NDArray[np.float64]


def analyze_dynamics(
    safetensor_files: str | Path | list[str | Path],
    options: DynamicsOptions = DynamicsOptions(),
) -> DynamicsResult:
    """Memory-map safetensor shards in Zig and return placeholder analysis arrays."""
    library = load_library("dynamics_analysis")

    class NativeOptions(ctypes.Structure):
        _fields_ = [
            ("frame_start", ctypes.c_size_t),
            ("frame_stop", ctypes.c_size_t),
            ("has_frame_stop", ctypes.c_bool),
            ("timestep", ctypes.c_double),
            ("max_lag", ctypes.c_size_t),
            ("has_max_lag", ctypes.c_bool),
        ]

    class NativeResult(ctypes.Structure):
        _fields_ = [
            ("elapsed_time", F64Buffer),
            ("com_velocity", F64Buffer),
            ("lag_time", F64Buffer),
            ("pearson", F64Buffer),
        ]

    run = library.dynamics_analysis_run
    run.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_size_t,
        ctypes.POINTER(NativeOptions),
        ctypes.POINTER(NativeResult),
    ]
    run.restype = ctypes.c_int
    release = library.dynamics_analysis_release
    release.argtypes = [ctypes.POINTER(NativeResult)]
    release.restype = None

    paths, _owners = encode_paths(safetensor_files)
    native_options = NativeOptions(
        options.frame_start,
        options.frame_stop or 0,
        options.frame_stop is not None,
        options.timestep,
        options.max_lag or 0,
        options.max_lag is not None,
    )
    native_result = NativeResult()
    status = run(paths, len(paths), ctypes.byref(native_options), ctypes.byref(native_result))
    raise_for_status(status, "dynamics_analysis")
    try:
        return DynamicsResult(
            elapsed_time=copy_f64(native_result.elapsed_time),
            com_velocity=copy_f64(native_result.com_velocity),
            lag_time=copy_f64(native_result.lag_time),
            pearson=copy_f64(native_result.pearson),
        )
    finally:
        release(ctypes.byref(native_result))
