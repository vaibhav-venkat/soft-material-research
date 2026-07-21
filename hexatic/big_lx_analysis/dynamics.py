"""Python binding for the Zig center-of-mass dynamics analysis."""

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

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


@dataclass(frozen=True)
class DynamicsOptions:
    frame_start: int = 0
    frame_stop: int | None = None
    timestep: float = 1.0
    max_lag: int | None = None
    device_ordinal: int = 0


@dataclass(frozen=True)
class DynamicsResult:
    elapsed_time: NDArray[np.float64]
    center: NDArray[np.float64]
    velocity: NDArray[np.float64]
    lag_indices: NDArray[np.intp]
    lag_time: NDArray[np.float64]
    pearson: NDArray[np.float64]
    origin_counts: NDArray[np.intp]


class _NativeOptions(ctypes.Structure):
    _fields_ = [
        ("frame_start", ctypes.c_size_t),
        ("frame_stop", ctypes.c_size_t),
        ("has_frame_stop", ctypes.c_bool),
        ("timestep", ctypes.c_double),
        ("max_lag", ctypes.c_size_t),
        ("has_max_lag", ctypes.c_bool),
        ("device_ordinal", ctypes.c_uint32),
    ]


class _NativeResult(ctypes.Structure):
    _fields_ = [
        ("elapsed_time", F64Buffer),
        ("center", F64Buffer),
        ("velocity", F64Buffer),
        ("lag_indices", UsizeBuffer),
        ("lag_times", F64Buffer),
        ("pearson", F64Buffer),
        ("origin_counts", UsizeBuffer),
        ("owner", ctypes.c_void_p),
    ]


def analyze_dynamics(
    static_file: str | Path,
    safetensor_files: str | Path | Sequence[str | Path],
    options: DynamicsOptions = DynamicsOptions(),
) -> DynamicsResult:
    """Run the complete Zig COM and lagged-correlation analysis."""
    library = load_library("dynamics_analysis")
    version = library.dynamics_analysis_api_version
    version.argtypes = []
    version.restype = ctypes.c_uint32
    if version() != 2:
        raise RuntimeError("dynamics_analysis native library has an incompatible ABI")

    run = library.dynamics_analysis_run
    run.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_size_t,
        ctypes.POINTER(_NativeOptions),
        ctypes.POINTER(_NativeResult),
    ]
    run.restype = ctypes.c_int
    release = library.dynamics_analysis_release
    release.argtypes = [ctypes.POINTER(_NativeResult)]
    release.restype = None

    paths, _owners = encode_paths(safetensor_files)
    static_path = os.fsencode(Path(static_file))
    native_options = _NativeOptions(
        options.frame_start,
        options.frame_stop or 0,
        options.frame_stop is not None,
        options.timestep,
        options.max_lag or 0,
        options.max_lag is not None,
        options.device_ordinal,
    )
    native_result = _NativeResult()
    status = run(
        static_path,
        paths,
        len(paths),
        ctypes.byref(native_options),
        ctypes.byref(native_result),
    )
    raise_for_status(status, "dynamics_analysis")
    try:
        return DynamicsResult(
            elapsed_time=copy_f64(native_result.elapsed_time),
            center=copy_f64(native_result.center),
            velocity=copy_f64(native_result.velocity),
            lag_indices=copy_usize(native_result.lag_indices),
            lag_time=copy_f64(native_result.lag_times),
            pearson=copy_f64(native_result.pearson),
            origin_counts=copy_usize(native_result.origin_counts),
        )
    finally:
        release(ctypes.byref(native_result))
