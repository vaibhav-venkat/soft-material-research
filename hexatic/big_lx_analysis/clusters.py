"""Python binding for cluster identification."""

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._native import UsizeBuffer, copy_usize, encode_paths, load_library, raise_for_status


@dataclass(frozen=True)
class ClusterOptions:
    frame_start: int = 0
    frame_stop: int | None = None
    lag_frames: int = 1
    minimum_particles: int = 2
    psi6_minimum: float = 0.7
    motion_cosine_minimum: float = 0.8


@dataclass(frozen=True)
class ClusterResult:
    structural_offsets: NDArray[np.intp]
    structural_particles: NDArray[np.intp]
    structural_frames: NDArray[np.intp]
    motion_offsets: NDArray[np.intp]
    motion_particles: NDArray[np.intp]
    motion_frames: NDArray[np.intp]


def analyze_clusters(
    safetensor_files: str | Path | list[str | Path],
    options: ClusterOptions = ClusterOptions(),
) -> ClusterResult:
    """Validate safetensors in Zig and return empty CSR cluster memberships."""
    library = load_library("cluster_analysis")

    class NativeOptions(ctypes.Structure):
        _fields_ = [
            ("frame_start", ctypes.c_size_t),
            ("frame_stop", ctypes.c_size_t),
            ("has_frame_stop", ctypes.c_bool),
            ("lag_frames", ctypes.c_size_t),
            ("minimum_particles", ctypes.c_size_t),
            ("psi6_minimum", ctypes.c_double),
            ("motion_cosine_minimum", ctypes.c_double),
        ]

    class NativeMembership(ctypes.Structure):
        _fields_ = [
            ("offsets", UsizeBuffer),
            ("particle_indices", UsizeBuffer),
            ("frame_indices", UsizeBuffer),
        ]

    class NativeResult(ctypes.Structure):
        _fields_ = [("structural", NativeMembership), ("motion", NativeMembership)]

    run = library.cluster_analysis_run
    run.argtypes = [
        ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t,
        ctypes.POINTER(NativeOptions), ctypes.POINTER(NativeResult),
    ]
    run.restype = ctypes.c_int
    release = library.cluster_analysis_release
    release.argtypes = [ctypes.POINTER(NativeResult)]
    release.restype = None

    paths, _owners = encode_paths(safetensor_files)
    native_options = NativeOptions(
        options.frame_start,
        options.frame_stop or 0,
        options.frame_stop is not None,
        options.lag_frames,
        options.minimum_particles,
        options.psi6_minimum,
        options.motion_cosine_minimum,
    )
    native_result = NativeResult()
    status = run(paths, len(paths), ctypes.byref(native_options), ctypes.byref(native_result))
    raise_for_status(status, "cluster_analysis")
    try:
        return ClusterResult(
            structural_offsets=copy_usize(native_result.structural.offsets),
            structural_particles=copy_usize(native_result.structural.particle_indices),
            structural_frames=copy_usize(native_result.structural.frame_indices),
            motion_offsets=copy_usize(native_result.motion.offsets),
            motion_particles=copy_usize(native_result.motion.particle_indices),
            motion_frames=copy_usize(native_result.motion.frame_indices),
        )
    finally:
        release(ctypes.byref(native_result))
