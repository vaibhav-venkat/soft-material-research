"""Python binding for structural crystal-cluster point samples."""

from __future__ import annotations

import ctypes
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hexatic.constants.cylinder import PARTICLE_DIAMETER

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
class ClusterOptions:
    frame_start: int = 0
    frame_stop: int | None = None
    psi6_minimum: float = 0.7
    misorientation_degrees: float = 5.0
    neighbor_radius_diameters: float = 1.7272
    minimum_particles: int = 2
    particle_diameter: float = PARTICLE_DIAMETER


@dataclass(frozen=True)
class ClusterResult:
    """Ragged, periodically unwrapped `(x, r theta)` cluster centers."""

    points: NDArray[np.float64]
    offsets: NDArray[np.intp]

    @property
    def clusters(self) -> tuple[NDArray[np.float64], ...]:
        return tuple(
            self.points[start:stop]
            for start, stop in zip(self.offsets[:-1], self.offsets[1:], strict=True)
        )


class _NativeOptions(ctypes.Structure):
    _fields_ = [
        ("frame_start", ctypes.c_size_t),
        ("frame_stop", ctypes.c_size_t),
        ("has_frame_stop", ctypes.c_bool),
        ("psi6_minimum", ctypes.c_double),
        ("misorientation_degrees", ctypes.c_double),
        ("neighbor_radius_diameters", ctypes.c_double),
        ("minimum_particles", ctypes.c_size_t),
        ("particle_diameter", ctypes.c_double),
    ]


class _NativeResult(ctypes.Structure):
    _fields_ = [
        ("points", F64Buffer),
        ("offsets", UsizeBuffer),
        ("owner", ctypes.c_void_p),
    ]


def analyze_clusters(
    static_file: str | Path,
    safetensor_files: str | Path | Sequence[str | Path],
    options: ClusterOptions = ClusterOptions(),
) -> ClusterResult:
    """Run Zig structural clustering and copy its ragged point samples."""
    library = load_library("cluster_analysis")
    version = library.cluster_analysis_api_version
    version.argtypes = []
    version.restype = ctypes.c_uint32
    if version() != 3:
        raise RuntimeError("cluster_analysis native library has an incompatible ABI")

    run = library.cluster_analysis_run
    run.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_size_t,
        ctypes.POINTER(_NativeOptions),
        ctypes.POINTER(_NativeResult),
    ]
    run.restype = ctypes.c_int
    release = library.cluster_analysis_release
    release.argtypes = [ctypes.POINTER(_NativeResult)]
    release.restype = None

    paths, _owners = encode_paths(safetensor_files)
    static_path = os.fsencode(Path(static_file))
    native_options = _NativeOptions(
        options.frame_start,
        options.frame_stop or 0,
        options.frame_stop is not None,
        options.psi6_minimum,
        options.misorientation_degrees,
        options.neighbor_radius_diameters,
        options.minimum_particles,
        options.particle_diameter,
    )
    native_result = _NativeResult()
    status = run(
        static_path,
        paths,
        len(paths),
        ctypes.byref(native_options),
        ctypes.byref(native_result),
    )
    raise_for_status(status, "cluster_analysis")
    try:
        points = copy_f64(native_result.points).reshape((-1, 2))
        offsets = copy_usize(native_result.offsets)
        if offsets.size == 0 or offsets[0] != 0 or offsets[-1] != points.shape[0]:
            raise RuntimeError("cluster_analysis returned invalid cluster offsets")
        return ClusterResult(points=points, offsets=offsets)
    finally:
        release(ctypes.byref(native_result))
