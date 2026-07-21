"""Python binding for structural crystal-cluster ratio samples."""

from __future__ import annotations

import ctypes
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._native import F64Buffer, copy_f64, encode_paths, load_library, raise_for_status


class ClusterRatioMode(IntEnum):
    """Normalization applied to each cluster area."""

    AREA_FRACTION = 0
    SQRT_AREA_FRACTION = 1


@dataclass(frozen=True)
class ClusterOptions:
    frame_start: int = 0
    frame_stop: int | None = None
    psi6_minimum: float = 0.7
    misorientation_degrees: float = 5.0
    neighbor_radius_diameters: float = 1.7272
    minimum_particles: int = 2
    particle_diameter: float = 1.0
    ratio_mode: ClusterRatioMode = ClusterRatioMode.SQRT_AREA_FRACTION


@dataclass(frozen=True)
class ClusterResult:
    """One selected area ratio per structural cluster."""

    ratios: NDArray[np.float64]


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
        ("ratio_mode", ctypes.c_uint8),
    ]


class _NativeResult(ctypes.Structure):
    _fields_ = [("ratios", F64Buffer), ("owner", ctypes.c_void_p)]


def analyze_clusters(
    static_file: str | Path,
    safetensor_files: str | Path | Sequence[str | Path],
    options: ClusterOptions = ClusterOptions(),
) -> ClusterResult:
    """Run Zig structural clustering and copy its ratio samples into NumPy."""
    library = load_library("cluster_analysis")
    version = library.cluster_analysis_api_version
    version.argtypes = []
    version.restype = ctypes.c_uint32
    if version() != 2:
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
        int(options.ratio_mode),
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
        return ClusterResult(ratios=copy_f64(native_result.ratios))
    finally:
        release(ctypes.byref(native_result))
