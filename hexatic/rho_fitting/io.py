"""Input loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ActiveMatterArrays:
    steps: np.ndarray
    coords: np.ndarray
    shell_mask: np.ndarray
    x_edges: np.ndarray
    x_centers: np.ndarray
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    active_direction: np.ndarray | None
    direction_cylindrical: np.ndarray | None
    flux_cylindrical: np.ndarray | None
    radius: float


@dataclass(frozen=True)
class GsdOrientationArrays:
    steps: np.ndarray
    orientation: np.ndarray


def load_active_matter_npz(path: Path, fallback_radius: float | None = None) -> ActiveMatterArrays:
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path) as data:
        coords = np.asarray(data["coords"], dtype=float)
        shell_mask = np.asarray(data["shell_mask"], dtype=bool)
        radius = _read_radius(data, fallback_radius)
        active_direction = (
            np.asarray(data["active_direction"], dtype=float)
            if "active_direction" in data.files
            else None
        )
        direction_cylindrical = (
            np.asarray(data["direction_cylindrical"], dtype=float)
            if "direction_cylindrical" in data.files
            else None
        )
        flux_cylindrical = (
            np.asarray(data["flux_cylindrical"], dtype=float)
            if "flux_cylindrical" in data.files
            else None
        )
        arrays = ActiveMatterArrays(
            steps=np.asarray(data["steps"]),
            coords=coords,
            shell_mask=shell_mask,
            x_edges=np.asarray(data["x_edges"], dtype=float),
            x_centers=_read_centers(data, "x"),
            theta_edges=np.asarray(data["theta_edges"], dtype=float),
            theta_centers=_read_centers(data, "theta"),
            active_direction=active_direction,
            direction_cylindrical=direction_cylindrical,
            flux_cylindrical=flux_cylindrical,
            radius=radius,
        )
    validate_active_matter_arrays(arrays)
    return arrays


def load_gsd_orientations(path: Path) -> GsdOrientationArrays:
    if not path.exists():
        raise FileNotFoundError(path)

    import gsd.hoomd

    steps: list[int] = []
    orientations: list[np.ndarray] = []
    with gsd.hoomd.open(name=str(path), mode="r") as source:
        for frame in source:
            orientation = frame.particles.orientation
            if orientation is None:
                raise ValueError(f"missing particle orientation in {path}")
            steps.append(int(frame.configuration.step))
            orientations.append(np.asarray(orientation, dtype=float))

    if not orientations:
        raise ValueError(f"empty GSD trajectory: {path}")
    return GsdOrientationArrays(
        steps=np.asarray(steps, dtype=np.int64),
        orientation=np.asarray(orientations, dtype=float),
    )


def validate_step_alignment(active: ActiveMatterArrays, gsd: GsdOrientationArrays | None) -> None:
    if gsd is None:
        return
    if active.steps.shape != gsd.steps.shape or not np.array_equal(active.steps, gsd.steps):
        raise ValueError("GSD and NPZ steps do not align")


def validate_active_matter_arrays(arrays: ActiveMatterArrays) -> None:
    if arrays.coords.ndim != 3 or arrays.coords.shape[-1] != 3:
        raise ValueError("coords must have shape (frames, particles, 3)")
    if arrays.steps.shape != (arrays.coords.shape[0],):
        raise ValueError("steps must match coords frame axis")
    if arrays.shell_mask.shape != arrays.coords.shape[:2]:
        raise ValueError("shell_mask must match coords frame/particle axes")
    if arrays.active_direction is not None and (
        arrays.active_direction.shape[:2] != arrays.coords.shape[:2]
        or arrays.active_direction.shape[-1] != 3
    ):
        raise ValueError("active_direction must match coords frame/particle axes")
    if arrays.direction_cylindrical is not None and (
        arrays.direction_cylindrical.shape[:2] != arrays.coords.shape[:2]
        or arrays.direction_cylindrical.shape[-1] != 3
    ):
        raise ValueError("direction_cylindrical must match coords frame/particle axes")
    if arrays.flux_cylindrical is not None and (
        arrays.flux_cylindrical.shape[:2] != arrays.coords.shape[:2]
        or arrays.flux_cylindrical.shape[-1] != 3
    ):
        raise ValueError("flux_cylindrical must match coords frame/particle axes")
    if arrays.x_edges.ndim != 1 or arrays.theta_edges.ndim != 1:
        raise ValueError("grid edges must be one-dimensional")
    if arrays.x_centers.shape != (arrays.x_edges.size - 1,):
        raise ValueError("x_centers must match x_edges")
    if arrays.theta_centers.shape != (arrays.theta_edges.size - 1,):
        raise ValueError("theta_centers must match theta_edges")
    if arrays.radius <= 0.0:
        raise ValueError("radius must be positive")


def _read_radius(data: np.lib.npyio.NpzFile, fallback_radius: float | None) -> float:
    for name in ("cylinder_radius", "radius"):
        if name in data.files:
            return float(np.asarray(data[name]).reshape(-1)[0])
    if fallback_radius is not None:
        return float(fallback_radius)
    raise ValueError("missing cylinder radius metadata")


def _read_centers(data: np.lib.npyio.NpzFile, axis: str) -> np.ndarray:
    center_name = f"{axis}_centers"
    edge_name = f"{axis}_edges"
    if center_name in data.files:
        return np.asarray(data[center_name], dtype=float)
    edges = np.asarray(data[edge_name], dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])
