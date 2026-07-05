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
            assert orientation is not None, f"missing particle orientation in {path}"
            steps.append(int(frame.configuration.step))
            orientations.append(np.asarray(orientation, dtype=float))

    assert orientations, f"empty GSD trajectory: {path}"
    return GsdOrientationArrays(
        steps=np.asarray(steps, dtype=np.int64),
        orientation=np.asarray(orientations, dtype=float),
    )


def validate_step_alignment(active: ActiveMatterArrays, gsd: GsdOrientationArrays | None) -> None:
    if gsd is None:
        return
    assert active.steps.shape == gsd.steps.shape and np.array_equal(active.steps, gsd.steps), (
        "GSD and NPZ steps do not align"
    )


def validate_active_matter_arrays(arrays: ActiveMatterArrays) -> None:
    assert arrays.coords.ndim == 3 and arrays.coords.shape[-1] == 3, "coords must have shape (frames, particles, 3)"
    assert arrays.steps.shape == (arrays.coords.shape[0],), "steps must match coords frame axis"
    assert arrays.shell_mask.shape == arrays.coords.shape[:2], "shell_mask must match coords frame/particle axes"
    _validate_optional_vectors(arrays)
    _validate_grid_axes(arrays)
    assert arrays.radius > 0.0, "radius must be positive"


def _validate_optional_vectors(arrays: ActiveMatterArrays) -> None:
    optional_vectors = {
        "active_direction": arrays.active_direction,
        "direction_cylindrical": arrays.direction_cylindrical,
        "flux_cylindrical": arrays.flux_cylindrical,
    }
    for name, values in optional_vectors.items():
        assert _optional_vector_matches(values, arrays.coords), f"{name} must match coords frame/particle axes"


def _validate_grid_axes(arrays: ActiveMatterArrays) -> None:
    assert arrays.x_edges.ndim == 1 and arrays.theta_edges.ndim == 1, "grid edges must be one-dimensional"
    assert arrays.x_centers.shape == (arrays.x_edges.size - 1,), "x_centers must match x_edges"
    assert arrays.theta_centers.shape == (arrays.theta_edges.size - 1,), "theta_centers must match theta_edges"


def _read_radius(data: np.lib.npyio.NpzFile, fallback_radius: float | None) -> float:
    for name in ("cylinder_radius", "radius"):
        if name in data.files:
            return float(np.asarray(data[name]).reshape(-1)[0])
    if fallback_radius is not None:
        return float(fallback_radius)
    assert False, "missing cylinder radius metadata"


def _read_centers(data: np.lib.npyio.NpzFile, axis: str) -> np.ndarray:
    center_name = f"{axis}_centers"
    edge_name = f"{axis}_edges"
    if center_name in data.files:
        return np.asarray(data[center_name], dtype=float)
    edges = np.asarray(data[edge_name], dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def _optional_vector_matches(values: np.ndarray | None, coords: np.ndarray) -> bool:
    return values is None or (values.shape[:2] == coords.shape[:2] and values.shape[-1] == 3)
