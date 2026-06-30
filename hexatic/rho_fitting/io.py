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
    theta_edges: np.ndarray
    active_direction: np.ndarray | None
    radius: float


def load_active_matter_npz(path: Path) -> ActiveMatterArrays:
    with np.load(path) as data:
        coords = np.asarray(data["coords"], dtype=float)
        shell_mask = np.asarray(data["shell_mask"], dtype=bool)
        radius = _read_radius(data)
        active_direction = (
            np.asarray(data["active_direction"], dtype=float)
            if "active_direction" in data.files
            else None
        )
        arrays = ActiveMatterArrays(
            steps=np.asarray(data["steps"]),
            coords=coords,
            shell_mask=shell_mask,
            x_edges=np.asarray(data["x_edges"], dtype=float),
            theta_edges=np.asarray(data["theta_edges"], dtype=float),
            active_direction=active_direction,
            radius=radius,
        )
    validate_active_matter_arrays(arrays)
    return arrays


def validate_active_matter_arrays(arrays: ActiveMatterArrays) -> None:
    if arrays.coords.ndim != 3 or arrays.coords.shape[-1] != 3:
        raise ValueError("coords must have shape (frames, particles, 3)")
    if arrays.shell_mask.shape != arrays.coords.shape[:2]:
        raise ValueError("shell_mask must match coords frame/particle axes")
    if arrays.active_direction is not None and arrays.active_direction.shape[:2] != arrays.coords.shape[:2]:
        raise ValueError("active_direction must match coords frame/particle axes")
    if arrays.x_edges.ndim != 1 or arrays.theta_edges.ndim != 1:
        raise ValueError("grid edges must be one-dimensional")


def _read_radius(data: np.lib.npyio.NpzFile) -> float:
    for name in ("cylinder_radius", "radius"):
        if name in data.files:
            return float(np.asarray(data[name]).reshape(-1)[0])
    raise ValueError("missing cylinder radius metadata")
