from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActiveMatterFields:
    coords: np.ndarray
    shell_mask: np.ndarray
    steps: np.ndarray
    x_edges: np.ndarray
    theta_edges: np.ndarray
    pocket_radius: float | None = None


def load_active_matter_fields(path: str | Path) -> ActiveMatterFields:
    source = Path(path)
    required = ("coords", "shell_mask", "steps", "x_edges", "theta_edges")
    with np.load(source, allow_pickle=False) as data:
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{source} is missing required arrays: {missing}")
        pocket_radius = None
        if "pocket_radius" in data:
            pocket_radius = float(np.asarray(data["pocket_radius"]))
        fields = ActiveMatterFields(
            coords=np.asarray(data["coords"]),
            shell_mask=np.asarray(data["shell_mask"], dtype=bool),
            steps=np.asarray(data["steps"]),
            x_edges=np.asarray(data["x_edges"], dtype=float),
            theta_edges=np.asarray(data["theta_edges"], dtype=float),
            pocket_radius=pocket_radius,
        )
    _validate_active_matter_fields(fields, source)
    return fields


def _validate_active_matter_fields(fields: ActiveMatterFields, source: Path) -> None:
    if fields.coords.ndim != 3 or fields.coords.shape[2] < 2:
        raise ValueError(f"{source} coords must have shape (frames, particles, >=2).")
    if fields.shell_mask.shape != fields.coords.shape[:2]:
        raise ValueError(
            f"{source} shell_mask shape {fields.shell_mask.shape} does not match "
            f"coords frame/particle shape {fields.coords.shape[:2]}."
        )
    if fields.steps.shape != (fields.coords.shape[0],):
        raise ValueError(f"{source} steps must have one entry per coordinate frame.")
    if fields.x_edges.ndim != 1 or fields.x_edges.size < 2:
        raise ValueError(f"{source} x_edges must be a one-dimensional edge array.")
    if fields.theta_edges.ndim != 1 or fields.theta_edges.size < 2:
        raise ValueError(f"{source} theta_edges must be a one-dimensional edge array.")
    if np.any(np.diff(fields.x_edges) <= 0.0):
        raise ValueError(f"{source} x_edges must be strictly increasing.")
    if np.any(np.diff(fields.theta_edges) <= 0.0):
        raise ValueError(f"{source} theta_edges must be strictly increasing.")


def load_cache(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def write_cache(
    path: str | Path,
    *,
    overwrite: bool = False,
    **arrays: Any,
) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"{destination} already exists; pass overwrite=True to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return destination
