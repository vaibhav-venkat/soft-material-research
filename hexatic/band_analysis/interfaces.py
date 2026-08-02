from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .density import SurfaceGrid


@dataclass(frozen=True)
class BandInterfaces:
    lower: np.ndarray
    upper: np.ndarray
    centerline: np.ndarray
    width: np.ndarray
    geometrically_complex: bool


def _circular_runs(indices: np.ndarray, size: int) -> list[np.ndarray]:
    present = np.zeros(size, dtype=bool)
    present[indices] = True
    starts = np.flatnonzero(present & ~np.roll(present, 1))
    if not len(starts):
        return [np.arange(size)] if np.any(present) else []
    runs = []
    for start in starts:
        run = [int(start)]
        cursor = (int(start) + 1) % size
        while present[cursor] and cursor != start:
            run.append(cursor)
            cursor = (cursor + 1) % size
        runs.append(np.asarray(run))
    return runs


def extract_interfaces(
    labels: np.ndarray,
    band_label: int,
    grid: SurfaceGrid,
) -> BandInterfaces:
    """Extract axial edges per s column, respecting the periodic x seam."""
    lower = np.full(grid.ns, np.nan)
    upper = np.full(grid.ns, np.nan)
    width = np.full(grid.ns, np.nan)
    centerline = np.full(grid.ns, np.nan)
    complex_geometry = False
    for j in range(grid.ns):
        indices = np.flatnonzero(labels[:, j] == band_label)
        runs = _circular_runs(indices, grid.nx)
        if not runs:
            continue
        if len(runs) > 1:
            complex_geometry = True
        run = max(runs, key=len)
        unwrapped = run.astype(float)
        unwrapped[unwrapped < run[0]] += grid.nx
        lower[j] = -0.5 * grid.lx + unwrapped[0] * grid.dx
        upper[j] = -0.5 * grid.lx + (unwrapped[-1] + 1.0) * grid.dx
        width[j] = len(run) * grid.dx
        centerline[j] = ((0.5 * (lower[j] + upper[j]) + 0.5 * grid.lx) % grid.lx) - 0.5 * grid.lx
    return BandInterfaces(lower, upper, centerline, width, complex_geometry)
