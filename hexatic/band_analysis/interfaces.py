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


def _unwrap_finite(values: np.ndarray, period: float) -> np.ndarray:
    """Put finite samples on a continuous periodic image in array order."""
    result = values.copy()
    finite = np.flatnonzero(np.isfinite(result))
    if not finite.size:
        return result
    previous = float(result[finite[0]])
    for index in finite[1:]:
        value = float(result[index])
        delta = (value - previous + 0.5 * period) % period - 0.5 * period
        previous += delta
        result[index] = previous
    return result


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
    raw_lower = np.full(grid.ns, np.nan)
    raw_upper = np.full(grid.ns, np.nan)
    width = np.full(grid.ns, np.nan)
    wrapped_centerline = np.full(grid.ns, np.nan)
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
        raw_lower[j] = -0.5 * grid.lx + unwrapped[0] * grid.dx
        raw_upper[j] = -0.5 * grid.lx + (unwrapped[-1] + 1.0) * grid.dx
        width[j] = len(run) * grid.dx
        raw_center = 0.5 * (raw_lower[j] + raw_upper[j])
        wrapped_centerline[j] = (
            (raw_center + 0.5 * grid.lx) % grid.lx - 0.5 * grid.lx
        )

    centerline = _unwrap_finite(wrapped_centerline, grid.lx)
    image_shift = centerline - wrapped_centerline
    # First move seam-crossing runs onto the image of their wrapped center,
    # then apply the same spatial unwrapping shift to both interfaces.
    raw_centerline = 0.5 * (raw_lower + raw_upper)
    base_shift = wrapped_centerline - raw_centerline
    lower = raw_lower + base_shift + image_shift
    upper = raw_upper + base_shift + image_shift
    return BandInterfaces(lower, upper, centerline, width, complex_geometry)
