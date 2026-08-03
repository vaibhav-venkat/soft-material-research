"""Reduce detected winding components to physical band observables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .components import BandComponent
from .density import SurfaceGrid


@dataclass(frozen=True)
class DetectedBand:
    """The physical quantities needed after density-based detection."""

    area: float
    axial_center: float
    mask: np.ndarray


def _periodic_axial_center(mask: np.ndarray, grid: SurfaceGrid) -> float:
    axial_cells = np.count_nonzero(mask, axis=1)
    phase = 2.0 * np.pi * (np.arange(grid.nx) + 0.5) / grid.nx
    moment = np.sum(axial_cells * np.exp(1j * phase))
    if abs(moment) < 1.0e-12:
        raise ValueError("band has no unique periodic axial center")
    return float((np.angle(moment) % (2.0 * np.pi)) * grid.lx / (2.0 * np.pi))


def characterize_band(
    component: BandComponent,
    labels: np.ndarray,
    grid: SurfaceGrid,
) -> DetectedBand:
    """Reduce a winding component to area, periodic center, and its mask."""
    mask = np.asarray(labels == component.label, dtype=bool)
    return DetectedBand(
        area=float(component.area_cells * grid.dx * grid.ds),
        axial_center=_periodic_axial_center(mask, grid),
        mask=mask,
    )
