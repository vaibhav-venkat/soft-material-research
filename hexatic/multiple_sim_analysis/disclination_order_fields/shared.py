from __future__ import annotations

from pathlib import Path

import numpy as np

from hexatic.constants import cylinder
from hexatic.radii_analysis.cases import HEXATIC_OUTPUT_DIR, RadiusCase

from ..common import (
    FRAME_START,
    FRAME_STOP,
    active_fields_path,
    load_active_fields,
    neighbor_counts_path,
)
from ..disclination import _load_neighbor_counts


CIRCUMFERENCE_REFERENCE_CASE_ID = "circ_60_0D"
LOCAL_CONTRAST_LENGTH = cylinder.ANALYSIS.neighbor_count_radius
LOCAL_PROFILE_BIN_EDGES = LOCAL_CONTRAST_LENGTH * np.arange(6, dtype=np.float64)
LOCAL_PROFILE_BIN_LABELS = np.asarray(
    ("< a", "a-2a", "2a-3a", "3a-4a", "4a-5a")
)
LOCAL_PROFILE_COLORS = ("#111111", "#0072b2", "#d55e00", "#009e73", "#cc0000")


def hexatic_order_path(case: RadiusCase) -> Path:
    return HEXATIC_OUTPUT_DIR / f"{case.case_id}_hexatic_order.txt"


def _load_hexatic_abs(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    table = np.loadtxt(path, dtype=np.float64)
    if table.ndim == 1:
        table = table[np.newaxis, :]
    if table.shape[1] < 6:
        raise ValueError(f"Hexatic order table is missing columns: {path}")

    frame_indices_table = table[:, 0].astype(np.int64)
    particle_indices = table[:, 2].astype(np.int64)
    if np.any(frame_indices_table >= shape[0]) or np.any(particle_indices >= shape[1]):
        raise ValueError(f"Hexatic order table does not match expected shape: {path}")

    psi_abs = np.full(shape, np.nan, dtype=np.float64)
    psi_abs[frame_indices_table, particle_indices] = table[:, 5]
    return psi_abs


def _disclination_mask(neighbor_counts: np.ndarray) -> np.ndarray:
    charges = cylinder.NEIGHBORS - neighbor_counts
    return np.abs(charges) == 1


def _validate_particle_frame_shape(
    name: str,
    values: np.ndarray,
    expected_shape: tuple[int, int],
) -> None:
    if values.shape[:2] != expected_shape:
        raise ValueError(
            f"{name} shape {values.shape[:2]} does not match "
            f"neighbor-count shape {expected_shape}."
        )


