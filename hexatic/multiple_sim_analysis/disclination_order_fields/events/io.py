from __future__ import annotations

from pathlib import Path

import numpy as np

from hexatic.radii_analysis.cases import RadiusCase

from ...common import NPZ_OUTPUT_DIR, PLOT_OUTPUT_DIR, save_metric_npz

EVENT_OUTPUT_STEM = "disclination_order_fields"
EVENT_NPZ_DIR = NPZ_OUTPUT_DIR / EVENT_OUTPUT_STEM
EVENT_PLOT_DIR = PLOT_OUTPUT_DIR / EVENT_OUTPUT_STEM


def event_metric_npz_path(metric_name: str, case: RadiusCase | None = None) -> Path:
    if case is None:
        return EVENT_NPZ_DIR / f"{metric_name}.npz"
    return EVENT_NPZ_DIR / f"{metric_name}_{case.case_id}.npz"


def event_plot_path(filename: str | Path) -> Path:
    return EVENT_PLOT_DIR / filename


def save_event_metric_npz(
    filename: str | Path,
    cases: tuple[RadiusCase, ...],
    metric_name: str,
    values: dict[str, np.ndarray],
    *,
    frame_start: int,
    frame_stop: int,
) -> None:
    save_metric_npz(
        filename,
        cases,
        metric_name,
        values,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
