from __future__ import annotations

import numpy as np

from hexatic.radii_analysis.cases import RadiusCase

from ..common import FRAME_START, FRAME_STOP
from .shared import CIRCUMFERENCE_REFERENCE_CASE_ID
from .events.runner import run as run_event_pipeline

SUMMARY_VALUE_NAMES = (
    "defect_track_count",
    "birth_event_count",
    "death_event_count",
    "annihilation_event_count",
)


def run(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    selected_cases = tuple(
        case
        for case in cases
        if not case.case_id.startswith("circ_")
        or case.case_id == CIRCUMFERENCE_REFERENCE_CASE_ID
    )
    if not selected_cases:
        print("skipped disclination_order_fields: no C = 60D or radius cases selected")
        return {name: np.asarray([], dtype=np.float64) for name in SUMMARY_VALUE_NAMES}

    return run_event_pipeline(
        selected_cases,
        frame_start=frame_start,
        frame_stop=frame_stop,
        overwrite=overwrite,
    )
