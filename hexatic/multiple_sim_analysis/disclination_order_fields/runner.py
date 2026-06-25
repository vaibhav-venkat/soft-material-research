from __future__ import annotations

import numpy as np

from hexatic.radii_analysis.cases import RadiusCase

from ..common import FRAME_START, FRAME_STOP
from .shared import CIRCUMFERENCE_REFERENCE_CASE_ID
from .shell_profiles import run_shell_profiles


def run(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    value_names = (
        "S_shell_profile",
        "hexatic_order_shell_profile",
        "chirality_shell_profile",
        "chirality_annulus",
    )
    cases = tuple(
        case
        for case in cases
        if not case.case_id.startswith("circ_")
        or case.case_id == CIRCUMFERENCE_REFERENCE_CASE_ID
    )
    if not cases:
        print("skipped disclination_order_fields: no C = 60D or radius cases selected")
        return {name: np.asarray([], dtype=np.float64) for name in value_names}

    return run_shell_profiles(
        cases,
        frame_start=frame_start,
        frame_stop=frame_stop,
        overwrite=overwrite,
    )
