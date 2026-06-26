from __future__ import annotations

import gsd.hoomd
import numpy as np

from hexatic.radii_analysis.cases import RadiusCase

from .best_fit import fit_payload
from .common import (
    FRAME_START,
    FRAME_STOP,
    NPZ_OUTPUT_DIR,
    active_fields_path,
    center_of_mass_x,
    center_of_mass_x_from_coords,
    finite_nanmean,
    frame_indices,
    load_active_fields,
    load_cached_metric_values,
    radii_for_cases,
    save_metric_npz,
    shell_mask_for_positions,
)


def x_com_values_for_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
) -> dict[str, float]:
    fields_path = active_fields_path(case)
    if fields_path.exists():
        return _x_com_values_from_active_fields(case, frame_start, frame_stop)

    all_values: list[float] = []
    shell_values: list[float] = []
    with gsd.hoomd.open(name=str(case.trajectory_gsd), mode="r") as source:
        selected = set(frame_indices(len(source), frame_start, frame_stop).tolist())
        for frame_idx, frame in enumerate(source):
            if frame_idx not in selected:
                continue
            positions = np.asarray(frame.particles.position, dtype=np.float64)
            box_length_x = float(frame.configuration.box[0])
            all_values.append(center_of_mass_x(positions, box_length_x))
            shell = shell_mask_for_positions(positions, case)
            shell_values.append(center_of_mass_x(positions[shell], box_length_x))
    return {
        "all": finite_nanmean(np.asarray(all_values, dtype=np.float64)),
        "shell": finite_nanmean(np.asarray(shell_values, dtype=np.float64)),
    }


def _x_com_values_from_active_fields(
    case: RadiusCase,
    frame_start: int,
    frame_stop: int,
) -> dict[str, float]:
    fields = load_active_fields(active_fields_path(case))
    selected = frame_indices(len(fields.steps), frame_start, frame_stop)
    box_length_x = float(fields.x_edges[-1] - fields.x_edges[0])
    all_values = []
    shell_values = []
    for frame_idx in selected:
        coords = np.asarray(fields.coords[frame_idx], dtype=np.float64)
        shell = np.asarray(fields.shell_mask[frame_idx], dtype=bool)
        all_values.append(center_of_mass_x_from_coords(coords, box_length_x))
        shell_values.append(center_of_mass_x_from_coords(coords[shell], box_length_x))
    return {
        "all": finite_nanmean(np.asarray(all_values, dtype=np.float64)),
        "shell": finite_nanmean(np.asarray(shell_values, dtype=np.float64)),
    }


def run(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    output_npz = NPZ_OUTPUT_DIR / "x_com.npz"
    value_names = ("all", "shell")
    arrays = load_cached_metric_values(
        output_npz,
        value_names,
        cases,
        frame_start,
        frame_stop,
        overwrite=overwrite,
    )
    if arrays is not None:
        print(f"using cached x_com values from {output_npz}")
        return arrays

    values = {"all": [], "shell": []}
    for case in cases:
        case_values = x_com_values_for_case(case, frame_start, frame_stop)
        values["all"].append(case_values["all"])
        values["shell"].append(case_values["shell"])
    arrays = {name: np.asarray(series, dtype=np.float64) for name, series in values.items()}
    _, payload = fit_payload(radii_for_cases(cases), arrays)
    save_metric_npz(
        output_npz,
        cases,
        "x_com",
        arrays,
        payload,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    return arrays
