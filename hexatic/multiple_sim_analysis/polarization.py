from __future__ import annotations

import gsd.hoomd
import numpy as np

from hexatic.active_matter_cylinder.math_utils import _active_direction_from_quaternion
from hexatic.radii_analysis.cases import RadiusCase

from .best_fit import fit_payload
from .common import (
    FRAME_START,
    FRAME_STOP,
    NPZ_OUTPUT_DIR,
    PLOT_OUTPUT_DIR,
    active_fields_path,
    finite_nanmean,
    frame_indices,
    load_active_fields,
    load_cached_metric_values,
    load_metric_fit_curves,
    radii_for_cases,
    save_metric_npz,
    shell_mask_for_positions,
)
from .numba_kernels import mean_by_population
from .plotting import plot_for_cases, plots_missing


def polarization_values_for_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
) -> dict[str, float]:
    fields_path = active_fields_path(case)
    if not fields_path.exists():
        return _polarization_values_from_gsd(case, frame_start, frame_stop)

    fields = load_active_fields(fields_path)
    px = np.asarray(fields.direction_cylindrical[..., 0], dtype=np.float64)
    shell = np.asarray(fields.shell_mask, dtype=bool)
    all_mean, shell_mean = mean_by_population(px, shell, frame_start, frame_stop)
    return {
        "all": all_mean,
        "shell": shell_mean,
    }


def _polarization_values_from_gsd(
    case: RadiusCase,
    frame_start: int,
    frame_stop: int,
) -> dict[str, float]:
    all_values: list[np.ndarray] = []
    shell_values: list[np.ndarray] = []
    with gsd.hoomd.open(name=str(case.trajectory_gsd), mode="r") as source:
        selected = set(frame_indices(len(source), frame_start, frame_stop).tolist())
        for frame_idx, frame in enumerate(source):
            if frame_idx not in selected:
                continue
            particles = frame.particles
            if particles.position is None or particles.orientation is None:
                continue
            positions = np.asarray(particles.position, dtype=np.float64)
            px = _active_direction_from_quaternion(particles.orientation)[:, 0]
            shell = shell_mask_for_positions(positions, case)
            all_values.append(px)
            shell_values.append(px[shell])

    return {
        "all": finite_nanmean(np.concatenate(all_values)) if all_values else np.nan,
        "shell": finite_nanmean(np.concatenate(shell_values)) if shell_values else np.nan,
    }


def run(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    output_npz = NPZ_OUTPUT_DIR / "polarization.npz"
    output_png = PLOT_OUTPUT_DIR / "polarization.png"
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
        if plots_missing(cases, output_png):
            fits = load_metric_fit_curves(output_npz, value_names)
            plot_for_cases(
                cases,
                arrays,
                output_png,
                title="Mean x polarization vs radius",
                ylabel="mean direction_cylindrical x",
                fits=fits,
            )
        print(f"using cached polarization values from {output_npz}")
        return arrays

    values = {"all": [], "shell": []}
    for case in cases:
        case_values = polarization_values_for_case(case, frame_start, frame_stop)
        values["all"].append(case_values["all"])
        values["shell"].append(case_values["shell"])
    arrays = {name: np.asarray(series, dtype=np.float64) for name, series in values.items()}
    fits, payload = fit_payload(radii_for_cases(cases), arrays)
    save_metric_npz(
        output_npz,
        cases,
        "polarization",
        arrays,
        payload,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    plot_for_cases(
        cases,
        arrays,
        output_png,
        title="Mean x polarization vs radius",
        ylabel="mean direction_cylindrical x",
        fits=fits,
    )
    return arrays
