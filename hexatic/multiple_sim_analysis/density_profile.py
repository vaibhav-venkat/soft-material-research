from __future__ import annotations

import math

import gsd.hoomd
import numpy as np

from hexatic.constants import cylinder
from hexatic.radii_analysis.cases import RadiusCase

from .best_fit import fit_payload
from .common import (
    FRAME_START,
    FRAME_STOP,
    NPZ_OUTPUT_DIR,
    PLOT_OUTPUT_DIR,
    active_fields_path,
    frame_indices,
    load_active_fields,
    load_cached_metric_values,
    load_metric_fit_curves,
    radii_for_cases,
    save_metric_npz,
    shell_mask_for_positions,
)
from .numba_kernels import shell_core_density_means
from .plotting import plot_for_cases


def shell_core_volumes(case: RadiusCase, box_length_x: float) -> tuple[float, float]:
    inner_radius = max(0.0, case.radius - cylinder.ANALYSIS.wall_cutoff)
    shell_area = math.pi * (case.radius**2 - inner_radius**2)
    core_area = math.pi * inner_radius**2
    return shell_area * box_length_x, core_area * box_length_x


def density_profile_values_for_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
) -> dict[str, float]:
    fields_path = active_fields_path(case)
    if fields_path.exists():
        return _density_profile_values_from_active_fields(case, frame_start, frame_stop)
    return _density_profile_values_from_gsd(case, frame_start, frame_stop)


def _density_profile_values_from_active_fields(
    case: RadiusCase,
    frame_start: int,
    frame_stop: int,
) -> dict[str, float]:
    fields = load_active_fields(active_fields_path(case))
    shell = np.asarray(fields.shell_mask, dtype=bool)
    box_length_x = float(fields.x_edges[-1] - fields.x_edges[0])
    shell_volume, core_volume = shell_core_volumes(case, box_length_x)
    return _density_values_from_shell_counts(
        shell,
        shell_volume,
        core_volume,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )


def _density_profile_values_from_gsd(
    case: RadiusCase,
    frame_start: int,
    frame_stop: int,
) -> dict[str, float]:
    shell_masks: list[np.ndarray] = []
    box_length_x: float | None = None
    with gsd.hoomd.open(name=str(case.trajectory_gsd), mode="r") as source:
        selected = set(frame_indices(len(source), frame_start, frame_stop).tolist())
        for frame_idx, frame in enumerate(source):
            if frame_idx not in selected:
                continue
            positions = np.asarray(frame.particles.position, dtype=np.float64)
            box_length_x = float(frame.configuration.box[0])
            shell_masks.append(shell_mask_for_positions(positions, case))
    if box_length_x is None:
        return {"shell": np.nan, "core": np.nan, "shell_minus_core": np.nan}
    shell_volume, core_volume = shell_core_volumes(case, box_length_x)
    return _density_values_from_shell_counts(shell_masks, shell_volume, core_volume)


def _density_values_from_shell_counts(
    shell_masks,
    shell_volume: float,
    core_volume: float,
    frame_start: int = 0,
    frame_stop: int | None = None,
) -> dict[str, float]:
    shell = np.asarray(shell_masks, dtype=bool)
    if shell.ndim == 1:
        shell = shell[np.newaxis, :]
    stop = shell.shape[0] if frame_stop is None else frame_stop
    shell_mean, core_mean, diff_mean = shell_core_density_means(
        shell,
        frame_start,
        stop,
        float(shell_volume),
        float(core_volume),
    )
    return {
        "shell": shell_mean,
        "core": core_mean,
        "shell_minus_core": diff_mean,
    }


def run(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    output_npz = NPZ_OUTPUT_DIR / "density_profile.npz"
    output_png = PLOT_OUTPUT_DIR / "density_profile.png"
    value_names = ("shell_minus_core", "shell", "core")
    arrays = load_cached_metric_values(
        output_npz,
        value_names,
        cases,
        frame_start,
        frame_stop,
        overwrite=overwrite,
    )
    if arrays is not None:
        if not output_png.exists():
            fits = load_metric_fit_curves(output_npz, value_names)
            plot_for_cases(
                cases,
                {"shell_minus_core": arrays["shell_minus_core"]},
                output_png,
                title="Shell-core density difference vs radius",
                ylabel=r"$\rho_{shell} - \rho_{core}$",
                fits={"shell_minus_core": fits["shell_minus_core"]},
            )
        print(f"using cached density_profile values from {output_npz}")
        return arrays

    values = {"shell_minus_core": [], "shell": [], "core": []}
    for case in cases:
        case_values = density_profile_values_for_case(case, frame_start, frame_stop)
        values["shell_minus_core"].append(case_values["shell_minus_core"])
        values["shell"].append(case_values["shell"])
        values["core"].append(case_values["core"])
    arrays = {name: np.asarray(series, dtype=np.float64) for name, series in values.items()}
    fits, payload = fit_payload(radii_for_cases(cases), arrays)
    save_metric_npz(
        output_npz,
        cases,
        "density_profile",
        arrays,
        payload,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    plot_for_cases(
        cases,
        {"shell_minus_core": arrays["shell_minus_core"]},
        output_png,
        title="Shell-core density difference vs radius",
        ylabel=r"$\rho_{shell} - \rho_{core}$",
        fits={"shell_minus_core": fits["shell_minus_core"]},
    )
    return arrays
