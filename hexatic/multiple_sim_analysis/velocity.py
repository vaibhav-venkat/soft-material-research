from __future__ import annotations

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
    finite_nanmean,
    frame_indices,
    load_active_fields,
    load_cached_metric_values,
    load_metric_fit_curves,
    minimum_image_delta,
    particle_masses,
    radii_for_cases,
    save_metric_npz,
    shell_mask_for_positions,
    unwrapped_x_positions,
)
from .plotting import plot_for_cases


def x_com_velocity_values_for_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
) -> dict[str, float]:
    fields_path = active_fields_path(case)
    if fields_path.exists():
        return _x_com_velocity_values_from_active_fields(case, frame_start, frame_stop)

    all_velocities: list[float] = []
    shell_velocities: list[float] = []
    previous_x: np.ndarray | None = None
    previous_step: int | None = None
    previous_box_length_x: float | None = None
    previous_masses: np.ndarray | None = None
    previous_shell: np.ndarray | None = None

    with gsd.hoomd.open(name=str(case.trajectory_gsd), mode="r") as source:
        for frame_idx, frame in enumerate(source):
            particles = frame.particles
            positions = np.asarray(particles.position, dtype=np.float64)
            box_length_x = float(frame.configuration.box[0])
            current_x = unwrapped_x_positions(particles, box_length_x)
            current_step = int(frame.configuration.step)
            current_masses = particle_masses(particles, positions.shape[0])
            current_shell = shell_mask_for_positions(positions, case)

            if (
                previous_x is not None
                and previous_step is not None
                and previous_box_length_x is not None
                and previous_masses is not None
                and previous_shell is not None
                and frame_start <= frame_idx < frame_stop
            ):
                delta_t = (current_step - previous_step) * float(cylinder.TIMESTEP)
                if delta_t > 0.0:
                    delta_x = minimum_image_delta(
                        current_x - previous_x,
                        previous_box_length_x,
                    )
                    all_velocities.append(
                        float(np.average(delta_x, weights=previous_masses) / delta_t)
                    )
                    shell = previous_shell & current_shell
                    if np.any(shell):
                        shell_velocities.append(
                            float(
                                np.average(
                                    delta_x[shell],
                                    weights=previous_masses[shell],
                                )
                                / delta_t
                            )
                        )
                    else:
                        shell_velocities.append(np.nan)

            previous_x = current_x
            previous_step = current_step
            previous_box_length_x = box_length_x
            previous_masses = current_masses
            previous_shell = current_shell

    return {
        "all": finite_nanmean(np.asarray(all_velocities, dtype=np.float64)),
        "shell": finite_nanmean(np.asarray(shell_velocities, dtype=np.float64)),
    }


def _x_com_velocity_values_from_active_fields(
    case: RadiusCase,
    frame_start: int,
    frame_stop: int,
) -> dict[str, float]:
    fields = load_active_fields(active_fields_path(case))
    selected_endpoints = set(frame_indices(len(fields.steps), frame_start, frame_stop).tolist())
    box_length_x = float(fields.x_edges[-1] - fields.x_edges[0])
    x_positions = np.asarray(fields.coords[..., 0], dtype=np.float64)
    shell_mask = np.asarray(fields.shell_mask, dtype=bool)
    steps = np.asarray(fields.steps, dtype=np.int64)
    all_velocities: list[float] = []
    shell_velocities: list[float] = []

    for frame_idx in range(1, len(steps)):
        if frame_idx not in selected_endpoints:
            continue
        delta_t = (steps[frame_idx] - steps[frame_idx - 1]) * float(cylinder.TIMESTEP)
        if delta_t <= 0.0:
            all_velocities.append(np.nan)
            shell_velocities.append(np.nan)
            continue
        delta_x = minimum_image_delta(
            x_positions[frame_idx] - x_positions[frame_idx - 1],
            box_length_x,
        )
        all_velocities.append(finite_nanmean(delta_x / delta_t))
        shell = shell_mask[frame_idx] & shell_mask[frame_idx - 1]
        shell_velocities.append(finite_nanmean((delta_x / delta_t)[shell]))

    return {
        "all": finite_nanmean(np.asarray(all_velocities, dtype=np.float64)),
        "shell": finite_nanmean(np.asarray(shell_velocities, dtype=np.float64)),
    }


def run(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    output_npz = NPZ_OUTPUT_DIR / "velocity.npz"
    output_png = PLOT_OUTPUT_DIR / "velocity.png"
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
        if not output_png.exists():
            fits = load_metric_fit_curves(output_npz, value_names)
            plot_for_cases(
                cases,
                arrays,
                output_png,
                title="Mean x COM velocity vs radius",
                ylabel="mean x COM velocity",
                fits=fits,
            )
        print(f"using cached velocity values from {output_npz}")
        return arrays

    values = {"all": [], "shell": []}
    for case in cases:
        case_values = x_com_velocity_values_for_case(case, frame_start, frame_stop)
        values["all"].append(case_values["all"])
        values["shell"].append(case_values["shell"])
    arrays = {name: np.asarray(series, dtype=np.float64) for name, series in values.items()}
    fits, payload = fit_payload(radii_for_cases(cases), arrays)
    save_metric_npz(
        output_npz,
        cases,
        "velocity",
        arrays,
        payload,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    plot_for_cases(
        cases,
        arrays,
        output_png,
        title="Mean x COM velocity vs radius",
        ylabel="mean x COM velocity",
        fits=fits,
    )
    return arrays
