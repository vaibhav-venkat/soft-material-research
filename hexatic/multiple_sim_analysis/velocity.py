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
    neighbor_counts_path,
    radii_for_cases,
    save_metric_npz,
    shell_mask_for_positions,
    unwrapped_x_positions,
)
from .disclination import _load_neighbor_counts
from .numba_kernels import defect_x_com_velocity, x_velocity_means_from_coords
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
    box_length_x = float(fields.x_edges[-1] - fields.x_edges[0])
    x_positions = np.asarray(fields.coords[..., 0], dtype=np.float64)
    shell_mask = np.asarray(fields.shell_mask, dtype=bool)
    steps = np.asarray(fields.steps, dtype=np.int64)
    all_mean, shell_mean = x_velocity_means_from_coords(
        x_positions,
        shell_mask,
        steps,
        box_length_x,
        float(cylinder.TIMESTEP),
        frame_start,
        frame_stop,
    )
    return {
        "all": all_mean,
        "shell": shell_mean,
    }


def _defect_x_com_velocity_from_coords(
    coords: np.ndarray,
    charges: np.ndarray,
    steps: np.ndarray,
    box_length_x: float,
    charge: int,
    frame_start: int,
    frame_stop: int,
) -> float:
    return defect_x_com_velocity(
        np.ascontiguousarray(coords[..., 0], dtype=np.float64),
        np.ascontiguousarray(charges, dtype=np.int64),
        np.ascontiguousarray(steps, dtype=np.int64),
        box_length_x,
        float(cylinder.TIMESTEP),
        int(charge),
        frame_start,
        frame_stop,
    )


def disclination_velocity_values_for_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
) -> dict[str, float]:
    counts = _load_neighbor_counts(neighbor_counts_path(case))
    charges = cylinder.NEIGHBORS - counts
    fields_path = active_fields_path(case)
    if fields_path.exists():
        fields = load_active_fields(fields_path)
        n_frames = min(len(fields.steps), charges.shape[0])
        coords = np.asarray(fields.coords[:n_frames], dtype=np.float64)
        steps = np.asarray(fields.steps[:n_frames], dtype=np.int64)
        charges = charges[:n_frames]
        box_length_x = float(fields.x_edges[-1] - fields.x_edges[0])
        return {
            "plus_1": _defect_x_com_velocity_from_coords(
                coords,
                charges,
                steps,
                box_length_x,
                charge=1,
                frame_start=frame_start,
                frame_stop=frame_stop,
            ),
            "minus_1": _defect_x_com_velocity_from_coords(
                coords,
                charges,
                steps,
                box_length_x,
                charge=-1,
                frame_start=frame_start,
                frame_stop=frame_stop,
            ),
        }

    coords: list[np.ndarray] = []
    steps: list[int] = []
    box_length_x: float | None = None
    with gsd.hoomd.open(name=str(case.trajectory_gsd), mode="r") as source:
        n_frames = min(len(source), charges.shape[0])
        for frame_idx in range(n_frames):
            frame = source[frame_idx]
            positions = np.asarray(frame.particles.position, dtype=np.float64)
            box_length_x = float(frame.configuration.box[0])
            theta = np.mod(np.arctan2(positions[:, 1], positions[:, 2]), 2.0 * np.pi)
            radius = np.linalg.norm(positions[:, 1:3], axis=1)
            coords.append(np.column_stack((positions[:, 0], theta, radius)))
            steps.append(int(frame.configuration.step))
    if box_length_x is None:
        return {"plus_1": np.nan, "minus_1": np.nan}
    coord_array = np.asarray(coords, dtype=np.float64)
    step_array = np.asarray(steps, dtype=np.int64)
    charges = charges[: len(step_array)]
    return {
        "plus_1": _defect_x_com_velocity_from_coords(
            coord_array,
            charges,
            step_array,
            box_length_x,
            charge=1,
            frame_start=frame_start,
            frame_stop=frame_stop,
        ),
        "minus_1": _defect_x_com_velocity_from_coords(
            coord_array,
            charges,
            step_array,
            box_length_x,
            charge=-1,
            frame_start=frame_start,
            frame_stop=frame_stop,
        ),
    }


def run_disclination_velocity(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    output_npz = NPZ_OUTPUT_DIR / "disclination_velocity.npz"
    output_png = PLOT_OUTPUT_DIR / "disclination_velocity.png"
    value_names = ("plus_1", "minus_1")
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
                title="Mean disclination x velocity vs radius",
                ylabel="mean disclination x velocity",
                fits=fits,
            )
        print(f"using cached disclination_velocity values from {output_npz}")
        return arrays

    values = {"plus_1": [], "minus_1": []}
    for case in cases:
        case_values = disclination_velocity_values_for_case(
            case,
            frame_start,
            frame_stop,
        )
        values["plus_1"].append(case_values["plus_1"])
        values["minus_1"].append(case_values["minus_1"])
    arrays = {name: np.asarray(series, dtype=np.float64) for name, series in values.items()}
    fits, payload = fit_payload(radii_for_cases(cases), arrays)
    save_metric_npz(
        output_npz,
        cases,
        "disclination_velocity",
        arrays,
        payload,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    plot_for_cases(
        cases,
        arrays,
        output_png,
        title="Mean disclination x velocity vs radius",
        ylabel="mean disclination x velocity",
        fits=fits,
    )
    return arrays


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
        run_disclination_velocity(
            cases,
            frame_start=frame_start,
            frame_stop=frame_stop,
            overwrite=overwrite,
        )
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
    run_disclination_velocity(
        cases,
        frame_start=frame_start,
        frame_stop=frame_stop,
        overwrite=overwrite,
    )
    return arrays
