from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from hexatic.constants import cylinder
from hexatic.radii_analysis.cases import RadiusCase

from ..common import (
    FRAME_START,
    FRAME_STOP,
    NPZ_OUTPUT_DIR,
    PLOT_OUTPUT_DIR,
    active_fields_path,
    case_ids_for_cases,
    group_names_for_cases,
    labels_for_cases,
    load_active_fields,
    neighbor_counts_path,
    radii_for_cases,
    save_metric_npz,
)
from ..disclination import _load_neighbor_counts
from ..plotting import plot_radius_values
from .shared import (
    LOCAL_CONTRAST_LENGTH,
    _cell_index,
    _disclination_mask,
    _validate_particle_frame_shape,
)

MOVING_VALUE_NAMES = (
    "abs_v_defect",
    "v_local_mean",
    "v_local_rms",
    "median_v_parallel",
    "median_v_perp",
)

MOVING_FRAME_VALUE_NAMES = (
    "frame_abs_v_defect",
    "frame_v_local_mean",
    "frame_v_local_rms",
    "frame_v_parallel",
    "frame_v_perp",
)


@njit(cache=True)
def _moving_defect_velocity_frame_values(
    coords: np.ndarray,
    disclination_mask: np.ndarray,
    steps: np.ndarray,
    frame_start: int,
    frame_stop: int,
    annulus_inner_radius: float,
    annulus_outer_radius: float,
    box_length_x: float,
    x_min: float,
    cylinder_radius: float,
    timestep: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start = min(max(frame_start, 1), coords.shape[0])
    stop = min(max(frame_stop, start), coords.shape[0])
    frame_count = max(0, stop - start)
    frame_abs_v_defect = np.full(frame_count, np.nan, dtype=np.float64)
    frame_v_local_mean = np.full(frame_count, np.nan, dtype=np.float64)
    frame_v_local_rms = np.full(frame_count, np.nan, dtype=np.float64)
    frame_v_parallel = np.full(frame_count, np.nan, dtype=np.float64)
    frame_v_perp = np.full(frame_count, np.nan, dtype=np.float64)
    if frame_count == 0 or annulus_inner_radius <= 0.0 or annulus_outer_radius <= 0.0:
        return (
            frame_abs_v_defect,
            frame_v_local_mean,
            frame_v_local_rms,
            frame_v_parallel,
            frame_v_perp,
        )

    n_particles = coords.shape[1]
    cell_size = annulus_inner_radius
    y_min = -cylinder_radius
    z_min = -cylinder_radius
    transverse_width = 2.0 * cylinder_radius
    n_x = max(1, int(math.ceil(box_length_x / cell_size)))
    n_y = max(1, int(math.ceil(transverse_width / cell_size)))
    n_z = max(1, int(math.ceil(transverse_width / cell_size)))
    n_cells = n_x * n_y * n_z
    search_cells = int(math.ceil(annulus_outer_radius / cell_size))
    inner_sq = annulus_inner_radius * annulus_inner_radius
    outer_sq = annulus_outer_radius * annulus_outer_radius

    for frame_idx in range(start, stop):
        output_idx = frame_idx - start
        delta_t = (steps[frame_idx] - steps[frame_idx - 1]) * timestep
        if delta_t <= 0.0:
            continue

        x_values = np.empty(n_particles, dtype=np.float64)
        y_values = np.empty(n_particles, dtype=np.float64)
        z_values = np.empty(n_particles, dtype=np.float64)
        finite_position = np.zeros(n_particles, dtype=np.bool_)
        head = np.full(n_cells, -1, dtype=np.int64)
        next_index = np.full(n_particles, -1, dtype=np.int64)

        for particle_idx in range(n_particles):
            x_i = coords[frame_idx, particle_idx, 0]
            theta_i = coords[frame_idx, particle_idx, 1]
            radius_i = coords[frame_idx, particle_idx, 2]
            if (
                not math.isfinite(x_i)
                or not math.isfinite(theta_i)
                or not math.isfinite(radius_i)
            ):
                continue

            y_i = radius_i * math.sin(theta_i)
            z_i = radius_i * math.cos(theta_i)
            x_values[particle_idx] = x_i
            y_values[particle_idx] = y_i
            z_values[particle_idx] = z_i
            finite_position[particle_idx] = True

            cell_idx = _cell_index(
                x_i,
                y_i,
                z_i,
                x_min,
                y_min,
                z_min,
                cell_size,
                n_x,
                n_y,
                n_z,
                box_length_x,
            )
            next_index[particle_idx] = head[cell_idx]
            head[cell_idx] = particle_idx

        abs_v_defect_sum = 0.0
        v_local_mean_sum = 0.0
        v_local_rms_sum = 0.0
        v_parallel_sum = 0.0
        v_perp_sum = 0.0
        speed_sample_count = 0
        component_sample_count = 0

        for defect_idx in range(n_particles):
            if not finite_position[defect_idx] or not disclination_mask[frame_idx, defect_idx]:
                continue

            prev_x = coords[frame_idx - 1, defect_idx, 0]
            prev_theta = coords[frame_idx - 1, defect_idx, 1]
            prev_radius = coords[frame_idx - 1, defect_idx, 2]
            if (
                not math.isfinite(prev_x)
                or not math.isfinite(prev_theta)
                or not math.isfinite(prev_radius)
            ):
                continue

            x_i = x_values[defect_idx]
            y_i = y_values[defect_idx]
            z_i = z_values[defect_idx]
            prev_y = prev_radius * math.sin(prev_theta)
            prev_z = prev_radius * math.cos(prev_theta)
            v_defect_x = x_i - prev_x
            if box_length_x > 0.0:
                v_defect_x -= box_length_x * round(v_defect_x / box_length_x)
            v_defect_y = y_i - prev_y
            v_defect_z = z_i - prev_z
            v_defect_x /= delta_t
            v_defect_y /= delta_t
            v_defect_z /= delta_t
            abs_v_defect = math.sqrt(
                v_defect_x * v_defect_x
                + v_defect_y * v_defect_y
                + v_defect_z * v_defect_z
            )
            if not math.isfinite(abs_v_defect):
                continue

            center_cell = _cell_index(
                x_i,
                y_i,
                z_i,
                x_min,
                y_min,
                z_min,
                cell_size,
                n_x,
                n_y,
                n_z,
                box_length_x,
            )
            center_x = center_cell // (n_y * n_z)
            center_y = (center_cell // n_z) % n_y
            center_z = center_cell % n_z

            local_vx_sum = 0.0
            local_vy_sum = 0.0
            local_vz_sum = 0.0
            local_speed_sum = 0.0
            local_velocity_count = 0

            for delta_x in range(-search_cells, search_cells + 1):
                cell_x = (center_x + delta_x) % n_x
                for delta_y in range(-search_cells, search_cells + 1):
                    cell_y = center_y + delta_y
                    if cell_y < 0 or cell_y >= n_y:
                        continue
                    for delta_z in range(-search_cells, search_cells + 1):
                        cell_z = center_z + delta_z
                        if cell_z < 0 or cell_z >= n_z:
                            continue

                        cell_idx = (cell_x * n_y + cell_y) * n_z + cell_z
                        particle_idx = head[cell_idx]
                        while particle_idx != -1:
                            dx = x_values[particle_idx] - x_i
                            if box_length_x > 0.0:
                                dx -= box_length_x * round(dx / box_length_x)
                            dy = y_values[particle_idx] - y_i
                            dz = z_values[particle_idx] - z_i
                            distance_sq = dx * dx + dy * dy + dz * dz
                            if inner_sq < distance_sq < outer_sq:
                                prev_particle_x = coords[frame_idx - 1, particle_idx, 0]
                                prev_particle_theta = coords[frame_idx - 1, particle_idx, 1]
                                prev_particle_radius = coords[frame_idx - 1, particle_idx, 2]
                                if (
                                    math.isfinite(prev_particle_x)
                                    and math.isfinite(prev_particle_theta)
                                    and math.isfinite(prev_particle_radius)
                                ):
                                    particle_vx = x_values[particle_idx] - prev_particle_x
                                    if box_length_x > 0.0:
                                        particle_vx -= box_length_x * round(
                                            particle_vx / box_length_x
                                        )
                                    prev_particle_y = prev_particle_radius * math.sin(
                                        prev_particle_theta
                                    )
                                    prev_particle_z = prev_particle_radius * math.cos(
                                        prev_particle_theta
                                    )
                                    particle_vx /= delta_t
                                    particle_vy = (
                                        y_values[particle_idx] - prev_particle_y
                                    ) / delta_t
                                    particle_vz = (
                                        z_values[particle_idx] - prev_particle_z
                                    ) / delta_t
                                    particle_speed = math.sqrt(
                                        particle_vx * particle_vx
                                        + particle_vy * particle_vy
                                        + particle_vz * particle_vz
                                    )
                                    if math.isfinite(particle_speed):
                                        local_vx_sum += particle_vx
                                        local_vy_sum += particle_vy
                                        local_vz_sum += particle_vz
                                        local_speed_sum += particle_speed
                                        local_velocity_count += 1

                            particle_idx = next_index[particle_idx]

            if local_velocity_count <= 0:
                continue

            local_vx = local_vx_sum / local_velocity_count
            local_vy = local_vy_sum / local_velocity_count
            local_vz = local_vz_sum / local_velocity_count
            v_local_mean = math.sqrt(
                local_vx * local_vx + local_vy * local_vy + local_vz * local_vz
            )
            v_local_rms = local_speed_sum / local_velocity_count
            if v_local_mean > 0.0:
                v_local_hat_x = local_vx / v_local_mean
                v_local_hat_y = local_vy / v_local_mean
                v_local_hat_z = local_vz / v_local_mean
                v_parallel = (
                    v_defect_x * v_local_hat_x
                    + v_defect_y * v_local_hat_y
                    + v_defect_z * v_local_hat_z
                )
                perp_x = v_defect_x - v_parallel * v_local_hat_x
                perp_y = v_defect_y - v_parallel * v_local_hat_y
                perp_z = v_defect_z - v_parallel * v_local_hat_z
                v_perp = math.sqrt(perp_x * perp_x + perp_y * perp_y + perp_z * perp_z)
            else:
                v_parallel = np.nan
                v_perp = np.nan

            abs_v_defect_sum += abs_v_defect
            v_local_mean_sum += v_local_mean
            v_local_rms_sum += v_local_rms
            speed_sample_count += 1
            if math.isfinite(v_parallel):
                v_parallel_sum += v_parallel
                v_perp_sum += v_perp
                component_sample_count += 1

        if speed_sample_count > 0:
            frame_abs_v_defect[output_idx] = abs_v_defect_sum / speed_sample_count
            frame_v_local_mean[output_idx] = v_local_mean_sum / speed_sample_count
            frame_v_local_rms[output_idx] = v_local_rms_sum / speed_sample_count
        if component_sample_count > 0:
            frame_v_parallel[output_idx] = v_parallel_sum / component_sample_count
            frame_v_perp[output_idx] = v_perp_sum / component_sample_count

    return (
        frame_abs_v_defect,
        frame_v_local_mean,
        frame_v_local_rms,
        frame_v_parallel,
        frame_v_perp,
    )


def _finite_mean(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan
    return float(np.mean(values[finite]))


def _finite_median(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan
    return float(np.median(values[finite]))


def moving_defect_velocity_values_for_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
) -> dict[str, np.ndarray]:
    neighbor_counts = _load_neighbor_counts(neighbor_counts_path(case))
    expected_shape = neighbor_counts.shape
    disclinations = _disclination_mask(neighbor_counts)

    fields = load_active_fields(active_fields_path(case))
    coords = np.asarray(fields.coords, dtype=np.float64)
    _validate_particle_frame_shape("coords", coords, expected_shape)
    steps = np.asarray(fields.steps, dtype=np.int64)
    n_frames = min(coords.shape[0], disclinations.shape[0], steps.shape[0])
    coords = coords[:n_frames]
    disclinations = disclinations[:n_frames]
    steps = steps[:n_frames]

    x_edges = np.asarray(fields.x_edges, dtype=np.float64)
    box_length_x = float(x_edges[-1] - x_edges[0])
    frame_values = _moving_defect_velocity_frame_values(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(disclinations, dtype=np.bool_),
        np.ascontiguousarray(steps, dtype=np.int64),
        frame_start,
        frame_stop,
        LOCAL_CONTRAST_LENGTH,
        3.0 * LOCAL_CONTRAST_LENGTH,
        box_length_x,
        float(x_edges[0]),
        float(case.radius),
        float(cylinder.TIMESTEP),
    )
    return {
        name: value
        for name, value in zip(MOVING_FRAME_VALUE_NAMES, frame_values)
    }


def _cache_matches(
    output_npz: Path,
    cases: tuple[RadiusCase, ...],
    frame_start: int,
    frame_stop: int,
) -> bool:
    with np.load(output_npz) as data:
        required = MOVING_VALUE_NAMES + MOVING_FRAME_VALUE_NAMES + (
            "case_ids",
            "frame_start",
            "frame_stop",
        )
        if not all(name in data for name in required):
            return False
        return (
            int(np.asarray(data["frame_start"]).item()) == int(frame_start)
            and int(np.asarray(data["frame_stop"]).item()) == int(frame_stop)
            and np.array_equal(np.asarray(data["case_ids"]), case_ids_for_cases(cases))
        )


def _load_cache(output_npz: Path) -> dict[str, np.ndarray]:
    with np.load(output_npz) as data:
        return {
            name: np.asarray(data[name], dtype=np.float64)
            for name in MOVING_VALUE_NAMES + MOVING_FRAME_VALUE_NAMES
        }


def _plot_speed_summary(
    cases: tuple[RadiusCase, ...],
    arrays: dict[str, np.ndarray],
    output_png: Path,
) -> None:
    plot_radius_values(
        radii_for_cases(cases),
        {
            "|v_defect|": arrays["abs_v_defect"],
            "v_local_mean": arrays["v_local_mean"],
            "v_local_rms": arrays["v_local_rms"],
        },
        output_png,
        "|v_defect| vs local annulus velocity",
        "velocity",
        case_labels=labels_for_cases(cases),
        group_names=group_names_for_cases(cases),
        series_colors=("#111111", "#0072b2", "#d55e00"),
    )


def _plot_median_value(
    cases: tuple[RadiusCase, ...],
    values: np.ndarray,
    output_png: Path,
    ylabel: str,
    title: str,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    radii = radii_for_cases(cases)
    labels = labels_for_cases(cases)

    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    ax.plot(radii, values, "o-", color="#111111", linewidth=1.5, markersize=5)
    ax.axhline(0.0, color="#777777", linewidth=1.0, alpha=0.6)
    for radius, label in zip(radii, labels):
        ax.annotate(
            str(label),
            (radius, ax.get_ylim()[0]),
            xytext=(0, -20),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7,
            rotation=35,
            alpha=0.75,
            clip_on=False,
        )
    ax.set_xlabel("Cylinder radius R")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.28)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def _plot_outputs_missing(output_pngs: tuple[Path, Path, Path]) -> bool:
    return any(not output_png.exists() for output_png in output_pngs)


def _plot_moving_values(
    cases: tuple[RadiusCase, ...],
    arrays: dict[str, np.ndarray],
    output_pngs: tuple[Path, Path, Path],
) -> None:
    _plot_speed_summary(cases, arrays, output_pngs[0])
    _plot_median_value(
        cases,
        arrays["median_v_parallel"],
        output_pngs[1],
        "median v_parallel",
        "Median defect velocity parallel to local flow",
    )
    _plot_median_value(
        cases,
        arrays["median_v_perp"],
        output_pngs[2],
        "median v_perp",
        "Median defect velocity perpendicular to local flow",
    )


def run_moving_defect_velocity_annulus(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    output_npz = NPZ_OUTPUT_DIR / "moving_defect_velocity_annulus.npz"
    output_pngs = (
        PLOT_OUTPUT_DIR / "moving_defect_velocity_annulus.png",
        PLOT_OUTPUT_DIR / "moving_defect_median_v_parallel.png",
        PLOT_OUTPUT_DIR / "moving_defect_median_v_perp.png",
    )
    if (
        not overwrite
        and output_npz.exists()
        and _cache_matches(output_npz, cases, frame_start, frame_stop)
    ):
        arrays = _load_cache(output_npz)
        if _plot_outputs_missing(output_pngs):
            _plot_moving_values(cases, arrays, output_pngs)
        print(f"using cached moving-defect velocity annulus values from {output_npz}")
        return arrays

    frame_chunks: dict[str, list[np.ndarray]] = {
        name: [] for name in MOVING_FRAME_VALUE_NAMES
    }
    values = {name: [] for name in MOVING_VALUE_NAMES}
    for case in cases:
        case_frame_values = moving_defect_velocity_values_for_case(
            case,
            frame_start,
            frame_stop,
        )
        for name in MOVING_FRAME_VALUE_NAMES:
            frame_chunks[name].append(case_frame_values[name])

        values["abs_v_defect"].append(
            _finite_mean(case_frame_values["frame_abs_v_defect"])
        )
        values["v_local_mean"].append(
            _finite_mean(case_frame_values["frame_v_local_mean"])
        )
        values["v_local_rms"].append(
            _finite_mean(case_frame_values["frame_v_local_rms"])
        )
        values["median_v_parallel"].append(
            _finite_median(case_frame_values["frame_v_parallel"])
        )
        values["median_v_perp"].append(
            _finite_median(case_frame_values["frame_v_perp"])
        )

    arrays = {
        name: np.asarray(series, dtype=np.float64)
        for name, series in values.items()
    }
    arrays.update(
        {
            name: (
                np.vstack(series)
                if series
                else np.empty((0, 0), dtype=np.float64)
            )
            for name, series in frame_chunks.items()
        }
    )
    save_metric_npz(
        output_npz,
        cases,
        "moving_defect_velocity_annulus",
        arrays,
        {
            "annulus_inner_radius": np.asarray(
                LOCAL_CONTRAST_LENGTH,
                dtype=np.float64,
            ),
            "annulus_outer_radius": np.asarray(
                3.0 * LOCAL_CONTRAST_LENGTH,
                dtype=np.float64,
            ),
        },
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    _plot_moving_values(cases, arrays, output_pngs)
    return arrays
