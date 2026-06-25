from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
)
from ..numba_kernels import moving_defect_frontback_chirality
from .shared import (
    LOCAL_CONTRAST_LENGTH,
    _disclination_mask,
    _load_hexatic_abs,
    _load_neighbor_counts,
    _validate_particle_frame_shape,
    hexatic_order_path,
)


def moving_defect_frontback_values_for_case(
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
    hexatic_abs = _load_hexatic_abs(hexatic_order_path(case), expected_shape)
    steps = np.asarray(fields.steps, dtype=np.int64)
    n_frames = min(
        coords.shape[0],
        disclinations.shape[0],
        hexatic_abs.shape[0],
        steps.shape[0],
    )
    coords = coords[:n_frames]
    disclinations = disclinations[:n_frames]
    hexatic_abs = hexatic_abs[:n_frames]
    steps = steps[:n_frames]

    x_edges = np.asarray(fields.x_edges, dtype=np.float64)
    box_length_x = float(x_edges[-1] - x_edges[0])
    (
        speed,
        abs_v_local,
        cos_v_defect_v_local,
        abs_v_residual,
        v_residual_x,
        v_residual_y,
        v_residual_z,
        abs_delta_chirality,
        velocity_direction,
        delta_chirality_sign,
        d_chi_x,
        d_chi_y,
        d_chi_z,
        d_psi6_x,
        d_psi6_y,
        d_psi6_z,
        v_residual_dot_d_chi_hat,
        v_residual_dot_d_psi6_hat,
        cos_v_residual_d_chi,
        cos_v_residual_d_psi6,
    ) = (
        moving_defect_frontback_chirality(
            np.ascontiguousarray(coords, dtype=np.float64),
            np.ascontiguousarray(disclinations, dtype=np.bool_),
            np.ascontiguousarray(hexatic_abs, dtype=np.float64),
            np.ascontiguousarray(steps, dtype=np.int64),
            frame_start,
            frame_stop,
            LOCAL_CONTRAST_LENGTH,
            3.0 * LOCAL_CONTRAST_LENGTH,
            cylinder.ANALYSIS.neighbor_count_radius,
            box_length_x,
            float(x_edges[0]),
            float(case.radius),
            float(cylinder.TIMESTEP),
        )
    )
    return {
        "speed": speed,
        "abs_v_local": abs_v_local,
        "cos_v_defect_v_local": cos_v_defect_v_local,
        "abs_v_residual": abs_v_residual,
        "v_residual_x": v_residual_x,
        "v_residual_y": v_residual_y,
        "v_residual_z": v_residual_z,
        "abs_delta_chirality": abs_delta_chirality,
        "velocity_direction": velocity_direction,
        "delta_chirality_sign": delta_chirality_sign,
        "d_chi_x": d_chi_x,
        "d_chi_y": d_chi_y,
        "d_chi_z": d_chi_z,
        "d_psi6_x": d_psi6_x,
        "d_psi6_y": d_psi6_y,
        "d_psi6_z": d_psi6_z,
        "v_residual_dot_d_chi_hat": v_residual_dot_d_chi_hat,
        "v_residual_dot_d_psi6_hat": v_residual_dot_d_psi6_hat,
        "cos_v_residual_d_chi": cos_v_residual_d_chi,
        "cos_v_residual_d_psi6": cos_v_residual_d_psi6,
    }


def _frontback_cache_matches(
    output_npz: Path,
    cases: tuple[RadiusCase, ...],
    frame_start: int,
    frame_stop: int,
) -> bool:
    with np.load(output_npz) as data:
        required = (
            "speed",
            "abs_v_local",
            "cos_v_defect_v_local",
            "abs_v_residual",
            "v_residual_x",
            "v_residual_y",
            "v_residual_z",
            "abs_delta_chirality",
            "velocity_direction",
            "delta_chirality_sign",
            "d_chi_x",
            "d_chi_y",
            "d_chi_z",
            "d_psi6_x",
            "d_psi6_y",
            "d_psi6_z",
            "v_residual_dot_d_chi_hat",
            "v_residual_dot_d_psi6_hat",
            "cos_v_residual_d_chi",
            "cos_v_residual_d_psi6",
            "sample_case_ids",
            "sample_radii",
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


def _load_frontback_cache(output_npz: Path) -> dict[str, np.ndarray]:
    with np.load(output_npz) as data:
        return {
            "speed": np.asarray(data["speed"], dtype=np.float64),
            "abs_v_local": np.asarray(data["abs_v_local"], dtype=np.float64),
            "cos_v_defect_v_local": np.asarray(
                data["cos_v_defect_v_local"],
                dtype=np.float64,
            ),
            "abs_v_residual": np.asarray(data["abs_v_residual"], dtype=np.float64),
            "v_residual_x": np.asarray(data["v_residual_x"], dtype=np.float64),
            "v_residual_y": np.asarray(data["v_residual_y"], dtype=np.float64),
            "v_residual_z": np.asarray(data["v_residual_z"], dtype=np.float64),
            "abs_delta_chirality": np.asarray(
                data["abs_delta_chirality"],
                dtype=np.float64,
            ),
            "velocity_direction": np.asarray(
                data["velocity_direction"],
                dtype=np.float64,
            ),
            "delta_chirality_sign": np.asarray(
                data["delta_chirality_sign"],
                dtype=np.float64,
            ),
            "d_chi_x": np.asarray(data["d_chi_x"], dtype=np.float64),
            "d_chi_y": np.asarray(data["d_chi_y"], dtype=np.float64),
            "d_chi_z": np.asarray(data["d_chi_z"], dtype=np.float64),
            "d_psi6_x": np.asarray(data["d_psi6_x"], dtype=np.float64),
            "d_psi6_y": np.asarray(data["d_psi6_y"], dtype=np.float64),
            "d_psi6_z": np.asarray(data["d_psi6_z"], dtype=np.float64),
            "v_residual_dot_d_chi_hat": np.asarray(
                data["v_residual_dot_d_chi_hat"],
                dtype=np.float64,
            ),
            "v_residual_dot_d_psi6_hat": np.asarray(
                data["v_residual_dot_d_psi6_hat"],
                dtype=np.float64,
            ),
            "cos_v_residual_d_chi": np.asarray(
                data["cos_v_residual_d_chi"],
                dtype=np.float64,
            ),
            "cos_v_residual_d_psi6": np.asarray(
                data["cos_v_residual_d_psi6"],
                dtype=np.float64,
            ),
            "sample_radii": np.asarray(data["sample_radii"], dtype=np.float64),
        }


def _save_frontback_cache(
    output_npz: Path,
    cases: tuple[RadiusCase, ...],
    arrays: dict[str, np.ndarray],
    frame_start: int,
    frame_stop: int,
) -> None:
    NPZ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        metric_name="moving_defect_frontback_chirality",
        case_ids=case_ids_for_cases(cases),
        labels=labels_for_cases(cases),
        group_names=group_names_for_cases(cases),
        radii=radii_for_cases(cases),
        frame_start=int(frame_start),
        frame_stop=int(frame_stop),
        core_radius=np.asarray(LOCAL_CONTRAST_LENGTH, dtype=np.float64),
        annulus_inner_radius=np.asarray(LOCAL_CONTRAST_LENGTH, dtype=np.float64),
        annulus_outer_radius=np.asarray(3.0 * LOCAL_CONTRAST_LENGTH, dtype=np.float64),
        chirality_radius=np.asarray(
            cylinder.ANALYSIS.neighbor_count_radius,
            dtype=np.float64,
        ),
        **arrays,
    )


def _median_by_radius(
    sample_radii: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite_radii = np.isfinite(sample_radii)
    radius_values = np.unique(sample_radii[finite_radii])
    medians = np.full(radius_values.shape, np.nan, dtype=np.float64)
    for radius_idx, radius in enumerate(radius_values):
        mask = (sample_radii == radius) & np.isfinite(values)
        if np.any(mask):
            medians[radius_idx] = float(np.median(values[mask]))
    return radius_values, medians


def _plot_local_vs_defect_speed(
    arrays: dict[str, np.ndarray],
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    speed = np.asarray(arrays["speed"], dtype=np.float64)
    local_speed = np.asarray(arrays["abs_v_local"], dtype=np.float64)
    sample_radii = np.asarray(arrays["sample_radii"], dtype=np.float64)
    finite = (
        np.isfinite(local_speed)
        & np.isfinite(speed)
        & np.isfinite(sample_radii)
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    scatter = ax.scatter(
        local_speed[finite],
        speed[finite],
        c=sample_radii[finite],
        cmap="viridis",
        s=20,
        alpha=0.75,
        edgecolors="none",
    )
    ax.set_xlabel("|v_local|")
    ax.set_ylabel("|v_defect|")
    ax.set_title("Defect speed vs local annulus speed")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="R")
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def _plot_defect_local_cosine_by_radius(
    arrays: dict[str, np.ndarray],
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    sample_radii = np.asarray(arrays["sample_radii"], dtype=np.float64)
    cosines = np.asarray(arrays["cos_v_defect_v_local"], dtype=np.float64)
    radii, median_cosine = _median_by_radius(sample_radii, cosines)

    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax.plot(
        radii,
        median_cosine,
        "o-",
        color="#111111",
        label="median cos(v_defect, v_local)",
    )
    ax.axhline(0.0, color="#777777", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("R")
    ax.set_ylabel("median cos(v_defect, v_local)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Defect-local velocity alignment vs radius")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def _plot_residual_speed_ratio_by_radius(
    arrays: dict[str, np.ndarray],
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    speed = np.asarray(arrays["speed"], dtype=np.float64)
    residual_speed = np.asarray(arrays["abs_v_residual"], dtype=np.float64)
    sample_radii = np.asarray(arrays["sample_radii"], dtype=np.float64)
    ratio = np.divide(
        residual_speed,
        speed,
        out=np.full(residual_speed.shape, np.nan, dtype=np.float64),
        where=np.isfinite(speed) & (speed > 0.0),
    )
    radii, median_ratio = _median_by_radius(sample_radii, ratio)

    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax.plot(
        radii,
        median_ratio,
        "o-",
        color="#111111",
        label="median |v_residual| / |v_defect|",
    )
    ax.set_xlabel("R")
    ax.set_ylabel("median |v_residual| / |v_defect|")
    ax.set_title("Residual velocity ratio vs radius")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def run_moving_frontback_chirality(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    output_npz = NPZ_OUTPUT_DIR / "moving_defect_frontback_chirality.npz"
    ratio_output_png = PLOT_OUTPUT_DIR / "moving_defect_residual_speed_ratio.png"
    speed_scatter_output_png = PLOT_OUTPUT_DIR / "moving_defect_local_speed_scatter.png"
    cosine_output_png = PLOT_OUTPUT_DIR / "moving_defect_local_velocity_cosine.png"
    if (
        not overwrite
        and output_npz.exists()
        and _frontback_cache_matches(output_npz, cases, frame_start, frame_stop)
    ):
        arrays = _load_frontback_cache(output_npz)
        if not ratio_output_png.exists():
            _plot_residual_speed_ratio_by_radius(arrays, ratio_output_png)
        if not speed_scatter_output_png.exists():
            _plot_local_vs_defect_speed(arrays, speed_scatter_output_png)
        if not cosine_output_png.exists():
            _plot_defect_local_cosine_by_radius(arrays, cosine_output_png)
        print(f"using cached moving-defect front/back chirality from {output_npz}")
        return arrays

    chunks: dict[str, list[np.ndarray]] = {
        "speed": [],
        "abs_v_local": [],
        "cos_v_defect_v_local": [],
        "abs_v_residual": [],
        "v_residual_x": [],
        "v_residual_y": [],
        "v_residual_z": [],
        "abs_delta_chirality": [],
        "velocity_direction": [],
        "delta_chirality_sign": [],
        "d_chi_x": [],
        "d_chi_y": [],
        "d_chi_z": [],
        "d_psi6_x": [],
        "d_psi6_y": [],
        "d_psi6_z": [],
        "v_residual_dot_d_chi_hat": [],
        "v_residual_dot_d_psi6_hat": [],
        "cos_v_residual_d_chi": [],
        "cos_v_residual_d_psi6": [],
        "sample_radii": [],
    }
    sample_case_ids: list[np.ndarray] = []
    for case in cases:
        case_values = moving_defect_frontback_values_for_case(
            case,
            frame_start,
            frame_stop,
        )
        n_samples = case_values["speed"].shape[0]
        for name in (
            "speed",
            "abs_v_local",
            "cos_v_defect_v_local",
            "abs_v_residual",
            "v_residual_x",
            "v_residual_y",
            "v_residual_z",
            "abs_delta_chirality",
            "velocity_direction",
            "delta_chirality_sign",
            "d_chi_x",
            "d_chi_y",
            "d_chi_z",
            "d_psi6_x",
            "d_psi6_y",
            "d_psi6_z",
            "v_residual_dot_d_chi_hat",
            "v_residual_dot_d_psi6_hat",
            "cos_v_residual_d_chi",
            "cos_v_residual_d_psi6",
        ):
            chunks[name].append(case_values[name])
        chunks["sample_radii"].append(
            np.full(n_samples, float(case.radius), dtype=np.float64)
        )
        sample_case_ids.append(np.full(n_samples, case.case_id))

    arrays = {
        name: (
            np.concatenate(series)
            if series
            else np.asarray([], dtype=np.float64)
        )
        for name, series in chunks.items()
    }
    arrays["sample_case_ids"] = (
        np.concatenate(sample_case_ids) if sample_case_ids else np.asarray([])
    )
    _save_frontback_cache(output_npz, cases, arrays, frame_start, frame_stop)
    _plot_residual_speed_ratio_by_radius(arrays, ratio_output_png)
    _plot_local_vs_defect_speed(arrays, speed_scatter_output_png)
    _plot_defect_local_cosine_by_radius(arrays, cosine_output_png)
    return arrays
