from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hexatic.constants import cylinder
from hexatic.radii_analysis.cases import HEXATIC_OUTPUT_DIR, RadiusCase

from .common import (
    FRAME_START,
    FRAME_STOP,
    NPZ_OUTPUT_DIR,
    PLOT_OUTPUT_DIR,
    active_fields_path,
    case_ids_for_cases,
    group_names_for_cases,
    labels_for_cases,
    load_active_fields,
    load_cached_metric_values,
    neighbor_counts_path,
    radii_for_cases,
    save_metric_npz,
)
from .disclination import _load_neighbor_counts
from .numba_kernels import (
    local_disclination_field_contrasts,
    local_disclination_field_profiles,
    moving_defect_frontback_chirality,
)
from .plotting import companion_circumference_plot_path, plot_radius_values


CIRCUMFERENCE_REFERENCE_CASE_ID = "circ_60_0D"
LOCAL_CONTRAST_LENGTH = cylinder.ANALYSIS.neighbor_count_radius
LOCAL_PROFILE_BIN_EDGES = LOCAL_CONTRAST_LENGTH * np.arange(6, dtype=np.float64)
LOCAL_PROFILE_BIN_LABELS = np.asarray(
    ("< a", "a-2a", "2a-3a", "3a-4a", "4a-5a")
)
LOCAL_PROFILE_COLORS = ("#111111", "#0072b2", "#d55e00", "#009e73", "#cc0000")


def hexatic_order_path(case: RadiusCase) -> Path:
    return HEXATIC_OUTPUT_DIR / f"{case.case_id}_hexatic_order.txt"


def _load_hexatic_abs(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    table = np.loadtxt(path, dtype=np.float64)
    if table.ndim == 1:
        table = table[np.newaxis, :]
    if table.shape[1] < 6:
        raise ValueError(f"Hexatic order table is missing columns: {path}")

    frame_indices_table = table[:, 0].astype(np.int64)
    particle_indices = table[:, 2].astype(np.int64)
    if np.any(frame_indices_table >= shape[0]) or np.any(particle_indices >= shape[1]):
        raise ValueError(f"Hexatic order table does not match expected shape: {path}")

    psi_abs = np.full(shape, np.nan, dtype=np.float64)
    psi_abs[frame_indices_table, particle_indices] = table[:, 5]
    return psi_abs


def _disclination_mask(neighbor_counts: np.ndarray) -> np.ndarray:
    charges = cylinder.NEIGHBORS - neighbor_counts
    return np.abs(charges) == 1


def _validate_particle_frame_shape(
    name: str,
    values: np.ndarray,
    expected_shape: tuple[int, int],
) -> None:
    if values.shape[:2] != expected_shape:
        raise ValueError(
            f"{name} shape {values.shape[:2]} does not match "
            f"neighbor-count shape {expected_shape}."
        )


def disclination_order_values_for_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
) -> dict[str, float]:
    neighbor_counts = _load_neighbor_counts(neighbor_counts_path(case))
    expected_shape = neighbor_counts.shape
    disclinations = _disclination_mask(neighbor_counts)

    fields = load_active_fields(active_fields_path(case))
    direction_cylindrical = np.asarray(fields.direction_cylindrical, dtype=np.float64)
    _validate_particle_frame_shape(
        "direction_cylindrical",
        direction_cylindrical,
        expected_shape,
    )

    hexatic_abs = _load_hexatic_abs(hexatic_order_path(case), expected_shape)
    coords = np.asarray(fields.coords, dtype=np.float64)
    _validate_particle_frame_shape("coords", coords, expected_shape)
    x_edges = np.asarray(fields.x_edges, dtype=np.float64)
    box_length_x = float(x_edges[-1] - x_edges[0])
    local_delta_s, local_delta_hexatic, local_delta_chirality = (
        local_disclination_field_contrasts(
            np.ascontiguousarray(coords, dtype=np.float64),
            np.ascontiguousarray(direction_cylindrical, dtype=np.float64),
            np.ascontiguousarray(hexatic_abs, dtype=np.float64),
            np.ascontiguousarray(disclinations, dtype=np.bool_),
            frame_start,
            frame_stop,
            LOCAL_CONTRAST_LENGTH,
            2.0 * LOCAL_CONTRAST_LENGTH,
            5.0 * LOCAL_CONTRAST_LENGTH,
            cylinder.ANALYSIS.neighbor_count_radius,
            box_length_x,
            float(x_edges[0]),
            float(case.radius),
        )
    )

    return {
        "delta_S_local": local_delta_s,
        "delta_hexatic_order_local": local_delta_hexatic,
        "delta_chirality_disclinations_local": local_delta_chirality,
    }


def disclination_shell_profile_values_for_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
) -> dict[str, np.ndarray]:
    neighbor_counts = _load_neighbor_counts(neighbor_counts_path(case))
    expected_shape = neighbor_counts.shape
    disclinations = _disclination_mask(neighbor_counts)

    fields = load_active_fields(active_fields_path(case))
    direction_cylindrical = np.asarray(fields.direction_cylindrical, dtype=np.float64)
    _validate_particle_frame_shape(
        "direction_cylindrical",
        direction_cylindrical,
        expected_shape,
    )

    hexatic_abs = _load_hexatic_abs(hexatic_order_path(case), expected_shape)
    coords = np.asarray(fields.coords, dtype=np.float64)
    _validate_particle_frame_shape("coords", coords, expected_shape)
    x_edges = np.asarray(fields.x_edges, dtype=np.float64)
    box_length_x = float(x_edges[-1] - x_edges[0])
    s_profile, hexatic_profile, chirality_profile, chirality_annulus = (
        local_disclination_field_profiles(
            np.ascontiguousarray(coords, dtype=np.float64),
            np.ascontiguousarray(direction_cylindrical, dtype=np.float64),
            np.ascontiguousarray(hexatic_abs, dtype=np.float64),
            np.ascontiguousarray(disclinations, dtype=np.bool_),
            frame_start,
            frame_stop,
            np.ascontiguousarray(LOCAL_PROFILE_BIN_EDGES, dtype=np.float64),
            cylinder.ANALYSIS.neighbor_count_radius,
            box_length_x,
            float(x_edges[0]),
            float(case.radius),
        )
    )

    return {
        "S_shell_profile": s_profile,
        "hexatic_order_shell_profile": hexatic_profile,
        "chirality_shell_profile": chirality_profile,
        "chirality_annulus": np.asarray(chirality_annulus, dtype=np.float64),
    }


def _plot_values(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "delta_S_local": values["delta_S_local"],
        "delta_hexatic_order_local": values["delta_hexatic_order_local"],
        "delta_chirality_disclinations_local": values[
            "delta_chirality_disclinations_local"
        ],
    }


def _plot_missing(cases: tuple[RadiusCase, ...], output_png: Path) -> bool:
    if not output_png.exists():
        return True
    group_names = group_names_for_cases(cases)
    if not bool(np.any(group_names == "circumference")):
        return False
    return not companion_circumference_plot_path(output_png).exists()


def _filter_values(
    values: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(series, dtype=np.float64)[mask]
        for name, series in values.items()
    }


def _plot_for_cases_without_fit(
    cases: tuple[RadiusCase, ...],
    values: dict[str, np.ndarray],
    output_png: Path,
) -> None:
    values = _plot_values(values)
    radii = radii_for_cases(cases)
    labels = labels_for_cases(cases)
    case_ids = case_ids_for_cases(cases)
    group_names = group_names_for_cases(cases)
    regular_mask = (group_names != "circumference") | (case_ids == "circ_60_0D")
    if not np.any(regular_mask):
        regular_mask = np.ones(len(cases), dtype=bool)

    plot_radius_values(
        radii[regular_mask],
        _filter_values(values, regular_mask),
        output_png,
        "Local disclination-field contrast vs radius",
        "core mean - annulus mean",
        case_labels=labels[regular_mask],
        group_names=group_names[regular_mask],
    )

    circumference_mask = group_names == "circumference"
    if not np.any(circumference_mask):
        return
    circumference_x = (
        2.0 * np.pi * radii[circumference_mask] / cylinder.PARTICLE_DIAMETER
    )
    plot_radius_values(
        circumference_x,
        _filter_values(values, circumference_mask),
        companion_circumference_plot_path(output_png),
        "Local disclination-field contrast vs radius (circumference sweep)",
        "core mean - annulus mean",
        case_labels=labels[circumference_mask],
        group_names=group_names[circumference_mask],
        xlabel="Circumference C / D",
    )


def _profile_plot_missing(output_pngs: tuple[Path, ...]) -> bool:
    return any(not output_png.exists() for output_png in output_pngs)


def _profile_series(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    return {
        str(label): values[:, bin_idx]
        for bin_idx, label in enumerate(LOCAL_PROFILE_BIN_LABELS)
    }


def _plot_profile_for_cases(
    cases: tuple[RadiusCase, ...],
    profile: np.ndarray,
    output_png: Path,
    title: str,
    ylabel: str,
) -> None:
    radii = radii_for_cases(cases)
    labels = labels_for_cases(cases)
    case_ids = case_ids_for_cases(cases)
    group_names = group_names_for_cases(cases)
    regular_mask = (group_names != "circumference") | (case_ids == "circ_60_0D")
    if not np.any(regular_mask):
        regular_mask = np.ones(len(cases), dtype=bool)

    plot_radius_values(
        radii[regular_mask],
        _profile_series(profile[regular_mask]),
        output_png,
        title,
        ylabel,
        case_labels=labels[regular_mask],
        group_names=group_names[regular_mask],
        series_colors=LOCAL_PROFILE_COLORS,
    )


def _plot_shell_profiles(
    cases: tuple[RadiusCase, ...],
    arrays: dict[str, np.ndarray],
    output_pngs: tuple[Path, Path, Path],
) -> None:
    _plot_profile_for_cases(
        cases,
        arrays["S_shell_profile"],
        output_pngs[0],
        "Local disclination S profile vs radius",
        "local S",
    )
    _plot_profile_for_cases(
        cases,
        arrays["hexatic_order_shell_profile"],
        output_pngs[1],
        "Local disclination hexatic profile vs radius",
        "local hexatic order",
    )
    _plot_profile_for_cases(
        cases,
        arrays["chirality_shell_profile"],
        output_pngs[2],
        "Local disclination chirality profile vs radius",
        "local mean(chirality * chirality)",
    )


def _mean_abs_defect_velocity(velocity_arrays: dict[str, np.ndarray]) -> np.ndarray:
    plus = np.abs(np.asarray(velocity_arrays["plus_1"], dtype=np.float64))
    minus = np.abs(np.asarray(velocity_arrays["minus_1"], dtype=np.float64))
    values = np.column_stack((plus, minus))
    finite = np.isfinite(values)
    counts = np.count_nonzero(finite, axis=1)
    sums = np.where(finite, values, 0.0).sum(axis=1)
    return np.divide(
        sums,
        counts,
        out=np.full(values.shape[0], np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _median_abs_defect_velocity(velocity_arrays: dict[str, np.ndarray]) -> np.ndarray:
    plus = np.abs(np.asarray(velocity_arrays["plus_1"], dtype=np.float64))
    minus = np.abs(np.asarray(velocity_arrays["minus_1"], dtype=np.float64))
    values = np.column_stack((plus, minus))
    finite = np.isfinite(values)
    medians = np.full(values.shape[0], np.nan, dtype=np.float64)
    for row_idx in range(values.shape[0]):
        if np.any(finite[row_idx]):
            medians[row_idx] = float(np.median(values[row_idx, finite[row_idx]]))
    return medians


def _disclination_velocity_arrays(
    cases: tuple[RadiusCase, ...],
    frame_start: int,
    frame_stop: int,
    overwrite: bool,
) -> dict[str, np.ndarray]:
    output_npz = NPZ_OUTPUT_DIR / "disclination_velocity.npz"
    value_names = ("plus_1", "minus_1")
    try:
        arrays = load_cached_metric_values(
            output_npz,
            value_names,
            cases,
            frame_start,
            frame_stop,
            overwrite=overwrite,
        )
    except FileExistsError:
        arrays = None
    if arrays is not None:
        return arrays

    from .velocity import disclination_velocity_values_for_case

    values = {name: [] for name in value_names}
    for case in cases:
        case_values = disclination_velocity_values_for_case(
            case,
            frame_start,
            frame_stop,
        )
        for name in value_names:
            values[name].append(case_values[name])
    return {
        name: np.asarray(series, dtype=np.float64)
        for name, series in values.items()
    }


def _plot_velocity_chirality_summary(
    cases: tuple[RadiusCase, ...],
    arrays: dict[str, np.ndarray],
    output_png: Path,
) -> None:
    plot_radius_values(
        radii_for_cases(cases),
        arrays,
        output_png,
        "|v_defect| and local chirality vs radius",
        "value",
        case_labels=labels_for_cases(cases),
        group_names=group_names_for_cases(cases),
        series_colors=("#111111", "#777777", "#0072b2", "#cc0000"),
    )


def run_velocity_chirality_summary(
    cases: tuple[RadiusCase, ...],
    shell_profiles: dict[str, np.ndarray],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    output_npz = NPZ_OUTPUT_DIR / "disclination_velocity_chirality.npz"
    output_png = PLOT_OUTPUT_DIR / "disclination_velocity_chirality.png"
    value_names = (
        "abs_v_defect",
        "median_abs_v_defect",
        "chirality_annulus",
        "chirality_annulus_minus_core",
    )
    velocity_arrays = _disclination_velocity_arrays(
        cases,
        frame_start,
        frame_stop,
        overwrite,
    )
    chirality_profile = np.asarray(
        shell_profiles["chirality_shell_profile"],
        dtype=np.float64,
    )
    chirality_core = chirality_profile[:, 0]
    chirality_annulus = np.asarray(
        shell_profiles["chirality_annulus"],
        dtype=np.float64,
    )
    arrays = {
        "abs_v_defect": _mean_abs_defect_velocity(velocity_arrays),
        "median_abs_v_defect": _median_abs_defect_velocity(velocity_arrays),
        "chirality_annulus": chirality_annulus,
        "chirality_annulus_minus_core": chirality_annulus - chirality_core,
    }
    save_metric_npz(
        output_npz,
        cases,
        "disclination_velocity_chirality",
        arrays,
        {
            "chirality_core_radius": np.asarray(
                LOCAL_CONTRAST_LENGTH,
                dtype=np.float64,
            ),
            "chirality_annulus_inner_radius": np.asarray(
                LOCAL_CONTRAST_LENGTH,
                dtype=np.float64,
            ),
            "chirality_annulus_outer_radius": np.asarray(
                3.0 * LOCAL_CONTRAST_LENGTH,
                dtype=np.float64,
            ),
        },
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    _plot_velocity_chirality_summary(cases, arrays, output_png)
    return arrays


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
    steps = np.asarray(fields.steps, dtype=np.int64)
    n_frames = min(coords.shape[0], disclinations.shape[0], steps.shape[0])
    coords = coords[:n_frames]
    disclinations = disclinations[:n_frames]
    steps = steps[:n_frames]

    x_edges = np.asarray(fields.x_edges, dtype=np.float64)
    box_length_x = float(x_edges[-1] - x_edges[0])
    (
        speed,
        abs_v_residual,
        abs_delta_chirality,
        velocity_direction,
        delta_chirality_sign,
    ) = (
        moving_defect_frontback_chirality(
            np.ascontiguousarray(coords, dtype=np.float64),
            np.ascontiguousarray(disclinations, dtype=np.bool_),
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
        "abs_v_residual": abs_v_residual,
        "abs_delta_chirality": abs_delta_chirality,
        "velocity_direction": velocity_direction,
        "delta_chirality_sign": delta_chirality_sign,
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
            "abs_v_residual",
            "abs_delta_chirality",
            "velocity_direction",
            "delta_chirality_sign",
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
            "abs_v_residual": np.asarray(data["abs_v_residual"], dtype=np.float64),
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
            "sample_radii": np.asarray(data["sample_radii"], dtype=np.float64),
        }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman_correlation(x_values: np.ndarray, y_values: np.ndarray) -> float:
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if np.count_nonzero(finite) < 2:
        return np.nan

    x_ranks = _average_ranks(np.asarray(x_values[finite], dtype=np.float64))
    y_ranks = _average_ranks(np.asarray(y_values[finite], dtype=np.float64))
    x_centered = x_ranks - np.mean(x_ranks)
    y_centered = y_ranks - np.mean(y_ranks)
    denominator = math.sqrt(
        float(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered))
    )
    if denominator == 0.0:
        return np.nan
    return float(np.sum(x_centered * y_centered) / denominator)


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


def _plot_moving_frontback_chirality(
    arrays: dict[str, np.ndarray],
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    speed = np.asarray(arrays["speed"], dtype=np.float64)
    abs_delta = np.asarray(arrays["abs_delta_chirality"], dtype=np.float64)
    sample_radii = np.asarray(arrays["sample_radii"], dtype=np.float64)

    finite_speed = np.isfinite(speed) & np.isfinite(abs_delta)
    finite_radii = np.isfinite(sample_radii)
    radius_values = np.unique(sample_radii[finite_radii])
    if radius_values.size == 0:
        radius_values = np.asarray([np.nan], dtype=np.float64)

    n_panels = radius_values.size
    n_cols = min(4, max(1, int(math.ceil(math.sqrt(n_panels)))))
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 2.8 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for axis_idx, radius in enumerate(radius_values):
        ax = axes.flat[axis_idx]
        if np.isfinite(radius):
            mask = finite_speed & (sample_radii == radius)
            radius_label = f"R = {radius:g}"
        else:
            mask = finite_speed & ~finite_radii
            radius_label = "R = nan"

        ax.scatter(
            speed[mask],
            abs_delta[mask],
            color="#111111",
            s=16,
            alpha=0.65,
            edgecolors="none",
        )
        rho = _spearman_correlation(speed[mask], abs_delta[mask])
        rho_label = "nan" if not np.isfinite(rho) else f"{rho:.3f}"
        ax.set_title(radius_label)
        ax.text(
            0.04,
            0.94,
            f"Spearman rho = {rho_label}\nn = {np.count_nonzero(mask)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
        ax.set_xlabel("|v_defect|")
        ax.set_ylabel("|Delta chi_frontback|")
        ax.grid(True, alpha=0.25)

    for axis_idx in range(n_panels, axes.size):
        axes.flat[axis_idx].set_visible(False)

    fig.suptitle("|v_defect| vs |Delta chi_frontback| by radius")
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def _plot_moving_frontback_residual(
    arrays: dict[str, np.ndarray],
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    residual_speed = np.asarray(arrays["abs_v_residual"], dtype=np.float64)
    abs_delta = np.asarray(arrays["abs_delta_chirality"], dtype=np.float64)
    sample_radii = np.asarray(arrays["sample_radii"], dtype=np.float64)

    finite_values = np.isfinite(residual_speed) & np.isfinite(abs_delta)
    finite_radii = np.isfinite(sample_radii)
    radius_values = np.unique(sample_radii[finite_radii])
    if radius_values.size == 0:
        radius_values = np.asarray([np.nan], dtype=np.float64)

    n_panels = radius_values.size
    n_cols = min(4, max(1, int(math.ceil(math.sqrt(n_panels)))))
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 2.8 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for axis_idx, radius in enumerate(radius_values):
        ax = axes.flat[axis_idx]
        if np.isfinite(radius):
            mask = finite_values & (sample_radii == radius)
            radius_label = f"R = {radius:g}"
        else:
            mask = finite_values & ~finite_radii
            radius_label = "R = nan"

        ax.scatter(
            residual_speed[mask],
            abs_delta[mask],
            color="#111111",
            s=16,
            alpha=0.65,
            edgecolors="none",
        )
        ax.set_title(radius_label)
        ax.text(
            0.04,
            0.94,
            f"n = {np.count_nonzero(mask)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
        ax.set_xlabel("|v_residual|")
        ax.set_ylabel("|Delta chi_frontback|")
        ax.grid(True, alpha=0.25)

    for axis_idx in range(n_panels, axes.size):
        axes.flat[axis_idx].set_visible(False)

    fig.suptitle("|v_residual| vs |Delta chi_frontback| by radius")
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def run_moving_frontback_chirality(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    output_npz = NPZ_OUTPUT_DIR / "moving_defect_frontback_chirality.npz"
    output_png = PLOT_OUTPUT_DIR / "moving_defect_frontback_chirality.png"
    residual_output_png = (
        PLOT_OUTPUT_DIR / "moving_defect_frontback_residual_chirality.png"
    )
    if (
        not overwrite
        and output_npz.exists()
        and _frontback_cache_matches(output_npz, cases, frame_start, frame_stop)
    ):
        arrays = _load_frontback_cache(output_npz)
        if not output_png.exists():
            _plot_moving_frontback_chirality(arrays, output_png)
        if not residual_output_png.exists():
            _plot_moving_frontback_residual(arrays, residual_output_png)
        print(f"using cached moving-defect front/back chirality from {output_npz}")
        return arrays

    chunks: dict[str, list[np.ndarray]] = {
        "speed": [],
        "abs_v_residual": [],
        "abs_delta_chirality": [],
        "velocity_direction": [],
        "delta_chirality_sign": [],
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
            "abs_v_residual",
            "abs_delta_chirality",
            "velocity_direction",
            "delta_chirality_sign",
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
    _plot_moving_frontback_chirality(arrays, output_png)
    _plot_moving_frontback_residual(arrays, residual_output_png)
    return arrays


def run_shell_profiles(
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
    output_npz = NPZ_OUTPUT_DIR / "disclination_order_shell_profiles.npz"
    output_pngs = (
        PLOT_OUTPUT_DIR / "disclination_S_shell_profile.png",
        PLOT_OUTPUT_DIR / "disclination_hexatic_order_shell_profile.png",
        PLOT_OUTPUT_DIR / "disclination_chirality_shell_profile.png",
    )
    try:
        arrays = load_cached_metric_values(
            output_npz,
            value_names,
            cases,
            frame_start,
            frame_stop,
            overwrite=overwrite,
        )
    except FileExistsError:
        arrays = None
    if arrays is not None:
        if _profile_plot_missing(output_pngs):
            _plot_shell_profiles(cases, arrays, output_pngs)
        print(f"using cached disclination shell profiles from {output_npz}")
        run_velocity_chirality_summary(
            cases,
            arrays,
            frame_start=frame_start,
            frame_stop=frame_stop,
            overwrite=overwrite,
        )
        run_moving_frontback_chirality(
            cases,
            frame_start=frame_start,
            frame_stop=frame_stop,
            overwrite=overwrite,
        )
        return arrays

    values = {name: [] for name in value_names}
    for case in cases:
        case_values = disclination_shell_profile_values_for_case(
            case,
            frame_start,
            frame_stop,
        )
        for name in value_names:
            values[name].append(case_values[name])

    arrays = {
        name: np.asarray(series, dtype=np.float64)
        for name, series in values.items()
    }
    save_metric_npz(
        output_npz,
        cases,
        "disclination_order_shell_profiles",
        arrays,
        {
            "local_profile_bin_edges": np.asarray(
                LOCAL_PROFILE_BIN_EDGES,
                dtype=np.float64,
            ),
            "local_profile_bin_labels": LOCAL_PROFILE_BIN_LABELS,
            "local_contrast_length": np.asarray(
                LOCAL_CONTRAST_LENGTH,
                dtype=np.float64,
            ),
            "chirality_radius": np.asarray(
                cylinder.ANALYSIS.neighbor_count_radius,
                dtype=np.float64,
            ),
        },
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    _plot_shell_profiles(cases, arrays, output_pngs)
    run_velocity_chirality_summary(
        cases,
        arrays,
        frame_start=frame_start,
        frame_stop=frame_stop,
        overwrite=overwrite,
    )
    run_moving_frontback_chirality(
        cases,
        frame_start=frame_start,
        frame_stop=frame_stop,
        overwrite=overwrite,
    )
    return arrays


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
