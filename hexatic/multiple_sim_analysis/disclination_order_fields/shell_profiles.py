from __future__ import annotations

from pathlib import Path

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
    load_cached_metric_values,
    neighbor_counts_path,
    radii_for_cases,
    save_metric_npz,
)
from ..disclination import _load_neighbor_counts
from ..plotting import plot_radius_values
from .moving import run_moving_defect_velocity_annulus
from .shared import (
    CIRCUMFERENCE_REFERENCE_CASE_ID,
    LOCAL_CONTRAST_LENGTH,
    LOCAL_PROFILE_BIN_EDGES,
    LOCAL_PROFILE_BIN_LABELS,
    LOCAL_PROFILE_COLORS,
    _disclination_mask,
    _load_hexatic_abs,
    _validate_particle_frame_shape,
    hexatic_order_path,
    local_disclination_field_profiles,
)
from .velocity_summary import run_velocity_chirality_summary


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
    regular_mask = (group_names != "circumference") | (
        case_ids == CIRCUMFERENCE_REFERENCE_CASE_ID
    )
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
        PLOT_OUTPUT_DIR / "disc" /  "disclination_S_shell_profile.png",
        PLOT_OUTPUT_DIR / "disc" / "disclination_hexatic_order_shell_profile.png",
        PLOT_OUTPUT_DIR / "disc"" / disclination_chirality_shell_profile.png",
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
        run_moving_defect_velocity_annulus(
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
    run_moving_defect_velocity_annulus(
        cases,
        frame_start=frame_start,
        frame_stop=frame_stop,
        overwrite=overwrite,
    )
    return arrays
