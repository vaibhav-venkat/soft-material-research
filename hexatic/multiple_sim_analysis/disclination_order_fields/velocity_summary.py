from __future__ import annotations

from pathlib import Path

import numpy as np

from hexatic.radii_analysis.cases import RadiusCase

from ..common import (
    FRAME_START,
    FRAME_STOP,
    NPZ_OUTPUT_DIR,
    PLOT_OUTPUT_DIR,
    load_cached_metric_values,
    radii_for_cases,
    labels_for_cases,
    group_names_for_cases,
    save_metric_npz,
)
from ..plotting import plot_radius_values
from .shared import LOCAL_CONTRAST_LENGTH


def _abs_defect_velocity_stats(
    velocity_arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    plus = np.abs(np.asarray(velocity_arrays["plus_1"], dtype=np.float64))
    minus = np.abs(np.asarray(velocity_arrays["minus_1"], dtype=np.float64))
    values = np.column_stack((plus, minus))
    finite = np.isfinite(values)
    counts = np.count_nonzero(finite, axis=1)
    sums = np.where(finite, values, 0.0).sum(axis=1)
    means = np.divide(
        sums,
        counts,
        out=np.full(values.shape[0], np.nan, dtype=np.float64),
        where=counts > 0,
    )
    medians = np.full(values.shape[0], np.nan, dtype=np.float64)
    for row_idx in range(values.shape[0]):
        if np.any(finite[row_idx]):
            medians[row_idx] = float(np.median(values[row_idx, finite[row_idx]]))
    return means, medians


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

    from ..velocity import disclination_velocity_values_for_case

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
    mean_abs_v_defect, median_abs_v_defect = _abs_defect_velocity_stats(velocity_arrays)
    arrays = {
        "abs_v_defect": mean_abs_v_defect,
        "median_abs_v_defect": median_abs_v_defect,
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

