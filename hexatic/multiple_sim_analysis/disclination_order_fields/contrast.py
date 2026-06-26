from __future__ import annotations

from pathlib import Path

import numpy as np

from hexatic.constants import cylinder
from hexatic.radii_analysis.cases import RadiusCase

from ..common import (
    FRAME_START,
    FRAME_STOP,
    active_fields_path,
    case_ids_for_cases,
    group_names_for_cases,
    labels_for_cases,
    load_active_fields,
    neighbor_counts_path,
    radii_for_cases,
)
from ..disclination import _load_neighbor_counts
from ..plotting import companion_circumference_plot_path, plot_radius_values
from .shared import (
    CIRCUMFERENCE_REFERENCE_CASE_ID,
    LOCAL_CONTRAST_LENGTH,
    _disclination_mask,
    _load_hexatic_abs,
    _validate_particle_frame_shape,
    hexatic_order_path,
    local_disclination_field_contrasts,
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
    regular_mask = (group_names != "circumference") | (
        case_ids == CIRCUMFERENCE_REFERENCE_CASE_ID
    )
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
