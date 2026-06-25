from __future__ import annotations

from pathlib import Path

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
    finite_nanmean,
    frame_indices,
    group_names_for_cases,
    labels_for_cases,
    load_active_fields,
    load_cached_metric_values,
    neighbor_counts_path,
    radii_for_cases,
    save_metric_npz,
)
from .disclination import _load_neighbor_counts
from .numba_kernels import disclination_translation_chirality_squared_mean
from .plotting import companion_circumference_plot_path, plot_radius_values


CIRCUMFERENCE_REFERENCE_CASE_ID = "circ_60_0D"


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


def _frame_mean_for_masked_values(
    values: np.ndarray,
    mask: np.ndarray,
    frame_start: int,
    frame_stop: int,
    *,
    square: bool = False,
) -> float:
    selected = frame_indices(values.shape[0], frame_start, frame_stop)
    frame_means: list[float] = []
    for frame_idx in selected:
        frame_values = np.asarray(values[frame_idx][mask[frame_idx]], dtype=np.float64)
        finite = np.isfinite(frame_values)
        if not np.any(finite):
            continue
        frame_values = frame_values[finite]
        if square:
            frame_values = frame_values * frame_values
        frame_means.append(float(np.mean(frame_values)))
    return finite_nanmean(np.asarray(frame_means, dtype=np.float64))


def _nematic_s_for_masked_particles(
    direction_cylindrical: np.ndarray,
    mask: np.ndarray,
    frame_start: int,
    frame_stop: int,
) -> float:
    selected = frame_indices(direction_cylindrical.shape[0], frame_start, frame_stop)
    frame_values: list[float] = []
    for frame_idx in selected:
        directions = np.asarray(
            direction_cylindrical[frame_idx][mask[frame_idx]],
            dtype=np.float64,
        )
        if directions.size == 0:
            continue
        p_x = directions[:, 0]
        p_theta = directions[:, 2]
        norm = np.hypot(p_x, p_theta)
        finite = np.isfinite(norm) & (norm > 0.0)
        if not np.any(finite):
            continue

        u_x = p_x[finite] / norm[finite]
        u_theta = p_theta[finite] / norm[finite]
        q_xx = np.mean(2.0 * u_x * u_x - 1.0)
        q_xtheta = np.mean(2.0 * u_x * u_theta)
        frame_values.append(float(np.hypot(q_xx, q_xtheta)))
    return finite_nanmean(np.asarray(frame_values, dtype=np.float64))


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

    return {
        "S": _nematic_s_for_masked_particles(
            direction_cylindrical,
            disclinations,
            frame_start,
            frame_stop,
        ),
        "hexatic_order": _frame_mean_for_masked_values(
            hexatic_abs,
            disclinations,
            frame_start,
            frame_stop,
        ),
        "chirality_disclinations": disclination_translation_chirality_squared_mean(
            np.ascontiguousarray(coords, dtype=np.float64),
            np.ascontiguousarray(disclinations, dtype=np.bool_),
            frame_start,
            frame_stop,
            cylinder.ANALYSIS.neighbor_count_radius,
            box_length_x,
        ),
    }


def _plot_values(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "S": values["S"],
        "hexatic_order": values["hexatic_order"],
        "chirality_disclinations": values["chirality_disclinations"],
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
        "Disclination-particle order metrics vs radius",
        "mean over disclination particles",
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
        "Disclination-particle order metrics vs radius (circumference sweep)",
        "mean over disclination particles",
        case_labels=labels[circumference_mask],
        group_names=group_names[circumference_mask],
        xlabel="Circumference C / D",
    )


def run(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    value_names = ("S", "hexatic_order", "chirality_disclinations")
    cases = tuple(
        case
        for case in cases
        if not case.case_id.startswith("circ_")
        or case.case_id == CIRCUMFERENCE_REFERENCE_CASE_ID
    )
    if not cases:
        print("skipped disclination_order_fields: no C = 60D or radius cases selected")
        return {name: np.asarray([], dtype=np.float64) for name in value_names}

    output_npz = NPZ_OUTPUT_DIR / "disclination_order_fields.npz"
    output_png = PLOT_OUTPUT_DIR / "disclination_order_fields.png"
    arrays = load_cached_metric_values(
        output_npz,
        value_names,
        cases,
        frame_start,
        frame_stop,
        overwrite=overwrite,
    )
    if arrays is not None:
        if _plot_missing(cases, output_png):
            _plot_for_cases_without_fit(cases, arrays, output_png)
        print(f"using cached disclination_order_fields values from {output_npz}")
        return arrays

    values = {name: [] for name in value_names}
    for case in cases:
        case_values = disclination_order_values_for_case(
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
        "disclination_order_fields",
        arrays,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    _plot_for_cases_without_fit(cases, arrays, output_png)
    return arrays
