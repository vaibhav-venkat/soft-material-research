from __future__ import annotations

import gsd.hoomd
import numpy as np

from hexatic.active_matter_cylinder.math_utils import (
    _active_direction_from_quaternion,
    _cylindrical_components,
)
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
from .numba_kernels import tangent_nematic_means
from .plotting import plot_for_cases, plots_missing


def nematic_values_for_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
) -> dict[str, float]:
    fields_path = active_fields_path(case)
    if fields_path.exists():
        fields = load_active_fields(fields_path)
        return _nematic_values_from_arrays(
            np.asarray(fields.direction_cylindrical, dtype=np.float64),
            np.asarray(fields.shell_mask, dtype=bool),
            frame_start,
            frame_stop,
        )
    return _nematic_values_from_gsd(case, frame_start, frame_stop)


def _nematic_values_from_arrays(
    direction_cylindrical: np.ndarray,
    shell_mask: np.ndarray,
    frame_start: int,
    frame_stop: int,
) -> dict[str, float]:
    s_shell, s_core, q_xx_shell, q_xtheta_shell = tangent_nematic_means(
        np.ascontiguousarray(direction_cylindrical, dtype=np.float64),
        np.ascontiguousarray(shell_mask, dtype=np.bool_),
        frame_start,
        frame_stop,
    )
    return {
        "s_shell": s_shell,
        "s_core": s_core,
        "q_xx_shell": q_xx_shell,
        "q_xtheta_shell": q_xtheta_shell,
    }


def _nematic_values_from_gsd(
    case: RadiusCase,
    frame_start: int,
    frame_stop: int,
) -> dict[str, float]:
    directions: list[np.ndarray] = []
    shell_masks: list[np.ndarray] = []
    with gsd.hoomd.open(name=str(case.trajectory_gsd), mode="r") as source:
        selected = set(frame_indices(len(source), frame_start, frame_stop).tolist())
        for frame_idx, frame in enumerate(source):
            if frame_idx not in selected:
                continue
            particles = frame.particles
            if particles.position is None or particles.orientation is None:
                continue
            positions = np.asarray(particles.position, dtype=np.float64)
            theta = np.mod(np.arctan2(positions[:, 1], positions[:, 2]), 2.0 * np.pi)
            active_direction = _active_direction_from_quaternion(particles.orientation)
            directions.append(_cylindrical_components(active_direction, theta))
            shell_masks.append(shell_mask_for_positions(positions, case))

    if not directions:
        return {
            "s_shell": np.nan,
            "s_core": np.nan,
            "q_xx_shell": np.nan,
            "q_xtheta_shell": np.nan,
        }
    return _nematic_values_from_arrays(
        np.asarray(directions, dtype=np.float64),
        np.asarray(shell_masks, dtype=np.bool_),
        0,
        len(directions),
    )


def run(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    output_npz = NPZ_OUTPUT_DIR / "nematic.npz"
    output_png = PLOT_OUTPUT_DIR / "nematic.png"
    value_names = ("s_shell", "s_core", "q_xx_shell", "q_xtheta_shell")
    arrays = load_cached_metric_values(
        output_npz,
        value_names,
        cases,
        frame_start,
        frame_stop,
        overwrite=overwrite,
    )
    if arrays is not None:
        if plots_missing(cases, output_png):
            fits = load_metric_fit_curves(output_npz, value_names)
            plot_for_cases(
                cases,
                arrays,
                output_png,
                title="Tangent nematic order vs radius",
                ylabel="2D tangent nematic order",
                fits=fits,
            )
        print(f"using cached nematic values from {output_npz}")
        return arrays

    values = {name: [] for name in value_names}
    for case in cases:
        case_values = nematic_values_for_case(case, frame_start, frame_stop)
        for name in value_names:
            values[name].append(case_values[name])

    arrays = {
        name: np.asarray(series, dtype=np.float64)
        for name, series in values.items()
    }
    fits, payload = fit_payload(radii_for_cases(cases), arrays)
    payload.update(
        {
            "tensor_convention": np.asarray("Q = <2 u_a u_b - delta_ab>"),
            "orientation_basis": np.asarray("u = normalized tangent (p_x, p_theta)"),
        }
    )
    save_metric_npz(
        output_npz,
        cases,
        "nematic",
        arrays,
        payload,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    plot_for_cases(
        cases,
        arrays,
        output_png,
        title="Tangent nematic order vs radius",
        ylabel="2D tangent nematic order",
        fits=fits,
    )
    return arrays
