from __future__ import annotations

import gsd.hoomd
import numpy as np

from hexatic.active_matter_cylinder.config import LOCAL_POCKET_RADIUS
from hexatic.active_matter_cylinder.math_utils import (
    _logged_particle_array,
    _pocket_vector_density,
)
from hexatic.constants import cylinder
from hexatic.radii_analysis.cases import RadiusCase

from .best_fit import fit_payload
from .common import (
    FRAME_START,
    FRAME_STOP,
    NPZ_OUTPUT_DIR,
    active_fields_path,
    finite_nanmean,
    frame_indices,
    load_active_fields,
    load_cached_metric_values,
    radii_for_cases,
    save_metric_npz,
    shell_mask_for_positions,
)
from .numba_kernels import mean_by_population


def force_density_values_for_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
) -> dict[str, float]:
    fields_path = active_fields_path(case)
    if not fields_path.exists():
        return _force_density_values_from_gsd(case, frame_start, frame_stop)

    fields = load_active_fields(fields_path)
    force_x = np.asarray(fields.force_density_cylindrical[..., 0], dtype=np.float64)
    shell = np.asarray(fields.shell_mask, dtype=bool)
    all_mean, shell_mean = mean_by_population(force_x, shell, frame_start, frame_stop)
    return {
        "all": all_mean,
        "shell": shell_mean,
    }


def _force_density_values_from_gsd(
    case: RadiusCase,
    frame_start: int,
    frame_stop: int,
) -> dict[str, float]:
    all_values: list[np.ndarray] = []
    shell_values: list[np.ndarray] = []
    with gsd.hoomd.open(name=str(case.trajectory_gsd), mode="r") as source:
        selected = set(frame_indices(len(source), frame_start, frame_stop).tolist())
        for frame_idx, frame in enumerate(source):
            if frame_idx not in selected:
                continue
            particles = frame.particles
            if particles.position is None:
                continue
            positions = np.asarray(particles.position, dtype=np.float64)
            forces = _logged_particle_array(frame, "forces", positions.shape[0])
            force_velocity = forces[:, :3] / float(cylinder.SIMULATION.gamma)
            pocket_force_density = _pocket_vector_density(
                positions,
                force_velocity,
                float(frame.configuration.box[0]),
                LOCAL_POCKET_RADIUS,
            )
            force_x = pocket_force_density[:, 0]
            shell = shell_mask_for_positions(positions, case)
            all_values.append(force_x)
            shell_values.append(force_x[shell])

    return {
        "all": finite_nanmean(np.concatenate(all_values)) if all_values else np.nan,
        "shell": finite_nanmean(np.concatenate(shell_values)) if shell_values else np.nan,
    }


def run(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    output_npz = NPZ_OUTPUT_DIR / "force_density.npz"
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
        print(f"using cached force_density values from {output_npz}")
        return arrays

    values = {"all": [], "shell": []}
    for case in cases:
        case_values = force_density_values_for_case(case, frame_start, frame_stop)
        values["all"].append(case_values["all"])
        values["shell"].append(case_values["shell"])
    arrays = {name: np.asarray(series, dtype=np.float64) for name, series in values.items()}
    _, payload = fit_payload(radii_for_cases(cases), arrays)
    save_metric_npz(
        output_npz,
        cases,
        "force_density",
        arrays,
        payload,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    return arrays
