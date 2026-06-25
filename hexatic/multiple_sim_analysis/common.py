from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gsd.hoomd
import numpy as np

from hexatic import analysis as hx
from hexatic.active_matter_cylinder.config import ActiveMatterFields
from hexatic.constants import cylinder
from hexatic.radii_analysis.cases import (
    HEXATIC_OUTPUT_DIR,
    NPZ_FIELDS_DIR,
    RadiusCase,
    all_cases,
    get_case,
)
from .best_fit import FitCurve

FRAME_START = 70
FRAME_STOP = 100
POPULATIONS = ("all", "shell")

ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ANALYSIS_DIR / "output"
NPZ_OUTPUT_DIR = OUTPUT_DIR / "npz"
PLOT_OUTPUT_DIR = OUTPUT_DIR / "plots"
FIT_OUTPUT_DIR = OUTPUT_DIR / "fits"


@dataclass(frozen=True)
class AggregateResult:
    metric_name: str
    radii: np.ndarray
    case_ids: np.ndarray
    labels: np.ndarray
    group_names: np.ndarray
    values: dict[str, np.ndarray]
    fit_payload: dict[str, np.ndarray | float | int | str]


def ensure_output_dirs() -> None:
    NPZ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def selected_cases(
    case_ids: Iterable[str],
    include_all: bool,
    include_long_axis: bool = False,
) -> tuple[RadiusCase, ...]:
    case_id_tuple = tuple(case_ids)
    if include_all:
        return all_cases(include_long_axis=include_long_axis)
    if case_id_tuple:
        return tuple(
            get_case(case_id, include_long_axis=include_long_axis)
            for case_id in case_id_tuple
        )
    raise SystemExit("Select --all or one or more --case values.")


def radii_for_cases(cases: tuple[RadiusCase, ...]) -> np.ndarray:
    return np.asarray([case.radius for case in cases], dtype=np.float64)


def case_ids_for_cases(cases: tuple[RadiusCase, ...]) -> np.ndarray:
    return np.asarray([case.case_id for case in cases])


def labels_for_cases(cases: tuple[RadiusCase, ...]) -> np.ndarray:
    return np.asarray([case.label or case.case_id for case in cases])


def group_names_for_cases(cases: tuple[RadiusCase, ...]) -> np.ndarray:
    groups = []
    for case in cases:
        if case.case_id.startswith("circ_"):
            groups.append("circumference")
        elif case.case_id.startswith("radius_"):
            groups.append("scaled_radius")
        else:
            groups.append("other")
    return np.asarray(groups)


def frame_indices(
    n_frames: int,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
) -> np.ndarray:
    start = min(max(frame_start, 0), n_frames)
    stop = min(max(frame_stop, start), n_frames)
    return np.arange(start, stop, dtype=np.int64)


def finite_nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan
    return float(np.mean(values[finite]))


def minimum_image_delta(delta: np.ndarray, box_length: float) -> np.ndarray:
    return delta - box_length * np.round(delta / box_length)


def particle_masses(particles, n_particles: int) -> np.ndarray:
    masses = getattr(particles, "mass", None)
    if masses is None:
        return np.ones(n_particles, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if masses.shape != (n_particles,):
        return np.ones(n_particles, dtype=np.float64)
    return masses


def unwrapped_x_positions(particles, box_length_x: float) -> np.ndarray:
    positions = np.asarray(particles.position, dtype=np.float64)
    x_positions = positions[:, 0].copy()
    images = getattr(particles, "image", None)
    if images is not None:
        images = np.asarray(images)
        if images.shape == positions.shape:
            x_positions += images[:, 0] * box_length_x
    return x_positions


def shell_mask_for_positions(positions: np.ndarray, case: RadiusCase) -> np.ndarray:
    return hx.get_dynamic_values(
        positions,
        contain_all=False,
        cylinder_radius=case.radius,
        cutoff=cylinder.ANALYSIS.wall_cutoff,
    ).shell_mask


def center_of_mass_x(positions: np.ndarray, box_length_x: float) -> float:
    if positions.size == 0:
        return np.nan
    coords = hx.get_new_coords(positions)
    center = hx.get_center_of_mass_x_theta(
        coords,
        periodic_x=True,
        box_length_x=box_length_x,
    )
    return float(center.x)


def center_of_mass_x_from_coords(coords: np.ndarray, box_length_x: float) -> float:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.size == 0:
        return np.nan
    center = hx.get_center_of_mass_x_theta(
        coords,
        periodic_x=True,
        box_length_x=box_length_x,
    )
    return float(center.x)


def active_fields_path(case: RadiusCase) -> Path:
    return NPZ_FIELDS_DIR / f"{case.case_id}_active_matter_fields.npz"


def translation_chirality_path(case: RadiusCase) -> Path:
    return NPZ_FIELDS_DIR / f"{case.case_id}_translation_chirality_fields.npz"


def neighbor_counts_path(case: RadiusCase) -> Path:
    return HEXATIC_OUTPUT_DIR / f"{case.case_id}_neighbor_counts.txt"


def hexatic_velocity_gsd_path(case: RadiusCase) -> Path:
    return HEXATIC_OUTPUT_DIR / f"{case.case_id}_hexatic_velocity.gsd"


def load_active_fields(path: str | Path) -> ActiveMatterFields:
    with np.load(path) as data:
        return ActiveMatterFields(
            steps=np.asarray(data["steps"]),
            x_edges=np.asarray(data["x_edges"]),
            x_centers=np.asarray(data["x_centers"]),
            theta_edges=np.asarray(data["theta_edges"]),
            theta_centers=np.asarray(data["theta_centers"]),
            coords=np.asarray(data["coords"]),
            shell_mask=np.asarray(data["shell_mask"]),
            rho=np.asarray(data["rho"]),
            active_direction=np.asarray(data["active_direction"]),
            direction_cylindrical=np.asarray(data["direction_cylindrical"]),
            polar_mean=np.asarray(data["polar_mean"]),
            polar_cylindrical=np.asarray(data["polar_cylindrical"]),
            flux_cylindrical=np.asarray(data["flux_cylindrical"]),
            force_density=np.asarray(data["force_density"]),
            force_density_cylindrical=np.asarray(data["force_density_cylindrical"]),
        )


def trajectory_frame_count(input_gsd: str | Path) -> int:
    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        return len(source)


def save_metric_npz(
    filename: str | Path,
    cases: tuple[RadiusCase, ...],
    metric_name: str,
    values: dict[str, np.ndarray],
    fit_payload: dict[str, np.ndarray | float | int | str] | None = None,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
) -> None:
    ensure_output_dirs()
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray | float | int | str] = {
        "metric_name": metric_name,
        "case_ids": case_ids_for_cases(cases),
        "labels": labels_for_cases(cases),
        "group_names": group_names_for_cases(cases),
        "radii": radii_for_cases(cases),
        "frame_start": int(frame_start),
        "frame_stop": int(frame_stop),
    }
    payload.update(values)
    if fit_payload:
        payload.update(fit_payload)
    np.savez_compressed(output_path, **payload)


def load_metric_values(
    filename: str | Path,
    value_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    with np.load(filename) as data:
        return {
            value_name: np.asarray(data[value_name], dtype=np.float64)
            for value_name in value_names
        }


def load_cached_metric_values(
    filename: str | Path,
    value_names: tuple[str, ...],
    cases: tuple[RadiusCase, ...],
    frame_start: int,
    frame_stop: int,
    overwrite: bool = False,
) -> dict[str, np.ndarray] | None:
    input_path = Path(filename)
    if overwrite or not input_path.exists():
        return None

    with np.load(input_path) as data:
        cached_case_ids = np.asarray(data["case_ids"])
        requested_case_ids = case_ids_for_cases(cases)
        cached_start = int(np.asarray(data["frame_start"]).item())
        cached_stop = int(np.asarray(data["frame_stop"]).item())
        has_values = all(value_name in data for value_name in value_names)

    if (
        not has_values
        or cached_start != int(frame_start)
        or cached_stop != int(frame_stop)
        or not np.array_equal(cached_case_ids, requested_case_ids)
    ):
        raise FileExistsError(
            f"Cached metric output {input_path} does not match the requested "
            "cases/frame window. Use --overwrite to regenerate it."
        )
    return load_metric_values(input_path, value_names)


def load_metric_fit_curves(
    filename: str | Path,
    value_names: tuple[str, ...],
) -> dict[str, FitCurve | None]:
    fits: dict[str, FitCurve | None] = {}
    with np.load(filename) as data:
        for value_name in value_names:
            prefix = f"fit_{value_name}"
            success_name = f"{prefix}_success"
            radii_name = f"{prefix}_radii"
            values_name = f"{prefix}_values"
            if (
                success_name in data
                and int(np.asarray(data[success_name]).item()) == 1
                and radii_name in data
                and values_name in data
            ):
                fits[value_name] = FitCurve(
                    radii=np.asarray(data[radii_name], dtype=np.float64),
                    values=np.asarray(data[values_name], dtype=np.float64),
                )
            else:
                fits[value_name] = None
    return fits
