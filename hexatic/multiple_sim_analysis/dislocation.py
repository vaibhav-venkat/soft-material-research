from __future__ import annotations

import gsd.hoomd
import numpy as np

from hexatic import analysis as hx
from hexatic.constants import cylinder
from hexatic.radii_analysis.cases import RadiusCase

from .best_fit import fit_payload
from .common import (
    FRAME_START,
    FRAME_STOP,
    NPZ_OUTPUT_DIR,
    PLOT_OUTPUT_DIR,
    finite_nanmean,
    frame_indices,
    hexatic_velocity_gsd_path,
    load_cached_metric_values,
    load_metric_fit_curves,
    neighbor_counts_path,
    radii_for_cases,
    save_metric_npz,
)
from .normalized_counts import (
    normalized_count_plots_missing,
    write_normalized_count_plots,
)
from .plotting import plot_for_cases, plots_missing
from .disclination import _load_neighbor_counts


def dislocation_value_for_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
) -> float:
    path = hexatic_velocity_gsd_path(case)
    if not path.exists():
        return _dislocation_value_from_neighbor_counts(case, frame_start, frame_stop)

    counts: list[int] = []
    with gsd.hoomd.open(name=str(path), mode="r") as source:
        selected = set(frame_indices(len(source), frame_start, frame_stop).tolist())
        for frame_idx, frame in enumerate(source):
            if frame_idx not in selected:
                continue
            orientation = getattr(frame.particles, "orientation", None)
            if orientation is None:
                counts.append(0)
                continue
            orientation = np.asarray(orientation)
            if orientation.ndim != 2 or orientation.shape[1] < 4:
                counts.append(0)
                continue
            counts.append(int(np.count_nonzero(orientation[:, 0] == 1)))
    return finite_nanmean(np.asarray(counts, dtype=np.float64))


def _dislocation_value_from_neighbor_counts(
    case: RadiusCase,
    frame_start: int,
    frame_stop: int,
) -> float:
    neighbor_counts = _load_neighbor_counts(neighbor_counts_path(case))
    selected = set(frame_indices(neighbor_counts.shape[0], frame_start, frame_stop).tolist())
    counts: list[int] = []
    with gsd.hoomd.open(name=str(case.trajectory_gsd), mode="r") as source:
        for frame_idx, frame in enumerate(source):
            if frame_idx not in selected:
                continue
            positions = np.asarray(frame.particles.position, dtype=np.float64)
            charges = cylinder.NEIGHBORS - neighbor_counts[frame_idx]
            dislocations = hx.identify_dislocation_particles_frame(
                positions,
                charges,
                pair_distance=cylinder.ANALYSIS.dislocation_pair_distance,
                box_length_x=float(frame.configuration.box[0]),
            )
            counts.append(int(np.count_nonzero(dislocations)))
    return finite_nanmean(np.asarray(counts, dtype=np.float64))


def run(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    output_npz = NPZ_OUTPUT_DIR / "dislocation.npz"
    output_png = PLOT_OUTPUT_DIR / "dislocation.png"
    value_names = ("dislocation",)
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
                title="Mean dislocation count vs radius",
                ylabel="mean dislocation count",
                fits=fits,
            )
        if normalized_count_plots_missing("dislocation"):
            write_normalized_count_plots(
                cases,
                arrays,
                metric_name="dislocation",
                title="Mean dislocation count",
            )
        print(f"using cached dislocation values from {output_npz}")
        return arrays

    values = {
        "dislocation": np.asarray(
            [
                dislocation_value_for_case(case, frame_start, frame_stop)
                for case in cases
            ],
            dtype=np.float64,
        )
    }
    fits, payload = fit_payload(radii_for_cases(cases), values)
    save_metric_npz(
        output_npz,
        cases,
        "dislocation",
        values,
        payload,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    plot_for_cases(
        cases,
        values,
        output_png,
        title="Mean dislocation count vs radius",
        ylabel="mean dislocation count",
        fits=fits,
    )
    write_normalized_count_plots(
        cases,
        values,
        metric_name="dislocation",
        title="Mean dislocation count",
    )
    return values
