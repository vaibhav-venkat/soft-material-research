from __future__ import annotations

import numpy as np

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
    load_cached_metric_values,
    load_metric_fit_curves,
    neighbor_counts_path,
    radii_for_cases,
    save_metric_npz,
)
from .plotting import plot_for_cases, plots_missing


def _load_neighbor_counts(path) -> np.ndarray:
    table = np.loadtxt(path, dtype=np.int64)
    if table.ndim == 1:
        table = table[np.newaxis, :]
    frame_indices_table = table[:, 0]
    particle_indices = table[:, 2]
    counts_flat = table[:, 3]
    n_frames = int(frame_indices_table.max()) + 1
    n_particles = int(particle_indices.max()) + 1
    counts = np.full((n_frames, n_particles), -1, dtype=np.int64)
    counts[frame_indices_table, particle_indices] = counts_flat
    if np.any(counts < 0):
        raise ValueError(f"Neighbor count table is incomplete: {path}")
    return counts


def disclination_values_for_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
) -> dict[str, float]:
    counts = _load_neighbor_counts(neighbor_counts_path(case))
    selected = frame_indices(counts.shape[0], frame_start, frame_stop)
    charges = cylinder.NEIGHBORS - counts[selected]
    plus_counts = np.count_nonzero(charges == 1, axis=1)
    minus_counts = np.count_nonzero(charges == -1, axis=1)
    return {
        "plus_1": finite_nanmean(plus_counts),
        "minus_1": finite_nanmean(minus_counts),
    }


def run(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    output_npz = NPZ_OUTPUT_DIR / "disclination.npz"
    output_png = PLOT_OUTPUT_DIR / "disclination.png"
    value_names = ("plus_1", "minus_1")
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
                title="Mean disclination counts vs radius",
                ylabel="mean count",
                fits=fits,
            )
        print(f"using cached disclination values from {output_npz}")
        return arrays

    values = {"plus_1": [], "minus_1": []}
    for case in cases:
        case_values = disclination_values_for_case(case, frame_start, frame_stop)
        values["plus_1"].append(case_values["plus_1"])
        values["minus_1"].append(case_values["minus_1"])
    arrays = {name: np.asarray(series, dtype=np.float64) for name, series in values.items()}
    fits, payload = fit_payload(radii_for_cases(cases), arrays)
    save_metric_npz(
        output_npz,
        cases,
        "disclination",
        arrays,
        payload,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    plot_for_cases(
        cases,
        arrays,
        output_png,
        title="Mean disclination counts vs radius",
        ylabel="mean count",
        fits=fits,
    )
    return arrays
