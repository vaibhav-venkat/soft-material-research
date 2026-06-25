from __future__ import annotations

import numpy as np

from hexatic.chirality.translation import compute_translation_chirality_trajectory
from hexatic.constants import cylinder
from hexatic.radii_analysis.cases import RadiusCase

from .best_fit import fit_payload
from .common import (
    FRAME_START,
    FRAME_STOP,
    NPZ_OUTPUT_DIR,
    PLOT_OUTPUT_DIR,
    load_cached_metric_values,
    load_metric_fit_curves,
    radii_for_cases,
    save_metric_npz,
    translation_chirality_path,
)
from .numba_kernels import mean_abs_frame_mean
from .plotting import plot_for_cases, plots_missing


def chirality_value_for_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
) -> float:
    path = translation_chirality_path(case)
    if path.exists():
        with np.load(path) as data:
            chirality = np.asarray(data["chirality"], dtype=np.float64)
    else:
        trajectory = compute_translation_chirality_trajectory(
            case.trajectory_gsd,
            neighborhood_radius=cylinder.ANALYSIS.neighbor_count_radius,
        )
        chirality = np.asarray(trajectory.chirality, dtype=np.float64)

    return mean_abs_frame_mean(chirality, frame_start, frame_stop)


def run(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    output_npz = NPZ_OUTPUT_DIR / "chirality.npz"
    output_png = PLOT_OUTPUT_DIR / "chirality.png"
    value_names = ("mean_abs",)
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
                title="Mean absolute translational chirality vs radius",
                ylabel="mean(mean(abs(chirality))) son",
                fits=fits,
            )
        print(f"using cached chirality values from {output_npz}")
        return arrays

    values = {
        "mean_abs": np.asarray(
            [
                chirality_value_for_case(case, frame_start, frame_stop)
                for case in cases
            ],
            dtype=np.float64,
        )
    }
    fits, payload = fit_payload(radii_for_cases(cases), values)
    save_metric_npz(
        output_npz,
        cases,
        "chirality",
        values,
        payload,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    plot_for_cases(
        cases,
        values,
        output_png,
        title="Mean absolute translational chirality vs radius",
        ylabel="mean(mean(abs(chirality)))",
        fits=fits,
    )
    return values
