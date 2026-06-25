from __future__ import annotations

import numpy as np

from hexatic.radii_analysis.cases import RadiusCase

from .best_fit import fit_payload
from .common import (
    FRAME_START,
    FRAME_STOP,
    NPZ_OUTPUT_DIR,
    PLOT_OUTPUT_DIR,
    active_fields_path,
    load_active_fields,
    load_cached_metric_values,
    load_metric_fit_curves,
    radii_for_cases,
    save_metric_npz,
)
from .numba_kernels import mean_by_population
from .plotting import plot_for_cases


def polarization_values_for_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
) -> dict[str, float]:
    fields = load_active_fields(active_fields_path(case))
    px = np.asarray(fields.direction_cylindrical[..., 0], dtype=np.float64)
    shell = np.asarray(fields.shell_mask, dtype=bool)
    all_mean, shell_mean = mean_by_population(px, shell, frame_start, frame_stop)
    return {
        "all": all_mean,
        "shell": shell_mean,
    }


def run(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    output_npz = NPZ_OUTPUT_DIR / "polarization.npz"
    output_png = PLOT_OUTPUT_DIR / "polarization.png"
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
        if not output_png.exists():
            fits = load_metric_fit_curves(output_npz, value_names)
            plot_for_cases(
                cases,
                arrays,
                output_png,
                title="Mean x polarization vs radius",
                ylabel="mean direction_cylindrical x",
                fits=fits,
            )
        print(f"using cached polarization values from {output_npz}")
        return arrays

    values = {"all": [], "shell": []}
    for case in cases:
        case_values = polarization_values_for_case(case, frame_start, frame_stop)
        values["all"].append(case_values["all"])
        values["shell"].append(case_values["shell"])
    arrays = {name: np.asarray(series, dtype=np.float64) for name, series in values.items()}
    fits, payload = fit_payload(radii_for_cases(cases), arrays)
    save_metric_npz(
        output_npz,
        cases,
        "polarization",
        arrays,
        payload,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    plot_for_cases(
        cases,
        arrays,
        output_png,
        title="Mean x polarization vs radius",
        ylabel="mean direction_cylindrical x",
        fits=fits,
    )
    return arrays
