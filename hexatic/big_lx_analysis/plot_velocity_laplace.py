from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson

from hexatic.big_lx.cases import DEFAULT_OUTPUT_ROOT, BigLxCase, CasePaths, get_case
from hexatic.big_lx.plot_center_of_mass import (
    CenterOfMassSeries,
    center_of_mass_series,
)


@dataclass(frozen=True)
class VelocityLaplaceTransform:
    case: BigLxCase
    r: np.ndarray
    omega: np.ndarray
    values: np.ndarray


def velocity_laplace_transform(
    case: BigLxCase,
    series: CenterOfMassSeries,
    *,
    r: np.ndarray,
    omega: np.ndarray,
) -> VelocityLaplaceTransform:
    time = np.asarray(series.elapsed_time, dtype=np.float64)
    velocity = np.asarray(series.x_velocity, dtype=np.float64)
    if time.ndim != 1 or velocity.shape != time.shape or time.size < 2:
        raise ValueError("COM velocity and time must be matching one-dimensional series")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("COM elapsed times must be strictly increasing")
    if r.ndim != 1 or omega.ndim != 1 or r.size < 2 or omega.size < 2:
        raise ValueError("r and omega must each contain at least two grid points")
    maximum_exponent = float(np.max(r) * time[-1])
    if maximum_exponent > 700.0:
        raise ValueError(
            "The requested positive r range overflows exp(r*t); reduce --r-max"
        )

    oscillatory = np.exp(1j * omega[:, None] * time[None, :])
    values = np.empty((omega.size, r.size), dtype=np.complex128)
    for r_index, real_part in enumerate(r):
        damped_velocity = np.exp(real_part * time) * velocity
        values[:, r_index] = simpson(
            oscillatory * damped_velocity[None, :],
            x=time,
            axis=1,
        )
    return VelocityLaplaceTransform(
        case=case,
        r=np.asarray(r, dtype=np.float64),
        omega=np.asarray(omega, dtype=np.float64),
        values=values,
    )


def plot_velocity_laplace(
    transforms: list[VelocityLaplaceTransform],
    output: Path,
    *,
    dpi: int = 180,
) -> Path:
    if not transforms:
        raise ValueError("At least one Laplace transform is required")
    if dpi < 1:
        raise ValueError("dpi must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)

    magnitudes = [np.abs(transform.values) for transform in transforms]
    global_maximum = max(float(np.max(magnitude)) for magnitude in magnitudes)
    if not np.isfinite(global_maximum) or global_maximum <= 0.0:
        raise ValueError("Laplace-transform magnitude is zero or non-finite")
    floor = global_maximum * 1.0e-12
    log_magnitudes = [
        np.log10(np.maximum(magnitude, floor)) for magnitude in magnitudes
    ]
    color_minimum = min(float(np.min(values)) for values in log_magnitudes)
    color_maximum = max(float(np.max(values)) for values in log_magnitudes)

    figure, axes = plt.subplots(
        1,
        len(transforms),
        figsize=(7 * len(transforms), 6),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    image = None
    for axis, transform, log_magnitude in zip(
        axes[0], transforms, log_magnitudes, strict=True
    ):
        image = axis.pcolormesh(
            transform.r,
            transform.omega,
            log_magnitude,
            shading="auto",
            cmap="magma",
            vmin=color_minimum,
            vmax=color_maximum,
            rasterized=True,
        )
        axis.axvline(0.0, color="white", linewidth=0.8, alpha=0.6)
        axis.axhline(0.0, color="white", linewidth=0.8, alpha=0.6)
        axis.set_title(transform.case.label)
        axis.set_xlabel(r"real part $r$")
    axes[0, 0].set_ylabel(r"imaginary part $\omega$")
    if image is None:
        raise RuntimeError("No Laplace heatmap was created")
    colorbar = figure.colorbar(image, ax=axes[0].tolist(), pad=0.03)
    colorbar.set_label(r"$\log_{10}|\hat v_x(r+i\omega)|$")
    figure.suptitle(
        r"Axial COM-velocity transform "
        r"$\hat v_x(r+i\omega)=\int_0^T e^{(r+i\omega)t}v_x(t)\,dt$"
    )
    figure.subplots_adjust(left=0.08, right=0.88, bottom=0.12, top=0.86, wspace=0.12)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the finite-time complex Laplace transform of axial Big-Lx "
            "COM velocity using SciPy Simpson integration."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Big-Lx case ID; repeat to create multiple heatmap panels.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--r-min", type=float)
    parser.add_argument("--r-max", type=float, default=0.0)
    parser.add_argument("--omega-min", type=float)
    parser.add_argument("--omega-max", type=float)
    parser.add_argument("--r-points", type=int, default=161)
    parser.add_argument("--omega-points", type=int, default=241)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if len(set(args.case)) != len(args.case):
        raise SystemExit("Each --case value must be unique")
    if args.r_points < 2 or args.omega_points < 2:
        raise SystemExit("--r-points and --omega-points must be at least two")
    cases = [get_case(case_id) for case_id in args.case]
    series_by_case = [
        center_of_mass_series(case, CasePaths(case, args.output_root).analysis_dir)
        for case in cases
    ]
    reference_duration = min(float(series.elapsed_time[-1]) for series in series_by_case)
    if reference_duration <= 0.0:
        raise ValueError("COM trajectories must span positive simulation time")
    nyquist_limits = [
        np.pi / float(np.min(np.diff(series.elapsed_time)))
        for series in series_by_case
    ]
    default_omega_max = min(min(nyquist_limits), 20.0 * np.pi / reference_duration)
    r_min = args.r_min if args.r_min is not None else -10.0 / reference_duration
    omega_min = (
        args.omega_min if args.omega_min is not None else -default_omega_max
    )
    omega_max = args.omega_max if args.omega_max is not None else default_omega_max
    if r_min >= args.r_max:
        raise SystemExit("The r range requires --r-min < --r-max")
    if omega_min >= omega_max:
        raise SystemExit("The omega range requires --omega-min < --omega-max")

    r = np.linspace(r_min, args.r_max, args.r_points, dtype=np.float64)
    omega = np.linspace(omega_min, omega_max, args.omega_points, dtype=np.float64)
    transforms = [
        velocity_laplace_transform(case, series, r=r, omega=omega)
        for case, series in zip(cases, series_by_case, strict=True)
    ]
    output = (
        args.output
        or args.output_root / "plots" / "big_lx_velocity_laplace.png"
    )
    result = plot_velocity_laplace(transforms, output, dpi=args.dpi)
    print(
        f"[big_lx_analysis.laplace] cases={len(cases)} "
        f"r=[{r[0]:.8g}, {r[-1]:.8g}] "
        f"omega=[{omega[0]:.8g}, {omega[-1]:.8g}] output={result}",
        flush=True,
    )


if __name__ == "__main__":
    main()
