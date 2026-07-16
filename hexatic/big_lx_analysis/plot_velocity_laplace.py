from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import simpson

from hexatic.big_lx.cases import DEFAULT_OUTPUT_ROOT, BigLxCase, CasePaths, get_case
from hexatic.big_lx.plot_center_of_mass import center_of_mass_series

from .correlations import lagged_pearson


@dataclass(frozen=True)
class VelocityCorrelationLaplaceTransform:
    case: BigLxCase
    r: np.ndarray
    omega: np.ndarray
    values: np.ndarray


def velocity_correlation_laplace_transform(
    case: BigLxCase,
    lag_times: np.ndarray,
    correlation: np.ndarray,
    *,
    r: np.ndarray,
    omega: np.ndarray,
) -> VelocityCorrelationLaplaceTransform:
    time = np.asarray(lag_times, dtype=np.float64)
    values_to_transform = np.asarray(correlation, dtype=np.float64)
    if (
        time.ndim != 1
        or values_to_transform.shape != time.shape
        or time.size < 2
    ):
        raise ValueError(
            "Velocity correlation and lag time must be matching one-dimensional series"
        )
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
        damped_correlation = np.exp(real_part * time) * values_to_transform
        values[:, r_index] = simpson(
            oscillatory * damped_correlation[None, :],
            x=time,
            axis=1,
        )
    return VelocityCorrelationLaplaceTransform(
        case=case,
        r=np.asarray(r, dtype=np.float64),
        omega=np.asarray(omega, dtype=np.float64),
        values=values,
    )


def plot_velocity_laplace(
    transforms: list[VelocityCorrelationLaplaceTransform],
    output: Path,
) -> Path:
    if not transforms:
        raise ValueError("At least one Laplace transform is required")
    if output.suffix.lower() != ".html":
        raise ValueError("Laplace surface output must use an .html suffix")
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

    figure = make_subplots(
        rows=1,
        cols=len(transforms),
        specs=[[{"type": "xy"} for _ in transforms]],
        subplot_titles=[transform.case.label for transform in transforms],
        horizontal_spacing=0.04,
    )
    for column, (transform, log_magnitude) in enumerate(
        zip(transforms, log_magnitudes, strict=True),
        start=1,
    ):
        figure.add_trace(
            go.Heatmap(
                x=transform.r,
                y=transform.omega,
                z=log_magnitude,
                coloraxis="coloraxis",
                name=transform.case.label,
                hovertemplate=(
                    "r=%{x:.5g}<br>omega=%{y:.5g}<br>"
                    "log10|C_v-hat|=%{z:.5g}<extra>%{fullData.name}</extra>"
                ),
            ),
            row=1,
            col=column,
        )
    figure.update_xaxes(title_text="real part r")
    figure.update_yaxes(title_text="imaginary part omega")
    figure.update_layout(
        title=(
            "Pearson velocity-correlation Laplace heatmap: "
            "C_v-hat(r+i omega) = integral exp((r+i omega) tau) C_v(tau) d tau"
        ),
        coloraxis={
            "colorscale": "Magma",
            "cmin": color_minimum,
            "cmax": color_maximum,
            "colorbar": {"title": {"text": "log10 |C_v-hat|"}},
        },
        width=max(900, 720 * len(transforms)),
        height=620,
        margin={"l": 20, "r": 80, "t": 90, "b": 20},
    )
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write an interactive Plotly heatmap of the finite-time complex "
            "Laplace transform of the axial COM-velocity Pearson correlation."
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
    parser.add_argument("--min-origins", type=int, default=10)
    parser.add_argument("--max-lag", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if len(set(args.case)) != len(args.case):
        raise SystemExit("Each --case value must be unique")
    if args.r_points < 2 or args.omega_points < 2:
        raise SystemExit("--r-points and --omega-points must be at least two")
    if args.min_origins < 2:
        raise SystemExit("--min-origins must be at least two")
    if args.max_lag is not None and args.max_lag < 1:
        raise SystemExit("--max-lag must be positive")
    cases = [get_case(case_id) for case_id in args.case]
    series_by_case = [
        center_of_mass_series(case, CasePaths(case, args.output_root).analysis_dir)
        for case in cases
    ]
    correlation_inputs: list[tuple[np.ndarray, np.ndarray]] = []
    for series in series_by_case:
        frame_count = series.elapsed_time.size
        if args.min_origins > frame_count:
            raise ValueError(
                f"min_origins={args.min_origins} exceeds {frame_count} frames"
            )
        maximum_lag = frame_count - args.min_origins
        if args.max_lag is not None:
            maximum_lag = min(maximum_lag, args.max_lag)
        if maximum_lag < 1:
            raise ValueError(
                "The selected min-origins/max-lag settings leave no positive lag"
            )
        lag_indices = np.arange(maximum_lag + 1, dtype=np.int64)
        time_spacing = np.diff(series.elapsed_time)
        if not np.allclose(time_spacing, time_spacing[0], rtol=1e-10, atol=1e-12):
            raise ValueError("Pearson correlation requires uniformly spaced samples")
        lag_times = lag_indices.astype(np.float64) * float(time_spacing[0])
        correlation = lagged_pearson(
            series.x_velocity,
            maximum_lag,
            "Axial COM velocity",
        )
        correlation_inputs.append((lag_times, correlation))
    reference_duration = min(float(times[-1]) for times, _ in correlation_inputs)
    if reference_duration <= 0.0:
        raise ValueError("COM trajectories must span positive simulation time")
    nyquist_limits = [
        np.pi / float(np.min(np.diff(times)))
        for times, _ in correlation_inputs
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
        velocity_correlation_laplace_transform(
            case,
            lag_times,
            correlation,
            r=r,
            omega=omega,
        )
        for case, (lag_times, correlation) in zip(
            cases, correlation_inputs, strict=True
        )
    ]
    output = (
        args.output
        or args.output_root / "plots" / "big_lx_velocity_correlation_laplace.html"
    )
    result = plot_velocity_laplace(transforms, output)
    print(
        f"[big_lx_analysis.laplace] cases={len(cases)} "
        f"r=[{r[0]:.8g}, {r[-1]:.8g}] "
        f"omega=[{omega[0]:.8g}, {omega[-1]:.8g}] output={result}",
        flush=True,
    )


if __name__ == "__main__":
    main()
