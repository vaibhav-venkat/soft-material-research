from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from safetensors import safe_open
from scipy.integrate import simpson

from hexatic.big_lx.cases import (
    DEFAULT_OUTPUT_ROOT,
    CasePaths as BigLxCasePaths,
    get_case as get_big_lx_case,
)
from hexatic.confinement_comparison.cases import (
    CasePaths as ConfinementCasePaths,
    get_case as get_confinement_case,
)
from hexatic.constants import cylinder

from .correlations import lagged_pearson


@dataclass(frozen=True)
class LaplaceCase:
    case_id: str
    label: str
    lx: float
    n_particles: int
    analysis_dir: Path


@dataclass(frozen=True)
class AxialCenterSeries:
    elapsed_time: np.ndarray
    x_velocity: np.ndarray


@dataclass(frozen=True)
class VelocityCorrelationLaplaceTransform:
    case: LaplaceCase
    r: np.ndarray
    omega: np.ndarray
    values: np.ndarray


def velocity_correlation_laplace_transform(
    case: LaplaceCase,
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


def _resolve_cases(
    mode: str,
    case_ids: list[str],
    output_root: Path,
) -> list[LaplaceCase]:
    result: list[LaplaceCase] = []
    for case_id in case_ids:
        if mode == "big-lx":
            case = get_big_lx_case(case_id)
            analysis_dir = BigLxCasePaths(case, output_root).analysis_dir
        elif mode == "confinement":
            case = get_confinement_case(case_id)
            analysis_dir = ConfinementCasePaths(case, output_root).analysis_dir
        else:
            raise ValueError(f"Unsupported analysis mode: {mode}")
        result.append(
            LaplaceCase(
                case_id=case.case_id,
                label=case.label,
                lx=case.lx,
                n_particles=case.n_particles,
                analysis_dir=analysis_dir,
            )
        )
    return result


def _axial_center_series(case: LaplaceCase, mode: str) -> AxialCenterSeries:
    manifest_path = case.analysis_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing analysis manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected_schema = (
        "hexatic.big_lx.analysis.v1"
        if mode == "big-lx"
        else "hexatic.confinement_comparison.analysis.v1"
    )
    if manifest.get("schema") != expected_schema:
        raise ValueError(f"Unsupported analysis schema in {manifest_path}")
    if manifest.get("complete") is not True:
        raise ValueError(f"Analysis is not marked complete: {manifest_path}")
    case_payload = manifest.get("case")
    if not isinstance(case_payload, dict) or case_payload.get("case_id") != case.case_id:
        raise ValueError(f"Analysis manifest does not match case {case.case_id}")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"Analysis manifest has no frame shards: {manifest_path}")

    expected_start = 0
    steps: list[int] = []
    centers: list[float] = []
    previous_wrapped: np.ndarray | None = None
    unwrapped: np.ndarray | None = None
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("Analysis manifest contains an invalid shard")
        start = shard.get("frame_start")
        stop = shard.get("frame_stop")
        filename = shard.get("file")
        if (
            not isinstance(start, int)
            or start != expected_start
            or not isinstance(stop, int)
            or stop <= start
            or not isinstance(filename, str)
        ):
            raise ValueError("Analysis shards are not contiguous from frame zero")
        shard_path = case.analysis_dir / filename
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing frame shard: {shard_path}")
        with safe_open(shard_path, framework="numpy") as tensors:
            keys = set(tensors.keys())
            coordinate_name = "coords" if "coords" in keys else "position_cartesian"
            if coordinate_name not in keys or "step" not in keys:
                raise KeyError(
                    f"{shard_path} lacks logical axial coordinates or step values"
                )
            coordinates = np.asarray(tensors.get_tensor(coordinate_name))
            shard_steps = np.asarray(tensors.get_tensor("step")).reshape(-1)
        frame_count = stop - start
        if (
            coordinates.ndim != 3
            or coordinates.shape[:2] != (frame_count, case.n_particles)
            or coordinates.shape[2] < 1
            or shard_steps.shape != (frame_count,)
        ):
            raise ValueError(f"Frame tensor shape mismatch in {shard_path}")
        for local_index in range(frame_count):
            wrapped = np.asarray(coordinates[local_index, :, 0], dtype=np.float64)
            if previous_wrapped is None:
                unwrapped = wrapped.copy()
            else:
                displacement = wrapped - previous_wrapped
                displacement -= case.lx * np.rint(displacement / case.lx)
                unwrapped += displacement
            previous_wrapped = wrapped.copy()
            if unwrapped is None:
                raise RuntimeError("Axial coordinate unwrapping was not initialized")
            steps.append(int(shard_steps[local_index]))
            centers.append(float(np.mean(unwrapped)))
        expected_start = stop

    declared_frames = manifest.get("frame_count")
    if not isinstance(declared_frames, int) or declared_frames != expected_start:
        raise ValueError("Manifest frame count does not match its shards")
    step_array = np.asarray(steps, dtype=np.int64)
    if step_array.size < 2 or np.any(np.diff(step_array) <= 0):
        raise ValueError("Analysis steps must be strictly increasing")
    elapsed_time = (
        step_array.astype(np.float64) - float(step_array[0])
    ) * cylinder.SIMULATION.timestep
    edge_order = 2 if elapsed_time.size >= 3 else 1
    velocity = np.gradient(
        np.asarray(centers, dtype=np.float64),
        elapsed_time,
        edge_order=edge_order,
    )
    return AxialCenterSeries(elapsed_time=elapsed_time, x_velocity=velocity)


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
        "--cases",
        dest="case",
        action="extend",
        nargs="+",
        required=True,
        help="Case IDs; pass one or more values to create heatmap panels.",
    )
    parser.add_argument(
        "--mode",
        choices=("big-lx", "confinement"),
        default="big-lx",
    )
    parser.add_argument(
        "--output-root",
        "--output-dir",
        "--output_dir",
        dest="output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
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
    cases = _resolve_cases(args.mode, args.case, args.output_root)
    series_by_case = [
        _axial_center_series(case, args.mode)
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
        or args.output_root
        / "plots"
        / f"{args.mode.replace('-', '_')}_velocity_correlation_laplace.html"
    )
    result = plot_velocity_laplace(transforms, output)
    print(
        f"[big_lx_analysis.laplace] mode={args.mode} cases={len(cases)} "
        f"r=[{r[0]:.8g}, {r[-1]:.8g}] "
        f"omega=[{omega[0]:.8g}, {omega[-1]:.8g}] output={result}",
        flush=True,
    )


if __name__ == "__main__":
    main()
