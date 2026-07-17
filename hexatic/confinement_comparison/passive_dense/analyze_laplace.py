from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, cast

import gsd.hoomd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from hexatic.big_lx_analysis.correlations import lagged_pearson
from hexatic.big_lx_analysis.plot_velocity_laplace import (
    LaplaceCase,
    VelocityCorrelationLaplaceTransform,
    plot_velocity_laplace,
    velocity_correlation_laplace_transform,
)
from hexatic.constants import cylinder

from .cases import all_cases


@dataclass(frozen=True)
class DiscoveredTrajectory:
    case_id: str
    label: str
    path: Path
    lx: float
    n_particles: int


@dataclass(frozen=True)
class GsdCenterSeries:
    trajectory: DiscoveredTrajectory
    frames: np.ndarray
    steps: np.ndarray
    elapsed_time: np.ndarray
    x_center: np.ndarray
    x_velocity: np.ndarray


@dataclass(frozen=True)
class CorrelationInput:
    series: GsdCenterSeries
    lag_times: np.ndarray
    correlation: np.ndarray


def _trajectory_directory(input_dir: Path) -> Path:
    root = input_dir.resolve()
    nested = root / "gsd"
    if nested.is_dir():
        return nested
    if root.is_dir():
        return root
    raise FileNotFoundError(f"Passive-dense input directory does not exist: {root}")


def scan_trajectories(
    input_dir: Path,
    selected_case_ids: list[str] | None = None,
) -> list[DiscoveredTrajectory]:
    trajectory_dir = _trajectory_directory(input_dir)
    registered = {case.case_id: case for case in all_cases()}
    requested = set(selected_case_ids or ())
    unknown = sorted(requested - registered.keys())
    if unknown:
        raise ValueError(f"Unknown passive-dense case(s): {', '.join(unknown)}")

    discovered: list[DiscoveredTrajectory] = []
    seen: set[str] = set()
    for path in sorted(trajectory_dir.glob("trajectory_*.gsd")):
        case_id = path.stem.removeprefix("trajectory_")
        case = registered.get(case_id)
        if case is None or (requested and case_id not in requested):
            continue
        if case_id in seen:
            raise ValueError(f"Duplicate trajectory for case {case_id}: {path}")
        with gsd.hoomd.open(name=str(path), mode="r") as trajectory:
            if len(trajectory) < 2:
                raise ValueError(f"At least two frames are required: {path}")
            first = trajectory[0]
            n_particles = int(first.particles.N)
            lx = float(first.configuration.box[0])
        if n_particles != case.n_particles:
            raise ValueError(
                f"Particle count mismatch for {case_id}: "
                f"GSD has {n_particles}, case expects {case.n_particles}"
            )
        if not np.isclose(lx, case.lx, rtol=1.0e-6, atol=1.0e-6):
            raise ValueError(
                f"Axial box mismatch for {case_id}: GSD has {lx}, case expects {case.lx}"
            )
        seen.add(case_id)
        discovered.append(
            DiscoveredTrajectory(
                case_id=case_id,
                label=case.label,
                path=path.resolve(),
                lx=lx,
                n_particles=n_particles,
            )
        )
    if requested:
        missing = sorted(requested - seen)
        if missing:
            raise FileNotFoundError(
                f"No trajectory GSD found for case(s): {', '.join(missing)}"
            )
    if not discovered:
        raise ValueError(f"No registered trajectory_*.gsd files found in {trajectory_dir}")
    return discovered


def center_of_mass_series(trajectory: DiscoveredTrajectory) -> GsdCenterSeries:
    with gsd.hoomd.open(name=str(trajectory.path), mode="r") as source:
        frame_count = len(source)
        frames = np.arange(frame_count, dtype=np.int64)
        steps = np.empty(frame_count, dtype=np.int64)
        centers = np.empty(frame_count, dtype=np.float64)
        previous_wrapped: np.ndarray | None = None
        unwrapped: np.ndarray | None = None
        for frame_index, frame in enumerate(source):
            if int(frame.particles.N) != trajectory.n_particles:
                raise ValueError(
                    f"Particle count changes at frame {frame_index}: {trajectory.path}"
                )
            frame_lx = float(frame.configuration.box[0])
            if not np.isclose(frame_lx, trajectory.lx, rtol=1.0e-6, atol=1.0e-6):
                raise ValueError(
                    f"Axial box changes at frame {frame_index}: {trajectory.path}"
                )
            positions = np.asarray(frame.particles.position)
            if positions.shape != (trajectory.n_particles, 3):
                raise ValueError(
                    f"Position shape mismatch at frame {frame_index}: {trajectory.path}"
                )
            wrapped = np.asarray(positions[:, 0], dtype=np.float64)
            if not np.all(np.isfinite(wrapped)):
                raise ValueError(
                    f"Non-finite x coordinates at frame {frame_index}: {trajectory.path}"
                )
            if previous_wrapped is None:
                unwrapped = wrapped.copy()
            else:
                displacement = wrapped - previous_wrapped
                displacement -= trajectory.lx * np.rint(displacement / trajectory.lx)
                if unwrapped is None:
                    raise RuntimeError("Particle unwrapping was not initialized")
                unwrapped += displacement
            previous_wrapped = wrapped.copy()
            if unwrapped is None:
                raise RuntimeError("Particle unwrapping was not initialized")
            steps[frame_index] = int(frame.configuration.step)
            centers[frame_index] = float(np.mean(unwrapped))

    if np.any(np.diff(steps) <= 0):
        raise ValueError(f"Simulation steps are not strictly increasing: {trajectory.path}")
    step_spacing = np.diff(steps)
    if not np.all(step_spacing == step_spacing[0]):
        raise ValueError(f"Simulation steps are not uniformly spaced: {trajectory.path}")
    elapsed_time = (
        steps.astype(np.float64) - float(steps[0])
    ) * cylinder.SIMULATION.timestep
    velocity = np.gradient(
        centers,
        elapsed_time,
        edge_order=2 if frame_count >= 3 else 1,
    )
    return GsdCenterSeries(
        trajectory=trajectory,
        frames=frames,
        steps=steps,
        elapsed_time=elapsed_time,
        x_center=centers,
        x_velocity=np.asarray(velocity, dtype=np.float64),
    )


def correlation_input(
    series: GsdCenterSeries,
    *,
    min_origins: int,
) -> CorrelationInput:
    frame_count = series.frames.size
    if min_origins < 2:
        raise ValueError("min_origins must be at least two")
    if min_origins > frame_count:
        raise ValueError(
            f"min_origins={min_origins} exceeds {frame_count} frames for "
            f"{series.trajectory.case_id}"
        )
    maximum_lag = frame_count - min_origins
    if maximum_lag < 1:
        raise ValueError(
            f"No positive correlation lag remains for {series.trajectory.case_id}"
        )
    spacing = np.diff(series.elapsed_time)
    dt = float(spacing[0])
    if not np.allclose(spacing, dt, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(
            f"COM samples are not uniformly spaced for {series.trajectory.case_id}"
        )
    lag_times = np.arange(maximum_lag + 1, dtype=np.float64) * dt
    correlation = lagged_pearson(
        series.x_velocity,
        maximum_lag,
        f"Axial COM velocity for {series.trajectory.case_id}",
    )
    return CorrelationInput(
        series=series,
        lag_times=lag_times,
        correlation=correlation,
    )


def _transform_grid(
    inputs: list[CorrelationInput],
    *,
    r_min: float | None,
    r_max: float,
    omega_min: float | None,
    omega_max: float | None,
    r_points: int,
    omega_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    reference_duration = min(float(item.lag_times[-1]) for item in inputs)
    if reference_duration <= 0.0:
        raise ValueError("Correlation series must span positive lag time")
    nyquist_limit = min(
        np.pi / float(np.diff(item.lag_times)[0]) for item in inputs
    )
    default_omega_max = min(nyquist_limit, 20.0 * np.pi / reference_duration)
    selected_r_min = -10.0 / reference_duration if r_min is None else r_min
    selected_omega_min = (
        -default_omega_max if omega_min is None else omega_min
    )
    selected_omega_max = default_omega_max if omega_max is None else omega_max
    if selected_r_min >= r_max:
        raise ValueError("The r range requires r_min < r_max")
    if selected_omega_min >= selected_omega_max:
        raise ValueError("The omega range requires omega_min < omega_max")
    if max(abs(selected_omega_min), abs(selected_omega_max)) > nyquist_limit * (
        1.0 + 1.0e-12
    ):
        raise ValueError(
            "The requested omega range exceeds the shared Nyquist limit "
            f"{nyquist_limit:g}"
        )
    return (
        np.linspace(selected_r_min, r_max, r_points, dtype=np.float64),
        np.linspace(
            selected_omega_min,
            selected_omega_max,
            omega_points,
            dtype=np.float64,
        ),
    )


def plot_center_series(series_by_case: list[GsdCenterSeries], output: Path, dpi: int) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = plt.colormaps["viridis"](
        np.linspace(0.08, 0.92, len(series_by_case))
    )
    figure, (center_axis, velocity_axis) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
    )
    for color, series in zip(colors, series_by_case, strict=True):
        center_axis.plot(
            series.elapsed_time,
            series.x_center,
            color=color,
            linewidth=1.5,
            label=f"{series.trajectory.label} ({series.frames.size} frames)",
        )
        velocity_axis.plot(
            series.elapsed_time,
            series.x_velocity,
            color=color,
            linewidth=1.3,
            label=series.trajectory.label,
        )
    center_axis.set_title("Unwrapped axial center of mass")
    center_axis.set_ylabel(r"unwrapped $x_{\mathrm{COM}}$")
    center_axis.grid(alpha=0.22)
    center_axis.legend(loc="best", fontsize=8)
    velocity_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
    velocity_axis.set_title("Axial center-of-mass velocity")
    velocity_axis.set_xlabel(r"elapsed simulation time $\tau$")
    velocity_axis.set_ylabel(r"$v_{x,\mathrm{COM}}$")
    velocity_axis.grid(alpha=0.22)
    velocity_axis.legend(loc="best", fontsize=8)
    figure.suptitle(
        "Passive-dense GSD COM and velocity\n"
        "per-particle minimum-image x unwrapping; full length per trajectory"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def _guard_outputs(outputs: tuple[Path, ...], overwrite: bool) -> None:
    existing = tuple(path for path in outputs if path.exists())
    if existing and not overwrite:
        names = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Pass --overwrite to replace analysis outputs:\n{names}")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)


def _write_numerical_output(
    output: Path,
    correlations: list[CorrelationInput],
    transforms: list[VelocityCorrelationLaplaceTransform],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for item, transform in zip(correlations, transforms, strict=True):
        prefix = item.series.trajectory.case_id
        arrays[f"{prefix}__frames"] = item.series.frames
        arrays[f"{prefix}__steps"] = item.series.steps
        arrays[f"{prefix}__elapsed_time"] = item.series.elapsed_time
        arrays[f"{prefix}__x_center"] = item.series.x_center
        arrays[f"{prefix}__x_velocity"] = item.series.x_velocity
        arrays[f"{prefix}__lag_times"] = item.lag_times
        arrays[f"{prefix}__velocity_pearson"] = item.correlation
        arrays[f"{prefix}__laplace_real"] = transform.values.real
        arrays[f"{prefix}__laplace_imag"] = transform.values.imag
    arrays["r"] = transforms[0].r
    arrays["omega"] = transforms[0].omega
    # NumPy's stub treats arbitrary string keys as potentially shadowing its
    # allow_pickle keyword; the generated keys above cannot do so.
    save_arrays = cast(Any, np.savez_compressed)
    save_arrays(output, **arrays)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan passive-dense trajectory GSDs, compute full-length unwrapped "
            "COM x-velocity series, and plot velocity-correlation Laplace heatmaps."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Passive-dense production root containing gsd/trajectory_*.gsd.",
    )
    parser.add_argument(
        "--case",
        "--cases",
        dest="case",
        action="extend",
        nargs="+",
        choices=tuple(case.case_id for case in all_cases()),
        help="Optional case filter; by default every registered trajectory is scanned.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-origins", type=int, default=10)
    parser.add_argument("--r-min", type=float)
    parser.add_argument("--r-max", type=float, default=0.0)
    parser.add_argument("--omega-min", type=float)
    parser.add_argument("--omega-max", type=float)
    parser.add_argument("--r-points", type=int, default=161)
    parser.add_argument("--omega-points", type=int, default=241)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.case and len(set(args.case)) != len(args.case):
        raise SystemExit("Each --case value must be unique")
    if args.r_points < 2 or args.omega_points < 2:
        raise SystemExit("--r-points and --omega-points must be at least two")
    if args.dpi < 1:
        raise SystemExit("--dpi must be positive")

    trajectories = scan_trajectories(args.input_dir, args.case)
    series_by_case = [center_of_mass_series(item) for item in trajectories]
    correlations = [
        correlation_input(series, min_origins=args.min_origins)
        for series in series_by_case
    ]
    r, omega = _transform_grid(
        correlations,
        r_min=args.r_min,
        r_max=args.r_max,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        r_points=args.r_points,
        omega_points=args.omega_points,
    )
    transforms = [
        velocity_correlation_laplace_transform(
            LaplaceCase(
                case_id=item.series.trajectory.case_id,
                label=item.series.trajectory.label,
                lx=item.series.trajectory.lx,
                n_particles=item.series.trajectory.n_particles,
                analysis_dir=item.series.trajectory.path.parent,
            ),
            item.lag_times,
            item.correlation,
            r=r,
            omega=omega,
        )
        for item in correlations
    ]

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.input_dir.resolve() / "gsd_laplace_analysis"
    )
    heatmap_output = output_dir / "velocity_correlation_laplace.html"
    center_output = output_dir / "com_x_velocity.png"
    numerical_output = output_dir / "com_velocity_laplace.npz"
    manifest_output = output_dir / "manifest.json"
    _guard_outputs(
        (heatmap_output, center_output, numerical_output, manifest_output),
        args.overwrite,
    )
    plot_velocity_laplace(transforms, heatmap_output)
    plot_center_series(series_by_case, center_output, args.dpi)
    _write_numerical_output(numerical_output, correlations, transforms)
    manifest = {
        "schema": "hexatic.confinement_comparison.passive_dense.gsd_laplace.v1",
        "input_dir": str(args.input_dir.resolve()),
        "complete": True,
        "min_origins": args.min_origins,
        "r": {"minimum": float(r[0]), "maximum": float(r[-1]), "points": len(r)},
        "omega": {
            "minimum": float(omega[0]),
            "maximum": float(omega[-1]),
            "points": len(omega),
        },
        "outputs": {
            "heatmap": str(heatmap_output),
            "com_x_velocity": str(center_output),
            "numerical": str(numerical_output),
        },
        "cases": [
            {
                "case_id": item.series.trajectory.case_id,
                "trajectory_gsd": str(item.series.trajectory.path),
                "n_particles": item.series.trajectory.n_particles,
                "frame_count": int(item.series.frames.size),
                "first_step": int(item.series.steps[0]),
                "last_step": int(item.series.steps[-1]),
                "correlation_lag_count": int(item.lag_times.size),
            }
            for item in correlations
        ],
    }
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n")
    for item in correlations:
        print(
            f"[passive_dense.gsd_laplace.case] "
            f"case={item.series.trajectory.case_id} "
            f"frames={item.series.frames.size} "
            f"lags={item.lag_times.size} "
            f"gsd={item.series.trajectory.path}",
            flush=True,
        )
    print(
        f"[passive_dense.gsd_laplace] cases={len(correlations)} "
        f"heatmap={heatmap_output} com={center_output} data={numerical_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
