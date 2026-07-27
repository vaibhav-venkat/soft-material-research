"""Plot full-coordinate mean-squared displacement from new-sims shards."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import freud
import numpy as np
from numpy.typing import NDArray
from safetensors import safe_open
import seaborn as sns

from hexatic.constants import cylinder


@dataclass(frozen=True)
class FullMsd:
    case_id: str
    elapsed_time: NDArray[np.float64]
    mean_squared_displacement: NDArray[np.float64]
    n_particles: int
    periodic_axes: tuple[str, ...]
    dimensions: int
    effective_tau_r: float
    theoretical_diffusivity: float
    x_only: bool


@dataclass(frozen=True)
class LongTimeFit:
    slope: float
    intercept: float
    diffusivity: float


def _load_manifest(shard_dir: Path) -> dict[str, Any]:
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "hexatic.new_sims.analysis.v1":
        raise ValueError(f"unsupported manifest schema in {manifest_path}")
    if manifest.get("complete") is not True:
        raise ValueError(f"analysis manifest is incomplete: {manifest_path}")
    case = manifest.get("case")
    if not isinstance(case, dict):
        raise ValueError(f"missing case metadata in {manifest_path}")
    coordinate_order = manifest.get("coordinate_order")
    if (
        not isinstance(coordinate_order, list)
        or not coordinate_order
        or coordinate_order[0] != "x"
    ):
        raise ValueError(
            f"expected axial x as the first coordinate in {manifest_path}"
        )
    return manifest


def _load_box_lengths(shard_dir: Path, dimensions: int) -> NDArray[np.float64]:
    static_path = shard_dir / "static.safetensors"
    if not static_path.is_file():
        raise FileNotFoundError(f"missing static tensors: {static_path}")
    with safe_open(static_path, framework="np") as source:
        if "box" not in set(source.keys()):
            raise KeyError(f"{static_path} has no 'box' tensor")
        box = np.asarray(source.get_tensor("box"), dtype=np.float64)
    if box.ndim != 1 or box.size < dimensions:
        raise ValueError(f"invalid box shape {box.shape} in {static_path}")
    lengths = box[:dimensions].copy()
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError(f"invalid box lengths {lengths} in {static_path}")
    return lengths


def _freud_full_msd(
    unwrapped_positions: NDArray[np.float64],
    particle_chunk: int,
) -> NDArray[np.float64]:
    calculator = freud.msd.MSD(box=None, mode="window")
    first_chunk = True
    for start in range(0, unwrapped_positions.shape[1], particle_chunk):
        stop = min(start + particle_chunk, unwrapped_positions.shape[1])
        positions = np.zeros(
            (unwrapped_positions.shape[0], stop - start, 3),
            dtype=np.float64,
        )
        positions[:, :, : unwrapped_positions.shape[2]] = (
            unwrapped_positions[:, start:stop]
        )
        calculator.compute(positions, reset=first_chunk)
        first_chunk = False
    return np.asarray(calculator.msd, dtype=np.float64)


def _load_full_msd(
    shard_dir: Path,
    frame_limit: int | None,
    particle_chunk: int,
    x_only: bool,
) -> FullMsd:
    manifest = _load_manifest(shard_dir)
    case = manifest["case"]
    case_id = case.get("case_id")
    if not isinstance(case_id, str):
        raise ValueError(f"missing case_id in {shard_dir / 'manifest.json'}")
    periodic_axes = case.get("periodic_axes", [])
    if not isinstance(periodic_axes, list):
        raise ValueError(f"invalid periodic_axes for case {case_id}")
    dimensions = int(case.get("dimensions", len(manifest["coordinate_order"])))
    if dimensions not in {2, 3}:
        raise ValueError(f"expected dimensions 2 or 3 for case {case_id}")
    diffusion_period = int(
        case.get(
            "rotational_diffusion_period",
            cylinder.SIMULATION.rotational_diffusion_period,
        )
    )
    if diffusion_period < 1:
        raise ValueError(
            f"invalid rotational_diffusion_period={diffusion_period} "
            f"for case {case_id}"
        )
    effective_tau_r = float(cylinder.SIMULATION.tau_r * diffusion_period)
    theoretical_diffusivity = (
        cylinder.SIMULATION.u0**2
        * effective_tau_r
        / (dimensions * (dimensions - 1))
    )
    coordinate_order = tuple(manifest["coordinate_order"][:dimensions])
    expected_order = ("x", "y") if dimensions == 2 else ("x", "y", "z")
    if coordinate_order != expected_order:
        raise ValueError(
            f"expected coordinate order {expected_order}, got {coordinate_order}"
        )
    periodic_axis_set = set(periodic_axes)
    unknown_periodic = periodic_axis_set.difference(coordinate_order)
    if unknown_periodic:
        raise ValueError(f"unknown periodic axes for {case_id}: {unknown_periodic}")
    box_lengths = _load_box_lengths(shard_dir, dimensions)
    timestep = float(case.get("timestep", cylinder.SIMULATION.timestep))
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError(f"invalid timestep={timestep} for case {case_id}")

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"manifest contains no shards: {shard_dir}")

    coordinate_arrays: list[NDArray[np.float64]] = []
    step_arrays: list[NDArray[np.int64]] = []
    loaded_frames = 0
    expected_start = 0
    n_particles: int | None = None

    for entry in shards:
        if not isinstance(entry, dict):
            raise ValueError(f"invalid shard entry in {shard_dir}")
        start = entry.get("frame_start")
        stop = entry.get("frame_stop")
        filename = entry.get("file")
        if start != expected_start or not isinstance(stop, int) or stop <= start:
            raise ValueError(f"non-contiguous shard range in {shard_dir}")
        if not isinstance(filename, str):
            raise ValueError(f"shard entry has no filename in {shard_dir}")
        shard_path = shard_dir / filename
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        with safe_open(shard_path, framework="np") as source:
            keys = set(source.keys())
            if "coords" not in keys or "step" not in keys:
                raise KeyError(f"{shard_path} must contain 'coords' and 'step'")
            coords = np.asarray(source.get_tensor("coords"), dtype=np.float64)
            shard_steps = np.asarray(source.get_tensor("step"), dtype=np.int64)
        if (
            coords.ndim != 3
            or coords.shape[1] < 1
            or coords.shape[2] < dimensions
        ):
            raise ValueError(f"invalid coords shape {coords.shape} in {shard_path}")
        if shard_steps.ndim != 1 or len(shard_steps) != len(coords):
            raise ValueError(f"step/coordinate length mismatch in {shard_path}")
        if len(coords) != stop - start:
            raise ValueError(f"shard range does not match tensors in {shard_path}")
        if not np.all(np.isfinite(coords[:, :, :dimensions])):
            raise ValueError(f"non-finite coordinate in {shard_path}")
        if n_particles is None:
            n_particles = coords.shape[1]
        elif coords.shape[1] != n_particles:
            raise ValueError(f"particle count changes in {shard_path}")

        take = len(coords)
        if frame_limit is not None:
            take = min(take, frame_limit - loaded_frames)
        coordinate_arrays.append(coords[:take, :, :dimensions].copy())
        step_arrays.append(shard_steps[:take].copy())
        loaded_frames += take
        expected_start = stop
        if frame_limit is not None and loaded_frames >= frame_limit:
            break

    if n_particles is None or not coordinate_arrays:
        raise ValueError(f"no frames loaded from {shard_dir}")
    unwrapped_positions = np.concatenate(coordinate_arrays, axis=0)
    step_values = np.concatenate(step_arrays)
    step_spacing = np.diff(step_values)
    if np.any(step_spacing <= 0):
        raise ValueError(f"simulation steps are not strictly increasing in {shard_dir}")
    if len(step_spacing) and not np.all(step_spacing == step_spacing[0]):
        raise ValueError(
            f"Freud windowed MSD requires uniform frame spacing in {shard_dir}"
        )

    for axis_index, axis_name in enumerate(coordinate_order):
        if x_only and axis_name != "x":
            continue
        if axis_name not in periodic_axis_set or len(unwrapped_positions) <= 1:
            continue
        length = box_lengths[axis_index]
        previous_wrapped = unwrapped_positions[0, :, axis_index].copy()
        for frame_index in range(1, len(unwrapped_positions)):
            current_wrapped = unwrapped_positions[
                frame_index, :, axis_index
            ].copy()
            displacement = current_wrapped - previous_wrapped
            displacement -= length * np.rint(displacement / length)
            unwrapped_positions[frame_index, :, axis_index] = (
                unwrapped_positions[frame_index - 1, :, axis_index]
                + displacement
            )
            previous_wrapped = current_wrapped

    # Freud's FFT result is translation invariant. Centering every particle's
    # fully unwrapped path at zero avoids cancellation from large box coordinates.
    if x_only:
        unwrapped_positions = unwrapped_positions[:, :, :1]
    unwrapped_positions -= unwrapped_positions[0]
    mean_squared_displacement = _freud_full_msd(
        unwrapped_positions,
        particle_chunk,
    )
    frame_time = float(step_spacing[0]) * timestep if len(step_spacing) else 0.0
    elapsed_time = np.arange(len(step_values), dtype=np.float64) * frame_time
    return FullMsd(
        case_id=case_id,
        elapsed_time=elapsed_time,
        mean_squared_displacement=mean_squared_displacement,
        n_particles=n_particles,
        periodic_axes=tuple(periodic_axes),
        dimensions=dimensions,
        effective_tau_r=effective_tau_r,
        theoretical_diffusivity=theoretical_diffusivity,
        x_only=x_only,
    )


def _long_time_fit(result: FullMsd, linear_start: float) -> LongTimeFit:
    selected = result.elapsed_time >= linear_start
    if np.count_nonzero(selected) < 2:
        raise ValueError(
            f"case {result.case_id} has fewer than two frames at or after "
            f"t={linear_start:g}"
        )
    design = np.column_stack(
        (
            result.elapsed_time[selected],
            np.ones(np.count_nonzero(selected), dtype=np.float64),
        )
    )
    slope, intercept = np.linalg.lstsq(
        design,
        result.mean_squared_displacement[selected],
        rcond=None,
    )[0]
    return LongTimeFit(
        slope=float(slope),
        intercept=float(intercept),
        diffusivity=(
            0.5 * float(slope)
            if result.x_only
            else float(slope) / (2.0 * result.dimensions)
        ),
    )


def _plot(
    series: list[tuple[str, FullMsd]],
    output: Path,
    linear_start: float,
) -> list[tuple[str, FullMsd, LongTimeFit]]:
    sns.set_theme(context="paper", style="ticks", font_scale=1.1)
    colors = sns.color_palette("colorblind", n_colors=len(series))
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 7.2),
        sharex=True,
        constrained_layout=True,
    )
    msd_axis, diffusivity_axis = axes
    fitted: list[tuple[str, FullMsd, LongTimeFit]] = []
    for color, (label, result) in zip(colors, series, strict=True):
        fit = _long_time_fit(result, linear_start)
        fitted.append((label, result, fit))
        msd_axis.plot(
            result.elapsed_time,
            result.mean_squared_displacement,
            color=color,
            linewidth=2.0,
            label=label,
        )
        fit_time = result.elapsed_time[result.elapsed_time >= linear_start]
        msd_axis.plot(
            fit_time,
            fit.slope * fit_time + fit.intercept,
            color=color,
            linestyle="--",
            linewidth=1.4,
            label=f"{label} linear fit",
        )
        diffusivity_axis.axhline(
            fit.diffusivity,
            color=color,
            linewidth=2.0,
            label=(
                rf"{label}: fitted $D_x={fit.diffusivity:.6g}$"
                if result.x_only
                else rf"{label}: fitted $D={fit.diffusivity:.6g}$"
            ),
        )
        diffusivity_axis.axhline(
            result.theoretical_diffusivity,
            color=color,
            linestyle=":",
            linewidth=2.0,
            label=(
                rf"{label}: $U_0^2\tau_r/[d(d-1)]"
                rf"={result.theoretical_diffusivity:.6g}$"
            ),
        )

    msd_axis.axvline(
        linear_start,
        color="0.45",
        linestyle="-.",
        linewidth=1.0,
        label=rf"fit start $t={linear_start:g}$",
    )
    x_only = series[0][1].x_only
    if x_only:
        msd_axis.set_title("Axial mean-squared displacement")
        msd_axis.set_ylabel(
            r"$\langle[x_i(t_0+\Delta t)-x_i(t_0)]^2\rangle_{i,t_0}$"
        )
    else:
        msd_axis.set_title("Full-coordinate mean-squared displacement")
        msd_axis.set_ylabel(
            r"$\langle|\mathbf{r}_i(t_0+\Delta t)-\mathbf{r}_i(t_0)|^2"
            r"\rangle_{i,t_0}$"
        )
    diffusivity_axis.set_title("Long-time effective diffusion")
    diffusivity_axis.set_xlabel(r"Lag time $\Delta t$")
    diffusivity_axis.set_ylabel(
        r"$D_x=\frac{1}{2}\,d\,\mathrm{MSD}_x/dt$"
        if x_only
        else r"$D=\frac{1}{2d}\,d\,\mathrm{MSD}/dt$"
    )
    for axis in axes:
        axis.grid(color="0.9", linewidth=0.7)
        axis.legend(frameon=False, fontsize="small")
        axis.set_xlim(left=0.0)
        axis.set_ylim(bottom=0.0)
        sns.despine(ax=axis)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="png",
        dpi=300,
        bbox_inches="tight",
        metadata={"Creator": "hexatic.new_sims_analysis"},
    )
    plt.close(figure)
    return fitted


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot full-coordinate particle mean-squared displacement from one or more "
            "new-sims safetensor output directories."
        )
    )
    parser.add_argument(
        "--shard-dir",
        action="append",
        type=Path,
        required=True,
        help="Analysis directory; repeat this option to overlay multiple cases.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Plot label; when used, provide one label per --shard-dir.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Maximum number of initial frames to use (default: all).",
    )
    parser.add_argument(
        "--particle-chunk",
        type=int,
        default=1024,
        help=(
            "Particles per Freud FFT batch, controlling peak memory "
            "(default: 1024)."
        ),
    )
    parser.add_argument(
        "--linear-start",
        type=float,
        default=40.0,
        help=(
            "Fit the long-time MSD slope from this time onward "
            "(default: 40)."
        ),
    )
    parser.add_argument(
        "--x-only",
        action="store_true",
        help=(
            "Use only axial x displacement and fit D_x=slope/2 instead of "
            "using the full d-dimensional displacement (default: false)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("full_msd.png"),
    )
    args = parser.parse_args()
    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be positive")
    if args.particle_chunk < 1:
        parser.error("--particle-chunk must be positive")
    if args.linear_start < 0.0:
        parser.error("--linear-start must be nonnegative")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")
    if args.label is not None and len(args.label) != len(args.shard_dir):
        parser.error("provide exactly one --label per --shard-dir")

    results = [
        _load_full_msd(
            shard_dir,
            args.frames,
            args.particle_chunk,
            args.x_only,
        )
        for shard_dir in args.shard_dir
    ]
    labels = (
        args.label
        if args.label is not None
        else [result.case_id for result in results]
    )
    fitted = _plot(
        list(zip(labels, results, strict=True)),
        args.output,
        args.linear_start,
    )
    print(f"wrote {args.output}")
    for label, result, fit in fitted:
        print(
            f"{label}: frames={len(result.elapsed_time)} "
            f"particles={result.n_particles} "
            f"periodic_axes={','.join(result.periodic_axes) or 'none'} "
            f"d={result.dimensions} tau_r={result.effective_tau_r:.8g} "
            f"mode={'x-only' if result.x_only else 'full'} "
            f"fitted_{'Dx' if result.x_only else 'D'}={fit.diffusivity:.8g} "
            f"theory_D={result.theoretical_diffusivity:.8g} "
            f"final_msd={result.mean_squared_displacement[-1]:.8g}"
        )


if __name__ == "__main__":
    main()
