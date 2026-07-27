"""Plot axial mean-squared displacement from new-sims safetensor shards."""

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
class AxialMsd:
    case_id: str
    elapsed_time: NDArray[np.float64]
    mean_squared_displacement: NDArray[np.float64]
    n_particles: int
    periodic_x: bool


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


def _load_lx(shard_dir: Path) -> float:
    static_path = shard_dir / "static.safetensors"
    if not static_path.is_file():
        raise FileNotFoundError(f"missing static tensors: {static_path}")
    with safe_open(static_path, framework="np") as source:
        if "box" not in set(source.keys()):
            raise KeyError(f"{static_path} has no 'box' tensor")
        box = np.asarray(source.get_tensor("box"), dtype=np.float64)
    if box.ndim != 1 or box.size < 1:
        raise ValueError(f"invalid box shape {box.shape} in {static_path}")
    lx = float(box[0])
    if not np.isfinite(lx) or lx <= 0.0:
        raise ValueError(f"invalid axial box length lx={lx} in {static_path}")
    return lx


def _freud_axial_msd(
    unwrapped_x: NDArray[np.float64],
    particle_chunk: int,
) -> NDArray[np.float64]:
    calculator = freud.msd.MSD(box=None, mode="window")
    first_chunk = True
    for start in range(0, unwrapped_x.shape[1], particle_chunk):
        stop = min(start + particle_chunk, unwrapped_x.shape[1])
        positions = np.zeros(
            (unwrapped_x.shape[0], stop - start, 3),
            dtype=np.float64,
        )
        positions[:, :, 0] = unwrapped_x[:, start:stop]
        calculator.compute(positions, reset=first_chunk)
        first_chunk = False
    return np.asarray(calculator.msd, dtype=np.float64)


def _load_axial_msd(
    shard_dir: Path,
    frame_limit: int | None,
    particle_chunk: int,
) -> AxialMsd:
    manifest = _load_manifest(shard_dir)
    case = manifest["case"]
    case_id = case.get("case_id")
    if not isinstance(case_id, str):
        raise ValueError(f"missing case_id in {shard_dir / 'manifest.json'}")
    periodic_axes = case.get("periodic_axes", [])
    if not isinstance(periodic_axes, list):
        raise ValueError(f"invalid periodic_axes for case {case_id}")
    periodic_x = "x" in periodic_axes
    lx = _load_lx(shard_dir)
    timestep = float(case.get("timestep", cylinder.SIMULATION.timestep))
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError(f"invalid timestep={timestep} for case {case_id}")

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"manifest contains no shards: {shard_dir}")

    axial_arrays: list[NDArray[np.float64]] = []
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
        if coords.ndim != 3 or coords.shape[1] < 1 or coords.shape[2] < 1:
            raise ValueError(f"invalid coords shape {coords.shape} in {shard_path}")
        if shard_steps.ndim != 1 or len(shard_steps) != len(coords):
            raise ValueError(f"step/coordinate length mismatch in {shard_path}")
        if len(coords) != stop - start:
            raise ValueError(f"shard range does not match tensors in {shard_path}")
        if not np.all(np.isfinite(coords[:, :, 0])):
            raise ValueError(f"non-finite axial coordinate in {shard_path}")
        if n_particles is None:
            n_particles = coords.shape[1]
        elif coords.shape[1] != n_particles:
            raise ValueError(f"particle count changes in {shard_path}")

        take = len(coords)
        if frame_limit is not None:
            take = min(take, frame_limit - loaded_frames)
        axial_arrays.append(coords[:take, :, 0].copy())
        step_arrays.append(shard_steps[:take].copy())
        loaded_frames += take
        expected_start = stop
        if frame_limit is not None and loaded_frames >= frame_limit:
            break

    if n_particles is None or not axial_arrays:
        raise ValueError(f"no frames loaded from {shard_dir}")
    unwrapped_x = np.concatenate(axial_arrays, axis=0)
    step_values = np.concatenate(step_arrays)
    step_spacing = np.diff(step_values)
    if np.any(step_spacing <= 0):
        raise ValueError(f"simulation steps are not strictly increasing in {shard_dir}")
    if len(step_spacing) and not np.all(step_spacing == step_spacing[0]):
        raise ValueError(
            f"Freud windowed MSD requires uniform frame spacing in {shard_dir}"
        )

    if periodic_x and len(unwrapped_x) > 1:
        previous_wrapped = unwrapped_x[0].copy()
        for frame_index in range(1, len(unwrapped_x)):
            current_wrapped = unwrapped_x[frame_index].copy()
            displacement = current_wrapped - previous_wrapped
            displacement -= lx * np.rint(displacement / lx)
            unwrapped_x[frame_index] = (
                unwrapped_x[frame_index - 1] + displacement
            )
            previous_wrapped = current_wrapped

    # Freud's FFT result is translation invariant. Centering every particle's
    # fully unwrapped path at zero avoids cancellation from large box coordinates.
    unwrapped_x -= unwrapped_x[0]
    mean_squared_displacement = _freud_axial_msd(
        unwrapped_x,
        particle_chunk,
    )
    frame_time = float(step_spacing[0]) * timestep if len(step_spacing) else 0.0
    elapsed_time = np.arange(len(step_values), dtype=np.float64) * frame_time
    return AxialMsd(
        case_id=case_id,
        elapsed_time=elapsed_time,
        mean_squared_displacement=mean_squared_displacement,
        n_particles=n_particles,
        periodic_x=periodic_x,
    )


def _plot(series: list[tuple[str, AxialMsd]], output: Path) -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.1)
    colors = sns.color_palette("colorblind", n_colors=len(series))
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for color, (label, result) in zip(colors, series, strict=True):
        axis.plot(
            result.elapsed_time,
            result.mean_squared_displacement,
            color=color,
            linewidth=2.0,
            label=label,
        )
    axis.set_title("Axial mean-squared displacement")
    axis.set_xlabel(r"Elapsed time $t$")
    axis.set_ylabel(
        r"$\langle[x_i(t_0+\Delta t)-x_i(t_0)]^2\rangle_{i,t_0}$"
    )
    axis.grid(color="0.9", linewidth=0.7)
    axis.legend(frameon=False)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot axial particle mean-squared displacement from one or more "
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
        "--output",
        type=Path,
        default=Path("axial_msd.png"),
    )
    args = parser.parse_args()
    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be positive")
    if args.particle_chunk < 1:
        parser.error("--particle-chunk must be positive")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")
    if args.label is not None and len(args.label) != len(args.shard_dir):
        parser.error("provide exactly one --label per --shard-dir")

    results = [
        _load_axial_msd(shard_dir, args.frames, args.particle_chunk)
        for shard_dir in args.shard_dir
    ]
    labels = (
        args.label
        if args.label is not None
        else [result.case_id for result in results]
    )
    _plot(list(zip(labels, results, strict=True)), args.output)
    print(f"wrote {args.output}")
    for label, result in zip(labels, results, strict=True):
        print(
            f"{label}: frames={len(result.elapsed_time)} "
            f"particles={result.n_particles} periodic_x={result.periodic_x} "
            f"final_msd={result.mean_squared_displacement[-1]:.8g}"
        )


if __name__ == "__main__":
    main()
