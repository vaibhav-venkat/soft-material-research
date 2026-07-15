from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from safetensors.numpy import load_file

from hexatic.constants import cylinder

from .cases import DEFAULT_OUTPUT_ROOT, BigLxCase, CasePaths, get_case


@dataclass(frozen=True)
class CenterOfMassSeries:
    frames: np.ndarray
    steps: np.ndarray
    elapsed_time: np.ndarray
    x_center: np.ndarray
    x_velocity: np.ndarray
    method: str


def _selected_frames(
    frame_count: int,
    start: int,
    stop: int | None,
    stride: int,
) -> np.ndarray:
    if start < 0:
        raise ValueError("start must be nonnegative")
    if stride < 1:
        raise ValueError("stride must be positive")
    selected_stop = frame_count if stop is None else min(stop, frame_count)
    frames = np.arange(start, selected_stop, stride, dtype=np.int64)
    if frames.size < 2:
        raise ValueError("At least two selected frames are required for velocity")
    return frames


def _available_frame_stop(manifest: dict[str, object]) -> int:
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("The analysis manifest contains no frame shards")
    expected_start = 0
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("The analysis manifest contains an invalid shard entry")
        frame_start = shard.get("frame_start")
        frame_stop = shard.get("frame_stop")
        if (
            frame_start != expected_start
            or not isinstance(frame_stop, int)
            or frame_stop <= expected_start
        ):
            raise ValueError("Analysis shards are not contiguous from frame zero")
        expected_start = frame_stop
    return expected_start


def _iter_shard_frames(
    analysis_dir: Path,
    manifest: dict[str, object],
    final_frame: int,
) -> Iterator[tuple[int, int, np.ndarray]]:
    for shard in manifest["shards"]:
        shard_start = int(shard["frame_start"])
        shard_stop = int(shard["frame_stop"])
        if shard_start > final_frame:
            break
        shard_path = analysis_dir / str(shard["file"])
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing frame shard: {shard_path}")
        tensors = load_file(shard_path)
        missing = [name for name in ("coords", "step") if name not in tensors]
        if missing:
            raise KeyError(f"{shard_path} is missing tensors: {', '.join(missing)}")
        coords = np.asarray(tensors["coords"])
        steps = np.asarray(tensors["step"]).reshape(-1)
        expected_frames = shard_stop - shard_start
        if coords.shape[0] != expected_frames or steps.size != expected_frames:
            raise ValueError(f"Frame count mismatch in shard: {shard_path}")
        local_stop = min(expected_frames, final_frame - shard_start + 1)
        for local_index in range(local_stop):
            yield (
                shard_start + local_index,
                int(steps[local_index]),
                np.asarray(coords[local_index, :, 0], dtype=np.float64),
            )
        del tensors, coords, steps


def center_of_mass_series(
    case: BigLxCase,
    analysis_dir: Path,
    *,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
) -> CenterOfMassSeries:
    manifest_path = analysis_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing analysis manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "hexatic.big_lx.analysis.v1":
        raise ValueError(f"Unsupported analysis schema in {manifest_path}")
    case_payload = manifest.get("case")
    if not isinstance(case_payload, dict) or case_payload.get("case_id") != case.case_id:
        raise ValueError("Analysis manifest case does not match the selected case")

    available_stop = _available_frame_stop(manifest)
    frames = _selected_frames(available_stop, start, stop, stride)
    selected = set(int(frame) for frame in frames)
    steps = np.empty(frames.size, dtype=np.int64)
    x_center = np.empty(frames.size, dtype=np.float64)
    previous_wrapped: np.ndarray | None = None
    unwrapped: np.ndarray | None = None
    output_index = 0

    for frame_index, step, wrapped_x in _iter_shard_frames(
        analysis_dir,
        manifest,
        int(frames[-1]),
    ):
        if wrapped_x.shape != (case.n_particles,):
            raise ValueError(
                f"Frame {frame_index} has {wrapped_x.size} particles, "
                f"but {case.case_id} expects {case.n_particles}"
            )
        if previous_wrapped is None:
            unwrapped = wrapped_x.copy()
        else:
            displacement = wrapped_x - previous_wrapped
            displacement -= case.lx * np.rint(displacement / case.lx)
            unwrapped += displacement
        previous_wrapped = wrapped_x.copy()

        if frame_index in selected:
            steps[output_index] = step
            x_center[output_index] = float(np.mean(unwrapped))
            output_index += 1

    if output_index != frames.size:
        raise RuntimeError(
            f"Read {output_index} of {frames.size} selected analysis frames"
        )

    if np.any(np.diff(steps) <= 0):
        raise ValueError("Selected trajectory steps must be strictly increasing")

    elapsed_time = (
        steps.astype(np.float64) - float(steps[0])
    ) * cylinder.SIMULATION.timestep
    edge_order = 2 if elapsed_time.size >= 3 else 1
    x_velocity = np.gradient(x_center, elapsed_time, edge_order=edge_order)
    return CenterOfMassSeries(
        frames=frames,
        steps=steps,
        elapsed_time=elapsed_time,
        x_center=np.asarray(x_center, dtype=np.float64),
        x_velocity=np.asarray(x_velocity, dtype=np.float64),
        method="per-particle minimum-image safetensor unwrapping",
    )


def plot_center_of_mass(
    case: BigLxCase,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    output: Path | None = None,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    dpi: int = 180,
) -> Path:
    paths = CasePaths(case, output_root)
    series = center_of_mass_series(
        case,
        paths.analysis_dir,
        start=start,
        stop=stop,
        stride=stride,
    )
    output_path = output or (
        output_root / "plots" / f"{case.case_id}_x_com_velocity.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(series.elapsed_time, series.x_center, color="tab:blue", linewidth=1.5)
    axes[0].set_ylabel("unwrapped x center of mass")
    axes[0].set_title(f"{case.label}: axial center of mass ({series.method})")
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        series.elapsed_time,
        series.x_velocity,
        color="tab:orange",
        linewidth=1.5,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("elapsed simulation time")
    axes[1].set_ylabel(r"$d\langle x\rangle/dt$")
    axes[1].set_title("Axial center-of-mass velocity")
    axes[1].grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    print(
        f"[big_lx.com] case={case.case_id} frames={series.frames.size} "
        f"method={series.method} output={output_path}",
        flush=True,
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot unwrapped axial center of mass and its velocity for one "
            "big-Lx film case."
        )
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.dpi < 1:
        raise ValueError("dpi must be positive")
    plot_center_of_mass(
        get_case(args.case),
        output_root=args.output_root,
        output=args.output,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
