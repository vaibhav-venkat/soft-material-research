from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import gsd.fl
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

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


def _read_inherited_chunk(
    trajectory: gsd.fl.GSDFile,
    frame: int,
    name: str,
) -> np.ndarray:
    for source_frame in range(frame, -1, -1):
        if trajectory.chunk_exists(frame=source_frame, name=name):
            return np.asarray(trajectory.read_chunk(frame=source_frame, name=name))
    raise KeyError(f"Missing required GSD chunk {name!r} at frame {frame}")


def _box_vectors(box: np.ndarray) -> np.ndarray:
    lx, ly, lz, xy, xz, yz = np.asarray(box, dtype=np.float64)
    return np.asarray(
        (
            (lx, 0.0, 0.0),
            (xy * ly, ly, 0.0),
            (xz * lz, yz * lz, lz),
        ),
        dtype=np.float64,
    )


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


def center_of_mass_series(
    case: BigLxCase,
    trajectory_gsd: Path,
    *,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
) -> CenterOfMassSeries:
    with gsd.fl.open(name=str(trajectory_gsd), mode="r") as trajectory:
        if trajectory.nframes == 0:
            raise ValueError(f"Trajectory contains no frames: {trajectory_gsd}")
        frames = _selected_frames(trajectory.nframes, start, stop, stride)
        steps = np.empty(frames.size, dtype=np.int64)
        image_centers = np.empty(frames.size, dtype=np.float64)
        circular_phases = np.empty(frames.size, dtype=np.float64)
        circular_coherence = np.empty(frames.size, dtype=np.float64)
        has_dynamic_images = False

        for output_index, frame_index_value in enumerate(frames):
            frame_index = int(frame_index_value)
            position = np.asarray(
                _read_inherited_chunk(
                    trajectory,
                    frame_index,
                    "particles/position",
                ),
                dtype=np.float64,
            )
            if position.ndim != 2 or position.shape[1] != 3:
                raise ValueError(
                    f"Frame {frame_index} has invalid particle positions "
                    f"with shape {position.shape}"
                )
            box = _read_inherited_chunk(
                trajectory,
                frame_index,
                "configuration/box",
            )
            lx = float(np.asarray(box)[0])
            if not np.isclose(lx, case.lx, rtol=1e-6, atol=1e-6):
                raise ValueError(
                    f"Frame {frame_index} has Lx={lx}, but {case.case_id} "
                    f"expects Lx={case.lx}"
                )

            if trajectory.chunk_exists(frame=frame_index, name="particles/image"):
                has_dynamic_images |= frame_index > 0
            try:
                image = np.asarray(
                    _read_inherited_chunk(
                        trajectory,
                        frame_index,
                        "particles/image",
                    ),
                    dtype=np.int64,
                )
            except KeyError:
                image = np.zeros(position.shape, dtype=np.int64)
            if image.shape != position.shape:
                raise ValueError(
                    f"Frame {frame_index} has particle images with shape "
                    f"{image.shape}, expected {position.shape}"
                )
            has_dynamic_images |= bool(np.any(image))
            unwrapped = position + image @ _box_vectors(box)
            image_centers[output_index] = float(np.mean(unwrapped[:, 0]))

            phases = 2.0 * np.pi * position[:, 0] / lx
            circular_order = np.mean(np.exp(1j * phases))
            circular_phases[output_index] = float(np.angle(circular_order))
            circular_coherence[output_index] = float(np.abs(circular_order))
            steps[output_index] = int(
                _read_inherited_chunk(
                    trajectory,
                    frame_index,
                    "configuration/step",
                ).reshape(-1)[0]
            )

    if np.any(np.diff(steps) <= 0):
        raise ValueError("Selected trajectory steps must be strictly increasing")

    if has_dynamic_images:
        x_center = image_centers
        method = "particle images"
    else:
        if np.any(circular_coherence < 1.0e-6):
            raise ValueError(
                "The GSD has no dynamic particle images and the periodic x center "
                "is undefined for at least one nearly uniform frame"
            )
        x_center = np.unwrap(circular_phases) * case.lx / (2.0 * np.pi)
        method = "unwrapped periodic circular mean"

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
        method=method,
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
    if not paths.trajectory_gsd.is_file():
        raise FileNotFoundError(f"Missing trajectory: {paths.trajectory_gsd}")
    series = center_of_mass_series(
        case,
        paths.trajectory_gsd,
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
