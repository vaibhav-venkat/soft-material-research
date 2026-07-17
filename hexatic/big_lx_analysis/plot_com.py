from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open

from hexatic.big_lx.cases import all_cases as all_big_lx_cases
from hexatic.big_lx.cases import get_case as get_big_lx_case
from hexatic.confinement_comparison.cases import (
    all_cases as all_confinement_cases,
)
from hexatic.confinement_comparison.cases import (
    get_case as get_confinement_case,
)
from hexatic.constants import cylinder


@dataclass(frozen=True)
class PlotCase:
    case_id: str
    label: str
    lx: float
    n_particles: int
    analysis_dir: Path


@dataclass(frozen=True)
class CenterOfMassSeries:
    case: PlotCase
    frames: np.ndarray
    steps: np.ndarray
    elapsed_time: np.ndarray
    x_center: np.ndarray
    x_velocity: np.ndarray


def _resolve_cases(
    case_ids: list[str],
    output_dir: Path,
    *,
    confined: bool,
) -> list[PlotCase]:
    safetensors_root = (output_dir / "safetensors_output").resolve()
    resolved: list[PlotCase] = []
    for case_id in case_ids:
        case = (
            get_confinement_case(case_id)
            if confined
            else get_big_lx_case(case_id)
        )
        resolved.append(
            PlotCase(
                case_id=case.case_id,
                label=case.label,
                lx=float(case.lx),
                n_particles=int(case.n_particles),
                analysis_dir=safetensors_root / case.case_id,
            )
        )
    return resolved


def _selected_frames(
    frame_count: int,
    start: int,
    stop: int | None,
    stride: int,
) -> np.ndarray:
    if start < 0:
        raise ValueError("--start must be nonnegative")
    if stride < 1:
        raise ValueError("--stride must be positive")
    selected_stop = frame_count if stop is None else min(stop, frame_count)
    frames = np.arange(start, selected_stop, stride, dtype=np.int64)
    if frames.size < 2:
        raise ValueError("At least two selected frames are required for velocity")
    return frames


def _validated_shards(
    manifest: dict[str, object],
    manifest_path: Path,
) -> tuple[list[dict[str, object]], int]:
    if manifest.get("complete") is not True:
        raise ValueError(f"Analysis is not marked complete: {manifest_path}")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"Analysis manifest has no frame shards: {manifest_path}")

    validated: list[dict[str, object]] = []
    expected_start = 0
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError(f"Invalid shard entry in {manifest_path}")
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
            raise ValueError(
                f"Analysis shards are not contiguous from frame zero: {manifest_path}"
            )
        validated.append(shard)
        expected_start = stop

    declared_frames = manifest.get("frame_count")
    if not isinstance(declared_frames, int) or declared_frames != expected_start:
        raise ValueError(f"Manifest frame count does not match shards: {manifest_path}")
    return validated, expected_start


def _contained_shard_path(analysis_dir: Path, filename: str) -> Path:
    root = analysis_dir.resolve()
    shard_path = (root / filename).resolve()
    try:
        shard_path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Shard path escapes the case safetensor directory: {filename}"
        ) from error
    if not shard_path.is_file():
        raise FileNotFoundError(f"Missing frame shard: {shard_path}")
    return shard_path


def center_of_mass_series(
    case: PlotCase,
    *,
    confined: bool,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
) -> CenterOfMassSeries:
    manifest_path = case.analysis_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing analysis manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected_schema = (
        "hexatic.confinement_comparison.analysis.v1"
        if confined
        else "hexatic.big_lx.analysis.v1"
    )
    if manifest.get("schema") != expected_schema:
        raise ValueError(f"Unsupported analysis schema in {manifest_path}")
    case_payload = manifest.get("case")
    if not isinstance(case_payload, dict) or case_payload.get("case_id") != case.case_id:
        raise ValueError(f"Analysis manifest does not match case {case.case_id}")

    shards, frame_count = _validated_shards(manifest, manifest_path)
    frames = _selected_frames(frame_count, start, stop, stride)
    selected_frames = set(int(frame) for frame in frames)
    final_frame = int(frames[-1])
    steps = np.empty(frames.size, dtype=np.int64)
    centers = np.empty(frames.size, dtype=np.float64)

    previous_wrapped: np.ndarray | None = None
    unwrapped: np.ndarray | None = None
    output_index = 0
    for shard in shards:
        shard_start = int(shard["frame_start"])
        shard_stop = int(shard["frame_stop"])
        if shard_start > final_frame:
            break
        shard_path = _contained_shard_path(case.analysis_dir, str(shard["file"]))
        with safe_open(shard_path, framework="numpy") as tensors:
            keys = set(tensors.keys())
            if confined:
                coordinate_name = (
                    "coords" if "coords" in keys else "position_cartesian"
                )
            else:
                coordinate_name = "coords"
            missing = [
                name
                for name in (coordinate_name, "step")
                if name not in keys
            ]
            if missing:
                raise KeyError(
                    f"{shard_path} is missing tensors: {', '.join(missing)}"
                )
            coordinates = np.asarray(tensors.get_tensor(coordinate_name))
            shard_steps = np.asarray(tensors.get_tensor("step")).reshape(-1)

        expected_frames = shard_stop - shard_start
        if (
            coordinates.ndim != 3
            or coordinates.shape[0] != expected_frames
            or coordinates.shape[1] != case.n_particles
            or coordinates.shape[2] < 1
            or shard_steps.shape != (expected_frames,)
        ):
            raise ValueError(f"Frame tensor shape mismatch in {shard_path}")

        local_stop = min(expected_frames, final_frame - shard_start + 1)
        for local_index in range(local_stop):
            frame_index = shard_start + local_index
            wrapped = np.asarray(
                coordinates[local_index, :, 0],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(wrapped)):
                raise ValueError(
                    f"Non-finite axial coordinates in frame {frame_index}"
                )
            if previous_wrapped is None:
                unwrapped = wrapped.copy()
            else:
                displacement = wrapped - previous_wrapped
                displacement -= case.lx * np.rint(displacement / case.lx)
                if unwrapped is None:
                    raise RuntimeError("Axial coordinate unwrapping was not initialized")
                unwrapped += displacement
            previous_wrapped = wrapped.copy()

            if frame_index in selected_frames:
                if unwrapped is None:
                    raise RuntimeError("Axial coordinate unwrapping was not initialized")
                steps[output_index] = int(shard_steps[local_index])
                centers[output_index] = float(np.mean(unwrapped))
                output_index += 1

    if output_index != frames.size:
        raise RuntimeError(f"Read {output_index} of {frames.size} selected frames")
    if np.any(np.diff(steps) <= 0):
        raise ValueError("Selected simulation steps must be strictly increasing")

    elapsed_time = (
        steps.astype(np.float64) - float(steps[0])
    ) * float(cylinder.SIMULATION.timestep)
    edge_order = 2 if elapsed_time.size >= 3 else 1
    velocity = np.gradient(centers, elapsed_time, edge_order=edge_order)
    return CenterOfMassSeries(
        case=case,
        frames=frames,
        steps=steps,
        elapsed_time=elapsed_time,
        x_center=centers,
        x_velocity=np.asarray(velocity, dtype=np.float64),
    )


def plot_com(
    series_by_case: list[CenterOfMassSeries],
    output: Path,
    *,
    confined: bool,
    dpi: int,
) -> Path:
    if not series_by_case:
        raise ValueError("At least one case is required")
    if output.suffix.lower() != ".png":
        raise ValueError("Static COM output must use a .png suffix")
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
            linewidth=1.6,
            label=series.case.label,
        )
        velocity_axis.plot(
            series.elapsed_time,
            series.x_velocity,
            color=color,
            linewidth=1.4,
            label=series.case.label,
        )

    center_axis.set_title("Unwrapped axial center of mass")
    center_axis.set_ylabel(r"unwrapped $x_{\mathrm{COM}}$")
    center_axis.grid(alpha=0.22)
    center_axis.legend(loc="best", fontsize=9)

    velocity_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
    velocity_axis.set_title("Axial center-of-mass velocity")
    velocity_axis.set_xlabel(r"elapsed simulation time $\tau$")
    velocity_axis.set_ylabel(r"$v_{x,\mathrm{COM}} = dx_{\mathrm{COM}}/dt$")
    velocity_axis.grid(alpha=0.22)
    velocity_axis.legend(loc="best", fontsize=9)

    mode_label = "confinement comparison" if confined else "Big-Lx"
    figure.suptitle(
        f"{mode_label} axial COM and velocity\n"
        "per-particle minimum-image safetensor unwrapping"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def _parse_args() -> argparse.Namespace:
    big_lx_ids = tuple(case.case_id for case in all_big_lx_cases())
    confinement_ids = tuple(case.case_id for case in all_confinement_cases())
    parser = argparse.ArgumentParser(
        description=(
            "Plot unwrapped axial COM and COM x-velocity from analysis "
            "safetensors for selected Big-Lx or confinement cases."
        ),
        epilog=(
            f"Big-Lx cases: {', '.join(big_lx_ids)}. "
            f"With --confined: {', '.join(confinement_ids)}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        required=True,
        help="Production-run root containing safetensors_output/.",
    )
    parser.add_argument(
        "--case",
        "--cases",
        dest="case",
        action="extend",
        nargs="+",
        choices=big_lx_ids + confinement_ids,
        required=True,
        help=(
            "One or more case IDs to plot; confinement IDs require --confined."
        ),
    )
    parser.add_argument(
        "--confined",
        action="store_true",
        help="Interpret --case values as confinement-comparison cases.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if len(set(args.case)) != len(args.case):
        raise SystemExit("Each case may be selected only once")
    if args.dpi < 1:
        raise SystemExit("--dpi must be positive")
    valid_case_ids = {
        case.case_id
        for case in (
            all_confinement_cases() if args.confined else all_big_lx_cases()
        )
    }
    invalid_case_ids = [
        case_id for case_id in args.case if case_id not in valid_case_ids
    ]
    if invalid_case_ids:
        mode = "confinement" if args.confined else "Big-Lx"
        raise SystemExit(
            f"Invalid {mode} case selection: {', '.join(invalid_case_ids)}"
        )
    cases = _resolve_cases(
        args.case,
        args.output_dir,
        confined=args.confined,
    )
    series = [
        center_of_mass_series(
            case,
            confined=args.confined,
            start=args.start,
            stop=args.stop,
            stride=args.stride,
        )
        for case in cases
    ]
    mode_name = "confinement" if args.confined else "big_lx"
    output = (
        args.output
        or args.output_dir / "plots" / f"{mode_name}_axial_com_velocity.png"
    )
    result = plot_com(
        series,
        output,
        confined=args.confined,
        dpi=args.dpi,
    )
    print(
        f"[big_lx_analysis.com] mode={mode_name} cases={len(cases)} "
        f"output={result}",
        flush=True,
    )


if __name__ == "__main__":
    main()
