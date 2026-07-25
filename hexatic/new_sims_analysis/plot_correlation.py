from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from safetensors import safe_open
import seaborn as sns

from hexatic.constants import cylinder


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
    return manifest


def _load_com_x(
    shard_dir: Path,
    manifest: dict[str, Any],
    frame_limit: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64], float]:
    """Load unwrapped COM x positions and simulation steps.

    Returns (com_x, steps, lx) where com_x is the per-frame mean of
    unwrapped axial particle coordinates and lx is the axial box length.
    """
    static_path = shard_dir / "static.safetensors"
    if not static_path.is_file():
        raise FileNotFoundError(f"missing static tensors: {static_path}")
    with safe_open(static_path, framework="np") as static_file:
        if "box" not in static_file.keys():
            raise KeyError(f"{static_path} has no 'box' tensor")
        box = np.asarray(static_file.get_tensor("box"), dtype=np.float64)
    lx = float(box[0])
    if not np.isfinite(lx) or lx <= 0.0:
        raise ValueError(f"invalid axial box length lx={lx}")

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("manifest contains no safetensor shards")

    com_x_frames: list[float] = []
    step_frames: list[int] = []
    loaded_frames = 0
    expected_start = 0
    # Per-particle unwrapping state, initialised on the first frame.
    previous_wrapped: NDArray[np.float64] | None = None
    unwrapped: NDArray[np.float64] | None = None

    for entry in shards:
        if not isinstance(entry, dict):
            raise ValueError("invalid shard entry in manifest")
        start = entry.get("frame_start")
        stop = entry.get("frame_stop")
        filename = entry.get("file")
        if start != expected_start or not isinstance(stop, int) or stop <= start:
            raise ValueError("safetensor shards are not contiguous")
        if not isinstance(filename, str):
            raise ValueError("shard entry has no filename")

        shard_path = shard_dir / filename
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        with safe_open(shard_path, framework="np") as shard:
            keys = set(shard.keys())
            if "coords" not in keys:
                raise KeyError(f"{shard_path} has no 'coords' tensor")
            if "step" not in keys:
                raise KeyError(f"{shard_path} has no 'step' tensor")
            coords = np.asarray(shard.get_tensor("coords"), dtype=np.float64)
            steps = np.asarray(shard.get_tensor("step"), dtype=np.int64)

        if coords.ndim != 3 or coords.shape[1] == 0 or coords.shape[2] == 0:
            raise ValueError(
                f"expected coords shape (frames, particles, components), "
                f"got {coords.shape}"
            )
        n_frames_in_shard = coords.shape[0]
        n_particles = coords.shape[1]
        if steps.ndim != 1 or len(steps) != n_frames_in_shard:
            raise ValueError(
                f"expected one simulation step per frame in {shard_path}"
            )
        if n_frames_in_shard != stop - start:
            raise ValueError(
                f"shard range [{start}, {stop}) does not match "
                f"{n_frames_in_shard} stored frames in {shard_path}"
            )
        if previous_wrapped is None:
            previous_wrapped = np.empty(n_particles, dtype=np.float64)
            unwrapped = np.empty(n_particles, dtype=np.float64)
        elif n_particles != len(previous_wrapped):
            raise ValueError("particle count changes between shards")
        assert unwrapped is not None

        take = frame_limit - loaded_frames
        coords = coords[:take]
        steps = steps[:take]
        actual_take = coords.shape[0]

        x_wrapped = coords[:, :, 0]  # shape: (frames, particles)
        for local_frame in range(actual_take):
            frame_x = x_wrapped[local_frame]
            if not np.all(np.isfinite(frame_x)):
                raise ValueError(
                    f"non-finite x coordinate at frame {start + local_frame}"
                )
            if loaded_frames == 0 and local_frame == 0:
                # The native big-Lx implementation takes the first stored
                # coordinates as the unwrapped reference frame.
                unwrapped[:] = frame_x
            else:
                # Displacement from the previous wrapped position, minus the
                # nearest integer number of box lengths.
                displacement = frame_x - previous_wrapped
                scaled = displacement / lx
                nearest_integer = np.copysign(
                    np.floor(np.abs(scaled) + 0.5),
                    scaled,
                )
                unwrapped += displacement - lx * nearest_integer
            previous_wrapped[:] = frame_x

            # Match the native big-Lx Kahan summation order.
            total = 0.0
            compensation = 0.0
            for coordinate in unwrapped:
                corrected = float(coordinate) - compensation
                updated = total + corrected
                compensation = (updated - total) - corrected
                total = updated
            com_x_frames.append(total / n_particles)
            step_frames.append(int(steps[local_frame]))

        loaded_frames += actual_take
        expected_start = stop
        if loaded_frames >= frame_limit:
            break

    if not com_x_frames:
        raise ValueError("no frames were loaded")
    if len(com_x_frames) < 3:
        raise ValueError(
            "at least three frames are required for velocity correlation"
        )
    return (
        np.array(com_x_frames, dtype=np.float64),
        np.array(step_frames, dtype=np.int64),
        lx,
    )


def _finite_difference(
    values: NDArray[np.float64],
    coords: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Second-order three-point finite difference on non-uniform coordinates.

    This replicates the Zig ``finiteDifference`` formula exactly so that
    the velocity signal matches the big_lx_analysis results.
    """
    n = len(values)
    if n != len(coords):
        raise ValueError("values and coordinates must have the same length")
    if n < 2:
        raise ValueError("need at least two samples for derivative")
    if not np.all(np.isfinite(values)):
        raise ValueError("values contain non-finite samples")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coordinates contain non-finite samples")
    if np.any(np.diff(coords) <= 0.0):
        raise ValueError("coordinates must be strictly increasing")
    derivative = np.empty(n, dtype=np.float64)

    if n == 2:
        slope = (values[1] - values[0]) / (coords[1] - coords[0])
        derivative[0] = slope
        derivative[1] = slope
        return derivative

    # Interior points: three-point Lagrange interpolation derivative.
    for i in range(1, n - 1):
        h_before = coords[i] - coords[i - 1]
        h_after = coords[i + 1] - coords[i]
        a = -h_after / (h_before * (h_before + h_after))
        b = (h_after - h_before) / (h_before * h_after)
        c = h_before / (h_after * (h_before + h_after))
        derivative[i] = a * values[i - 1] + b * values[i] + c * values[i + 1]

    # First point: forward three-point.
    h1 = coords[1] - coords[0]
    h2 = coords[2] - coords[1]
    derivative[0] = (
        -(2.0 * h1 + h2) / (h1 * (h1 + h2)) * values[0]
        + (h1 + h2) / (h1 * h2) * values[1]
        - h1 / (h2 * (h1 + h2)) * values[2]
    )

    # Last point: backward three-point.
    h_before = coords[n - 2] - coords[n - 3]
    h_last = coords[n - 1] - coords[n - 2]
    derivative[n - 1] = (
        h_last / (h_before * (h_before + h_last)) * values[n - 3]
        - (h_last + h_before) / (h_before * h_last) * values[n - 2]
        + (2.0 * h_last + h_before) / (h_last * (h_before + h_last)) * values[n - 1]
    )
    return derivative


def _pearson_correlation(
    velocity: NDArray[np.float64],
    max_lag: int,
) -> NDArray[np.float64]:
    """Lagged Pearson correlation of a velocity series.

    Returns pearson array of length max_lag + 1 where
    pearson[t] = corr(v[0 : T-t], v[t : T]).
    """
    n = len(velocity)
    if n < 3:
        raise ValueError("need at least three velocity samples")
    if not np.all(np.isfinite(velocity)):
        raise ValueError("velocity contains non-finite samples")
    if max_lag > n - 2:
        raise ValueError(
            f"max_lag ({max_lag}) exceeds available lags "
            f"({n - 2} for {n} samples)"
        )
    if max_lag < 0:
        raise ValueError("max_lag must be nonnegative")

    pearson = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        sample_count = n - lag
        left = velocity[:sample_count]
        right = velocity[lag : lag + sample_count]
        count = float(sample_count)
        sum_left = np.sum(left)
        sum_right = np.sum(right)
        sum_left_sq = np.dot(left, left)
        sum_right_sq = np.dot(right, right)
        sum_product = np.dot(left, right)
        covariance = sum_product - sum_left * sum_right / count
        variance_left = sum_left_sq - sum_left * sum_left / count
        variance_right = sum_right_sq - sum_right * sum_right / count
        if variance_left <= 0.0 or variance_right <= 0.0:
            raise ValueError(
                f"constant (zero-variance) velocity window at lag={lag}"
            )
        coefficient = covariance / np.sqrt(variance_left * variance_right)
        pearson[lag] = float(np.clip(coefficient, -1.0, 1.0))
    return pearson


def _lag_times(
    elapsed_time: NDArray[np.float64],
    max_lag: int,
) -> NDArray[np.float64]:
    """Return the native big-Lx lag grid after validating uniform spacing."""
    if len(elapsed_time) < 3:
        raise ValueError("at least three frames are required")
    spacing = float(elapsed_time[1] - elapsed_time[0])
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("invalid simulation time spacing")
    actual_spacing = np.diff(elapsed_time)
    tolerance = np.maximum(
        1.0e-12,
        1.0e-10 * np.maximum(np.abs(spacing), np.abs(actual_spacing)),
    )
    if (
        not np.all(np.isfinite(elapsed_time))
        or np.any(np.abs(actual_spacing - spacing) > tolerance)
    ):
        raise ValueError("simulation frames must have uniform, increasing spacing")
    return np.arange(max_lag + 1, dtype=np.float64) * spacing


def _elapsed_time(
    steps: NDArray[np.int64],
    timestep: float,
) -> NDArray[np.float64]:
    initial = steps[0]
    return (steps.astype(np.float64) - float(initial)) * timestep


def _case_identity(
    input_dir: Path,
    manifest: dict[str, Any],
) -> tuple[str, str]:
    """Return the same case label used by big-Lx and a safe output slug."""
    case = manifest["case"]
    case_id = case.get("case_id")
    label = case.get("label")
    if not isinstance(case_id, str) or not case_id:
        case_id = input_dir.resolve().name
    if not isinstance(label, str) or not label:
        label = case_id
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", case_id).strip("._")
    if not slug:
        raise ValueError(f"cannot derive an output filename for {input_dir}")
    return label, slug


def _plot_correlation(
    lag_time: NDArray[np.float64],
    pearson: NDArray[np.float64],
    label: str,
    output: Path,
) -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.1)
    color = sns.color_palette("colorblind")[0]
    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    axis.plot(lag_time, pearson, color=color, lw=2.0, label=label)
    axis.axhline(0.0, color="0.65", lw=0.9)
    axis.set_title("Lagged axial COM-velocity Pearson correlation")
    axis.set_xlabel("Lag time")
    axis.set_ylabel("Pearson correlation")
    axis.set_ylim(-1.05, 1.05)
    axis.grid(axis="y", color="0.9", lw=0.7)
    axis.legend(frameon=False, ncol=2)
    sns.despine(ax=axis)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="svg",
        bbox_inches="tight",
        metadata={"Creator": "hexatic.new_sims_analysis"},
    )
    plt.close(figure)


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        action="append",
        type=Path,
        required=True,
        help="Safetensor output directory (repeatable); one SVG per input.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("correlation_output"),
        help="Directory for output SVGs (default: correlation_output/).",
    )
    parser.add_argument(
        "--frames",
        type=_nonnegative_int,
        default=1000,
        help="Maximum number of initial frames to use (default: 1000).",
    )
    parser.add_argument(
        "--max-lag",
        type=_nonnegative_int,
        default=998,
        help="Maximum lag in frames (default: 998).",
    )
    parser.add_argument(
        "--timestep",
        type=_positive_float,
        default=None,
        help=(
            "Override the simulation timestep (default: case metadata, then "
            "the cylinder constant)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.frames < 3:
        raise ValueError("--frames must be at least 3")

    for index, input_dir in enumerate(args.input_dir, start=1):
        print(
            f"[correlation] {index}/{len(args.input_dir)} dir={input_dir}",
            flush=True,
        )
        manifest = _load_manifest(input_dir)
        label, slug = _case_identity(input_dir, manifest)

        # An explicit CLI value wins; otherwise use the recorded case value,
        # falling back to the same cylinder constant as big-Lx.
        case_meta = manifest.get("case", {})
        stored_timestep = case_meta.get("timestep")
        if args.timestep is not None:
            timestep = args.timestep
        elif stored_timestep is not None:
            timestep = float(stored_timestep)
            if timestep <= 0.0 or not np.isfinite(timestep):
                raise ValueError(
                    f"invalid timestep in manifest for {input_dir}: "
                    f"{stored_timestep!r}"
                )
        else:
            timestep = float(cylinder.SIMULATION.timestep)

        com_x, steps, lx = _load_com_x(input_dir, manifest, args.frames)
        print(
            f"[correlation] loaded {len(com_x)} frames, "
            f"lx={lx:.6g}",
            flush=True,
        )
        elapsed = _elapsed_time(steps, timestep)
        velocity = _finite_difference(com_x, elapsed)

        # Cap max_lag to what the velocity series actually supports, then
        # use the same effective value for Pearson *and* lag-time arrays.
        available_max_lag = len(velocity) - 2
        if available_max_lag < 0:
            raise ValueError(
                f"need at least 2 velocity samples (got {len(velocity)})"
            )
        effective_max_lag = args.max_lag
        if effective_max_lag > available_max_lag:
            print(
                f"[correlation] capping --max-lag {effective_max_lag} -> "
                f"{available_max_lag} (only {len(velocity)} velocity samples)",
                flush=True,
            )
            effective_max_lag = available_max_lag

        pearson = _pearson_correlation(velocity, effective_max_lag)
        lag_time = _lag_times(elapsed, effective_max_lag)

        output_path = args.output_dir / f"velocity_correlation_{slug}.svg"
        _plot_correlation(lag_time, pearson, label, output_path)
        print(f"[correlation] wrote {output_path}", flush=True)

    print(f"[correlation] done — {len(args.input_dir)} plot(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
