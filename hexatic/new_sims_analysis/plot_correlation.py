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
from scipy.signal import fftconvolve
import seaborn as sns

from hexatic.constants import cylinder

SUPPORTED_MANIFEST_SCHEMAS = {
    "hexatic.big_lx.analysis.v1",
    "hexatic.new_sims.analysis.v1",
}

# Cylindrical component order used throughout: (axial, azimuthal, radial)
# for coordinates, matching the ``coords`` tensor layout (x, theta, r).
CYLINDRICAL_COORDINATE_ORDER = ["x", "theta", "r"]


def _load_manifest(shard_dir: Path) -> dict[str, Any]:
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    schema = manifest.get("schema")
    if schema not in SUPPORTED_MANIFEST_SCHEMAS:
        supported = ", ".join(sorted(SUPPORTED_MANIFEST_SCHEMAS))
        raise ValueError(
            f"unsupported manifest schema {schema!r} in {manifest_path}; "
            f"expected one of: {supported}"
        )
    if manifest.get("complete") is not True:
        raise ValueError(f"analysis manifest is incomplete: {manifest_path}")
    case = manifest.get("case")
    if not isinstance(case, dict):
        raise ValueError(f"missing case metadata in {manifest_path}")
    # big_lx runs are always cylindrical and record no coordinate_order; the
    # new_sims schema records one, and only the cylindrical layout is
    # supported here.
    coordinate_order = manifest.get("coordinate_order")
    if (
        coordinate_order is not None
        and list(coordinate_order) != CYLINDRICAL_COORDINATE_ORDER
    ):
        raise ValueError(
            f"expected cylindrical coordinate_order "
            f"{CYLINDRICAL_COORDINATE_ORDER}, got {coordinate_order!r}"
        )
    return manifest


def _load_lx(static_path: Path) -> float:
    """Load axial box length from either supported static-tensor schema."""
    if not static_path.is_file():
        raise FileNotFoundError(f"missing static tensors: {static_path}")
    with safe_open(static_path, framework="np") as static_file:
        keys = set(static_file.keys())
        if "lx" in keys:
            lx_tensor = np.asarray(static_file.get_tensor("lx"))
            if lx_tensor.size != 1:
                raise ValueError(
                    f"expected scalar 'lx' tensor in {static_path}, "
                    f"got shape {lx_tensor.shape}"
                )
            lx = float(lx_tensor.reshape(-1)[0])
        elif "box" in keys:
            box = np.asarray(static_file.get_tensor("box"))
            if box.ndim != 1 or box.size < 1:
                raise ValueError(
                    f"expected one-dimensional 'box' tensor in {static_path}, "
                    f"got shape {box.shape}"
                )
            lx = float(box[0])
        else:
            raise KeyError(f"{static_path} has neither 'lx' nor 'box' tensor")
    if not np.isfinite(lx) or lx <= 0.0:
        raise ValueError(f"invalid axial box length lx={lx}")
    return lx


class _PeriodicComUnwrapper:
    """Streaming unwrap of a wrapped periodic coordinate into a COM series.

    Mirrors the native big-Lx implementation: the first frame pushed becomes
    the unwrapped reference, later frames advance by the minimum-image
    displacement, and each mean uses the same Kahan summation order. Used for
    the axial coordinate (period lx) and the azimuthal one (period 2 pi).
    """

    def __init__(self, n_particles: int, period: float) -> None:
        if n_particles < 1:
            raise ValueError("need at least one particle")
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError(f"invalid period {period}")
        self._n_particles = n_particles
        self._period = period
        self._previous = np.empty(n_particles, dtype=np.float64)
        self._unwrapped = np.empty(n_particles, dtype=np.float64)
        self._started = False

    def push(self, values: NDArray[np.float64], frame_index: int) -> float:
        if len(values) != self._n_particles:
            raise ValueError(f"particle count changes at frame {frame_index}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite coordinate at frame {frame_index}")
        if not self._started:
            self._unwrapped[:] = values
            self._started = True
        else:
            # Displacement from the previous wrapped position, minus the
            # nearest integer number of periods.
            displacement = values - self._previous
            scaled = displacement / self._period
            nearest_integer = np.copysign(
                np.floor(np.abs(scaled) + 0.5),
                scaled,
            )
            self._unwrapped += displacement - self._period * nearest_integer
        self._previous[:] = values

        # Match the native big-Lx Kahan summation order.
        total = 0.0
        compensation = 0.0
        for coordinate in self._unwrapped:
            corrected = float(coordinate) - compensation
            updated = total + corrected
            compensation = (updated - total) - corrected
            total = updated
        return total / self._n_particles


def _cartesian_directions(
    shard: Any,
    shard_path: Path,
) -> NDArray[np.float64]:
    """Load the per-particle Cartesian orientation vectors from a shard.

    ``new_sims`` shards store these as ``polarization`` and ``big_lx`` shards
    as ``active_direction``; both are Cartesian (x, y, z) unit vectors.
    """
    keys = set(shard.keys())
    name = (
        "polarization"
        if "polarization" in keys
        else "active_direction"
        if "active_direction" in keys
        else None
    )
    if name is None:
        raise KeyError(
            f"{shard_path} has neither 'polarization' nor 'active_direction'; "
            f"orientation correlation is undefined for this case"
        )
    directions = np.asarray(shard.get_tensor(name), dtype=np.float64)
    if directions.ndim != 3 or directions.shape[2] != 3:
        raise ValueError(
            f"expected {name} shape (frames, particles, 3) in {shard_path}, "
            f"got {directions.shape}"
        )
    if not np.all(np.isfinite(directions)):
        raise ValueError(f"{name} contains non-finite values in {shard_path}")
    norms = np.linalg.norm(directions, axis=2)
    if not np.allclose(norms, 1.0, rtol=2e-4, atol=2e-4):
        raise ValueError(f"{name} in {shard_path} is not unit-normalized")
    return directions


def _load_frame_series(
    shard_dir: Path,
    manifest: dict[str, Any],
    frame_limit: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float32],
    NDArray[np.int64],
    float,
]:
    """Load the per-frame COM coordinates and per-particle orientations.

    Returns ``(com, q, steps, lx)`` where ``com`` is ``(frames, 3)`` in
    cylindrical (x, theta, r) component order and ``q`` is
    ``(frames, particles, 3)`` in Cartesian (x, y, z) components. The axial
    coordinate is unwrapped over the box length and the azimuthal one over
    2 pi; the radial coordinate needs no unwrapping.

    Note that ``q`` holds every particle for every frame, so memory scales as
    frames x particles; ``--frames`` is the knob for keeping it bounded.
    """
    static_path = shard_dir / "static.safetensors"
    lx = _load_lx(static_path)

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("manifest contains no safetensor shards")

    com_frames: list[tuple[float, float, float]] = []
    q_arrays: list[NDArray[np.float32]] = []
    step_frames: list[int] = []
    loaded_frames = 0
    expected_start = 0
    # Per-particle unwrapping state, initialised on the first frame.
    axial_unwrapper: _PeriodicComUnwrapper | None = None
    azimuthal_unwrapper: _PeriodicComUnwrapper | None = None

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
            directions = _cartesian_directions(shard, shard_path)

        if coords.ndim != 3 or coords.shape[1] == 0 or coords.shape[2] != 3:
            raise ValueError(
                f"expected cylindrical coords shape (frames, particles, 3), "
                f"got {coords.shape}"
            )
        n_frames_in_shard = coords.shape[0]
        n_particles = coords.shape[1]
        if directions.shape != coords.shape:
            raise ValueError(
                f"orientation shape {directions.shape} does not match coords "
                f"shape {coords.shape} in {shard_path}"
            )
        if steps.ndim != 1 or len(steps) != n_frames_in_shard:
            raise ValueError(
                f"expected one simulation step per frame in {shard_path}"
            )
        if n_frames_in_shard != stop - start:
            raise ValueError(
                f"shard range [{start}, {stop}) does not match "
                f"{n_frames_in_shard} stored frames in {shard_path}"
            )
        if axial_unwrapper is None:
            axial_unwrapper = _PeriodicComUnwrapper(n_particles, lx)
            azimuthal_unwrapper = _PeriodicComUnwrapper(
                n_particles, 2.0 * np.pi
            )
        assert azimuthal_unwrapper is not None

        take = frame_limit - loaded_frames
        coords = coords[:take]
        steps = steps[:take]
        directions = directions[:take]
        actual_take = coords.shape[0]

        theta = coords[:, :, 1]
        q_arrays.append(directions.astype(np.float32))

        for local_frame in range(actual_take):
            frame_index = start + local_frame
            com_frames.append(
                (
                    axial_unwrapper.push(coords[local_frame, :, 0], frame_index),
                    azimuthal_unwrapper.push(theta[local_frame], frame_index),
                    float(np.mean(coords[local_frame, :, 2])),
                )
            )
            step_frames.append(int(steps[local_frame]))

        loaded_frames += actual_take
        expected_start = stop
        if loaded_frames >= frame_limit:
            break

    if len(com_frames) < 3:
        raise ValueError(
            "at least three frames are required for velocity correlation"
        )
    return (
        np.array(com_frames, dtype=np.float64),
        np.concatenate(q_arrays, axis=0),
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


def _finite_difference_vector(
    values: NDArray[np.float64],
    coords: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Component-wise finite difference of a (frames, components) series."""
    return np.stack(
        [_finite_difference(values[:, axis], coords) for axis in range(values.shape[1])],
        axis=1,
    )


def _normalized_autocorrelation(
    series: NDArray[np.float64],
    max_lag: int,
) -> NDArray[np.float64]:
    """Unbiased, lag-zero-normalized autocorrelation of a scalar or vector.

    ``series`` is either (frames,) or (frames, components). For the vector
    case, products are dot products summed over components.
    """
    if series.ndim == 1:
        series = series[:, None]
    if series.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D series, got shape {series.shape}")
    n = len(series)
    if n < 3:
        raise ValueError("need at least three samples")
    if not np.all(np.isfinite(series)):
        raise ValueError("series contains non-finite samples")
    if max_lag < 0:
        raise ValueError("max_lag must be nonnegative")
    if max_lag > n - 2:
        raise ValueError(
            f"max_lag ({max_lag}) exceeds available lags "
            f"({n - 2} for {n} samples)"
        )

    correlation = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        sample_count = n - lag
        left = series[:sample_count]
        right = series[lag : lag + sample_count]
        correlation[lag] = float(np.sum(left * right)) / sample_count
    if correlation[0] == 0.0:
        raise ValueError("cannot normalize an autocorrelation with zero lag value")
    return correlation / correlation[0]


def _distinct_particle_correlation(
    q: NDArray[np.float32],
    max_lag: int,
) -> NDArray[np.float64]:
    """Distinct-particle orientation correlation, normalized to 1 at lag 0.

    Returns the j != i part of the double sum,

        1/N^2 sum_i sum_{j != i} q_i(t) . q_j(t + lag),

    obtained as the full double sum minus the self (i == j) term. The full
    term factorizes through the per-frame mean and is cheap; the self term
    does not factorize, so it needs one FFT per particle chunk.

    Like the other curves, the unbiased lagged product is normalized by its
    lag-zero value.
    """
    n_frames, n_particles, _ = q.shape
    if max_lag < 0 or max_lag > n_frames - 2:
        raise ValueError(f"max_lag {max_lag} out of range for {n_frames} frames")

    pair_counts = np.arange(
        n_frames, n_frames - max_lag - 1, -1, dtype=np.float64
    )

    mean_q = q.mean(axis=1, dtype=np.float64)
    full = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        count = n_frames - lag
        full[lag] = np.sum(mean_q[:count] * mean_q[lag:])
    full /= pair_counts

    # Chunk the particles so a long run does not need one huge FFT temporary.
    self_total = np.zeros(max_lag + 1, dtype=np.float64)
    for start in range(0, n_particles, 512):
        chunk = q[:, start : start + 512].astype(np.float64)
        raw = fftconvolve(chunk, chunk[::-1], mode="full", axes=0)
        self_total += raw[n_frames - 1 : n_frames + max_lag].sum(axis=(1, 2))
    self_term = self_total / (pair_counts * n_particles**2)

    distinct = full - self_term
    if distinct[0] == 0.0:
        raise ValueError("distinct-particle correlation has zero lag-0 value")
    return distinct / distinct[0]


def _self_particle_orientation_correlation(
    q: NDArray[np.float32],
    max_lag: int,
) -> NDArray[np.float64]:
    """Return the normalized autocorrelation of each particle with itself.

    The averaged products are ``q_i(t) . q_i(t + lag)`` with no cross-particle
    terms. FFT particle chunks compute all requested lags together.
    """
    n_frames, n_particles, _ = q.shape
    if max_lag < 0 or max_lag > n_frames - 2:
        raise ValueError(f"max_lag {max_lag} out of range for {n_frames} frames")

    total = np.zeros(max_lag + 1, dtype=np.float64)
    for start in range(0, n_particles, 512):
        chunk = q[:, start : start + 512]
        raw = fftconvolve(chunk, chunk[::-1], mode="full", axes=0)
        total += raw[n_frames - 1 : n_frames + max_lag].sum(axis=(1, 2))
    pair_counts = np.arange(
        n_frames, n_frames - max_lag - 1, -1, dtype=np.float64
    )
    correlation = total / (pair_counts * n_particles)
    if correlation[0] == 0.0:
        raise ValueError("self-particle autocorrelation has zero lag-0 value")
    return correlation / correlation[0]


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
    velocity_x: NDArray[np.float64],
    velocity_3d: NDArray[np.float64],
    orientation_x: NDArray[np.float64],
    orientation_3d: NDArray[np.float64],
    orientation_distinct: NDArray[np.float64],
    orientation_self: NDArray[np.float64],
    label: str,
    output: Path,
) -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.1)
    palette = sns.color_palette("colorblind")
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)

    curves = (
        (velocity_x, palette[0], "-", r"COM velocity $v_x$"),
        (velocity_3d, palette[1], "-", r"COM velocity $\vec{v}\cdot\vec{v}$"),
        (orientation_x, palette[2], "--", r"Orientation $q_x$"),
        (
            orientation_3d,
            palette[3],
            "--",
            r"Orientation $\vec{q}_i\cdot\vec{q}_j$, all $i,j$",
        ),
        (
            orientation_distinct,
            palette[4],
            ":",
            r"Orientation $\vec{q}_i\cdot\vec{q}_j$, $j \neq i$",
        ),
        (
            orientation_self,
            palette[5],
            "-.",
            r"Orientation $\vec{q}_i(t)\cdot\vec{q}_i(t+\Delta t)$",
        ),
    )
    for correlation, color, style, name in curves:
        axis.plot(lag_time, correlation, color=color, ls=style, lw=2.0, label=name)

    axis.axhline(0.0, color="0.65", lw=0.9)
    axis.set_title(f"Normalized autocorrelation — {label}")
    axis.set_xlabel("Lag time")
    axis.set_ylabel("Normalized autocorrelation")
    axis.set_ylim(-1.05, 1.05)
    axis.set_xlim(0.0, lag_time[-1])
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

        com, q, steps, lx = _load_frame_series(
            input_dir, manifest, args.frames
        )
        print(
            f"[correlation] loaded {len(com)} frames, "
            f"{q.shape[1]} particles, lx={lx:.6g}",
            flush=True,
        )
        elapsed = _elapsed_time(steps, timestep)
        velocity = _finite_difference_vector(com, elapsed)

        # Cap max_lag to what the velocity series actually supports, then
        # use the same effective value for correlation and lag-time arrays.
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

        # The full double sum over particle pairs factorizes exactly through
        # the per-frame mean orientation, so these two curves cost O(T N).
        mean_q = q.mean(axis=1, dtype=np.float64)
        velocity_x = _normalized_autocorrelation(
            velocity[:, 0], effective_max_lag
        )
        velocity_3d = _normalized_autocorrelation(
            velocity, effective_max_lag
        )
        orientation_x = _normalized_autocorrelation(
            mean_q[:, 0], effective_max_lag
        )
        orientation_3d = _normalized_autocorrelation(
            mean_q, effective_max_lag
        )
        orientation_distinct = _distinct_particle_correlation(
            q, effective_max_lag
        )
        orientation_self = _self_particle_orientation_correlation(
            q, effective_max_lag
        )
        lag_time = _lag_times(elapsed, effective_max_lag)

        output_path = args.output_dir / f"velocity_correlation_{slug}.svg"
        _plot_correlation(
            lag_time,
            velocity_x,
            velocity_3d,
            orientation_x,
            orientation_3d,
            orientation_distinct,
            orientation_self,
            label,
            output_path,
        )
        print(f"[correlation] wrote {output_path}", flush=True)

    print(f"[correlation] done — {len(args.input_dir)} plot(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
