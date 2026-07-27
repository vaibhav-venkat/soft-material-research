from __future__ import annotations

import argparse
from dataclasses import dataclass
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


class _AxialComUnwrapper:
    """Streaming unwrap of wrapped axial coordinates into a COM x series.

    Mirrors the native big-Lx implementation: the first frame pushed becomes
    the unwrapped reference, later frames advance by the minimum-image
    displacement, and each mean uses the same Kahan summation order.
    """

    def __init__(self, n_particles: int, lx: float) -> None:
        if n_particles < 1:
            raise ValueError("need at least one particle")
        self._n_particles = n_particles
        self._lx = lx
        self._previous = np.empty(n_particles, dtype=np.float64)
        self._unwrapped = np.empty(n_particles, dtype=np.float64)
        self._started = False

    def push(self, frame_x: NDArray[np.float64], frame_index: int) -> float:
        if len(frame_x) != self._n_particles:
            raise ValueError(f"particle count changes at frame {frame_index}")
        if not np.all(np.isfinite(frame_x)):
            raise ValueError(f"non-finite x coordinate at frame {frame_index}")
        if not self._started:
            self._unwrapped[:] = frame_x
            self._started = True
        else:
            # Displacement from the previous wrapped position, minus the
            # nearest integer number of box lengths.
            displacement = frame_x - self._previous
            scaled = displacement / self._lx
            nearest_integer = np.copysign(
                np.floor(np.abs(scaled) + 0.5),
                scaled,
            )
            self._unwrapped += displacement - self._lx * nearest_integer
        self._previous[:] = frame_x

        total = 0.0
        compensation = 0.0
        for coordinate in self._unwrapped:
            corrected = float(coordinate) - compensation
            updated = total + corrected
            compensation = (updated - total) - corrected
            total = updated
        return total / self._n_particles


def _axial_directions(
    shard: Any,
    shard_path: Path,
) -> NDArray[np.float64]:
    """Return the axial component of the per-particle orientation vectors.

    ``new_sims`` shards store these as ``polarization`` and ``big_lx`` shards
    as ``active_direction``; both are Cartesian unit vectors whose component 0
    is the axial (x) direction.
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
    return directions[:, :, 0]


def _load_frame_series(
    shard_dir: Path,
    manifest: dict[str, Any],
    frame_limit: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], float]:
    """Load unwrapped COM x positions, mean axial orientation, and steps.

    Returns (com_x, mean_qx, steps, lx) where com_x is the per-frame mean of
    unwrapped axial particle coordinates, mean_qx is the per-frame mean axial
    orientation component, and lx is the axial box length.

    ``mean_qx`` is the exact factorization of the cross-particle orientation
    correlation: because

        1/N^2 sum_i sum_j q_i,x(t1) q_j,x(t2) = mean_qx(t1) * mean_qx(t2),

    the naive O(T N^2) double sum reduces to an O(T N) per-frame mean with no
    approximation.
    """
    static_path = shard_dir / "static.safetensors"
    lx = _load_lx(static_path)

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("manifest contains no safetensor shards")

    com_x_frames: list[float] = []
    mean_qx_frames: list[float] = []
    step_frames: list[int] = []
    loaded_frames = 0
    expected_start = 0
    # Per-particle unwrapping state, initialised on the first frame.
    unwrapper: _AxialComUnwrapper | None = None

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
            axial_directions = _axial_directions(shard, shard_path)

        if coords.ndim != 3 or coords.shape[1] == 0 or coords.shape[2] == 0:
            raise ValueError(
                f"expected coords shape (frames, particles, components), "
                f"got {coords.shape}"
            )
        n_frames_in_shard = coords.shape[0]
        n_particles = coords.shape[1]
        if axial_directions.shape != (n_frames_in_shard, n_particles):
            raise ValueError(
                f"orientation shape {axial_directions.shape} does not match "
                f"coords shape {coords.shape[:2]} in {shard_path}"
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
        if unwrapper is None:
            unwrapper = _AxialComUnwrapper(n_particles, lx)

        take = frame_limit - loaded_frames
        coords = coords[:take]
        steps = steps[:take]
        axial_directions = axial_directions[:take]
        actual_take = coords.shape[0]

        x_wrapped = coords[:, :, 0]  # shape: (frames, particles)
        for local_frame in range(actual_take):
            frame_index = start + local_frame
            com_x_frames.append(
                unwrapper.push(x_wrapped[local_frame], frame_index)
            )
            # No native reference implementation to bit-match here, so a plain
            # float64 mean is enough for the orientation series.
            mean_qx_frames.append(float(np.mean(axial_directions[local_frame])))
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
        np.array(mean_qx_frames, dtype=np.float64),
        np.array(step_frames, dtype=np.int64),
        lx,
    )


@dataclass(frozen=True)
class GsdSeries:
    """Per-frame COM decomposition inputs read straight from a trajectory."""

    com_x: NDArray[np.float64]
    mean_qx: NDArray[np.float64]
    qx: NDArray[np.float32]  # (frames, particles), for the self term
    mean_wall_fx: NDArray[np.float64]
    steps: NDArray[np.int64]
    lx: float


def _quaternion_axial_x(orientation: NDArray[np.float64]) -> NDArray[np.float64]:
    """Axial component of the body-frame +x axis rotated into the lab frame.

    HOOMD stores quaternions as (w, x, y, z) and the active force points along
    the body-frame +x axis, so only the first row of the rotation matrix is
    needed here.
    """
    if orientation.ndim != 2 or orientation.shape[1] != 4:
        raise ValueError(
            f"expected quaternion shape (particles, 4), got {orientation.shape}"
        )
    norms = np.linalg.norm(orientation, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("frame contains a zero quaternion")
    quaternion = orientation / norms[:, None]
    y = quaternion[:, 2]
    z = quaternion[:, 3]
    return 1.0 - 2.0 * (y * y + z * z)


def _logged_wall_force_x(frame: Any, n_particles: int) -> NDArray[np.float64]:
    """Sum the axial component of every logged wall-force array in a frame.

    ``big_lx`` logs the pair force and the wall force under separate keys, so
    the wall contribution can be isolated by name.
    """
    log = getattr(frame, "log", None)
    if not log:
        raise ValueError(
            "GSD frame has no logger data; wall forces are only logged by "
            "hexatic/big_lx/simulate_case.py, not by hexatic/new_sims"
        )
    total = np.zeros(n_particles, dtype=np.float64)
    matched = False
    for key, value in log.items():
        name = str(key).lower()
        if "force" not in name or "wall" not in name:
            continue
        array = np.asarray(value, dtype=np.float64)
        if array.shape[:1] != (n_particles,) or array.ndim != 2:
            continue
        total += array[:, 0]
        matched = True
    if not matched:
        available = ", ".join(str(key) for key in log)
        raise ValueError(f"no logged wall forces; available keys: {available}")
    return total


def _load_gsd_series(gsd_path: Path, frame_limit: int) -> GsdSeries:
    """Read COM x, axial orientations, and mean axial wall force from a GSD."""
    import gsd.hoomd

    com_x_frames: list[float] = []
    mean_qx_frames: list[float] = []
    wall_fx_frames: list[float] = []
    qx_frames: list[NDArray[np.float32]] = []
    step_frames: list[int] = []
    unwrapper: _AxialComUnwrapper | None = None
    lx = 0.0

    with gsd.hoomd.open(str(gsd_path), mode="r") as trajectory:
        for frame_index, frame in enumerate(trajectory):
            if frame_index >= frame_limit:
                break
            n_particles = int(frame.particles.N)
            if unwrapper is None:
                box = np.asarray(frame.configuration.box, dtype=np.float64)
                lx = float(box[0])
                if not np.isfinite(lx) or lx <= 0.0:
                    raise ValueError(f"invalid axial box length lx={lx}")
                unwrapper = _AxialComUnwrapper(n_particles, lx)

            positions = np.asarray(frame.particles.position, dtype=np.float64)
            orientation = np.asarray(
                frame.particles.orientation, dtype=np.float64
            )
            qx = _quaternion_axial_x(orientation)
            step = frame.configuration.step
            if step is None:
                raise ValueError(f"frame {frame_index} has no simulation step")

            com_x_frames.append(unwrapper.push(positions[:, 0], frame_index))
            mean_qx_frames.append(float(np.mean(qx)))
            qx_frames.append(qx.astype(np.float32))
            wall_fx_frames.append(
                float(np.mean(_logged_wall_force_x(frame, n_particles)))
            )
            step_frames.append(int(step))

    if len(com_x_frames) < 3:
        raise ValueError(
            f"{gsd_path} has fewer than three usable frames "
            f"({len(com_x_frames)})"
        )
    return GsdSeries(
        com_x=np.array(com_x_frames, dtype=np.float64),
        mean_qx=np.array(mean_qx_frames, dtype=np.float64),
        qx=np.stack(qx_frames, axis=0),
        mean_wall_fx=np.array(wall_fx_frames, dtype=np.float64),
        steps=np.array(step_frames, dtype=np.int64),
        lx=lx,
    )


def _lagged_covariance(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    max_lag: int,
) -> NDArray[np.float64]:
    """Return cov(left[t], right[t + lag]) for lag in [0, max_lag].

    Both series are centred on their own global mean and every lag uses the
    same 1/(T - lag) weight, so covariances built this way stay *additive*:
    if v = a + b then cov_vv = cov_aa + cov_ab + cov_ba + cov_bb exactly.
    Pearson correlation does not have this property, which is why the
    decomposition panel works in covariance units.
    """
    n = len(left)
    if len(right) != n:
        raise ValueError("series must have the same length")
    if max_lag < 0 or max_lag > n - 2:
        raise ValueError(f"max_lag {max_lag} out of range for {n} samples")
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    covariance = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        count = n - lag
        covariance[lag] = (
            np.dot(left_centered[:count], right_centered[lag : lag + count])
            / count
        )
    return covariance


def _self_orientation_covariance(
    qx: NDArray[np.float32],
    max_lag: int,
) -> NDArray[np.float64]:
    """Particle-averaged self covariance (1/N) sum_i cov(q_i,x(t), q_i,x(t+lag)).

    This is the i == j part of the double sum. Unlike the factorized
    cross-particle term it does not collapse, so it needs one FFT per particle
    chunk, exactly as in plot_ideal_orientation_autocorrelation.
    """
    n_frames, n_particles = qx.shape
    if max_lag < 0 or max_lag > n_frames - 2:
        raise ValueError(f"max_lag {max_lag} out of range for {n_frames} frames")
    centered = (qx - qx.mean(axis=0, keepdims=True)).astype(np.float64)
    total = np.zeros(max_lag + 1, dtype=np.float64)
    # Chunk the particles so a long run does not need one huge FFT temporary.
    for start in range(0, n_particles, 512):
        chunk = centered[:, start : start + 512]
        raw = fftconvolve(chunk, chunk[::-1], mode="full", axes=0)
        total += raw[n_frames - 1 : n_frames + max_lag].sum(axis=1)
    pair_counts = np.arange(
        n_frames, n_frames - max_lag - 1, -1, dtype=np.float64
    )
    return total / (pair_counts * n_particles)


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
    velocity_pearson: NDArray[np.float64],
    propulsion_pearson: NDArray[np.float64],
    residual_pearson: NDArray[np.float64],
    label: str,
    output: Path,
) -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.1)
    palette = sns.color_palette("colorblind")
    figure, axes = plt.subplots(
        2, 1, figsize=(8.2, 8.0), sharex=True, constrained_layout=True
    )

    axes[0].plot(
        lag_time,
        velocity_pearson,
        color=palette[0],
        lw=2.0,
        label=r"COM velocity $V_x$",
    )
    axes[0].plot(
        lag_time,
        propulsion_pearson,
        color=palette[1],
        lw=2.0,
        label=r"Propulsion $U_0 \langle q_x \rangle$",
    )
    axes[0].plot(
        lag_time,
        residual_pearson,
        color=palette[2],
        lw=2.0,
        label=r"Residual $V_x - U_0 \langle q_x \rangle$",
    )
    axes[0].set_title(f"Lagged axial Pearson correlation — {label}")
    axes[0].set_ylabel("Pearson correlation")
    axes[0].set_ylim(-1.05, 1.05)

    axes[1].plot(
        lag_time,
        velocity_pearson - propulsion_pearson,
        color=palette[3],
        lw=2.0,
        label=r"$C_{V_x} - C_{U_0 \langle q_x \rangle}$",
    )
    axes[1].set_title("Correlation difference")
    axes[1].set_ylabel("Difference")
    axes[1].set_xlabel("Lag time")

    for axis in axes:
        axis.axhline(0.0, color="0.65", lw=0.9)
        axis.grid(axis="y", color="0.9", lw=0.7)
        axis.set_xlim(0.0, lag_time[-1])
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


def _plot_decomposition(
    lag_time: NDArray[np.float64],
    velocity_cov: NDArray[np.float64],
    propulsion_cov: NDArray[np.float64],
    cross_qf_cov: NDArray[np.float64],
    cross_fq_cov: NDArray[np.float64],
    wall_cov: NDArray[np.float64],
    factorized_qq: NDArray[np.float64],
    self_qq: NDArray[np.float64],
    label: str,
    output: Path,
) -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.1)
    palette = sns.color_palette("colorblind")
    figure, axes = plt.subplots(
        2, 1, figsize=(8.2, 8.4), sharex=True, constrained_layout=True
    )

    # Normalizing every term by the same constant preserves additivity.
    scale = velocity_cov[0]
    if scale == 0.0:
        raise ValueError("COM velocity has zero variance")
    cross_cov = cross_qf_cov + cross_fq_cov
    model_sum = propulsion_cov + cross_cov + wall_cov
    axes[0].plot(
        lag_time,
        velocity_cov / scale,
        color=palette[0],
        lw=2.2,
        label=r"$C_{V_xV_x}$ (finite difference)",
    )
    axes[0].plot(
        lag_time,
        propulsion_cov / scale,
        color=palette[1],
        lw=1.8,
        label=r"$U_0^2\,C_{QQ}$",
    )
    axes[0].plot(
        lag_time,
        cross_cov / scale,
        color=palette[2],
        lw=1.8,
        label=r"$(U_0/\gamma)(C_{QF} + C_{FQ})$",
    )
    axes[0].plot(
        lag_time,
        wall_cov / scale,
        color=palette[3],
        lw=1.8,
        label=r"$C_{FF}/\gamma^2$ (wall)",
    )
    axes[0].plot(
        lag_time,
        model_sum / scale,
        color="0.2",
        ls="--",
        lw=1.4,
        label="Sum of terms",
    )
    axes[0].set_title(f"Axial COM-velocity covariance decomposition — {label}")
    axes[0].set_ylabel(r"Covariance / $C_{V_xV_x}(0)$")

    axes[1].plot(
        lag_time,
        factorized_qq / factorized_qq[0],
        color=palette[1],
        lw=2.0,
        label=r"Cross-particle $\langle q_x\rangle(0)\langle q_x\rangle(t)$",
    )
    axes[1].plot(
        lag_time,
        self_qq / self_qq[0],
        color=palette[4],
        lw=2.0,
        ls="--",
        label=r"Self $\frac{1}{N}\sum_i q_{i,x}(0)q_{i,x}(t)$",
    )
    axes[1].set_title(
        "Orientation correlation: cross-particle vs self "
        "(overlap ⇒ distinct particles are uncorrelated)"
    )
    axes[1].set_ylabel("Normalized covariance")
    axes[1].set_xlabel("Lag time")
    axes[1].set_ylim(-1.05, 1.05)

    for axis in axes:
        axis.axhline(0.0, color="0.65", lw=0.9)
        axis.grid(axis="y", color="0.9", lw=0.7)
        axis.set_xlim(0.0, lag_time[-1])
        axis.legend(frameon=False, ncol=2, fontsize="small")
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
        default=None,
        help="Safetensor output directory (repeatable); one SVG per input.",
    )
    parser.add_argument(
        "--gsd",
        action="append",
        type=Path,
        default=None,
        help=(
            "Trajectory GSD (repeatable) for the covariance decomposition. "
            "Self-contained — no --input-dir needed. Requires logged wall "
            "forces, which only hexatic/big_lx simulations write."
        ),
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
    parser.add_argument(
        "--u0",
        type=_positive_float,
        default=float(cylinder.SIMULATION.u0),
        help=(
            "Self-propulsion speed used to split the COM velocity into "
            "propulsion and residual parts (default: the cylinder constant)."
        ),
    )
    return parser.parse_args()


def _effective_max_lag(requested: int, n_samples: int) -> int:
    available = n_samples - 2
    if available < 0:
        raise ValueError(f"need at least 2 samples (got {n_samples})")
    if requested > available:
        print(
            f"[correlation] capping --max-lag {requested} -> {available} "
            f"(only {n_samples} samples)",
            flush=True,
        )
        return available
    return requested


def _run_decomposition(gsd_path: Path, args: argparse.Namespace) -> Path:
    """Covariance decomposition of the COM x-velocity read from a GSD."""
    series = _load_gsd_series(gsd_path, args.frames)
    timestep = (
        args.timestep
        if args.timestep is not None
        else float(cylinder.SIMULATION.timestep)
    )
    gamma = float(cylinder.SIMULATION.gamma)
    print(
        f"[decomposition] loaded {len(series.com_x)} frames, "
        f"{series.qx.shape[1]} particles, lx={series.lx:.6g}",
        flush=True,
    )

    elapsed = _elapsed_time(series.steps, timestep)
    velocity = _finite_difference(series.com_x, elapsed)
    max_lag = _effective_max_lag(args.max_lag, len(velocity))

    # v_x = U_0 Q + F/gamma with Q = <q_x> and F = <F_wall,x>, so the
    # covariance splits into four additive terms. The two cross terms are
    # generally unequal at nonzero lag: <Q(0)F(t)> != <F(0)Q(t)>.
    orientation = series.mean_qx
    wall_force = series.mean_wall_fx
    velocity_cov = _lagged_covariance(velocity, velocity, max_lag)
    factorized_qq = _lagged_covariance(orientation, orientation, max_lag)
    propulsion_cov = args.u0**2 * factorized_qq
    cross_qf_cov = (args.u0 / gamma) * _lagged_covariance(
        orientation, wall_force, max_lag
    )
    cross_fq_cov = (args.u0 / gamma) * _lagged_covariance(
        wall_force, orientation, max_lag
    )
    wall_cov = _lagged_covariance(wall_force, wall_force, max_lag) / gamma**2
    self_qq = _self_orientation_covariance(series.qx, max_lag)

    lag_time = _lag_times(elapsed, max_lag)
    output_path = args.output_dir / f"velocity_decomposition_{gsd_path.stem}.svg"
    _plot_decomposition(
        lag_time,
        velocity_cov,
        propulsion_cov,
        cross_qf_cov,
        cross_fq_cov,
        wall_cov,
        factorized_qq,
        self_qq,
        gsd_path.stem,
        output_path,
    )

    residual = velocity_cov[0] - (
        propulsion_cov[0] + cross_qf_cov[0] + cross_fq_cov[0] + wall_cov[0]
    )
    print(
        f"[decomposition] lag-0 closure residual "
        f"{residual / velocity_cov[0]:+.3e} of C_VV(0)",
        flush=True,
    )
    return output_path


def main() -> None:
    args = _parse_args()
    if args.frames < 3:
        raise ValueError("--frames must be at least 3")
    input_dirs = args.input_dir or []
    gsd_paths = args.gsd or []
    if not input_dirs and not gsd_paths:
        raise ValueError("pass at least one --input-dir or --gsd")

    for index, gsd_path in enumerate(gsd_paths, start=1):
        print(
            f"[decomposition] {index}/{len(gsd_paths)} gsd={gsd_path}",
            flush=True,
        )
        written = _run_decomposition(gsd_path, args)
        print(f"[decomposition] wrote {written}", flush=True)

    for index, input_dir in enumerate(input_dirs, start=1):
        print(
            f"[correlation] {index}/{len(input_dirs)} dir={input_dir}",
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

        com_x, mean_qx, steps, lx = _load_frame_series(
            input_dir, manifest, args.frames
        )
        print(
            f"[correlation] loaded {len(com_x)} frames, "
            f"lx={lx:.6g}",
            flush=True,
        )
        elapsed = _elapsed_time(steps, timestep)
        velocity = _finite_difference(com_x, elapsed)

        # Cap max_lag to what the velocity series actually supports, then
        # use the same effective value for Pearson *and* lag-time arrays.
        effective_max_lag = _effective_max_lag(args.max_lag, len(velocity))

        # v_x(t) = U_0 <q_x>(t) + <F_x>(t)/gamma holds frame by frame, so the
        # residual isolates the non-propulsive contribution. Internal pair
        # forces cancel in the COM sum by Newton's third law, so for a
        # periodic cylinder only the wall force can survive in <F_x>.
        propulsion = args.u0 * mean_qx
        residual = velocity - propulsion

        pearson = _pearson_correlation(velocity, effective_max_lag)
        propulsion_pearson = _pearson_correlation(propulsion, effective_max_lag)
        residual_pearson = _pearson_correlation(residual, effective_max_lag)
        lag_time = _lag_times(elapsed, effective_max_lag)

        output_path = args.output_dir / f"velocity_correlation_{slug}.svg"
        _plot_correlation(
            lag_time,
            pearson,
            propulsion_pearson,
            residual_pearson,
            label,
            output_path,
        )
        print(f"[correlation] wrote {output_path}", flush=True)

    total = len(input_dirs) + len(gsd_paths)
    print(f"[correlation] done — {total} plot(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
