"""Plot orientation autocorrelations for the ideal-cylinder simulation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from safetensors import safe_open
import seaborn as sns
from scipy.signal import fftconvolve
from scipy.stats import linregress

from hexatic.constants import cylinder

IDEAL_CASE_ID = "ideal_abp_cylinder"


@dataclass(frozen=True)
class ExponentialFit:
    amplitude: float
    rate: float
    r_squared: float


def _load_manifest(shard_dir: Path) -> dict[str, Any]:
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "hexatic.new_sims.analysis.v1":
        raise ValueError(f"unsupported manifest schema in {manifest_path}")

    case = manifest.get("case")
    if not isinstance(case, dict):
        raise ValueError(f"missing case metadata in {manifest_path}")
    assert case.get("geometry_kind") == IDEAL_CASE_ID, (
        "this plot is only defined for the ideal_abp_cylinder case"
    )
    return manifest


def _load_polarization(
    shard_dir: Path,
    manifest: dict[str, Any],
    frame_limit: int,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    polarization_order = manifest.get("polarization_order")
    if polarization_order is not None and polarization_order != ["x", "y", "z"]:
        raise ValueError(
            "expected Cartesian polarization_order ['x', 'y', 'z'], "
            f"got {polarization_order!r}"
        )
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("manifest contains no safetensor shards")

    arrays: list[NDArray[np.float32]] = []
    step_arrays: list[NDArray[np.int64]] = []
    tensor_name: str | None = None
    loaded_frames = 0
    expected_start = 0
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
            current_name = (
                "polarization"
                if "polarization" in keys
                else "orientation"
                if "orientation" in keys
                else None
            )
            if current_name is None:
                raise KeyError(
                    f"{shard_path} has neither 'polarization' nor 'orientation'"
                )
            if tensor_name is None:
                tensor_name = current_name
            elif current_name != tensor_name:
                raise ValueError("orientation tensor name changes between shards")
            values = np.asarray(shard.get_tensor(current_name), dtype=np.float32)
            if "step" not in keys:
                raise KeyError(f"{shard_path} has no 'step' tensor")
            steps = np.asarray(shard.get_tensor("step"), dtype=np.int64)
        values = _cartesian_directions(values, current_name)
        if steps.ndim != 1 or len(steps) != len(values):
            raise ValueError(
                f"expected one simulation step per frame in {shard_path}"
            )
        take = frame_limit - loaded_frames
        arrays.append(values[:take])
        step_arrays.append(steps[:take])
        loaded_frames += min(len(values), frame_limit - loaded_frames)
        expected_start = stop
        if loaded_frames >= frame_limit:
            break

    if not arrays:
        raise ValueError("no polarization frames were loaded")
    return np.concatenate(arrays, axis=0), np.concatenate(step_arrays)


def _cartesian_directions(
    values: NDArray[np.float32],
    tensor_name: str,
) -> NDArray[np.float32]:
    """Return Cartesian (x, y, z) directions from stored vectors or quaternions."""
    if values.ndim != 3:
        raise ValueError(
            f"expected {tensor_name} shape (frames, particles, components), "
            f"got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{tensor_name} contains non-finite values")

    if tensor_name == "polarization":
        if values.shape[2] != 3:
            raise ValueError(
                f"expected Cartesian polarization shape (..., 3), got {values.shape}"
            )
        directions = values
    else:
        if values.shape[2] != 4:
            raise ValueError(
                f"expected HOOMD quaternion shape (..., 4), got {values.shape}"
            )
        # HOOMD stores quaternions as (w, x, y, z). The active direction is the
        # body-frame +x unit vector rotated into the lab Cartesian frame.
        norms = np.linalg.norm(values, axis=2, keepdims=True)
        if np.any(norms == 0.0):
            raise ValueError("orientation contains a zero quaternion")
        quaternion = values / norms
        w, x, y, z = (quaternion[:, :, index] for index in range(4))
        directions = np.stack(
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y + w * z),
                2.0 * (x * z - w * y),
            ),
            axis=2,
        )

    direction_norms = np.linalg.norm(directions, axis=2)
    if not np.allclose(direction_norms, 1.0, rtol=2e-4, atol=2e-4):
        raise ValueError(
            f"{tensor_name} does not describe unit Cartesian orientation vectors"
        )
    return np.asarray(directions, dtype=np.float32)


def _orientation_autocorrelations(
    q: NDArray[np.float32],
    max_lag: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return unbiased, lag-zero-normalized correlations for q and q_x."""
    if max_lag >= len(q):
        raise ValueError(
            f"--max-lag must be smaller than the {len(q)} loaded frames"
        )

    # Work in particle chunks so a 1,000-frame run does not require a single
    # multi-gigabyte FFT temporary.
    q_sum = np.zeros(max_lag + 1, dtype=np.float64)
    qx_sum = np.zeros(max_lag + 1, dtype=np.float64)
    for start in range(0, q.shape[1], 512):
        chunk = q[:, start : start + 512]
        raw = fftconvolve(chunk, chunk[::-1], mode="full", axes=0)
        positive_lags = raw[len(q) - 1 : len(q) + max_lag]
        q_sum += positive_lags.sum(axis=(1, 2), dtype=np.float64)
        qx_sum += positive_lags[:, :, 0].sum(axis=1, dtype=np.float64)

    pair_counts = np.arange(len(q), len(q) - max_lag - 1, -1, dtype=np.float64)
    normalization = pair_counts * q.shape[1]
    q_corr = q_sum / normalization
    qx_corr = qx_sum / normalization
    if q_corr[0] == 0.0 or qx_corr[0] == 0.0:
        raise ValueError("cannot normalize an autocorrelation with zero lag variance")
    return q_corr / q_corr[0], qx_corr / qx_corr[0]


def _lag_times(
    steps: NDArray[np.int64],
    max_lag: int,
    timestep: float,
) -> NDArray[np.float64]:
    if len(steps) < 2:
        raise ValueError("at least two frames are required to determine time spacing")
    step_spacing = np.diff(steps)
    if np.any(step_spacing <= 0) or not np.all(step_spacing == step_spacing[0]):
        raise ValueError("simulation frames must have uniform, increasing step spacing")
    return (
        np.arange(max_lag + 1, dtype=np.float64)
        * float(step_spacing[0])
        * timestep
    )


def _fit_exponential(
    time: NDArray[np.float64],
    correlation: NDArray[np.float64],
    tau_r: float,
    minimum_correlation: float,
) -> ExponentialFit:
    # Fit only the initial contiguous decay. Once the correlation reaches the
    # threshold (or crosses zero), later positive noise excursions stay excluded.
    invalid = ~np.isfinite(correlation) | (correlation <= minimum_correlation)
    stopping_points = np.flatnonzero(invalid)
    stop = int(stopping_points[0]) if stopping_points.size else len(correlation)
    if stop < 2:
        raise ValueError(
            "exponential regression needs at least two initial correlation "
            f"values above {minimum_correlation}"
        )
    scaled_time = time[:stop] / tau_r
    result = linregress(scaled_time, np.log(correlation[:stop]))
    return ExponentialFit(
        amplitude=float(np.exp(result.intercept)),
        rate=float(-result.slope),
        r_squared=float(result.rvalue**2),
    )


def _plot(
    lag_time: NDArray[np.float64],
    q_corr: NDArray[np.float64],
    qx_corr: NDArray[np.float64],
    q_fit: ExponentialFit,
    qx_fit: ExponentialFit,
    tau_r: float,
    output: Path,
) -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.1)
    color = sns.color_palette("colorblind")[0]
    figure, axes = plt.subplots(
        2, 1, figsize=(7.2, 7.0), sharex=True, constrained_layout=True
    )
    fits = (q_fit, qx_fit)
    correlations = (q_corr, qx_corr)
    for axis, correlation, fit in zip(axes, correlations, fits, strict=True):
        axis.plot(lag_time, correlation, color=color, linewidth=2.0, label="Data")
        axis.plot(
            lag_time,
            fit.amplitude * np.exp(-fit.rate * lag_time / tau_r),
            color="0.2",
            linestyle="--",
            linewidth=1.5,
            label="Log-linear fit",
        )
        axis.text(
            0.98,
            0.95,
            f"A = {fit.amplitude:.6g}\n"
            f"b = {fit.rate:.6g}\n"
            f"$R^2$ = {fit.r_squared:.6g}",
            transform=axis.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
        )
        axis.legend(frameon=False)

    axes[0].set_title(r"Orientation autocorrelation")
    axes[0].set_ylabel(r"$C_q(\Delta t)$")
    axes[1].set_title(r"Axial-orientation autocorrelation")
    axes[1].set_ylabel(r"$C_{q_x}(\Delta t)$")
    axes[1].set_xlabel(r"Lag time $\Delta t$")

    for axis in axes:
        axis.axhline(0.0, color="0.65", linewidth=0.8, zorder=0)
        axis.grid(axis="y", color="0.9", linewidth=0.7)
        axis.set_xlim(0.0, lag_time[-1])
        sns.despine(ax=axis)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="svg",
        bbox_inches="tight",
        metadata={"Creator": "hexatic.new_sims_analysis"},
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ideal-cylinder orientation autocorrelations."
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        required=True,
        help="Folder containing manifest.json and the safetensor frame shards.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=1000,
        help="Maximum number of initial frames to use (default: 1000).",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=900,
        help="Largest nonnegative lag in frames (default: 900).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ideal_orientation_autocorrelation.svg"),
        help="Output SVG path.",
    )
    parser.add_argument(
        "--fit-min-correlation",
        type=float,
        default=0.01,
        help=(
            "Stop the initial-decay fit when C first reaches this value "
            "(default: 0.01)."
        ),
    )
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.max_lag < 0:
        parser.error("--max-lag must be nonnegative")
    if not 0.0 < args.fit_min_correlation < 1.0:
        parser.error("--fit-min-correlation must be between 0 and 1")

    manifest = _load_manifest(args.shard_dir)
    q, steps = _load_polarization(args.shard_dir, manifest, args.frames)
    q_corr, qx_corr = _orientation_autocorrelations(q, args.max_lag)
    case = manifest["case"]
    timestep = float(case.get("timestep", cylinder.SIMULATION.timestep))
    tau_r = float(cylinder.SIMULATION.tau_r)
    if timestep <= 0.0 or tau_r <= 0.0:
        raise ValueError("timestep and tau_r must be positive")
    lag_time = _lag_times(steps, args.max_lag, timestep)
    q_fit = _fit_exponential(
        lag_time, q_corr, tau_r, args.fit_min_correlation
    )
    qx_fit = _fit_exponential(
        lag_time, qx_corr, tau_r, args.fit_min_correlation
    )
    _plot(lag_time, q_corr, qx_corr, q_fit, qx_fit, tau_r, args.output)
    print(f"wrote {args.output} using {len(q)} frames and {q.shape[1]} particles")
    print(
        f"q: A={q_fit.amplitude:.8g}, b={q_fit.rate:.8g}, "
        f"R^2={q_fit.r_squared:.8g}"
    )
    print(
        f"q_x: A={qx_fit.amplitude:.8g}, b={qx_fit.rate:.8g}, "
        f"R^2={qx_fit.r_squared:.8g}"
    )


if __name__ == "__main__":
    main()
