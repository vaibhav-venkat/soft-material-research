"""Plot orientation autocorrelations for the ideal-cylinder simulation."""

from __future__ import annotations

import argparse
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

IDEAL_CASE_ID = "ideal_abp_cylinder"


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
) -> NDArray[np.float32]:
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
        values = _cartesian_directions(values, current_name)
        arrays.append(values[: frame_limit - loaded_frames])
        loaded_frames += min(len(values), frame_limit - loaded_frames)
        expected_start = stop
        if loaded_frames >= frame_limit:
            break

    if not arrays:
        raise ValueError("no polarization frames were loaded")
    return np.concatenate(arrays, axis=0)


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


def _plot(
    q_corr: NDArray[np.float64],
    qx_corr: NDArray[np.float64],
    output: Path,
) -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.1)
    color = sns.color_palette("colorblind")[0]
    lags = np.arange(len(q_corr))
    figure, axes = plt.subplots(
        2, 1, figsize=(7.2, 7.0), sharex=True, constrained_layout=True
    )
    axes[0].plot(lags, q_corr, color=color, linewidth=2.0)
    axes[0].set_title(r"Orientation autocorrelation")
    axes[0].set_ylabel(r"$C_q(\Delta t)$")
    axes[1].plot(lags, qx_corr, color=color, linewidth=2.0)
    axes[1].set_title(r"Axial-orientation autocorrelation")
    axes[1].set_ylabel(r"$C_{q_x}(\Delta t)$")
    axes[1].set_xlabel(r"Lag $\Delta t$ (frames)")

    for axis in axes:
        axis.axhline(0.0, color="0.65", linewidth=0.8, zorder=0)
        axis.grid(axis="y", color="0.9", linewidth=0.7)
        axis.set_xlim(0, len(q_corr) - 1)
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
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.max_lag < 0:
        parser.error("--max-lag must be nonnegative")

    manifest = _load_manifest(args.shard_dir)
    q = _load_polarization(args.shard_dir, manifest, args.frames)
    q_corr, qx_corr = _orientation_autocorrelations(q, args.max_lag)
    _plot(q_corr, qx_corr, args.output)
    print(f"wrote {args.output} using {len(q)} frames and {q.shape[1]} particles")


if __name__ == "__main__":
    main()
