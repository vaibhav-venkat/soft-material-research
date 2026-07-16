from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open
from scipy.signal import correlate
from scipy.stats import pearsonr

from hexatic.big_lx.cases import BigLxCase, CasePaths
from hexatic.big_lx.plot_center_of_mass import center_of_mass_series
from hexatic.constants import cylinder


@dataclass(frozen=True)
class CorrelationSeries:
    case: BigLxCase
    lag_indices: np.ndarray
    lag_times: np.ndarray
    velocity: np.ndarray
    velocity_pearson: np.ndarray
    psi6_autocorrelation: np.ndarray
    psi6_pearson: np.ndarray
    time_origin_counts: np.ndarray
    shell_particle_counts: np.ndarray


class ShardRecord(TypedDict):
    file: str
    frame_start: int
    frame_stop: int


def _load_manifest(analysis_dir: Path, case: BigLxCase) -> dict[str, object]:
    manifest_path = analysis_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing analysis manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "hexatic.big_lx.analysis.v1":
        raise ValueError(f"Unsupported analysis schema in {manifest_path}")
    if manifest.get("complete") is not True:
        raise ValueError(f"Analysis is not marked complete: {manifest_path}")
    case_payload = manifest.get("case")
    if not isinstance(case_payload, dict) or case_payload.get("case_id") != case.case_id:
        raise ValueError(f"Analysis manifest does not match case {case.case_id}")
    return manifest


def _validated_shards(
    analysis_dir: Path,
    manifest: dict[str, object],
) -> tuple[list[ShardRecord], int]:
    payload = manifest.get("shards")
    if not isinstance(payload, list) or not payload:
        raise ValueError("The analysis manifest contains no frame shards")
    shards: list[ShardRecord] = []
    expected_start = 0
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("The analysis manifest contains an invalid shard entry")
        start = entry.get("frame_start")
        stop = entry.get("frame_stop")
        filename = entry.get("file")
        if (
            not isinstance(start, int)
            or start != expected_start
            or not isinstance(stop, int)
            or stop <= expected_start
            or not isinstance(filename, str)
        ):
            raise ValueError("Analysis shards are not contiguous from frame zero")
        path = analysis_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing frame shard: {path}")
        shards.append(
            {"file": filename, "frame_start": start, "frame_stop": stop}
        )
        expected_start = stop
    frame_count = manifest.get("frame_count")
    if not isinstance(frame_count, int) or frame_count != expected_start:
        raise ValueError(
            "Analysis manifest frame_count does not match the contiguous shards"
        )
    return shards, frame_count


def _load_mean_hexatic_series(
    case: BigLxCase,
    analysis_dir: Path,
    manifest: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shards, frame_count = _validated_shards(analysis_dir, manifest)
    if frame_count < 2:
        raise ValueError("At least two analysis frames are required")
    steps = np.empty(frame_count, dtype=np.int64)
    mean_magnitudes = np.empty(frame_count, dtype=np.float64)
    shell_counts = np.empty(frame_count, dtype=np.int64)
    required = ("psi_real", "psi_imag", "hexatic_shell_mask", "step")

    for shard in shards:
        start = shard["frame_start"]
        stop = shard["frame_stop"]
        path = analysis_dir / shard["file"]
        with safe_open(path, framework="numpy") as tensors:
            keys = set(tensors.keys())
            missing = [name for name in required if name not in keys]
            if missing:
                raise KeyError(f"{path} is missing tensors: {', '.join(missing)}")
            real = np.asarray(tensors.get_tensor("psi_real"))
            imaginary = np.asarray(tensors.get_tensor("psi_imag"))
            mask = np.asarray(tensors.get_tensor("hexatic_shell_mask"), dtype=np.bool_)
            shard_steps = np.asarray(tensors.get_tensor("step")).reshape(-1)

        expected_shape = (stop - start, case.n_particles)
        if real.shape != expected_shape or imaginary.shape != expected_shape:
            raise ValueError(f"Hexatic tensor shape mismatch in {path}")
        if mask.shape != expected_shape:
            raise ValueError(f"Hexatic shell-mask shape mismatch in {path}")
        if shard_steps.shape != (stop - start,):
            raise ValueError(f"Step tensor shape mismatch in {path}")

        magnitudes = np.hypot(real, imaginary)
        counts = np.count_nonzero(mask, axis=1).astype(np.int64)
        if np.any(counts == 0):
            local_index = int(np.flatnonzero(counts == 0)[0])
            raise ValueError(f"No shell-valid hexatic particles in frame {start + local_index}")
        totals = np.sum(magnitudes, where=mask, axis=1, dtype=np.float64)
        mean_magnitudes[start:stop] = totals / counts
        shell_counts[start:stop] = counts
        steps[start:stop] = shard_steps.astype(np.int64, copy=False)

    if np.any(np.diff(steps) <= 0):
        raise ValueError("Trajectory steps must be strictly increasing")
    return steps, mean_magnitudes, shell_counts


def _normalized_autocorrelation(values: np.ndarray, max_lag: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    full = correlate(values, values, mode="full", method="fft")
    start = values.size - 1
    sums = full[start : start + max_lag + 1]
    origins = values.size - np.arange(max_lag + 1, dtype=np.int64)
    means = sums / origins
    normalization = float(means[0])
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError(f"{name} has zero or invalid lag-zero power")
    result = means / normalization
    result[0] = 1.0
    return result


def lagged_pearson(values: np.ndarray, max_lag: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty(max_lag + 1, dtype=np.float64)
    result[0] = 1.0
    for lag in range(1, max_lag + 1):
        statistic = float(pearsonr(values[:-lag], values[lag:]).statistic)
        if not np.isfinite(statistic):
            raise ValueError(f"{name} has zero variance at lag {lag}")
        result[lag] = statistic
    return np.clip(result, -1.0, 1.0)


def analyze_correlations(
    case: BigLxCase,
    *,
    output_root: Path,
    min_origins: int = 10,
    max_lag: int | None = None,
    absolute: bool = False,
) -> CorrelationSeries:
    if min_origins < 2:
        raise ValueError("min_origins must be at least two for Pearson correlation")
    if max_lag is not None and max_lag < 0:
        raise ValueError("max_lag must be nonnegative")

    analysis_dir = CasePaths(case, output_root).analysis_dir
    manifest = _load_manifest(analysis_dir, case)
    steps, mean_psi6, shell_counts = _load_mean_hexatic_series(
        case, analysis_dir, manifest
    )
    com = center_of_mass_series(case, analysis_dir)
    if not np.array_equal(steps, com.steps):
        raise ValueError("Hexatic and COM step arrays do not match")

    frame_count = steps.size
    if min_origins > frame_count:
        raise ValueError(
            f"min_origins={min_origins} exceeds the {frame_count} available frames"
        )
    selected_max_lag = frame_count - min_origins
    if max_lag is not None:
        selected_max_lag = min(selected_max_lag, max_lag)

    step_spacing = np.diff(steps)
    if not np.all(step_spacing == step_spacing[0]):
        raise ValueError("Correlation analysis requires uniformly spaced steps")
    lag_indices = np.arange(selected_max_lag + 1, dtype=np.int64)
    lag_times = (
        lag_indices.astype(np.float64)
        * float(step_spacing[0])
        * cylinder.SIMULATION.timestep
    )
    velocity = _normalized_autocorrelation(
        com.x_velocity, selected_max_lag, "Axial COM velocity"
    )
    velocity_pearson = lagged_pearson(
        com.x_velocity, selected_max_lag, "Axial COM velocity"
    )
    if absolute:
        velocity = np.abs(velocity)
        velocity_pearson = np.abs(velocity_pearson)
    psi6_autocorrelation = _normalized_autocorrelation(
        mean_psi6, selected_max_lag, "Mean hexatic magnitude"
    )
    psi6_pearson = lagged_pearson(
        mean_psi6, selected_max_lag, "Mean hexatic magnitude"
    )
    return CorrelationSeries(
        case=case,
        lag_indices=lag_indices,
        lag_times=lag_times,
        velocity=velocity,
        velocity_pearson=velocity_pearson,
        psi6_autocorrelation=psi6_autocorrelation,
        psi6_pearson=psi6_pearson,
        time_origin_counts=frame_count - lag_indices,
        shell_particle_counts=shell_counts,
    )


def plot_correlations(
    series_by_case: list[CorrelationSeries],
    output: Path,
    *,
    dpi: int = 180,
    absolute: bool = False,
) -> Path:
    if not series_by_case:
        raise ValueError("At least one correlation series is required")
    if dpi < 1:
        raise ValueError("dpi must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, (autocorrelation_axis, pearson_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 11),
        sharex=True,
    )
    magnitude_axis = autocorrelation_axis.twinx()
    colors = plt.colormaps["viridis"](
        np.linspace(0.1, 0.9, len(series_by_case))
    )
    for color, series in zip(colors, series_by_case, strict=True):
        label = series.case.label
        autocorrelation_axis.plot(
            series.lag_times,
            series.velocity,
            color=color,
            linestyle="-",
            linewidth=1.7,
            label=rf"$C_v$: {label}",
        )
        magnitude_axis.plot(
            series.lag_times,
            series.psi6_autocorrelation,
            color=color,
            linestyle="--",
            linewidth=1.7,
            label=rf"$C_{{|\psi_6|}}$: {label}",
        )
        pearson_axis.plot(
            series.lag_times,
            series.velocity_pearson,
            color=color,
            linestyle="-",
            linewidth=1.7,
            label=rf"$r_v$: {label}",
        )
        pearson_axis.plot(
            series.lag_times,
            series.psi6_pearson,
            color=color,
            linestyle="--",
            linewidth=1.7,
            label=rf"$r_{{|\psi_6|}}$: {label}",
        )

    autocorrelation_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    magnitude_axis.axhline(1.0, color="black", linewidth=0.8, alpha=0.3)
    magnitude_axis.set_ylim(0.9, 1.05)
    autocorrelation_axis.set_ylabel("normalized axial COM-velocity correlation")
    magnitude_axis.set_ylabel(r"normalized $|\psi_6|$ autocorrelation")
    velocity_label = r"$|C_v|$" if absolute else r"$C_v$"
    autocorrelation_axis.set_title(
        "Regular normalized autocorrelations "
        rf"(`scipy.signal.correlate`, FFT; {velocity_label})"
    )
    autocorrelation_axis.grid(alpha=0.2)
    handles, labels = autocorrelation_axis.get_legend_handles_labels()
    magnitude_handles, magnitude_labels = magnitude_axis.get_legend_handles_labels()
    autocorrelation_axis.legend(
        handles + magnitude_handles,
        labels + magnitude_labels,
        loc="best",
    )

    pearson_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    pearson_axis.axhline(1.0, color="black", linewidth=0.8, alpha=0.3)
    pearson_axis.set_ylim(-1.05, 1.05)
    pearson_axis.set_xlabel("lag time")
    pearson_axis.set_ylabel("Pearson correlation coefficient")
    pearson_axis.set_title(
        "Lag-specific axial COM-velocity and shell-mean hexatic correlations "
        "(`scipy.stats.pearsonr`)"
    )
    pearson_axis.grid(alpha=0.2)
    pearson_axis.legend(loc="best")

    figure.suptitle("Big-Lx velocity and hexatic-magnitude lag correlations")
    figure.tight_layout()
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output
