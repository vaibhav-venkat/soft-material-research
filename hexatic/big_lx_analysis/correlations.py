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
from scipy.fft import irfft, next_fast_len, rfft

from hexatic.big_lx.cases import BigLxCase, CasePaths
from hexatic.big_lx.plot_center_of_mass import center_of_mass_series
from hexatic.constants import cylinder


@dataclass(frozen=True)
class CorrelationSeries:
    case: BigLxCase
    lag_indices: np.ndarray
    lag_times: np.ndarray
    velocity: np.ndarray
    psi6: np.ndarray
    time_origin_counts: np.ndarray
    psi6_pair_counts: np.ndarray


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
            {
                "file": filename,
                "frame_start": start,
                "frame_stop": stop,
            }
        )
        expected_start = stop
    frame_count = manifest.get("frame_count")
    if not isinstance(frame_count, int) or frame_count != expected_start:
        raise ValueError(
            "Analysis manifest frame_count does not match the contiguous shards"
        )
    return shards, frame_count


def _load_hexatic_series(
    case: BigLxCase,
    analysis_dir: Path,
    manifest: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shards, frame_count = _validated_shards(analysis_dir, manifest)
    if frame_count < 2:
        raise ValueError("At least two analysis frames are required")
    magnitudes = np.empty((frame_count, case.n_particles), dtype=np.float32)
    masks = np.empty((frame_count, case.n_particles), dtype=np.bool_)
    steps = np.empty(frame_count, dtype=np.int64)
    required = ("psi_real", "psi_imag", "hexatic_shell_mask", "step")

    for shard in shards:
        start = shard["frame_start"]
        stop = shard["frame_stop"]
        path = analysis_dir / str(shard["file"])
        with safe_open(path, framework="numpy") as tensors:
            keys = set(tensors.keys())
            missing = [name for name in required if name not in keys]
            if missing:
                raise KeyError(f"{path} is missing tensors: {', '.join(missing)}")
            real = np.asarray(tensors.get_tensor("psi_real"))
            imaginary = np.asarray(tensors.get_tensor("psi_imag"))
            mask = np.asarray(tensors.get_tensor("hexatic_shell_mask"))
            shard_steps = np.asarray(tensors.get_tensor("step")).reshape(-1)
        expected_shape = (stop - start, case.n_particles)
        if real.shape != expected_shape or imaginary.shape != expected_shape:
            raise ValueError(f"Hexatic tensor shape mismatch in {path}")
        if mask.shape != expected_shape:
            raise ValueError(f"Hexatic shell-mask shape mismatch in {path}")
        if shard_steps.shape != (stop - start,):
            raise ValueError(f"Step tensor shape mismatch in {path}")
        np.hypot(real, imaginary, out=magnitudes[start:stop])
        masks[start:stop] = mask.astype(np.bool_, copy=False)
        magnitudes[start:stop] *= masks[start:stop]
        steps[start:stop] = shard_steps.astype(np.int64, copy=False)

    if np.any(np.diff(steps) <= 0):
        raise ValueError("Trajectory steps must be strictly increasing")
    return steps, magnitudes, masks


def _autocorrelation_sum(values: np.ndarray, fft_length: int) -> np.ndarray:
    transformed = rfft(values, n=fft_length)
    return irfft(np.conjugate(transformed) * transformed, n=fft_length)[: values.size]


def _velocity_correlation(velocity: np.ndarray, max_lag: int) -> np.ndarray:
    fft_length = next_fast_len(2 * velocity.size - 1)
    sums = _autocorrelation_sum(np.asarray(velocity, dtype=np.float64), fft_length)
    origins = velocity.size - np.arange(max_lag + 1, dtype=np.int64)
    means = sums[: max_lag + 1] / origins
    normalization = float(means[0])
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("Axial COM velocity has zero or invalid lag-zero power")
    result = means / normalization
    result[0] = 1.0
    return result


def _hexatic_correlation(
    magnitudes: np.ndarray,
    masks: np.ndarray,
    max_lag: int,
    particle_block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count, particle_count = magnitudes.shape
    fft_length = next_fast_len(2 * frame_count - 1)
    frequency_count = fft_length // 2 + 1
    numerator_spectrum = np.zeros(frequency_count, dtype=np.float64)
    pair_count_spectrum = np.zeros(frequency_count, dtype=np.float64)

    for start in range(0, particle_count, particle_block_size):
        stop = min(start + particle_block_size, particle_count)
        value_transform = rfft(
            magnitudes[:, start:stop],
            n=fft_length,
            axis=0,
            workers=-1,
        )
        numerator_spectrum += np.sum(
            np.real(np.conjugate(value_transform) * value_transform), axis=1
        )
        del value_transform
        mask_transform = rfft(
            masks[:, start:stop].astype(np.float32),
            n=fft_length,
            axis=0,
            workers=-1,
        )
        pair_count_spectrum += np.sum(
            np.real(np.conjugate(mask_transform) * mask_transform), axis=1
        )
        del mask_transform

    numerator = irfft(numerator_spectrum, n=fft_length)[: max_lag + 1]
    pair_counts_float = irfft(pair_count_spectrum, n=fft_length)[: max_lag + 1]
    pair_counts = np.rint(pair_counts_float).astype(np.int64)
    if np.any(pair_counts <= 0):
        first = int(np.flatnonzero(pair_counts <= 0)[0])
        raise ValueError(f"No shell-valid particle pairs at lag {first}")
    means = numerator / pair_counts
    normalization = float(means[0])
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("Hexatic magnitude has zero or invalid lag-zero power")
    result = means / normalization
    result[0] = 1.0
    return result, pair_counts


def analyze_correlations(
    case: BigLxCase,
    *,
    output_root: Path,
    min_origins: int = 10,
    max_lag: int | None = None,
    particle_block_size: int = 4096,
    absolute: bool = False,
) -> CorrelationSeries:
    if min_origins < 1:
        raise ValueError("min_origins must be positive")
    if max_lag is not None and max_lag < 0:
        raise ValueError("max_lag must be nonnegative")
    if particle_block_size < 1:
        raise ValueError("particle_block_size must be positive")

    analysis_dir = CasePaths(case, output_root).analysis_dir
    manifest = _load_manifest(analysis_dir, case)
    steps, magnitudes, masks = _load_hexatic_series(case, analysis_dir, manifest)
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
    velocity = _velocity_correlation(com.x_velocity, selected_max_lag)
    if absolute:
        velocity = np.abs(velocity)
    psi6, psi6_pair_counts = _hexatic_correlation(
        magnitudes,
        masks,
        selected_max_lag,
        particle_block_size,
    )
    return CorrelationSeries(
        case=case,
        lag_indices=lag_indices,
        lag_times=lag_times,
        velocity=velocity,
        psi6=psi6,
        time_origin_counts=frame_count - lag_indices,
        psi6_pair_counts=psi6_pair_counts,
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
    figure, axis = plt.subplots(figsize=(11, 7))
    colors = plt.colormaps["viridis"](
        np.linspace(0.1, 0.9, len(series_by_case))
    )
    for color, series in zip(colors, series_by_case, strict=True):
        label = series.case.label
        axis.plot(
            series.lag_times,
            series.velocity,
            color=color,
            linestyle="-",
            linewidth=1.7,
            label=rf"$C_v$: {label}",
        )
        axis.plot(
            series.lag_times,
            series.psi6,
            color=color,
            linestyle="--",
            linewidth=1.7,
            label=rf"$C_{{|\psi_6|}}$: {label}",
        )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    axis.axhline(1.0, color="black", linewidth=0.8, alpha=0.3)
    axis.set_xlabel("lag time")
    axis.set_ylabel("normalized correlation")
    velocity_label = r"$|C_v|$" if absolute else r"$C_v$"
    axis.set_title(
        f"Big-Lx axial COM velocity and hexatic-magnitude correlations ({velocity_label})"
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output
