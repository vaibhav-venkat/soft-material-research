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
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import find_peaks
from scipy.signal.windows import hann

from hexatic.constants import cylinder

from .cases import DEFAULT_OUTPUT_ROOT, BigLxCase, CasePaths, all_cases


@dataclass(frozen=True)
class CenterOfMassSeries:
    frames: np.ndarray
    steps: np.ndarray
    elapsed_time: np.ndarray
    x_center: np.ndarray
    x_velocity: np.ndarray
    method: str


@dataclass(frozen=True)
class VelocitySpectrum:
    omega: np.ndarray
    real_coefficient: np.ndarray
    power: np.ndarray
    peak_omega: np.ndarray
    peak_period: np.ndarray
    peak_power: np.ndarray
    denoised_velocity: np.ndarray
    noise_floor: float
    threshold: float


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


def velocity_spectrum(
    series: CenterOfMassSeries,
    *,
    noise_factor: float,
    minimum_cycles: float,
    peak_false_alarm_probability: float,
    zoom_fraction: float,
) -> VelocitySpectrum:
    time_spacing = np.diff(series.elapsed_time)
    dt = float(time_spacing[0])
    if not np.allclose(time_spacing, dt, rtol=1e-10, atol=1e-12):
        raise ValueError("FFT requires uniformly spaced trajectory samples")

    velocity_mean = float(np.mean(series.x_velocity))
    centered_velocity = series.x_velocity - velocity_mean
    coefficients = fft(centered_velocity, norm="forward")
    frequencies = fftfreq(centered_velocity.size, d=dt)
    omega = 2.0 * np.pi * frequencies

    negative_indices = (-np.arange(coefficients.size)) % coefficients.size

    window = hann(centered_velocity.size, sym=False)
    windowed_mean = float(np.sum(centered_velocity * window) / np.sum(window))
    windowed_velocity = (centered_velocity - windowed_mean) * window
    windowed_coefficients = fft(windowed_velocity, norm="forward")
    windowed_coefficients /= float(np.mean(window))
    windowed_negative_coefficients = windowed_coefficients[negative_indices]
    windowed_power = np.real(
        windowed_coefficients * windowed_negative_coefficients
    )
    windowed_power = np.maximum(windowed_power, 0.0)

    observation_time = float(series.elapsed_time[-1] - series.elapsed_time[0])
    minimum_omega = 2.0 * np.pi * minimum_cycles / observation_time
    nyquist_omega = np.pi / dt
    maximum_omega = min(
        nyquist_omega,
        max(zoom_fraction * nyquist_omega, 4.0 * minimum_omega),
    )
    displayed = (omega >= minimum_omega) & (omega <= maximum_omega)
    displayed_omega = omega[displayed]
    displayed_coefficients = windowed_coefficients[displayed]
    displayed_power = windowed_power[displayed]
    if displayed_omega.size < 3:
        raise ValueError(
            "Too few positive-frequency bins remain after applying the "
            "minimum-cycle and near-zero zoom limits"
        )

    coefficient_scale = float(np.max(np.abs(displayed_coefficients)))
    power_scale = float(np.max(displayed_power))
    normalized_real_coefficient = np.real(displayed_coefficients) / max(
        coefficient_scale,
        np.finfo(np.float64).tiny,
    )
    normalized_power = displayed_power / max(
        power_scale,
        np.finfo(np.float64).tiny,
    )

    noise_mean = float(np.median(displayed_power) / np.log(2.0))
    independent_bins = displayed_power.size
    false_alarm_multiplier = -np.log(
        1.0
        - (1.0 - peak_false_alarm_probability) ** (1.0 / independent_bins)
    )
    false_alarm_level = false_alarm_multiplier * noise_mean
    peak_indices, _ = find_peaks(
        displayed_power,
        height=false_alarm_level,
        prominence=noise_mean,
    )
    peak_omega = displayed_omega[peak_indices]
    peak_period = 2.0 * np.pi / peak_omega
    peak_power = normalized_power[peak_indices]

    absolute_omega = np.abs(omega)
    high_frequency = absolute_omega >= 0.75 * float(np.max(absolute_omega))
    noise_floor = float(np.median(np.abs(coefficients[high_frequency])))
    threshold = noise_factor * noise_floor
    keep = np.abs(coefficients) >= threshold
    keep |= keep[negative_indices]
    filtered_coefficients = np.where(keep, coefficients, 0.0)
    denoised_velocity = (
        np.real(ifft(filtered_coefficients, norm="forward")) + velocity_mean
    )

    return VelocitySpectrum(
        omega=displayed_omega,
        real_coefficient=normalized_real_coefficient,
        power=normalized_power,
        peak_omega=peak_omega,
        peak_period=peak_period,
        peak_power=peak_power,
        denoised_velocity=np.asarray(denoised_velocity, dtype=np.float64),
        noise_floor=noise_floor,
        threshold=threshold,
    )


def _plot_velocity_summary(
    series_by_case: dict[BigLxCase, CenterOfMassSeries],
    *,
    output_path: Path,
    noise_factor: float,
    minimum_cycles: float,
    peak_false_alarm_probability: float,
    zoom_fraction: float,
    dpi: int,
) -> None:
    cases = tuple(series_by_case)
    circumferences = tuple(
        sorted({case.circumference_diameters for case in cases})
    )
    multipliers = tuple(sorted({case.lx_multiplier for case in cases}))
    colors = plt.colormaps["viridis"](
        np.linspace(0.1, 0.9, len(multipliers))
    )
    color_by_multiplier = dict(zip(multipliers, colors, strict=True))
    line_style_by_circumference = {
        circumference: line_style
        for circumference, line_style in zip(
            circumferences,
            ("-", "--"),
            strict=True,
        )
    }
    spectrum_by_case = {
        case: velocity_spectrum(
            series,
            noise_factor=noise_factor,
            minimum_cycles=minimum_cycles,
            peak_false_alarm_probability=peak_false_alarm_probability,
            zoom_fraction=zoom_fraction,
        )
        for case, series in series_by_case.items()
    }

    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    maximum_axis = axes[0, 0]
    real_axis = axes[0, 1]
    power_axis = axes[1, 0]
    denoised_axis = axes[1, 1]

    for circumference in circumferences:
        circumference_cases = sorted(
            (
                case
                for case in cases
                if case.circumference_diameters == circumference
            ),
            key=lambda case: case.lx_multiplier,
        )
        maximum_axis.plot(
            [case.lx_multiplier for case in circumference_cases],
            [
                float(np.max(np.abs(series_by_case[case].x_velocity)))
                for case in circumference_cases
            ],
            marker="o",
            linewidth=1.8,
            label=f"C = {circumference:g}D",
        )

    for case in sorted(
        cases,
        key=lambda item: (item.circumference_diameters, item.lx_multiplier),
    ):
        series = series_by_case[case]
        spectrum = spectrum_by_case[case]
        color = color_by_multiplier[case.lx_multiplier]
        line_style = line_style_by_circumference[case.circumference_diameters]
        label = f"C={case.circumference_diameters:g}D, Lx={case.lx_multiplier}x"
        real_axis.plot(
            spectrum.omega,
            spectrum.real_coefficient,
            color=color,
            linestyle=line_style,
            linewidth=1.1,
            label=label,
        )
        power_axis.plot(
            spectrum.omega,
            spectrum.power,
            color=color,
            linestyle=line_style,
            linewidth=1.1,
        )
        power_axis.scatter(
            spectrum.peak_omega,
            spectrum.peak_power,
            color=color,
            marker="o",
            s=24,
            zorder=4,
        )
        for peak_omega, peak_period, peak_power in zip(
            spectrum.peak_omega,
            spectrum.peak_period,
            spectrum.peak_power,
            strict=True,
        ):
            power_axis.annotate(
                f"T={peak_period:.2g}",
                xy=(peak_omega, peak_power),
                xytext=(4, 5),
                textcoords="offset points",
                color=color,
                fontsize=7,
                rotation=35,
            )
        denoised_axis.plot(
            series.elapsed_time,
            series.x_velocity,
            color=color,
            linestyle=line_style,
            linewidth=0.7,
            alpha=0.18,
        )
        denoised_axis.plot(
            series.elapsed_time,
            spectrum.denoised_velocity,
            color=color,
            linestyle=line_style,
            linewidth=1.4,
        )

    maximum_axis.set_xticks(multipliers)
    maximum_axis.set_xlabel(r"$L_x$ multiplier")
    maximum_axis.set_ylabel(r"$\max_t |v_x(t)|$")
    maximum_axis.set_title("Maximum axial COM speed")
    maximum_axis.grid(alpha=0.2)
    maximum_axis.legend()

    real_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
    real_axis.set_xlabel(r"angular frequency $\omega$")
    real_axis.set_ylabel(r"normalized $\mathrm{Re}\,V(\omega)$")
    real_axis.set_title("Hann-windowed real COM-velocity FFT")
    real_axis.grid(alpha=0.2)
    real_axis.legend(fontsize="small", ncol=2)

    power_axis.set_xlabel(r"angular frequency $\omega$")
    power_axis.set_ylabel(r"normalized $V(\omega)V(-\omega)$")
    power_axis.set_title(
        "Positive-frequency Fourier magnitude "
        f"(at least {minimum_cycles:g} cycles; "
        f"FAP <= {100.0 * peak_false_alarm_probability:g}%)"
    )
    power_axis.set_yscale("log")
    power_axis.set_ylim(1e-6, 1.5)
    power_axis.grid(alpha=0.2)

    denoised_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
    denoised_axis.set_xlabel("elapsed simulation time")
    denoised_axis.set_ylabel(r"$v_x$")
    denoised_axis.set_title(
        "FFT/IFFT-denoised COM velocity "
        f"(threshold = {noise_factor:g}x high-frequency noise floor)"
    )
    denoised_axis.grid(alpha=0.2)

    figure.suptitle(
        "Big-Lx COM-velocity maxima, Fourier spectra, and denoising\n"
        "Positive frequencies near zero; each spectrum normalized independently; "
        "color: Lx multiplier; solid: C=60D; dashed: C=60.5D"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)

    for case, spectrum in spectrum_by_case.items():
        if spectrum.peak_period.size:
            periods = ",".join(f"{period:.8g}" for period in spectrum.peak_period)
        else:
            periods = "none"
        print(
            f"[big_lx.fft] case={case.case_id} significant_periods={periods} "
            f"false_alarm_probability={peak_false_alarm_probability:g}",
            flush=True,
        )


def plot_center_of_mass(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    output: Path | None = None,
    spectral_output: Path | None = None,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    dpi: int = 180,
    fft_noise_factor: float = 4.0,
    fft_minimum_cycles: float = 3.0,
    fft_peak_false_alarm_probability: float = 0.01,
    fft_zoom_fraction: float = 0.1,
) -> tuple[Path, Path]:
    cases = all_cases()
    series_by_case = {
        case: center_of_mass_series(
            case,
            CasePaths(case, output_root).analysis_dir,
            start=start,
            stop=stop,
            stride=stride,
        )
        for case in cases
    }
    circumferences = tuple(
        sorted({case.circumference_diameters for case in cases})
    )
    output_path = output or output_root / "plots" / "all_cases_x_com_velocity.png"
    spectral_output_path = spectral_output or (
        output_root / "plots" / "all_cases_x_com_velocity_spectral.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spectral_output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        2,
        len(circumferences),
        figsize=(16, 9),
        sharex="col",
        squeeze=False,
    )
    colors = plt.colormaps["viridis"](
        np.linspace(0.1, 0.9, len({case.lx_multiplier for case in cases}))
    )

    for column, circumference in enumerate(circumferences):
        circumference_cases = sorted(
            (
                case
                for case in cases
                if case.circumference_diameters == circumference
            ),
            key=lambda case: case.lx_multiplier,
        )
        com_axis = axes[0, column]
        velocity_axis = axes[1, column]
        for color, case in zip(colors, circumference_cases, strict=True):
            series = series_by_case[case]
            label = f"Lx = {case.lx_multiplier}x"
            com_axis.plot(
                series.elapsed_time,
                series.x_center,
                color=color,
                linewidth=1.5,
                label=label,
            )
            velocity_axis.plot(
                series.elapsed_time,
                series.x_velocity,
                color=color,
                linewidth=1.5,
                label=label,
            )

        circumference_label = f"C = {circumference:g}D"
        com_axis.set_title(f"{circumference_label}: axial center of mass")
        com_axis.set_ylabel("unwrapped x center of mass")
        com_axis.grid(alpha=0.2)
        com_axis.legend(title="Axial size")

        velocity_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        velocity_axis.set_title(f"{circumference_label}: axial COM velocity")
        velocity_axis.set_xlabel("elapsed simulation time")
        velocity_axis.set_ylabel(r"$d\langle x\rangle/dt$")
        velocity_axis.grid(alpha=0.2)

    figure.suptitle(
        "Big-Lx axial center of mass and velocity "
        f"({next(iter(series_by_case.values())).method})"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    _plot_velocity_summary(
        series_by_case,
        output_path=spectral_output_path,
        noise_factor=fft_noise_factor,
        minimum_cycles=fft_minimum_cycles,
        peak_false_alarm_probability=fft_peak_false_alarm_probability,
        zoom_fraction=fft_zoom_fraction,
        dpi=dpi,
    )
    print(
        f"[big_lx.com] cases={len(series_by_case)} "
        f"method={next(iter(series_by_case.values())).method} "
        f"output={output_path} spectral_output={spectral_output_path}",
        flush=True,
    )
    return output_path, spectral_output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot unwrapped axial center of mass and velocity for all big-Lx "
            "film cases together with a four-panel Fourier summary."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--spectral-output", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--fft-noise-factor", type=float, default=4.0)
    parser.add_argument("--fft-minimum-cycles", type=float, default=3.0)
    parser.add_argument(
        "--fft-peak-false-alarm-probability",
        type=float,
        default=0.01,
    )
    parser.add_argument("--fft-zoom-fraction", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.dpi < 1:
        raise ValueError("dpi must be positive")
    if args.fft_noise_factor <= 0.0:
        raise ValueError("fft-noise-factor must be positive")
    if args.fft_minimum_cycles <= 0.0:
        raise ValueError("fft-minimum-cycles must be positive")
    if not 0.0 < args.fft_peak_false_alarm_probability < 1.0:
        raise ValueError("fft-peak-false-alarm-probability must be between 0 and 1")
    if not 0.0 < args.fft_zoom_fraction <= 1.0:
        raise ValueError("fft-zoom-fraction must be in (0, 1]")
    plot_center_of_mass(
        output_root=args.output_root,
        output=args.output,
        spectral_output=args.spectral_output,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        dpi=args.dpi,
        fft_noise_factor=args.fft_noise_factor,
        fft_minimum_cycles=args.fft_minimum_cycles,
        fft_peak_false_alarm_probability=args.fft_peak_false_alarm_probability,
        fft_zoom_fraction=args.fft_zoom_fraction,
    )


if __name__ == "__main__":
    main()
