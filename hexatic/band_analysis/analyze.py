from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from hexatic.constants import cylinder

from .components import label_dilute_bands
from .characterization import N_MODES, characterize_band
from .density import SurfaceGrid, make_density_batch_kernel, validate_gpu
from .dynamics import MAX_FRAME_LAG, add_stochastic_statistics
from .distribution import plot_distribution
from .interfaces import extract_interfaces
from .io import frame_numbers, iter_frames, load_metadata
from .plots import plot_characterization
from .storage import (
    build_characterization_tensors,
    load_characterization,
    save_characterization,
)
from .tracking import BandTracker, TrackedFrame


def analyze(
    input_dir: Path,
    *,
    output_dir: Path | None = None,
    n_mult: float = 1.0,
    rho_c: float = 0.5,
    shell_epsilon_d: float = 0.05,
    smoothing_sigma: float = 1.0,
    minimum_area: int = 4,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    timestep: float = float(cylinder.SIMULATION.timestep),
    frame_batch_size: int = 16,
    persistence_frames: int = 5,
    maximum_frame_lag: int = MAX_FRAME_LAG,
    overwrite: bool = False,
) -> Path:
    analysis_started = time.perf_counter()

    def progress(message: str) -> None:
        elapsed = time.perf_counter() - analysis_started
        print(f"[band_analysis] {message} elapsed={elapsed:.1f}s", flush=True)

    metadata = load_metadata(input_dir)
    output_dir = output_dir or input_dir / "band_analysis_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if n_mult <= 0.0:
        raise ValueError("n_mult must be positive")
    if not math.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("timestep must be finite and positive")
    if frame_batch_size < 1:
        raise ValueError("frame_batch_size must be positive")
    if persistence_frames < 1:
        raise ValueError("persistence_frames must be positive")
    if maximum_frame_lag < 2:
        raise ValueError("maximum_frame_lag must be at least 2")
    grid = SurfaceGrid(
        lx=metadata.lx,
        circumference=metadata.circumference,
        nx=max(1, round(n_mult * metadata.lx / metadata.particle_diameter)),
        ns=max(1, round(n_mult * metadata.circumference / metadata.particle_diameter)),
    )
    if grid.ns < 2 * N_MODES + 1:
        raise ValueError(
            f"at least {2 * N_MODES + 1} circumferential bins are required "
            f"for modes 1..{N_MODES}; got {grid.ns}"
        )
    selected_frames = frame_numbers(metadata, start, stop, stride)
    if not selected_frames:
        raise ValueError("the selected frame range is empty")
    configuration = {
        "input_manifest_sha256": hashlib.sha256(
            json.dumps(metadata.manifest, sort_keys=True).encode()
        ).hexdigest(),
        "grid": asdict(grid),
        "n_mult": n_mult,
        "rho_c": rho_c,
        "shell_epsilon_d": shell_epsilon_d,
        "smoothing_sigma": smoothing_sigma,
        "minimum_area": minimum_area,
        "start": start,
        "stop": stop,
        "stride": stride,
        "timestep": timestep,
        "frame_batch_size": frame_batch_size,
        "persistence_frames": persistence_frames,
        "overlap_threshold": 0.05,
        "maximum_frame_lag": maximum_frame_lag,
        "target_bins": 20,
        "minimum_bin_samples": 50,
        "preferred_bin_samples": 100,
    }
    characterization_path = output_dir / "band_characterization.safetensors"
    result_path = output_dir / "analysis.json"
    if characterization_path.exists() and not overwrite:
        progress(f"stage=cache load path={characterization_path}")
        tensors, cached_configuration = load_characterization(characterization_path)
        if cached_configuration != configuration:
            raise ValueError(
                "characterization cache parameters or input manifest changed; "
                "rerun with --overwrite"
            )
        progress("stage=plots start source=cache")
        characterization_plots = plot_characterization(
            tensors, output_dir, progress=progress
        )
        existing: dict[str, object] = {}
        if result_path.exists():
            existing = json.loads(result_path.read_text())
        existing.update(
            {
                "schema": "hexatic.band_analysis.v13",
                "input_dir": str(input_dir),
                "configuration": configuration,
                "outputs": {
                    "density_distribution": "density_distribution.png",
                    "band_characterization": characterization_path.name,
                    "stochastic_plots": characterization_plots,
                },
            }
        )
        result_path.write_text(json.dumps(existing, indent=2) + "\n")
        progress(f"stage=complete wrote={result_path}")
        return result_path

    progress(
        f"stage=frames start selected={len(selected_frames)} "
        f"grid={grid.nx}x{grid.ns}"
    )
    validate_gpu()
    density_batch_for = make_density_batch_kernel(
        grid,
        radius=metadata.radius,
        particle_diameter=metadata.particle_diameter,
        shell_epsilon=shell_epsilon_d * metadata.particle_diameter,
        smoothing_sigma=smoothing_sigma,
    )
    area_fraction_samples: list[np.ndarray] = []
    frame_records: list[dict[str, object]] = []
    tracked_frames: list[TrackedFrame] = []
    tracker = BandTracker(grid.lx)
    particle_area = math.pi * metadata.particle_diameter**2 / 4.0
    shell_upper_radius = metadata.radius + shell_epsilon_d * metadata.particle_diameter

    frame_iterator = iter_frames(metadata, set(selected_frames))
    while batch := list(islice(frame_iterator, frame_batch_size)):
        coordinates = np.stack([item[2] for item in batch])
        density_device, shell_count_device, outside_device = density_batch_for(
            jnp.asarray(coordinates, dtype=jnp.float32)
        )
        densities, shell_counts, outside = jax.device_get(
            (density_device, shell_count_device, outside_device)
        )
        for (frame_index, step, _), density, shell_count_value, is_outside in zip(
            batch, densities, shell_counts, outside, strict=True
        ):
            if bool(is_outside):
                raise AssertionError(
                    f"found a particle outside R + eps = {shell_upper_radius:.8g}"
                )
            density = np.asarray(density)
            shell_count = int(shell_count_value)
            area_fraction_samples.append((density * particle_area).ravel())
            labels, components = label_dilute_bands(
                density < rho_c,
                minimum_area=minimum_area,
            )
            characterized = []
            for component in components:
                interfaces = extract_interfaces(labels, component.label, grid)
                characterized.append(
                    characterize_band(component, interfaces, grid)
                )
            masks = [labels == component.label for component in components]
            tracked, events = tracker.update(characterized, masks)
            tracked_frames.append(
                TrackedFrame(frame_index, step, tuple(tracked), events)
            )
            event_counts = Counter(event.code.name.lower() for event in events)
            frame_records.append(
                {
                    "frame": frame_index,
                    "step": step,
                    "shell_particles": shell_count,
                    "band_count": len(tracked),
                    "track_ids": [band.track_id for band in tracked],
                    "segment_ids": [band.segment_id for band in tracked],
                    "events": dict(event_counts),
                }
            )
            print(
                f"[band_analysis] frame={frame_index} shell={shell_count} "
                f"bands={len(components)}",
                flush=True,
            )

    progress(f"stage=frames complete count={len(tracked_frames)}")
    progress("stage=distribution concatenate")
    area_fractions = np.concatenate(area_fraction_samples)
    phi_c = rho_c * particle_area
    progress("stage=distribution plot_start")
    features = plot_distribution(
        area_fractions,
        threshold=phi_c,
        output=output_dir / "density_distribution.png",
    )
    progress("stage=distribution complete")
    progress("stage=storage build_padded_tensors_start")
    characterization_tensors = build_characterization_tensors(
        tracked_frames,
        ns=grid.ns,
        circumference=grid.circumference,
        timestep=timestep,
        run_steps=metadata.run_steps,
        trajectory_write_period=metadata.trajectory_write_period,
        persistence_frames=persistence_frames,
    )
    progress("stage=storage build_padded_tensors_complete")
    add_stochastic_statistics(
        characterization_tensors,
        max_frame_lag=maximum_frame_lag,
        progress=progress,
    )
    compute_provenance = {
        "jax_backend": jax.default_backend(),
        "jax_device": str(jax.devices()[0]),
        "gpu_operations": (
            "batched density deposition; "
            "distribution histogram; net and area-heterogeneity reductions; "
            "conditional moments; "
            "fixed-count event-free area-CV drift; "
            "raw and normalized neighbor-relative area drift; "
            "neighbor-relative area-increment drift; "
            "area transition matrices and Chapman-Kolmogorov discrepancy; "
            "axial MSD; area covariance and rates"
        ),
        "cpu_operations": (
            "connected components; morphology reductions and Fourier modes; "
            "Hungarian assignment; event topology; "
            "neighbor topology; distribution peak detection; plotting"
        ),
    }
    progress(f"stage=storage save_start path={characterization_path}")
    save_characterization(
        characterization_path,
        characterization_tensors,
        configuration=configuration,
        compute_provenance=compute_provenance,
    )
    progress("stage=storage save_complete")
    progress("stage=plots start source=fresh_analysis")
    characterization_plots = plot_characterization(
        characterization_tensors, output_dir, progress=progress
    )
    result = {
        "schema": "hexatic.band_analysis.v13",
        "input_dir": str(input_dir),
        "configuration": configuration,
        "compute_provenance": compute_provenance,
        "grid": {**asdict(grid), "dx": grid.dx, "ds": grid.ds},
        "n_mult": n_mult,
        "rho_c": rho_c,
        "phi_c": phi_c,
        "shell_epsilon_d": shell_epsilon_d,
        "smoothing_sigma_bins": smoothing_sigma,
        "minimum_area_cells": minimum_area,
        "distribution_peaks_phi": features.peaks.tolist(),
        "distribution_valleys_phi": features.valleys.tolist(),
        "frames": frame_records,
        "outputs": {
            "density_distribution": "density_distribution.png",
            "band_characterization": characterization_path.name,
            "stochastic_plots": characterization_plots,
        },
    }
    progress(f"stage=json write_start path={result_path}")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    progress(f"stage=complete wrote={result_path}")
    return result_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify dilute bands from a big-lx safetensor analysis directory."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--n-mult",
        type=float,
        default=1.0,
        help=(
            "Grid-resolution multiplier: Nx=round(n_mult*Lx/D) and "
            "Ns=round(n_mult*C/D); default 1.0."
        ),
    )
    parser.add_argument("--rho-c", type=float, default=0.5)
    parser.add_argument("--shell-epsilon-d", type=float, default=0.05)
    parser.add_argument("--smoothing-sigma", type=float, default=1.0)
    parser.add_argument("--minimum-area", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--timestep",
        type=float,
        default=float(cylinder.SIMULATION.timestep),
        help="Simulation time per integration step (default: cylinder constants).",
    )
    parser.add_argument(
        "--frame-batch-size",
        type=int,
        default=16,
        help="Number of trajectory frames deposited per GPU call (default: 16).",
    )
    parser.add_argument(
        "--persistence-frames",
        type=int,
        default=5,
        help="Consecutive analyzed frames required for a stable segment (default: 5).",
    )
    parser.add_argument(
        "--maximum-frame-lag",
        type=int,
        default=MAX_FRAME_LAG,
        help=(
            "Largest analyzed frame lag for conditional drift and diffusion "
            f"(default: {MAX_FRAME_LAG})."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute characterization instead of plotting the compatible cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = analyze(**vars(args))
    print(f"[band_analysis] wrote {result}", flush=True)


if __name__ == "__main__":
    main()
