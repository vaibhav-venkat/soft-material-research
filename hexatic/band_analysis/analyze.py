from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from hexatic.constants import cylinder

from .components import label_dilute_bands
from .characterization import N_MODES, characterize_band
from .density import SurfaceGrid, make_density_kernel, validate_gpu
from .dynamics import add_stochastic_statistics
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
    overwrite: bool = False,
) -> Path:
    metadata = load_metadata(input_dir)
    output_dir = output_dir or input_dir / "band_analysis_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if n_mult <= 0.0:
        raise ValueError("n_mult must be positive")
    if not math.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("timestep must be finite and positive")
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
        "persistence_frames": 5,
        "overlap_threshold": 0.05,
        "maximum_frame_lag": 10,
        "target_bins": 20,
        "minimum_bin_samples": 50,
        "preferred_bin_samples": 100,
    }
    characterization_path = output_dir / "band_characterization.safetensors"
    result_path = output_dir / "analysis.json"
    if characterization_path.exists() and not overwrite:
        tensors, cached_configuration = load_characterization(characterization_path)
        if cached_configuration != configuration:
            raise ValueError(
                "characterization cache parameters or input manifest changed; "
                "rerun with --overwrite"
            )
        characterization_plots = plot_characterization(tensors, output_dir)
        existing: dict[str, object] = {}
        if result_path.exists():
            existing = json.loads(result_path.read_text())
        existing.update(
            {
                "schema": "hexatic.band_analysis.v3",
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
        return result_path

    validate_gpu()
    density_for = make_density_kernel(
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

    for frame_index, step, coords in iter_frames(metadata, set(selected_frames)):
        density_device, shell_count_device, outside_device = density_for(
            jnp.asarray(coords, dtype=jnp.float32)
        )
        if bool(jax.device_get(outside_device)):
            raise AssertionError(
                f"found a particle outside R + eps = {shell_upper_radius:.8g}"
            )
        density = np.asarray(jax.device_get(density_device))
        shell_count = int(jax.device_get(shell_count_device))
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
            f"[band_analysis] frame={frame_index} shell={shell_count} bands={len(components)}",
            flush=True,
        )

    area_fractions = np.concatenate(area_fraction_samples)
    phi_c = rho_c * particle_area
    features = plot_distribution(
        area_fractions,
        threshold=phi_c,
        output=output_dir / "density_distribution.png",
    )
    characterization_tensors = build_characterization_tensors(
        tracked_frames,
        ns=grid.ns,
        circumference=grid.circumference,
        timestep=timestep,
        run_steps=metadata.run_steps,
        trajectory_write_period=metadata.trajectory_write_period,
    )
    add_stochastic_statistics(characterization_tensors)
    compute_provenance = {
        "jax_backend": jax.default_backend(),
        "jax_device": str(jax.devices()[0]),
        "gpu_operations": (
            "density deposition; morphology reductions and Fourier modes; "
            "distribution histogram; net reductions; conditional moments; "
            "axial MSD; area covariance and rates"
        ),
        "cpu_operations": (
            "connected components; Hungarian assignment; event topology; "
            "neighbor topology; distribution peak detection; plotting"
        ),
    }
    save_characterization(
        characterization_path,
        characterization_tensors,
        configuration=configuration,
        compute_provenance=compute_provenance,
    )
    characterization_plots = plot_characterization(
        characterization_tensors, output_dir
    )
    result = {
        "schema": "hexatic.band_analysis.v3",
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
    result_path.write_text(json.dumps(result, indent=2) + "\n")
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
