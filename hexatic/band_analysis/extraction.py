"""Extract stable band segments from one simulation directory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import islice
import logging
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .characterization import characterize_band
from .components import label_dilute_bands
from .density import SurfaceGrid, make_density_batch_kernel, validate_gpu
from .io import InputMetadata, frame_numbers, iter_frames
from .segments import StableSegment, build_stable_segments
from .storage import load_seed_segments, save_seed_segments
from .tracking import BandTracker, DetectionFrame


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionConfig:
    grid_multiplier: float = 1.0
    density_threshold: float = 0.5
    shell_epsilon_d: float = 0.05
    smoothing_sigma: float = 1.0
    minimum_area_cells: int = 4
    start: int = 0
    stop: int | None = None
    stride: int = 1
    timestep: float = 1.0e-6
    frame_batch_size: int = 16
    persistence_frames: int = 5
    overlap_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.grid_multiplier <= 0.0 or self.smoothing_sigma <= 0.0:
            raise ValueError("grid multiplier and smoothing sigma must be positive")
        if not math.isfinite(self.timestep) or self.timestep <= 0.0:
            raise ValueError("timestep must be finite and positive")
        if self.minimum_area_cells < 1 or self.frame_batch_size < 1:
            raise ValueError("minimum area and frame batch size must be positive")
        if self.persistence_frames < 1 or self.stride < 1:
            raise ValueError("persistence and stride must be positive")


def extraction_settings(
    metadata: InputMetadata, config: ExtractionConfig, grid: SurfaceGrid | None = None
) -> dict[str, object]:
    return {
        "extraction": asdict(config),
        "grid": asdict(grid if grid is not None else _grid(metadata, config)),
        "physical_case": metadata.compatibility_fingerprint,
    }


def _grid(metadata: InputMetadata, config: ExtractionConfig) -> SurfaceGrid:
    return SurfaceGrid(
        lx=metadata.lx,
        circumference=metadata.circumference,
        nx=max(
            1,
            round(config.grid_multiplier * metadata.lx / metadata.particle_diameter),
        ),
        ns=max(
            1,
            round(
                config.grid_multiplier
                * metadata.circumference
                / metadata.particle_diameter
            ),
        ),
    )


def extract_seed_segments(
    metadata: InputMetadata,
    cache_path: Path,
    *,
    seed_id: str,
    config: ExtractionConfig,
    overwrite: bool = False,
) -> list[StableSegment]:
    """Extract or reuse one seed without modifying its input directory."""
    grid = _grid(metadata, config)
    settings = extraction_settings(metadata, config, grid)
    if cache_path.exists() and not overwrite:
        logger.info("reusing segment cache: %s", cache_path)
        segments, _ = load_seed_segments(
            cache_path,
            source_fingerprint=metadata.source_fingerprint,
            settings=settings,
        )
        return segments

    selected = frame_numbers(metadata, config.start, config.stop, config.stride)
    if not selected:
        raise ValueError(f"selected frame range is empty for {metadata.input_dir}")
    validate_gpu()
    logger.info(
        "depositing density and detecting bands in %d frames on the active GPU",
        len(selected),
    )
    density_for = make_density_batch_kernel(
        grid,
        radius=metadata.radius,
        particle_diameter=metadata.particle_diameter,
        shell_epsilon=config.shell_epsilon_d * metadata.particle_diameter,
        smoothing_sigma=config.smoothing_sigma,
    )
    detections: list[DetectionFrame] = []
    frames = iter_frames(metadata, set(selected))
    processed = 0
    progress_interval = max(config.frame_batch_size, math.ceil(len(selected) / 10))
    next_progress = progress_interval
    while batch := list(islice(frames, config.frame_batch_size)):
        coordinates = np.stack([item[2] for item in batch])
        densities_device, _, outside_device = density_for(
            jnp.asarray(coordinates, dtype=jnp.float32)
        )
        densities, outside = jax.device_get((densities_device, outside_device))
        for (frame_index, step, _), density, is_outside in zip(
            batch, densities, outside, strict=True
        ):
            if bool(is_outside):
                upper = (
                    metadata.radius
                    + config.shell_epsilon_d * metadata.particle_diameter
                )
                raise ValueError(f"particle outside extraction shell r={upper:.8g}")
            labels, components = label_dilute_bands(
                np.asarray(density) < config.density_threshold,
                minimum_area=config.minimum_area_cells,
            )
            bands = tuple(
                characterize_band(component, labels, grid)
                for component in components
            )
            detections.append(
                DetectionFrame(
                    frame_index,
                    step,
                    config.timestep,
                    bands,
                )
            )
        processed += len(batch)
        if processed >= next_progress or processed == len(selected):
            logger.info("extraction progress: %d/%d frames", processed, len(selected))
            while next_progress <= processed:
                next_progress += progress_interval
    logger.info("tracking detected bands and debouncing topology events")
    tracker = BandTracker(
        overlap_threshold=config.overlap_threshold,
        persistence_frames=config.persistence_frames,
    )
    segments = build_stable_segments(
        tracker.track(detections),
        seed_id=seed_id,
        persistence_frames=config.persistence_frames,
    )
    save_seed_segments(
        cache_path,
        segments,
        seed_id=seed_id,
        source_fingerprint=metadata.source_fingerprint,
        settings=settings,
    )
    logger.info("saved segment cache: %s", cache_path)
    return segments
