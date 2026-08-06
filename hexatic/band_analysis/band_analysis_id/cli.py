"""Command line interface for branched band identification and plots."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import logging
from pathlib import Path
from typing import Sequence

from hexatic.constants import cylinder

from .extraction import ExtractionConfig, detect_bands, track_bands
from .io import InputMetadata, load_gsd_metadata, load_metadata
from .plots import plot_area_dynamics, plot_lifetimes
from .statistics import TrackingSeries, calculate_statistics
from .tracking import DetectionFrame, TrackedFrame


logger = logging.getLogger(__name__)
PREFIX_DURATION = 100.0


def build_parser() -> argparse.ArgumentParser:
    defaults = ExtractionConfig(timestep=float(cylinder.SIMULATION.timestep))
    parser = argparse.ArgumentParser(
        description="Identify bands across shared GSD prefixes and continuation branches."
    )
    for group in (0, 1):
        parser.add_argument(f"--gsd-{group}", type=Path, required=True)
        parser.add_argument(
            f"--input-dir-{group}",
            type=Path,
            action="append",
            required=True,
            help=f"continuation of --gsd-{group}; repeat for independent seeds",
        )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-frame", type=int, help="exclusive maximum continuation frame index"
    )
    parser.add_argument("--time-increment", type=float, default=1.0)
    extraction = parser.add_argument_group("band identification")
    for name, value in asdict(defaults).items():
        if name in {"start", "stop"}:
            continue
        extraction.add_argument(
            "--" + name.replace("_", "-"), type=type(value), default=value
        )
    return parser


def _config(args: argparse.Namespace, stop: int | None) -> ExtractionConfig:
    defaults = ExtractionConfig(timestep=float(cylinder.SIMULATION.timestep))
    values = {
        name: getattr(args, name)
        for name in asdict(defaults)
        if name not in {"start", "stop"}
    }
    return ExtractionConfig(**values, start=0, stop=stop)


def _prefix_time(detections: list[DetectionFrame]) -> list[DetectionFrame]:
    count = len(detections)
    return [
        replace(
            detection,
            frame_index=index,
            tau_override=PREFIX_DURATION * index / count,
        )
        for index, detection in enumerate(detections)
    ]


def _continuation_time(
    detections: list[DetectionFrame], frame_offset: int
) -> list[DetectionFrame]:
    return [
        replace(
            detection,
            frame_index=frame_offset + index,
            tau_override=PREFIX_DURATION + float(detection.frame_index),
        )
        for index, detection in enumerate(detections)
    ]


def _track(
    detections: list[DetectionFrame], config: ExtractionConfig
) -> list[TrackedFrame]:
    return track_bands(
        detections,
        config.persistence_frames,
        config.overlap_threshold,
    )


def _group_series(
    group: int,
    gsd_path: Path,
    input_dirs: list[Path],
    geometry: InputMetadata,
    args: argparse.Namespace,
) -> list[TrackingSeries]:
    prefix_config = _config(args, None)
    continuation_config = _config(args, args.max_frame)
    prefix_metadata = load_gsd_metadata(gsd_path, geometry)
    logger.info("group %d: identifying shared GSD prefix", group)
    prefix = _prefix_time(detect_bands(prefix_metadata, config=prefix_config))
    series = [TrackingSeries(_track(prefix, prefix_config), initial_bands_are_born=True)]
    for index, input_dir in enumerate(input_dirs, start=1):
        logger.info(
            "group %d: identifying continuation %d/%d: %s",
            group,
            index,
            len(input_dirs),
            input_dir,
        )
        continuation = detect_bands(
            load_metadata(input_dir, geometry), config=continuation_config
        )
        combined = [
            *prefix,
            *_continuation_time(continuation, len(prefix)),
        ]
        series.append(
            TrackingSeries(
                _track(combined, continuation_config),
                include_from_tau=PREFIX_DURATION,
                initial_bands_are_born=True,
            )
        )
    return series


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.max_frame is not None and args.max_frame < 1:
        raise ValueError("--max-frame must be positive")
    geometry = load_gsd_metadata(args.gsd_0)
    series = [
        *_group_series(0, args.gsd_0, args.input_dir_0, geometry, args),
        *_group_series(1, args.gsd_1, args.input_dir_1, geometry, args),
    ]
    stats = calculate_statistics(
        series,
        particle_diameter=geometry.particle_diameter,
        lag=args.time_increment,
    )
    logger.info(
        "statistics ready: %d area samples, %d increments, "
        "%d complete deaths, %d split lifetimes, %d merge lifetimes",
        stats.areas.size,
        stats.delta_areas.size,
        stats.lifetimes.size,
        stats.split_lifetimes.size,
        stats.merge_lifetimes.size,
    )
    logger.info(
        "raw confirmed topology: %d splits, %d merges",
        stats.raw_splits,
        stats.raw_merges,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return plot_area_dynamics(stats, output_dir), plot_lifetimes(stats, output_dir)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[band_analysis_id] %(message)s")
    outputs = run(build_parser().parse_args(argv))
    logger.info("complete: %s, %s", *outputs)
