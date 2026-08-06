"""Command line interface for band identification statistics and plots."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import logging
from pathlib import Path
from typing import Sequence

from hexatic.constants import cylinder

from .extraction import ExtractionConfig, identify_bands
from .io import load_gsd_metadata, load_metadata
from .plots import plot_area_dynamics, plot_lifetimes
from .statistics import calculate_statistics


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    defaults = ExtractionConfig(timestep=float(cylinder.SIMULATION.timestep))
    parser = argparse.ArgumentParser(
        description="Identify dilute bands and plot area and lifetime statistics."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        required=True,
        help="analysis directory, or GSD trajectory with --gsd; repeat per seed",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gsd", action="store_true", help="interpret each input as a GSD trajectory"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="analysis directory supplying geometry for GSD inputs",
    )
    parser.add_argument(
        "--max-frame",
        type=int,
        help="exclusive maximum trajectory frame index",
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


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.gsd and args.base_dir is None:
        raise ValueError("--base-dir is required with --gsd")
    if not args.gsd and args.base_dir is not None:
        raise ValueError("--base-dir is only used with --gsd")
    if args.max_frame is not None and args.max_frame < 1:
        raise ValueError("--max-frame must be positive")
    base_dir = args.base_dir
    if base_dir is not None and base_dir.is_file():
        base_dir = base_dir.parent
    metadata = (
        [load_gsd_metadata(base_dir, path) for path in args.input_dir]
        if args.gsd and base_dir is not None
        else [load_metadata(path) for path in args.input_dir]
    )
    config_values = {
        name: getattr(args, name)
        for name in asdict(
            ExtractionConfig(timestep=float(cylinder.SIMULATION.timestep))
        )
        if name not in {"start", "stop"}
    }
    config = ExtractionConfig(
        **config_values,
        start=0,
        stop=args.max_frame,
    )
    tracked = []
    for index, item in enumerate(metadata, start=1):
        logger.info("identifying seed %d/%d", index, len(metadata))
        tracked.append(identify_bands(item, config=config))
    stats = calculate_statistics(
        tracked,
        particle_diameter=metadata[0].particle_diameter,
        lag=args.time_increment,
    )
    logger.info(
        "statistics ready: %d area samples, %d increments, "
        "%d deaths, %d splits, %d merges",
        stats.areas.size,
        stats.delta_areas.size,
        stats.lifetimes.size,
        stats.split_lifetimes.size,
        stats.merge_lifetimes.size,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return plot_area_dynamics(stats, output_dir), plot_lifetimes(stats, output_dir)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[band_analysis_id] %(message)s")
    outputs = run(build_parser().parse_args(argv))
    logger.info("complete: %s, %s", *outputs)
