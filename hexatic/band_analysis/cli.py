"""Parse command-line options and run the cached analysis workflow."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import logging
from pathlib import Path
from typing import Sequence

from hexatic.constants import cylinder

from .bayesian import MCMCConfig
from .extraction import (
    ExtractionConfig,
    extract_seed_segments,
    validate_extraction_config,
)
from .io import InputMetadata, load_metadata, require_compatible_seeds
from .plots import plot_lag, plot_summary
from .reporting import write_configuration, write_metrics, write_report
from .segments import StableSegment
from .workflow import AnalysisConfig, run_analysis


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit the coupled stable-band area SDE across simulation seeds."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        required=True,
        help="training big-lx analysis directory; repeat for multiple seeds",
    )
    parser.add_argument(
        "--holdout-dir",
        type=Path,
        action="append",
        default=[],
        help="held-out big-lx analysis directory; repeat for multiple seeds",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lag", dest="lags", type=int, action="append")
    parser.add_argument("--base-lag", type=int, default=2)
    parser.add_argument("--optimizer-starts", type=int, default=8)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1_500)
    parser.add_argument("--draws", type=int, default=1_500)
    parser.add_argument("--target-accept", type=float, default=0.90)
    parser.add_argument("--retry-warmup", type=int, default=3_000)
    parser.add_argument("--retry-target-accept", type=float, default=0.95)
    parser.add_argument("--predictive-draws", type=int, default=200)
    parser.add_argument("--paths-per-segment", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    extraction = parser.add_argument_group("stable-band extraction")
    extraction.add_argument("--grid-multiplier", type=float, default=1.0)
    extraction.add_argument("--density-threshold", type=float, default=0.5)
    extraction.add_argument("--shell-epsilon-d", type=float, default=0.05)
    extraction.add_argument("--smoothing-sigma", type=float, default=1.0)
    extraction.add_argument("--minimum-area-cells", type=int, default=4)
    extraction.add_argument("--start", type=int, default=0)
    extraction.add_argument("--stop", type=int)
    extraction.add_argument("--stride", type=int, default=1)
    extraction.add_argument(
        "--timestep", type=float, default=float(cylinder.SIMULATION.timestep)
    )
    extraction.add_argument("--frame-batch-size", type=int, default=16)
    extraction.add_argument("--persistence-frames", type=int, default=5)
    extraction.add_argument("--overlap-threshold", type=float, default=0.05)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rebuild this command's segment and lag artifacts",
    )
    return parser


def _seed_id(metadata: InputMetadata, role: str, index: int) -> str:
    case = metadata.manifest["case"]
    return (
        f"{role}_{index}_{case.get('case_id', metadata.input_dir.name)}"
        f"_seed_{case.get('seed', 'unknown')}"
    )


def _extract_all(
    metadata: list[InputMetadata],
    role: str,
    output_dir: Path,
    config: ExtractionConfig,
    overwrite: bool,
) -> dict[str, list[StableSegment]]:
    extracted = {}
    for index, item in enumerate(metadata):
        seed_id = _seed_id(item, role, index)
        cache = output_dir / "segments" / f"{seed_id}.safetensors"
        logger.info(
            "extracting %s seed %d/%d: %s",
            role,
            index + 1,
            len(metadata),
            item.input_dir,
        )
        extracted[seed_id] = extract_seed_segments(
            item,
            cache,
            seed_id=seed_id,
            config=config,
            overwrite=overwrite,
        )
        logger.info(
            "%s seed %d/%d ready: %d stable segments",
            role,
            index + 1,
            len(metadata),
            len(extracted[seed_id]),
        )
    return extracted


def _configs(args: argparse.Namespace) -> tuple[ExtractionConfig, AnalysisConfig]:
    extraction = ExtractionConfig(
        grid_multiplier=args.grid_multiplier,
        density_threshold=args.density_threshold,
        shell_epsilon_d=args.shell_epsilon_d,
        smoothing_sigma=args.smoothing_sigma,
        minimum_area_cells=args.minimum_area_cells,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        timestep=args.timestep,
        frame_batch_size=args.frame_batch_size,
        persistence_frames=args.persistence_frames,
        overlap_threshold=args.overlap_threshold,
    )
    validate_extraction_config(extraction)
    mcmc = MCMCConfig(
        chains=args.chains,
        warmup=args.warmup,
        draws=args.draws,
        target_accept=args.target_accept,
        retry_warmup=args.retry_warmup,
        retry_target_accept=args.retry_target_accept,
    )
    analysis = AnalysisConfig(
        lags=tuple(args.lags or (1, 2)),
        base_lag=args.base_lag,
        optimizer_seed=args.seed,
        optimizer_starts=args.optimizer_starts,
        mcmc=mcmc,
        predictive_draws=args.predictive_draws,
        paths_per_segment=args.paths_per_segment,
    )
    return extraction, analysis


def run(args: argparse.Namespace) -> Path:
    training_paths = [path.resolve() for path in args.input_dir]
    holdout_paths = [path.resolve() for path in args.holdout_dir]
    overlap = set(training_paths) & set(holdout_paths)
    if overlap:
        raise ValueError(
            "training and holdout directories must be disjoint: "
            + ", ".join(map(str, sorted(overlap)))
        )
    extraction_config, analysis_config = _configs(args)
    logger.info(
        "loading %d training and %d holdout inputs",
        len(training_paths),
        len(holdout_paths),
    )
    training_metadata = [load_metadata(path) for path in training_paths]
    holdout_metadata = [load_metadata(path) for path in holdout_paths]
    compatibility = require_compatible_seeds(
        [*training_metadata, *holdout_metadata]
    )
    logger.info("input manifests are compatible")
    output_dir = args.output_dir.resolve()
    configuration = {
        "schema": "hexatic.band_analysis.configuration.v1",
        "training_inputs": [str(path) for path in training_paths],
        "holdout_inputs": [str(path) for path in holdout_paths],
        "output_dir": str(output_dir),
        "case_compatibility_fingerprint": compatibility,
        "extraction": asdict(extraction_config),
        "analysis": asdict(analysis_config),
    }
    training_by_seed = _extract_all(
        training_metadata,
        "training",
        output_dir,
        extraction_config,
        args.overwrite,
    )
    holdouts = _extract_all(
        holdout_metadata,
        "holdout",
        output_dir,
        extraction_config,
        args.overwrite,
    )
    training = [
        segment for segments in training_by_seed.values() for segment in segments
    ]
    outcomes = run_analysis(
        training,
        holdouts,
        output_dir,
        config=analysis_config,
        overwrite=args.overwrite,
    )
    logger.info("writing configuration, plots, metrics, and report")
    write_configuration(output_dir, configuration)
    for outcome in outcomes:
        logger.info("rendering lag %d plots", outcome.lag)
        plot_lag(outcome, output_dir)
    logger.info("rendering cross-lag and holdout plots")
    plot_summary(outcomes, output_dir)
    write_metrics(output_dir, outcomes, analysis_config)
    return write_report(output_dir, outcomes, analysis_config)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[band_analysis] %(message)s")
    args = build_parser().parse_args(argv)
    report = run(args)
    logger.info("complete: report=%s", report)
