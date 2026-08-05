"""Parse command-line options and run the cached analysis workflow."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import logging
from pathlib import Path
from typing import Sequence

from hexatic.constants import cylinder

from ..detection.chains import EventChain, globally_pad_chains
from ..fitting.bayesian import MCMCConfig
from ..detection.extraction import ExtractionConfig, SeedExtraction, extract_seed_data
from .io import InputMetadata, load_metadata, require_compatible_seeds
from .reporting import FitOutcomes, write_configuration, write_metrics, write_report
from .workflow import AnalysisConfig, run_analysis


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    analysis = AnalysisConfig()
    mcmc = MCMCConfig()
    extraction_defaults = ExtractionConfig(
        timestep=float(cylinder.SIMULATION.timestep)
    )
    parser = argparse.ArgumentParser(
        description="Fit clean-segment and event-chain band-area SDEs side by side."
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
    fit = parser.add_mutually_exclusive_group()
    fit.add_argument(
        "--event",
        action="store_true",
        help="run only the masked event-chain fit",
    )
    fit.add_argument(
        "--clean",
        action="store_true",
        help="run only the clean fixed-identity-segment fit",
    )
    parser.add_argument(
        "--lag",
        dest="lags",
        type=int,
        choices=analysis.lags,
        action="append",
        help="fitting lag; repeat to override the default lags "
        + ", ".join(map(str, analysis.lags)),
    )
    parser.add_argument("--base-lag", type=int, default=analysis.base_lag)
    parser.add_argument(
        "--optimizer-starts", type=int, default=analysis.optimizer_starts
    )
    parser.add_argument(
        "--optimizer-max-steps", type=int, default=analysis.optimizer_max_steps
    )
    parser.add_argument("--chains", type=int, default=mcmc.chains)
    parser.add_argument("--warmup", type=int, default=mcmc.warmup)
    parser.add_argument("--draws", type=int, default=mcmc.draws)
    parser.add_argument("--target-accept", type=float, default=mcmc.target_accept)
    parser.add_argument("--retry-warmup", type=int, default=mcmc.retry_warmup)
    parser.add_argument("--retry-draws", type=int, default=mcmc.retry_draws)
    parser.add_argument(
        "--retry-target-accept", type=float, default=mcmc.retry_target_accept
    )
    parser.add_argument(
        "--predictive-draws", type=int, default=analysis.predictive_draws
    )
    parser.add_argument(
        "--paths-per-segment", type=int, default=analysis.paths_per_segment
    )
    parser.add_argument("--seed", type=int, default=analysis.optimizer_seed)

    extraction = parser.add_argument_group("stable-band extraction")
    for name, value in asdict(extraction_defaults).items():
        flag = "--" + name.replace("_", "-")
        if name == "stop":
            extraction.add_argument(flag, type=int)
        else:
            extraction.add_argument(flag, type=type(value), default=value)
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
) -> dict[str, SeedExtraction]:
    extracted = {}
    for index, item in enumerate(metadata):
        seed_id = _seed_id(item, role, index)
        segment_cache = (
            output_dir / "extraction" / "clean" / f"{seed_id}.safetensors"
        )
        event_cache = (
            output_dir / "extraction" / "event" / f"{seed_id}.safetensors"
        )
        logger.info(
            "extracting %s seed %d/%d: %s",
            role,
            index + 1,
            len(metadata),
            item.input_dir,
        )
        extracted[seed_id] = extract_seed_data(
            item,
            segment_cache,
            event_cache,
            seed_id=seed_id,
            config=config,
            overwrite=overwrite,
        )
        logger.info(
            "%s seed %d/%d ready: %d clean segments, %d event chains",
            role,
            index + 1,
            len(metadata),
            len(extracted[seed_id].segments),
            len(extracted[seed_id].event_chains),
        )
    return extracted


def _global_event_data(
    training: dict[str, SeedExtraction],
    holdouts: dict[str, SeedExtraction],
) -> tuple[list[EventChain], dict[str, list[EventChain]]]:
    chains = [
        chain
        for extracted in (*training.values(), *holdouts.values())
        for chain in extracted.event_chains
    ]
    padded = globally_pad_chains(chains)
    by_seed: dict[str, list[EventChain]] = {
        seed_id: [] for seed_id in (*training, *holdouts)
    }
    for chain in padded:
        by_seed[chain.seed_id].append(chain)
    training_chains = [chain for seed_id in training for chain in by_seed[seed_id]]
    holdout_chains = {
        seed_id: by_seed[seed_id] for seed_id in holdouts if by_seed[seed_id]
    }
    return training_chains, holdout_chains


def _configs(args: argparse.Namespace) -> tuple[ExtractionConfig, AnalysisConfig]:
    extraction = ExtractionConfig(
        **{field.name: getattr(args, field.name) for field in fields(ExtractionConfig)}
    )
    mcmc = MCMCConfig(
        chains=args.chains,
        warmup=args.warmup,
        draws=args.draws,
        target_accept=args.target_accept,
        retry_warmup=args.retry_warmup,
        retry_draws=args.retry_draws,
        retry_target_accept=args.retry_target_accept,
    )
    analysis = AnalysisConfig(
        lags=tuple(args.lags) if args.lags else AnalysisConfig.lags,
        base_lag=args.base_lag,
        optimizer_seed=args.seed,
        optimizer_starts=args.optimizer_starts,
        optimizer_max_steps=args.optimizer_max_steps,
        mcmc=mcmc,
        predictive_draws=args.predictive_draws,
        paths_per_segment=args.paths_per_segment,
    )
    return extraction, analysis


def run(args: argparse.Namespace) -> Path:
    fits = ("event",) if args.event else ("clean",) if args.clean else ("clean", "event")
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
        "training_inputs": [str(path) for path in training_paths],
        "holdout_inputs": [str(path) for path in holdout_paths],
        "output_dir": str(output_dir),
        "case_compatibility_fingerprint": compatibility,
        "fits": list(fits),
        "extraction": asdict(extraction_config),
        "analysis": asdict(analysis_config),
    }
    training_extractions = _extract_all(
        training_metadata,
        "training",
        output_dir,
        extraction_config,
        args.overwrite,
    )
    holdout_extractions = _extract_all(
        holdout_metadata,
        "holdout",
        output_dir,
        extraction_config,
        args.overwrite,
    )
    outcomes: FitOutcomes = {}
    if "clean" in fits:
        clean_training = [
            segment
            for extracted in training_extractions.values()
            for segment in extracted.segments
        ]
        clean_holdouts = {
            seed_id: extracted.segments
            for seed_id, extracted in holdout_extractions.items()
            if extracted.segments
        }
        outcomes["clean"] = run_analysis(
            "clean",
            clean_training,
            clean_holdouts,
            output_dir,
            config=analysis_config,
            overwrite=args.overwrite,
        )
    if "event" in fits:
        event_training, event_holdouts = _global_event_data(
            training_extractions, holdout_extractions
        )
        outcomes["event"] = run_analysis(
            "event",
            event_training,
            event_holdouts,
            output_dir,
            config=analysis_config,
            overwrite=args.overwrite,
        )
    logger.info("writing configuration, plots, metrics, and report")
    # Imported here so matplotlib is only loaded when a run reaches rendering.
    from .plots import plot_lag, plot_summary

    write_configuration(output_dir, configuration)
    for fit_outcomes in outcomes.values():
        for outcome in fit_outcomes:
            logger.info("rendering %s lag %d plots", outcome.fit, outcome.lag)
            plot_lag(outcome, output_dir)
        plot_summary(fit_outcomes, output_dir)
    if "event" in fits:
        from .event_plots import plot_total_area_by_band_count

        plot_total_area_by_band_count(event_training, event_holdouts, output_dir)
    write_metrics(output_dir, outcomes, analysis_config)
    return write_report(output_dir, outcomes, analysis_config)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[band_analysis] %(message)s")
    args = build_parser().parse_args(argv)
    report = run(args)
    logger.info("complete: report=%s", report)
