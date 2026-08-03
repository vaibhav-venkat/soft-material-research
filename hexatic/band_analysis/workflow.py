"""Coordinate sequential lag inference with resumable cached artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import arviz as az
import numpy as np
from safetensors.numpy import load_file, save_file

from .bayesian import MCMCConfig, run_bayesian_inference, save_inference_data
from .inference import OptimizationResult, optimize_parameters
from .model import PARAMETER_NAMES, prepare_training_transitions
from .segments import StableSegment
from .storage import fingerprint
from .validation import ValidationResult, validate_posterior


LAG_CACHE_SCHEMA = "hexatic.band_lag.v2"


@dataclass(frozen=True)
class AnalysisConfig:
    lags: tuple[int, ...] = (1, 2, 3, 5, 10)
    base_lag: int = 2
    optimizer_seed: int = 0
    optimizer_starts: int = 8
    mcmc: MCMCConfig = MCMCConfig()
    predictive_draws: int = 200
    paths_per_segment: int = 200

    def __post_init__(self) -> None:
        if self.base_lag not in self.lags:
            raise ValueError("base lag must occur in configured lags")
        if len(set(self.lags)) != len(self.lags):
            raise ValueError("configured lags must be unique")
        if self.optimizer_starts < 2:
            raise ValueError("optimizer starts must be at least two")


@dataclass(frozen=True)
class LagOutcome:
    lag: int
    accepted: bool
    cached: bool
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    idata: az.InferenceData


def segments_fingerprint(segments: list[StableSegment]) -> str:
    digest = hashlib.sha256()
    for segment in segments:
        digest.update(segment.seed_id.encode())
        digest.update(np.asarray(segment.track_ids, dtype=np.int64).tobytes())
        for values in (segment.frame_indices, segment.tau, segment.areas):
            contiguous = np.ascontiguousarray(values)
            digest.update(str(contiguous.dtype).encode())
            digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _paths(output_dir: Path, lag: int) -> tuple[Path, Path, Path]:
    directory = output_dir / f"lag_{lag}"
    return (
        directory / "arrays.safetensors",
        directory / "posterior.nc",
        directory / "metrics.json",
    )


def _save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(
        {key: np.ascontiguousarray(value) for key, value in arrays.items()}, temporary
    )
    temporary.replace(path)


def _posterior_intervals(idata: az.InferenceData) -> dict[str, dict[str, float]]:
    intervals: dict[str, dict[str, float]] = {}
    for name in PARAMETER_NAMES:
        values = np.asarray(idata.posterior[name]).reshape(-1)
        hdi = np.asarray(az.hdi(values, hdi_prob=0.95))
        intervals[name] = {
            "median": float(np.median(values)),
            "hdi_low": float(hdi[0]),
            "hdi_high": float(hdi[1]),
        }
    return intervals


def _optimization_arrays(result: OptimizationResult) -> dict[str, np.ndarray]:
    arrays = {
        "bfgs_raw_parameters": np.stack([run.raw_parameters for run in result.runs]),
        "bfgs_parameters": np.stack([run.parameters for run in result.runs]),
        "bfgs_objective": np.asarray([run.objective for run in result.runs]),
        "bfgs_gradient_norm": np.asarray([run.gradient_norm for run in result.runs]),
        "empirical_parameters": result.empirical_parameters,
    }
    if result.best.hessian is not None:
        arrays.update(
            {
                "hessian": result.best.hessian.matrix,
                "hessian_eigenvalues": result.best.hessian.eigenvalues,
                "hessian_eigenvectors": result.best.hessian.eigenvectors,
            }
        )
    return arrays


def _prefixed(validation: ValidationResult, prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}{key}": value for key, value in validation.arrays.items()}


def _compute_lag(
    lag: int,
    training: list[StableSegment],
    holdouts: dict[str, list[StableSegment]],
    config: AnalysisConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray], az.InferenceData]:
    prepared = prepare_training_transitions(training, lag)
    optimization = optimize_parameters(
        prepared.blocks,
        seed=config.optimizer_seed + lag,
        starts=config.optimizer_starts,
    )
    bayesian = run_bayesian_inference(
        prepared.blocks,
        optimization.empirical_parameters,
        optimization.best.parameters,
        prepared.scaling,
        config=config.mcmc,
        seed=config.optimizer_seed + lag,
    )
    arrays = _optimization_arrays(optimization)
    arrays["normalized_posterior"] = bayesian.normalized_samples
    validation_metrics: dict[str, Any] = {}
    if bayesian.diagnostics.accepted:
        training_validation = validate_posterior(
            training,
            lag,
            prepared.scaling,
            bayesian.normalized_samples,
            predictive_draws=config.predictive_draws,
            paths_per_segment=config.paths_per_segment,
            seed=config.optimizer_seed + lag,
        )
        arrays.update(_prefixed(training_validation, "training_"))
        validation_metrics["training"] = training_validation.metrics
        holdout_metrics: dict[str, Any] = {}
        for index, (seed_id, segments) in enumerate(holdouts.items()):
            result = validate_posterior(
                segments,
                lag,
                prepared.scaling,
                bayesian.normalized_samples,
                predictive_draws=config.predictive_draws,
                paths_per_segment=config.paths_per_segment,
                seed=config.optimizer_seed + lag + index + 1,
            )
            arrays.update(_prefixed(result, f"holdout_{index}_"))
            holdout_metrics[seed_id] = result.metrics
        if holdouts:
            aggregate = validate_posterior(
                [segment for segments in holdouts.values() for segment in segments],
                lag,
                prepared.scaling,
                bayesian.normalized_samples,
                predictive_draws=config.predictive_draws,
                paths_per_segment=config.paths_per_segment,
                seed=config.optimizer_seed + lag + len(holdouts) + 1,
            )
            arrays.update(_prefixed(aggregate, "holdout_aggregate_"))
            validation_metrics["holdout"] = {
                "aggregate": aggregate.metrics,
                "per_seed": holdout_metrics,
            }
        else:
            validation_metrics["holdout"] = {"aggregate": None, "per_seed": {}}
    hessian = optimization.best.hessian
    metadata: dict[str, Any] = {
        "accepted": bayesian.diagnostics.accepted,
        "retried": bayesian.retried,
        "mcmc_config": asdict(bayesian.config),
        "diagnostics": bayesian.diagnostics.to_dict(),
        "posterior": _posterior_intervals(bayesian.idata),
        "scaling": asdict(prepared.scaling),
        "optimization": {
            "best_objective": optimization.best.objective,
            "best_gradient_norm": optimization.best.gradient_norm,
            "result": optimization.best.result,
            "condition_number": hessian.condition_number if hessian else None,
            "nonpositive_hessian": hessian.nonpositive if hessian else None,
            "weak_hessian": hessian.weak if hessian else None,
        },
        "validation": validation_metrics,
        "data": {
            "training_seeds": len({segment.seed_id for segment in training}),
            "training_segments": len(training),
            "training_transitions": int(
                sum(block.current.shape[0] for block in prepared.blocks)
            ),
            "weighted_training_transitions": float(
                sum(np.asarray(block.weight).sum() for block in prepared.blocks)
            ),
            "holdout_seeds": len(holdouts),
        },
    }
    return metadata, arrays, bayesian.idata


def _load_cache(
    output_dir: Path, lag: int, expected_fingerprint: str
) -> LagOutcome | None:
    arrays_path, posterior_path, metrics_path = _paths(output_dir, lag)
    if not metrics_path.exists():
        return None
    metadata = json.loads(metrics_path.read_text())
    if metadata.get("schema") != LAG_CACHE_SCHEMA:
        raise ValueError(f"incompatible lag cache {metrics_path}")
    if metadata.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"settings or inputs changed for lag cache {metrics_path}")
    if (
        metadata.get("status") != "complete"
        or not arrays_path.exists()
        or not posterior_path.exists()
    ):
        return None
    return LagOutcome(
        lag=lag,
        accepted=bool(metadata["accepted"]),
        cached=True,
        metadata=metadata,
        arrays=load_file(arrays_path),
        idata=az.from_netcdf(posterior_path, engine="h5netcdf"),
    )


def run_analysis(
    training: list[StableSegment],
    holdouts: dict[str, list[StableSegment]],
    output_dir: Path,
    *,
    config: AnalysisConfig = AnalysisConfig(),
    overwrite: bool = False,
    compute_lag: Callable[
        [int, list[StableSegment], dict[str, list[StableSegment]], AnalysisConfig],
        tuple[dict[str, Any], dict[str, np.ndarray], az.InferenceData],
    ] = _compute_lag,
) -> list[LagOutcome]:
    """Process and atomically cache lags one at a time in configured order."""
    if not training:
        raise ValueError("at least one training segment is required")
    settings = asdict(config)
    data_fingerprint = fingerprint(
        {
            "training": segments_fingerprint(training),
            "holdouts": {
                seed_id: segments_fingerprint(segments)
                for seed_id, segments in holdouts.items()
            },
            "settings": settings,
        }
    )
    outcomes: list[LagOutcome] = []
    for lag in config.lags:
        lag_fingerprint = fingerprint({"analysis": data_fingerprint, "lag": lag})
        if not overwrite:
            cached = _load_cache(output_dir, lag, lag_fingerprint)
            if cached is not None:
                outcomes.append(cached)
                continue
        metadata, arrays, idata = compute_lag(lag, training, holdouts, config)
        arrays_path, posterior_path, metrics_path = _paths(output_dir, lag)
        _save_arrays(arrays_path, arrays)
        save_inference_data(posterior_path, idata)
        completed = {
            **metadata,
            "schema": LAG_CACHE_SCHEMA,
            "status": "complete",
            "lag": lag,
            "fingerprint": lag_fingerprint,
        }
        _save_json(metrics_path, completed)
        outcomes.append(
            LagOutcome(lag, bool(metadata["accepted"]), False, completed, arrays, idata)
        )
    return outcomes


def stable_lags(outcomes: list[LagOutcome], base_lag: int = 2) -> tuple[int, ...]:
    """Apply the symmetric median-in-HDI stability rule to accepted lags."""
    by_lag = {outcome.lag: outcome for outcome in outcomes}
    base = by_lag.get(base_lag)
    if base is None or not base.accepted:
        return ()
    stable: list[int] = []
    for outcome in outcomes:
        if not outcome.accepted:
            continue
        pair_stable = True
        for name in PARAMETER_NAMES:
            reference = base.metadata["posterior"][name]
            candidate = outcome.metadata["posterior"][name]
            pair_stable &= (
                reference["hdi_low"]
                <= candidate["median"]
                <= reference["hdi_high"]
                and candidate["hdi_low"]
                <= reference["median"]
                <= candidate["hdi_high"]
            )
        if pair_stable:
            stable.append(outcome.lag)
    return tuple(stable)
