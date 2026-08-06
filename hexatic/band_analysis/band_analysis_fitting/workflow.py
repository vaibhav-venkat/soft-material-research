"""Coordinate clean-segment and event-chain inference with isolated caches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Literal, cast

import arviz as az
import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file

from ..band_analysis_id.chains import EventChain
from ..band_analysis_id.segments import StableSegment
from .bayesian import MCMCConfig, run_bayesian_inference, save_inference_data
from .event_validation import validate_event_posterior
from .inference import OptimizationResult, optimize_parameters
from .masked_model import (
    MaskedTrainingTransitions,
    prepare_training_transitions as prepare_event_transitions,
)
from .model import (
    EVENT_PARAMETER_NAMES,
    PARAMETER_NAMES,
    TrainingTransitions,
    prepare_training_transitions as prepare_clean_transitions,
)
from .validation import ValidationResult, validate_posterior
from ..band_analysis_id.storage import fingerprint, write_arrays, write_json


logger = logging.getLogger(__name__)

FitPath = Literal["clean", "event"]
AnalysisData = list[StableSegment] | list[EventChain]
HoldoutData = dict[str, list[StableSegment]] | dict[str, list[EventChain]]
PreparedData = TrainingTransitions | MaskedTrainingTransitions


@dataclass(frozen=True)
class AnalysisConfig:
    lags: tuple[int, ...] = (1, 2, 3, 5, 10)
    base_lag: int = 2
    optimizer_seed: int = 0
    optimizer_starts: int = 8
    optimizer_max_steps: int = 10_000
    mcmc: MCMCConfig = MCMCConfig()
    predictive_draws: int = 200
    paths_per_segment: int = 200

    def __post_init__(self) -> None:
        if any(lag < 1 for lag in self.lags):
            raise ValueError("fitting lags must be positive")
        if self.base_lag not in self.lags:
            raise ValueError("base lag must occur in configured lags")
        if len(set(self.lags)) != len(self.lags):
            raise ValueError("configured lags must be unique")
        if self.optimizer_starts < 2:
            raise ValueError("optimizer starts must be at least two")
        if self.optimizer_max_steps < 1:
            raise ValueError("optimizer max steps must be positive")


@dataclass(frozen=True)
class LagOutcome:
    fit: FitPath
    lag: int
    accepted: bool
    cached: bool
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    idata: az.InferenceData

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return EVENT_PARAMETER_NAMES if self.fit == "event" else PARAMETER_NAMES


def _update_array(digest: Any, values: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values)
    digest.update(str(contiguous.dtype).encode())
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())


def data_fingerprint(values: AnalysisData, fit: FitPath) -> str:
    digest = hashlib.sha256(fit.encode())
    for value in values:
        digest.update(value.seed_id.encode())
        if isinstance(value, StableSegment):
            digest.update(np.asarray(value.track_ids, dtype=np.int64).tobytes())
            arrays = (value.frame_indices, value.tau, value.areas)
        else:
            arrays = (
                value.slot_ids,
                value.frame_indices,
                value.tau,
                value.areas,
                value.mask,
            )
        for array in arrays:
            _update_array(digest, array)
        if isinstance(value, EventChain):
            for step in value.events:
                for name in (
                    "H",
                    "g",
                    "C",
                    "o",
                    "sigma_e_rows",
                    "mask_next",
                    "y",
                ):
                    _update_array(digest, getattr(step, name))
    return digest.hexdigest()


def _paths(output_dir: Path, fit: FitPath, lag: int) -> tuple[Path, Path, Path]:
    directory = output_dir / fit / f"lag_{lag}"
    return (
        directory / "arrays.safetensors",
        directory / "posterior.nc",
        directory / "metrics.json",
    )


def _posterior_intervals(
    idata: az.InferenceData, names: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    intervals = {}
    for name in names:
        values = np.asarray(idata.posterior[name]).reshape(-1)
        hdi = np.asarray(az.hdi(values, prob=0.95))
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


def _prepare(training: AnalysisData, lag: int, fit: FitPath) -> PreparedData:
    if fit == "event":
        return prepare_event_transitions(cast(list[EventChain], training), lag)
    return prepare_clean_transitions(cast(list[StableSegment], training), lag)


def _validate(
    fit: FitPath,
    values: AnalysisData,
    lag: int,
    prepared: PreparedData,
    samples: np.ndarray,
    config: AnalysisConfig,
    seed: int,
) -> ValidationResult:
    if fit == "event":
        assert isinstance(prepared, MaskedTrainingTransitions)
        sigma_b_squared = float(np.asarray(prepared.sigma_b_squared))
        return validate_event_posterior(
            cast(list[EventChain], values),
            lag,
            prepared.scaling,
            samples,
            sigma_b_squared=sigma_b_squared,
            predictive_draws=config.predictive_draws,
            paths_per_chain=config.paths_per_segment,
            seed=seed,
        )
    return validate_posterior(
        cast(list[StableSegment], values),
        lag,
        prepared.scaling,
        samples,
        predictive_draws=config.predictive_draws,
        paths_per_segment=config.paths_per_segment,
        seed=seed,
    )


def _prefixed(validation: ValidationResult, prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}{key}": value for key, value in validation.arrays.items()}


def _slope_data(prepared: PreparedData) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if isinstance(prepared, TrainingTransitions):
        return {}, {"fitted_in_likelihood": True}
    slopes = [np.asarray(slope) for slope in prepared.conservative_slopes]
    arrays = {
        "conservative_segment_slope": np.concatenate(slopes)
        * prepared.scaling.area
        / prepared.scaling.time
        if slopes
        else np.empty(0),
        "conservative_segment_slope_offset": np.cumsum(
            [0, *(len(slope) for slope in slopes)], dtype=np.int64
        ),
    }
    if isinstance(prepared, MaskedTrainingTransitions):
        masks = [np.asarray(mask, dtype=bool) for mask in prepared.slope_masks]
        arrays["conservative_segment_slope_mask"] = (
            np.concatenate(masks) if masks else np.empty(0, dtype=bool)
        )
        degrees = sum(int(mask.sum()) - 1 for mask in masks)
        informative = sum(int(mask.sum()) > 1 for mask in masks)
    else:
        degrees = sum(len(slope) - 1 for slope in slopes)
        informative = sum(len(slope) > 1 for slope in slopes)
    normalized_variance = float(np.asarray(prepared.sigma_b_squared))
    metadata = {
        "sigma_b_squared": normalized_variance
        * (prepared.scaling.area / prepared.scaling.time) ** 2,
        "normalized_sigma_b_squared": normalized_variance,
        "informative_runs": informative,
        "degrees_of_freedom": degrees,
    }
    return arrays, metadata


def _compute_lag(
    fit: FitPath,
    lag: int,
    training: AnalysisData,
    holdouts: HoldoutData,
    config: AnalysisConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray], az.InferenceData]:
    logger.info("%s lag %d: preparing normalized transitions", fit, lag)
    prepared = _prepare(training, lag, fit)
    transition_count = sum(block.current.shape[0] for block in prepared.blocks)
    optimization = optimize_parameters(
        prepared,
        seed=config.optimizer_seed + lag,
        starts=config.optimizer_starts,
        max_steps=config.optimizer_max_steps,
    )
    bayesian = run_bayesian_inference(
        prepared,
        optimization.empirical_parameters,
        optimization.best.parameters,
        prepared.scaling,
        config=config.mcmc,
        seed=config.optimizer_seed + lag,
    )
    arrays = _optimization_arrays(optimization)
    arrays["normalized_posterior"] = bayesian.normalized_samples
    slope_arrays, slope_metadata = _slope_data(prepared)
    arrays.update(slope_arrays)
    validation_metrics: dict[str, Any] = {}
    if bayesian.diagnostics.accepted:
        result = _validate(
            fit,
            training,
            lag,
            prepared,
            bayesian.normalized_samples,
            config,
            config.optimizer_seed + lag,
        )
        arrays.update(_prefixed(result, "training_"))
        validation_metrics["training"] = result.metrics
        holdout_metrics = {}
        for index, (seed_id, values) in enumerate(holdouts.items()):
            result = _validate(
                fit,
                values,
                lag,
                prepared,
                bayesian.normalized_samples,
                config,
                config.optimizer_seed + lag + index + 1,
            )
            arrays.update(_prefixed(result, f"holdout_{index}_"))
            holdout_metrics[seed_id] = result.metrics
        if holdouts:
            aggregate = cast(
                AnalysisData,
                [value for values in holdouts.values() for value in values],
            )
            result = _validate(
                fit,
                aggregate,
                lag,
                prepared,
                bayesian.normalized_samples,
                config,
                config.optimizer_seed + lag + len(holdouts) + 1,
            )
            arrays.update(_prefixed(result, "holdout_aggregate_"))
            aggregate_metrics: dict[str, Any] | None = result.metrics
        else:
            aggregate_metrics = None
        validation_metrics["holdout"] = {
            "aggregate": aggregate_metrics,
            "per_seed": holdout_metrics,
        }

    physical_dt = np.concatenate([np.asarray(block.dt) for block in prepared.blocks])
    physical_dt *= prepared.scaling.time
    hessian = optimization.best.hessian
    names = EVENT_PARAMETER_NAMES if fit == "event" else PARAMETER_NAMES
    weighted_events = 0.0
    event_rows = 0
    n_max = None
    if isinstance(prepared, MaskedTrainingTransitions):
        weighted_events = float(
            sum(
                np.sum(np.asarray(block.weight) * np.asarray(block.event))
                for block in prepared.blocks
            )
        )
        event_rows = int(
            sum(np.asarray(block.sigma_e_rows, dtype=bool).sum() for block in prepared.blocks)
        )
        n_max = int(prepared.blocks[0].current.shape[1])
    metadata: dict[str, Any] = {
        "fit": fit,
        "accepted": bayesian.diagnostics.accepted,
        "retried": bayesian.retried,
        "mcmc_config": asdict(bayesian.config),
        "diagnostics": bayesian.diagnostics.to_dict(),
        "posterior": _posterior_intervals(bayesian.idata, names),
        "scaling": asdict(prepared.scaling),
        "conservative_slope": slope_metadata,
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
            "training_seeds": len({value.seed_id for value in training}),
            "training_units": len(training),
            "training_transitions": int(transition_count),
            "weighted_training_transitions": float(
                sum(np.asarray(block.weight).sum() for block in prepared.blocks)
            ),
            "weighted_event_transitions": weighted_events,
            "event_mass_rows": event_rows,
            "n_max": n_max,
            "physical_dt_min": float(np.min(physical_dt)),
            "physical_dt_median": float(np.median(physical_dt)),
            "physical_dt_max": float(np.max(physical_dt)),
            "holdout_seeds": len(holdouts),
        },
    }
    return metadata, arrays, bayesian.idata


def _load_cache(
    output_dir: Path, fit: FitPath, lag: int, expected_fingerprint: str
) -> LagOutcome | None:
    arrays_path, posterior_path, metrics_path = _paths(output_dir, fit, lag)
    if not metrics_path.exists():
        return None
    metadata = json.loads(metrics_path.read_text())
    if metadata.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"settings or inputs changed for lag cache {metrics_path}")
    if (
        metadata.get("status") != "complete"
        or not arrays_path.exists()
        or not posterior_path.exists()
    ):
        return None
    with safe_open(arrays_path, framework="numpy") as handle:
        arrays_metadata = dict(handle.metadata())
    if arrays_metadata.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"incompatible lag arrays cache {arrays_path}")
    arrays = load_file(arrays_path)
    names = EVENT_PARAMETER_NAMES if fit == "event" else PARAMETER_NAMES
    posterior = arrays.get("normalized_posterior")
    if posterior is not None and posterior.shape[-1] != len(names):
        logger.info("%s lag %d: cached posterior predates the model, refitting", fit, lag)
        return None
    idata = az.from_netcdf(posterior_path, engine="h5netcdf")
    if idata.posterior.attrs.get("band_analysis_fingerprint") != expected_fingerprint:
        raise ValueError(f"incompatible lag posterior cache {posterior_path}")
    return LagOutcome(
        fit, lag, bool(metadata["accepted"]), True, metadata, arrays, idata
    )


def run_analysis(
    fit: FitPath,
    training: AnalysisData,
    holdouts: HoldoutData,
    output_dir: Path,
    *,
    config: AnalysisConfig = AnalysisConfig(),
    overwrite: bool = False,
) -> list[LagOutcome]:
    """Run one independently usable fit path under its own output namespace."""
    if not training:
        raise ValueError(f"at least one training {fit} unit is required")
    analysis_fingerprint = fingerprint(
        {
            "fit": fit,
            "training": data_fingerprint(training, fit),
            "holdouts": {
                seed_id: data_fingerprint(values, fit)
                for seed_id, values in holdouts.items()
            },
            "settings": asdict(config),
        }
    )
    outcomes = []
    for lag in config.lags:
        lag_fingerprint = fingerprint(
            {"fit": fit, "analysis": analysis_fingerprint, "lag": lag}
        )
        if not overwrite:
            cached = _load_cache(output_dir, fit, lag, lag_fingerprint)
            if cached is not None:
                required = (
                    "training_path_segment_break"
                    if fit == "event"
                    else "training_transfer_rate_discrepancy_null"
                )
                if cached.accepted and required not in cached.arrays:
                    logger.info(
                        "%s lag %d: refreshing cached validation arrays",
                        fit,
                        lag,
                    )
                    prepared = _prepare(training, lag, fit)
                    validation = _validate(
                        fit,
                        training,
                        lag,
                        prepared,
                        cached.arrays["normalized_posterior"],
                        config,
                        config.optimizer_seed + lag,
                    )
                    arrays = {
                        **cached.arrays,
                        **_prefixed(validation, "training_"),
                    }
                    metadata = {
                        **cached.metadata,
                        "validation": {
                            **cached.metadata["validation"],
                            "training": validation.metrics,
                        },
                    }
                    arrays_path, _, metrics_path = _paths(output_dir, fit, lag)
                    write_arrays(
                        arrays_path,
                        arrays,
                        {"fingerprint": lag_fingerprint, "fit": fit},
                    )
                    write_json(metrics_path, metadata)
                    cached = LagOutcome(
                        fit,
                        lag,
                        cached.accepted,
                        True,
                        metadata,
                        arrays,
                        cached.idata,
                    )
                outcomes.append(cached)
                continue
        metadata, arrays, idata = _compute_lag(
            fit, lag, training, holdouts, config
        )
        arrays_path, posterior_path, metrics_path = _paths(output_dir, fit, lag)
        write_json(
            metrics_path,
            {"status": "incomplete", "fit": fit, "lag": lag, "fingerprint": lag_fingerprint},
        )
        write_arrays(arrays_path, arrays, {"fingerprint": lag_fingerprint, "fit": fit})
        save_inference_data(posterior_path, idata, fingerprint=lag_fingerprint)
        completed = {
            **metadata,
            "status": "complete",
            "lag": lag,
            "fingerprint": lag_fingerprint,
        }
        write_json(metrics_path, completed)
        outcomes.append(
            LagOutcome(fit, lag, bool(metadata["accepted"]), False, completed, arrays, idata)
        )
    return outcomes


def stable_lags(outcomes: list[LagOutcome], base_lag: int = 2) -> tuple[int, ...]:
    """Apply the symmetric median-in-HDI stability rule within one fit path."""
    fits = {outcome.fit for outcome in outcomes}
    if len(fits) > 1:
        raise ValueError("lag stability must be evaluated within one fit path")
    by_lag = {outcome.lag: outcome for outcome in outcomes}
    base = by_lag.get(base_lag)
    if base is None or not base.accepted:
        return ()
    stable = []
    for outcome in outcomes:
        if not outcome.accepted:
            continue
        if all(
            base.metadata["posterior"][name]["hdi_low"]
            <= outcome.metadata["posterior"][name]["median"]
            <= base.metadata["posterior"][name]["hdi_high"]
            and outcome.metadata["posterior"][name]["hdi_low"]
            <= base.metadata["posterior"][name]["median"]
            <= outcome.metadata["posterior"][name]["hdi_high"]
            for name in outcome.parameter_names
        ):
            stable.append(outcome.lag)
    return tuple(stable)
