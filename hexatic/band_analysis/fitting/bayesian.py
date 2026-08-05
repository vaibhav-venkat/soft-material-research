"""Sample coupled area-model posteriors and assess chain convergence."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from pathlib import Path
from typing import Any

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

from .masked_model import (
    MaskedTrainingTransitions,
    parameter_negative_log_likelihood as masked_parameter_negative_log_likelihood,
)
from .model import (
    PARAMETER_NAMES,
    Scaling,
    TrainingTransitions,
    parameter_negative_log_likelihood as clean_parameter_negative_log_likelihood,
)
from ..pipeline.storage import atomic_path


logger = logging.getLogger(__name__)

TrainingData = TrainingTransitions | MaskedTrainingTransitions


def _parameter_names(data: TrainingData) -> tuple[str, ...]:
    return (
        PARAMETER_NAMES
        if isinstance(data, MaskedTrainingTransitions)
        else PARAMETER_NAMES[:-1]
    )


@dataclass(frozen=True)
class MCMCConfig:
    chains: int = 4
    warmup: int = 1_500
    draws: int = 1_500
    target_accept: float = 0.90
    retry_warmup: int = 3_000
    retry_target_accept: float = 0.95


@dataclass(frozen=True)
class MCMCDiagnostics:
    rhat: dict[str, float]
    ess_bulk: dict[str, float]
    ess_tail: dict[str, float]
    divergences: int
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "rhat": self.rhat,
            "ess_bulk": self.ess_bulk,
            "ess_tail": self.ess_tail,
            "divergences": self.divergences,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class BayesianResult:
    idata: az.InferenceData
    normalized_samples: np.ndarray
    diagnostics: MCMCDiagnostics
    retried: bool
    config: MCMCConfig


def coupled_area_model(
    data: TrainingData, empirical: np.ndarray
) -> None:
    """Empirical-centered priors and the slope-detrended AR(1) likelihood."""
    names = _parameter_names(data)
    samples = []
    for index, name in enumerate(names):
        if name == "sigma_E":
            samples.append(numpyro.sample(name, dist.HalfNormal(empirical[index])))
        else:
            samples.append(
                numpyro.sample(
                    name,
                    dist.LogNormal(
                        jnp.log(empirical[index]), 0.5 if index == 4 else 1.0
                    ),
                )
            )
    parameters = jnp.stack(samples)
    if isinstance(data, MaskedTrainingTransitions):
        objective = masked_parameter_negative_log_likelihood(parameters, data)
    else:
        objective = clean_parameter_negative_log_likelihood(parameters, data)
    numpyro.factor(
        "kalman_marginal_likelihood",
        -objective,
    )


def _run_once(
    data: TrainingData,
    empirical: np.ndarray,
    initial: np.ndarray,
    config: MCMCConfig,
    seed: int,
) -> MCMC:
    names = _parameter_names(data)
    initial_values = dict(zip(names, initial, strict=True))
    kernel = NUTS(
        coupled_area_model,
        dense_mass=True,
        target_accept_prob=config.target_accept,
        init_strategy=init_to_value(values=initial_values),
    )
    sampler = MCMC(
        kernel,
        num_warmup=config.warmup,
        num_samples=config.draws,
        num_chains=config.chains,
        chain_method="vectorized",
        progress_bar=True,
    )
    generator = np.random.default_rng(seed)
    offsets = generator.normal(
        0.0, 0.02, (config.chains, len(names))
    )
    offsets -= offsets.mean(axis=0, keepdims=True)
    unconstrained = np.log(initial)[None, :] + offsets
    initial_parameters = {
        name: jnp.asarray(
            unconstrained[:, index]
            if config.chains > 1
            else unconstrained[0, index]
        )
        for index, name in enumerate(names)
    }
    sampler.run(
        jax.random.key(seed),
        data,
        jnp.asarray(empirical),
        init_params=initial_parameters,
        extra_fields=("potential_energy", "energy", "accept_prob", "num_steps"),
    )
    return sampler


def _diagnostics(
    idata: az.InferenceData, names: tuple[str, ...]
) -> MCMCDiagnostics:
    summary = az.summary(idata, var_names=list(names), round_to=None)
    rhat = {
        name: float(summary.loc[name, "r_hat"]) for name in names
    }
    bulk = {
        name: float(summary.loc[name, "ess_bulk"])
        for name in names
    }
    tail = {
        name: float(summary.loc[name, "ess_tail"])
        for name in names
    }
    divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    accepted = (
        all(value < 1.01 for value in rhat.values())
        and all(value >= 400.0 for value in bulk.values())
        and all(value >= 400.0 for value in tail.values())
        and divergences == 0
    )
    return MCMCDiagnostics(rhat, bulk, tail, divergences, accepted)


def _physical_idata(
    sampler: MCMC, scaling: Scaling, names: tuple[str, ...]
) -> az.InferenceData:
    idata = az.from_numpyro(sampler)
    for name, factor in zip(
        names, scaling.parameter_factors[: len(names)], strict=True
    ):
        idata.posterior[name] = idata.posterior[name] * factor
    idata.posterior.attrs["parameter_units"] = (
        "tau_p: physical time; kappa_T: inverse physical time; "
        "diffusions: area^2/time; A_T_star and sigma_E: area"
    )
    return idata


def _normalized_samples(sampler: MCMC, names: tuple[str, ...]) -> np.ndarray:
    samples = sampler.get_samples(group_by_chain=True)
    return np.stack(
        [np.asarray(samples[name]) for name in names], axis=-1
    )


def run_bayesian_inference(
    data: TrainingData,
    empirical: np.ndarray,
    initial: np.ndarray,
    scaling: Scaling,
    *,
    config: MCMCConfig = MCMCConfig(),
    seed: int = 0,
) -> BayesianResult:
    """Run NUTS and perform exactly one prescribed retry after diagnostic failure."""
    logger.info(
        "NUTS sampling global persistent-rate parameters: %d chains, %d warmup, "
        "%d draws, "
        "target acceptance %.2f",
        config.chains,
        config.warmup,
        config.draws,
        config.target_accept,
    )
    names = _parameter_names(data)
    sampler = _run_once(data, empirical, initial, config, seed)
    idata = _physical_idata(sampler, scaling, names)
    diagnostics = _diagnostics(idata, names)
    retried = not diagnostics.accepted
    logger.info(
        "NUTS diagnostics: accepted=%s divergences=%d",
        diagnostics.accepted,
        diagnostics.divergences,
    )
    used_config = config
    if retried:
        used_config = replace(
            config,
            warmup=config.retry_warmup,
            target_accept=config.retry_target_accept,
        )
        logger.info(
            "retrying NUTS: %d warmup, target acceptance %.2f",
            used_config.warmup,
            used_config.target_accept,
        )
        sampler = _run_once(data, empirical, initial, used_config, seed + 1)
        idata = _physical_idata(sampler, scaling, names)
        diagnostics = _diagnostics(idata, names)
        logger.info(
            "NUTS retry diagnostics: accepted=%s divergences=%d",
            diagnostics.accepted,
            diagnostics.divergences,
        )
    return BayesianResult(
        idata=idata,
        normalized_samples=_normalized_samples(sampler, names),
        diagnostics=diagnostics,
        retried=retried,
        config=used_config,
    )


def save_inference_data(
    path: Path, idata: az.InferenceData, *, fingerprint: str | None = None
) -> None:
    """Atomically write physical-unit posterior samples."""
    if fingerprint is not None:
        idata.posterior.attrs["band_analysis_fingerprint"] = fingerprint
    with atomic_path(path) as temporary:
        idata.to_netcdf(temporary, engine="h5netcdf")
