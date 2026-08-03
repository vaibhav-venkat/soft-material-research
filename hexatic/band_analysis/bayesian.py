"""Sample coupled area-model posteriors and assess chain convergence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

from .model import PARAMETER_NAMES, Scaling, TransitionBlock, transition


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
    blocks: tuple[TransitionBlock, ...], empirical: np.ndarray
) -> None:
    """Empirical-centered priors and the shared weighted transition likelihood."""
    parameters = jnp.stack(
        [
            numpyro.sample(
                name,
                dist.LogNormal(jnp.log(empirical[index]), 0.5 if index == 4 else 1.0),
            )
            for index, name in enumerate(PARAMETER_NAMES)
        ]
    )
    for index, block in enumerate(blocks):
        means, covariances = jax.vmap(transition, in_axes=(0, 0, None))(
            block.current, block.dt, parameters
        )
        log_density = dist.MultivariateNormal(means, covariance_matrix=covariances).log_prob(
            block.following
        )
        numpyro.factor(f"transitions_{index}", jnp.sum(block.weight * log_density))


def _run_once(
    blocks: tuple[TransitionBlock, ...],
    empirical: np.ndarray,
    initial: np.ndarray,
    config: MCMCConfig,
    seed: int,
) -> MCMC:
    initial_values = dict(zip(PARAMETER_NAMES, initial, strict=True))
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
        progress_bar=False,
    )
    generator = np.random.default_rng(seed)
    offsets = generator.normal(0.0, 0.02, (config.chains, len(PARAMETER_NAMES)))
    offsets -= offsets.mean(axis=0, keepdims=True)
    unconstrained = np.log(initial)[None, :] + offsets
    initial_parameters = {
        name: jnp.asarray(
            unconstrained[:, index]
            if config.chains > 1
            else unconstrained[0, index]
        )
        for index, name in enumerate(PARAMETER_NAMES)
    }
    sampler.run(
        jax.random.key(seed),
        blocks,
        jnp.asarray(empirical),
        init_params=initial_parameters,
        extra_fields=("potential_energy", "energy", "accept_prob", "num_steps"),
    )
    return sampler


def _diagnostics(idata: az.InferenceData) -> MCMCDiagnostics:
    summary = az.summary(idata, var_names=list(PARAMETER_NAMES), round_to=None)
    rhat = {name: float(summary.loc[name, "r_hat"]) for name in PARAMETER_NAMES}
    bulk = {name: float(summary.loc[name, "ess_bulk"]) for name in PARAMETER_NAMES}
    tail = {name: float(summary.loc[name, "ess_tail"]) for name in PARAMETER_NAMES}
    divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    accepted = (
        all(value < 1.01 for value in rhat.values())
        and all(value >= 400.0 for value in bulk.values())
        and all(value >= 400.0 for value in tail.values())
        and divergences == 0
    )
    return MCMCDiagnostics(rhat, bulk, tail, divergences, accepted)


def _physical_idata(sampler: MCMC, scaling: Scaling) -> az.InferenceData:
    idata = az.from_numpyro(sampler)
    factors = np.asarray(
        [
            1.0 / scaling.time,
            1.0 / scaling.time,
            scaling.area**2 / scaling.time,
            scaling.area**2 / scaling.time,
            scaling.area,
        ]
    )
    for name, factor in zip(PARAMETER_NAMES, factors, strict=True):
        idata.posterior[name] = idata.posterior[name] * factor
    idata.posterior.attrs["parameter_units"] = (
        "rates: inverse physical time; diffusions: area^2/time; A_T_star: area"
    )
    return idata


def _normalized_samples(sampler: MCMC) -> np.ndarray:
    samples = sampler.get_samples(group_by_chain=True)
    return np.stack([np.asarray(samples[name]) for name in PARAMETER_NAMES], axis=-1)


def run_bayesian_inference(
    blocks: tuple[TransitionBlock, ...],
    empirical: np.ndarray,
    initial: np.ndarray,
    scaling: Scaling,
    *,
    config: MCMCConfig = MCMCConfig(),
    seed: int = 0,
) -> BayesianResult:
    """Run NUTS and perform exactly one prescribed retry after diagnostic failure."""
    sampler = _run_once(blocks, empirical, initial, config, seed)
    idata = _physical_idata(sampler, scaling)
    diagnostics = _diagnostics(idata)
    retried = not diagnostics.accepted
    used_config = config
    if retried:
        used_config = replace(
            config,
            warmup=config.retry_warmup,
            target_accept=config.retry_target_accept,
        )
        sampler = _run_once(blocks, empirical, initial, used_config, seed + 1)
        idata = _physical_idata(sampler, scaling)
        diagnostics = _diagnostics(idata)
    return BayesianResult(
        idata=idata,
        normalized_samples=_normalized_samples(sampler),
        diagnostics=diagnostics,
        retried=retried,
        config=used_config,
    )


def save_inference_data(
    path: Path, idata: az.InferenceData, *, fingerprint: str | None = None
) -> None:
    """Atomically write physical-unit posterior samples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fingerprint is not None:
        idata.posterior.attrs["band_analysis_fingerprint"] = fingerprint
    temporary = path.with_suffix(path.suffix + ".tmp")
    idata.to_netcdf(temporary, engine="h5netcdf")
    temporary.replace(path)
