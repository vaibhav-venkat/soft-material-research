"""Posterior predictive checks and forward simulation for event chains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from ..detection.chains import EventChain, event_transitions
from ..detection.events import EventStep
from .masked_model import (
    build_transition_blocks,
    normalize_event_chain,
    transition,
    transition_log_density,
)
from .model import Scaling
from .validation import ValidationResult


class _Schedule(NamedTuple):
    tau: jax.Array
    mask: jax.Array
    H: jax.Array
    g: jax.Array
    C: jax.Array
    o: jax.Array
    sigma_e_rows: jax.Array
    event: jax.Array
    created: jax.Array


@dataclass(frozen=True)
class EventSimulation:
    """A simulated chain plus the latent continuous state at every event boundary."""

    chain: EventChain
    continuous_mean: np.ndarray  # (T - 1, N_max), before b, u, and noise
    pre_event_areas: np.ndarray  # (T - 1, N_max)
    persistent_rates: np.ndarray  # (T - 1, N_max), rate used in each transition
    slopes: np.ndarray  # (T - 1, N_max), run slope used in each transition
    event_noise: np.ndarray  # (T - 1, N_max), before H


class _SimulationArrays(NamedTuple):
    areas: jax.Array
    continuous_mean: jax.Array
    pre_event: jax.Array
    rates: jax.Array
    slopes: jax.Array
    event_noise: jax.Array
    observation: jax.Array


def _schedule(chain: EventChain) -> _Schedule:
    previous_ids = chain.slot_ids[:-1]
    following_ids = chain.slot_ids[1:]
    created = (following_ids >= 0) & ~np.any(
        following_ids[:, :, None] == previous_ids[:, None, :], axis=2
    )
    return _Schedule(
        tau=jnp.asarray(chain.tau, dtype=jnp.float64),
        mask=jnp.asarray(chain.mask, dtype=jnp.float64),
        H=jnp.asarray([step.H for step in chain.events], dtype=jnp.float64),
        g=jnp.asarray([step.g for step in chain.events], dtype=jnp.float64),
        C=jnp.asarray([step.C for step in chain.events], dtype=jnp.float64),
        o=jnp.asarray([step.o for step in chain.events], dtype=jnp.float64),
        sigma_e_rows=jnp.asarray(
            [step.sigma_e_rows for step in chain.events], dtype=jnp.float64
        ),
        event=jnp.asarray(event_transitions(chain)),
        created=jnp.asarray(created),
    )


def _projection(mask: jax.Array) -> jax.Array:
    return jnp.diag(mask) - jnp.outer(mask, mask) / jnp.sum(mask)


def _projected_normal(
    key: jax.Array, mask: jax.Array, standard_deviation: jax.Array
) -> jax.Array:
    normal = jax.random.normal(key, mask.shape)
    return standard_deviation * (_projection(mask) @ normal)


@jax.jit
def _simulate(
    initial_area: jax.Array,
    schedule: _Schedule,
    parameters: jax.Array,
    sigma_b_squared: jax.Array,
    key: jax.Array,
) -> _SimulationArrays:
    tau_p, kappa_total, diffusion_u, diffusion_total, area_star, sigma_event = (
        parameters
    )
    initial_keys = jax.random.split(key, 3)
    initial_mask = schedule.mask[0]
    initial_rate = _projected_normal(
        initial_keys[0], initial_mask, jnp.sqrt(diffusion_u / tau_p)
    )
    initial_slope = _projected_normal(
        initial_keys[1], initial_mask, jnp.sqrt(sigma_b_squared)
    )

    inputs = (
        jnp.diff(schedule.tau),
        schedule.mask[:-1],
        schedule.mask[1:],
        schedule.H,
        schedule.g,
        schedule.C,
        schedule.o,
        schedule.sigma_e_rows,
        schedule.event,
        schedule.created,
        jax.random.split(initial_keys[2], len(schedule.H)),
    )

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        values: tuple[jax.Array, ...],
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, ...]]:
        area, rate, slope = carry
        dt, mask, mask_next, H, g, C, o, event_rows, is_event, created, step_key = (
            values
        )
        total = jnp.dot(mask, area)
        allocation = mask / jnp.sum(mask)
        total_increment = -kappa_total * (total - area_star) * dt
        continuous_mean = mask * (area + allocation * total_increment)

        total_key, event_key, rate_key, slope_key = jax.random.split(step_key, 4)
        total_noise = jnp.sqrt(2.0 * diffusion_total * dt) * jax.random.normal(
            total_key
        )
        pre_event = mask * (
            continuous_mean + (slope + rate) * dt + allocation * total_noise
        )
        event_noise = event_rows * sigma_event * jax.random.normal(
            event_key, mask.shape
        )
        following = mask_next * (H @ (pre_event + event_noise) + g)
        observation = o * (C @ (pre_event + event_noise) + g)

        decay = jnp.exp(-dt / tau_p)
        innovation_scale = jnp.sqrt(diffusion_u / tau_p * (1.0 - decay**2))
        evolved_rate = decay * rate + _projected_normal(
            rate_key, mask, innovation_scale
        )
        transformed_rate = _projection(mask_next) @ (
            following - (H @ area + g)
        ) / dt
        stationary_created = (
            jnp.sqrt(diffusion_u / tau_p)
            * jax.random.normal(rate_key, mask.shape)
        )
        event_rate = _projection(mask_next) @ jnp.where(
            created, stationary_created, transformed_rate
        )
        following_rate = jnp.where(is_event, event_rate, evolved_rate)
        new_slope = _projected_normal(
            slope_key, mask_next, jnp.sqrt(sigma_b_squared)
        )
        following_slope = jnp.where(is_event, new_slope, slope)
        following_rate *= mask_next
        following_slope *= mask_next
        outputs = (
            following,
            continuous_mean,
            pre_event,
            rate,
            slope,
            event_noise,
            observation,
        )
        return (following, following_rate, following_slope), outputs

    _, outputs = jax.lax.scan(
        step, (initial_area * initial_mask, initial_rate, initial_slope), inputs
    )
    following, mean, pre_event, rates, slopes, event_noise, observation = outputs
    return _SimulationArrays(
        areas=jnp.concatenate((initial_area[None] * initial_mask, following)),
        continuous_mean=mean,
        pre_event=pre_event,
        rates=rates,
        slopes=slopes,
        event_noise=event_noise,
        observation=observation,
    )


def simulate_event_chain(
    schedule: EventChain,
    parameters: np.ndarray,
    *,
    sigma_b_squared: float = 0.0,
    seed: int = 0,
) -> EventSimulation:
    """Simulate one masked raw-area path over an observed topology schedule.

    ``parameters`` and the schedule must use the same area and time units. Event
    mass noise is drawn on ``sigma_e_rows`` before applying ``H``; a split therefore
    distributes the scalar noise with its exogenous daughter fractions.
    """
    values = np.asarray(parameters, dtype=np.float64)
    if (
        values.shape != (6,)
        or not np.all(np.isfinite(values))
        or values[0] <= 0.0
        or values[4] <= 0.0
        or np.any(values[[1, 2, 3, 5]] < 0.0)
    ):
        raise ValueError("parameters must contain valid nonnegative SDE values")
    if sigma_b_squared < 0.0 or not np.isfinite(sigma_b_squared):
        raise ValueError("sigma_b_squared must be finite and nonnegative")
    simulated = _simulate(
        jnp.asarray(schedule.areas[0], dtype=jnp.float64),
        _schedule(schedule),
        jnp.asarray(values),
        jnp.asarray(sigma_b_squared),
        jax.random.key(seed),
    )
    areas = np.asarray(simulated.areas)
    observations = np.asarray(simulated.observation)
    events = tuple(
        EventStep(
            H=step.H,
            g=step.g,
            C=step.C,
            o=step.o,
            sigma_e_rows=step.sigma_e_rows,
            mask_next=step.mask_next,
            y=observations[index],
        )
        for index, step in enumerate(schedule.events)
    )
    chain = EventChain(
        seed_id=schedule.seed_id,
        slot_ids=schedule.slot_ids,
        frame_indices=schedule.frame_indices,
        tau=schedule.tau,
        areas=areas,
        mask=schedule.mask,
        events=events,
    )
    return EventSimulation(
        chain=chain,
        continuous_mean=np.asarray(simulated.continuous_mean),
        pre_event_areas=np.asarray(simulated.pre_event),
        persistent_rates=np.asarray(simulated.rates),
        slopes=np.asarray(simulated.slopes),
        event_noise=np.asarray(simulated.event_noise),
    )


def _posterior_matrix(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 3:
        values = values.reshape(-1, values.shape[-1])
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("event posterior samples must have shape (..., 6)")
    return values


def _draw_indices(sample_count: int, draws: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.choice(sample_count, size=draws, replace=sample_count < draws)


def _predict_draw(
    parameters: jax.Array,
    key: jax.Array,
    current: jax.Array,
    mask: jax.Array,
    mask_next: jax.Array,
    H: jax.Array,
    g: jax.Array,
    C: jax.Array,
    o: jax.Array,
    event_rows: jax.Array,
    dt: jax.Array,
    sigma_b_squared: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    means, covariances = jax.vmap(
        transition, in_axes=(0, 0, 0, None, None)
    )(current, mask, dt, parameters, sigma_b_squared)
    continuous_key, event_key = jax.random.split(key)
    continuous_noise = jax.random.normal(continuous_key, current.shape)
    pre_event = means + jnp.einsum(
        "tij,tj->ti", jnp.linalg.cholesky(covariances), continuous_noise
    )
    pre_event *= mask
    event_noise = parameters[5] * event_rows * jax.random.normal(
        event_key, current.shape
    )
    following = mask_next * (
        jnp.einsum("tij,tj->ti", H, pre_event + event_noise) + g
    )
    observation = o * (
        jnp.einsum("tij,tj->ti", C, pre_event + event_noise) + g
    )
    return following, observation


_predict_draws = jax.jit(jax.vmap(_predict_draw, in_axes=(0, 0) + (None,) * 10))
_density_rows = jax.vmap(
    transition_log_density,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None),
)
_density_draws = jax.jit(jax.vmap(_density_rows, in_axes=(None,) * 8 + (0, None)))


def _one_step(
    chains: list[EventChain],
    lag: int,
    posterior: np.ndarray,
    sigma_b_squared: float,
    draws: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    selected = posterior[_draw_indices(len(posterior), draws, seed)]
    keys = jax.random.split(jax.random.key(seed + 1), draws)
    observed_parts = []
    mean_parts = []
    low_parts = []
    high_parts = []
    mask_parts = []
    event_parts = []
    observation_parts = []
    observation_mask_parts = []
    log_scores = []
    informative_parts = []
    covered = 0
    values = 0
    for block in build_transition_blocks(chains, lag):
        predictions, predicted_observations = _predict_draws(
            jnp.asarray(selected),
            keys,
            block.current,
            block.mask,
            block.mask_next,
            block.H,
            block.g,
            block.C,
            block.o,
            block.sigma_e_rows,
            block.dt,
            jnp.asarray(sigma_b_squared),
        )
        prediction = np.asarray(predictions)
        following = np.asarray(block.following)
        active = np.asarray(block.mask_next, dtype=bool)
        low, high = np.quantile(prediction, [0.025, 0.975], axis=0)
        covered += int(
            np.count_nonzero(active & (following >= low) & (following <= high))
        )
        values += int(active.sum())
        draw_logp = np.asarray(
            _density_draws(
                block.y,
                block.current,
                block.mask,
                block.dt,
                block.C,
                block.g,
                block.o,
                block.sigma_e_rows,
                jnp.asarray(selected),
                jnp.asarray(sigma_b_squared),
            )
        )
        informative = np.asarray(block.o).sum(axis=1)
        informative_parts.append(informative)
        log_scores.append(logsumexp(draw_logp, axis=0) - np.log(draws))
        observed_parts.append(following.ravel())
        mean_parts.append(prediction.mean(axis=0).ravel())
        low_parts.append(low.ravel())
        high_parts.append(high.ravel())
        mask_parts.append(active.ravel())
        event_parts.append(
            np.repeat(np.asarray(block.event), int(block.current.shape[1]))
        )
        observation_parts.append(
            np.asarray(predicted_observations).mean(axis=0).ravel()
        )
        observation_mask_parts.append(np.asarray(block.o, dtype=bool).ravel())
    observed = np.concatenate(observed_parts)
    mean = np.concatenate(mean_parts)
    active = np.concatenate(mask_parts)
    arrays = {
        "one_step_observed": observed,
        "one_step_mean": mean,
        "one_step_low": np.concatenate(low_parts),
        "one_step_high": np.concatenate(high_parts),
        "one_step_mask": active,
        "one_step_event": np.concatenate(event_parts),
        "one_step_observation_mean": np.concatenate(observation_parts),
        "one_step_observation_mask": np.concatenate(observation_mask_parts),
    }
    metrics = {
        "predictive_log_score": float(np.mean(np.concatenate(log_scores))),
        "coverage_95": float(covered / values),
        "one_step_rmse": float(
            np.sqrt(np.mean((observed[active] - mean[active]) ** 2))
        ),
        "inactive_prediction_max_abs": float(
            np.max(np.abs(mean[~active]), initial=0.0)
        ),
        "mean_informative_rows": float(np.mean(np.concatenate(informative_parts))),
    }
    return arrays, metrics


def _paths(
    chains: list[EventChain],
    posterior: np.ndarray,
    sigma_b_squared: float,
    paths_per_chain: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    generator = np.random.default_rng(seed)
    path_areas = []
    path_masks = []
    path_tau = []
    offsets = [0]
    negative = 0
    generated = 0
    for chain_index, chain in enumerate(chains):
        indices = generator.choice(
            len(posterior), paths_per_chain, replace=len(posterior) < paths_per_chain
        )
        keys = jax.random.split(
            jax.random.key(seed + chain_index + 1), paths_per_chain
        )
        simulations = jax.vmap(_simulate, in_axes=(None, None, 0, None, 0))(
            jnp.asarray(chain.areas[0]),
            _schedule(chain),
            jnp.asarray(posterior[indices]),
            jnp.asarray(sigma_b_squared),
            keys,
        )
        areas = np.asarray(simulations.areas)
        mask = np.broadcast_to(chain.mask, areas.shape)
        path_areas.append(areas.ravel())
        path_masks.append(mask.ravel())
        path_tau.append(np.tile(chain.tau, paths_per_chain))
        row_size = chain.n_steps * chain.n_max
        offsets.extend(offsets[-1] + row_size * np.arange(1, paths_per_chain + 1))
        negative += int(np.count_nonzero((areas < 0.0) & mask))
        generated += int(mask.sum())
    arrays = {
        "path_area": np.concatenate(path_areas),
        "path_mask": np.concatenate(path_masks),
        "path_tau": np.concatenate(path_tau),
        "path_offset": np.asarray(offsets, dtype=np.int64),
    }
    metrics: dict[str, float | int] = {
        "negative_area_count": negative,
        "generated_area_count": generated,
        "negative_area_fraction": float(negative / generated) if generated else 0.0,
    }
    return arrays, metrics


def validate_event_posterior(
    chains: list[EventChain],
    lag: int,
    scaling: Scaling,
    normalized_samples: np.ndarray,
    *,
    sigma_b_squared: float = 0.0,
    predictive_draws: int = 200,
    paths_per_chain: int = 200,
    seed: int = 0,
) -> ValidationResult:
    """Validate a frozen six-parameter posterior on observed event schedules."""
    if sigma_b_squared < 0.0 or not np.isfinite(sigma_b_squared):
        raise ValueError("sigma_b_squared must be finite and nonnegative")
    posterior = _posterior_matrix(normalized_samples)
    normalized = [normalize_event_chain(chain, scaling) for chain in chains]
    one_step_arrays, one_step_metrics = _one_step(
        normalized, lag, posterior, sigma_b_squared, predictive_draws, seed
    )
    path_arrays, path_metrics = _paths(
        normalized, posterior, sigma_b_squared, paths_per_chain, seed + 2
    )
    area_keys = {
        "one_step_observed",
        "one_step_mean",
        "one_step_low",
        "one_step_high",
        "one_step_observation_mean",
        "path_area",
    }
    arrays = {
        name: values * scaling.area if name in area_keys else values
        for name, values in {**one_step_arrays, **path_arrays}.items()
    }
    arrays["path_tau"] *= scaling.time
    metrics = {**one_step_metrics, **path_metrics}
    metrics["one_step_rmse"] *= scaling.area
    metrics["inactive_prediction_max_abs"] *= scaling.area
    metrics["predictive_log_score"] -= metrics.pop(
        "mean_informative_rows"
    ) * np.log(scaling.area)
    return ValidationResult(arrays=arrays, metrics=metrics)
