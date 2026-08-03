"""Define scaling, transitions, and likelihoods for the coupled area SDE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from dynamax.linear_gaussian_ssm import (
    ParamsLGSSM,
    ParamsLGSSMDynamics,
    ParamsLGSSMEmissions,
    ParamsLGSSMInitial,
    lgssm_filter,
)

from .segments import StableSegment


jax.config.update("jax_enable_x64", True)

PARAMETER_NAMES = ("tau_p", "kappa_T", "D_u", "D_T", "A_T_star")
FITTED_PARAMETER_NAMES = PARAMETER_NAMES
JITTER = 1e-10


class TransitionBlock(NamedTuple):
    current: jax.Array
    following: jax.Array
    dt: jax.Array
    weight: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainingTransitions:
    scaling: Scaling
    blocks: tuple[TransitionBlock, ...]
    sequences: tuple[StateSpaceSequence, ...]


class StateSpaceSequence(NamedTuple):
    conservative: jax.Array
    tau: jax.Array
    weight: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Scaling:
    area: float
    time: float

    @classmethod
    def from_training(cls, segments: list[StableSegment]) -> Scaling:
        totals = np.concatenate([segment.areas.sum(axis=1) for segment in segments])
        intervals = np.concatenate([np.diff(segment.tau) for segment in segments])
        return cls(area=float(np.median(totals)), time=float(np.median(intervals)))

    def normalize_segment(self, segment: StableSegment) -> StableSegment:
        return StableSegment(
            seed_id=segment.seed_id,
            track_ids=segment.track_ids,
            frame_indices=segment.frame_indices.copy(),
            tau=segment.tau / self.time,
            areas=segment.areas / self.area,
        )

    def parameters_to_physical(self, normalized: np.ndarray) -> np.ndarray:
        tau_p, kappa_T, diffusion_u, diffusion_total, area_star = normalized
        return np.asarray(
            [
                tau_p * self.time,
                kappa_T / self.time,
                diffusion_u * self.area**2 / self.time,
                diffusion_total * self.area**2 / self.time,
                area_star * self.area,
            ],
            dtype=np.float64,
        )

    def parameters_to_normalized(self, physical: np.ndarray) -> np.ndarray:
        tau_p, kappa_T, diffusion_u, diffusion_total, area_star = physical
        return np.asarray(
            [
                tau_p / self.time,
                kappa_T * self.time,
                diffusion_u * self.time / self.area**2,
                diffusion_total * self.time / self.area**2,
                area_star / self.area,
            ],
            dtype=np.float64,
        )


def prepare_training_transitions(
    segments: list[StableSegment], lag: int
) -> TrainingTransitions:
    """Freeze training scaling before constructing normalized transitions."""
    scaling = Scaling.from_training(segments)
    normalized = [scaling.normalize_segment(segment) for segment in segments]
    return TrainingTransitions(
        scaling=scaling,
        blocks=build_transition_blocks(normalized, lag),
        sequences=build_state_space_sequences(normalized, lag),
    )


def _helmert_basis(n_bands: int) -> np.ndarray:
    basis = np.zeros((n_bands, n_bands - 1), dtype=np.float64)
    for column in range(n_bands - 1):
        count = column + 1
        basis[:count, column] = 1.0 / np.sqrt(count * (count + 1))
        basis[count, column] = -count / np.sqrt(count * (count + 1))
    return basis


def build_state_space_sequences(
    segments: list[StableSegment], lag: int
) -> tuple[StateSpaceSequence, ...]:
    sequences = []
    for segment in segments:
        if segment.n_bands == 1:
            continue
        basis = _helmert_basis(segment.n_bands)
        for phase in range(lag):
            areas = segment.areas[phase::lag]
            tau = segment.tau[phase::lag]
            if len(tau) < 2:
                continue
            sequences.append(
                StateSpaceSequence(
                    conservative=jnp.asarray(areas @ basis, dtype=jnp.float64),
                    tau=jnp.asarray(tau, dtype=jnp.float64),
                    weight=jnp.asarray(1.0 / lag, dtype=jnp.float64),
                )
            )
    return tuple(sequences)


def build_transition_blocks(
    segments: list[StableSegment], lag: int
) -> tuple[TransitionBlock, ...]:
    """Use every lagged transition wholly contained in a clean segment."""
    grouped: dict[int, list[tuple[np.ndarray, np.ndarray, float]]] = {}
    for segment in segments:
        for start in range(len(segment.tau) - lag):
            grouped.setdefault(segment.n_bands, []).append(
                (
                    segment.areas[start],
                    segment.areas[start + lag],
                    float(segment.tau[start + lag] - segment.tau[start]),
                )
            )
    return tuple(
        TransitionBlock(
            current=jnp.asarray([item[0] for item in transitions], dtype=jnp.float64),
            following=jnp.asarray(
                [item[1] for item in transitions], dtype=jnp.float64
            ),
            dt=jnp.asarray([item[2] for item in transitions], dtype=jnp.float64),
            weight=jnp.full(len(transitions), 1.0 / lag, dtype=jnp.float64),
        )
        for _, transitions in sorted(grouped.items())
    )


def positive_parameters(raw_parameters: jax.Array) -> jax.Array:
    return jax.nn.softplus(raw_parameters) + 1e-12


def raw_parameters(parameters: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(parameters, dtype=np.float64) - 1e-12, 1e-12)
    return values + np.log(-np.expm1(-values))


def transition(
    area: jax.Array, dt: jax.Array, parameters: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Unconditional one-step moments used by posterior predictive checks."""
    tau_p, kappa_total, diffusion_u, diffusion_total, area_star = parameters
    n_bands = area.shape[-1]
    projection = jnp.eye(n_bands, dtype=jnp.float64) - jnp.ones(
        (n_bands, n_bands), dtype=jnp.float64
    ) / n_bands
    total = jnp.sum(area)
    allocation = jnp.ones(n_bands, dtype=jnp.float64) / n_bands
    drift = -kappa_total * (total - area_star) * allocation
    mean = area + drift * dt
    persistent_variance = diffusion_u * dt**2 / tau_p
    covariance = persistent_variance * projection + 2.0 * dt * (
        diffusion_total * jnp.outer(allocation, allocation)
    )
    return mean, covariance + JITTER * jnp.eye(n_bands, dtype=jnp.float64)


def transition_log_density(
    following: jax.Array,
    area: jax.Array,
    dt: jax.Array,
    parameters: jax.Array,
) -> jax.Array:
    mean, covariance = transition(area, dt, parameters)
    cholesky = jnp.linalg.cholesky(covariance)
    whitened = jax.scipy.linalg.solve_triangular(
        cholesky, following - mean, lower=True
    )
    dimension = area.shape[-1]
    return -0.5 * (
        dimension * jnp.log(2.0 * jnp.pi)
        + 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
        + jnp.dot(whitened, whitened)
    )


def parameter_negative_log_likelihood(
    parameters: jax.Array, data: TrainingTransitions
) -> jax.Array:
    tau_p, kappa_total, diffusion_u, diffusion_total, area_star = parameters
    objective = jnp.asarray(0.0, dtype=jnp.float64)
    for block in data.blocks:
        current = jnp.sum(block.current, axis=1)
        following = jnp.sum(block.following, axis=1)
        mean = current - kappa_total * (current - area_star) * block.dt
        variance = 2.0 * diffusion_total * block.dt + JITTER
        log_density = -0.5 * (
            jnp.log(2.0 * jnp.pi * variance) + (following - mean) ** 2 / variance
        )
        objective -= jnp.sum(block.weight * log_density)
    for sequence in data.sequences:
        objective -= sequence.weight * _persistent_log_likelihood(
            sequence, tau_p, diffusion_u
        )
    return objective


def _persistent_log_likelihood(
    sequence: StateSpaceSequence, tau_p: jax.Array, diffusion_u: jax.Array
) -> jax.Array:
    params = _persistent_lgssm(sequence, tau_p, diffusion_u)
    return lgssm_filter(params, sequence.conservative).marginal_loglik


def _persistent_lgssm(
    sequence: StateSpaceSequence, tau_p: jax.Array, diffusion_u: jax.Array
) -> ParamsLGSSM:
    observations = sequence.conservative
    dimension = observations.shape[1]
    identity = jnp.eye(dimension, dtype=jnp.float64)
    zeros = jnp.zeros_like(identity)
    dt = jnp.diff(sequence.tau)
    dt_full = jnp.concatenate((dt, dt[-1:]))
    decay = jnp.exp(-dt_full / tau_p)
    weights = jax.vmap(
        lambda step, persistence: jnp.block(
            [[identity, step * identity], [zeros, persistence * identity]]
        )
    )(dt_full, decay)
    variance_u = diffusion_u / tau_p
    noise_u = variance_u * (1.0 - decay**2)
    covariances = jax.vmap(
        lambda variance: jnp.block(
            [[JITTER * identity, zeros], [zeros, variance * identity]]
        )
    )(noise_u)
    emission = jnp.concatenate((identity, zeros), axis=1)
    return ParamsLGSSM(
        initial=ParamsLGSSMInitial(
            mean=jnp.concatenate((observations[0], jnp.zeros(dimension))),
            cov=jnp.block(
                [[JITTER * identity, zeros], [zeros, variance_u * identity]]
            ),
        ),
        dynamics=ParamsLGSSMDynamics(
            weights=weights,
            bias=jnp.zeros(2 * dimension),
            input_weights=jnp.zeros((2 * dimension, 0)),
            cov=covariances,
        ),
        emissions=ParamsLGSSMEmissions(
            weights=emission,
            bias=jnp.zeros(dimension),
            input_weights=jnp.zeros((dimension, 0)),
            cov=JITTER * identity,
        ),
    )


def persistent_innovations(
    sequence: StateSpaceSequence, tau_p: jax.Array, diffusion_u: jax.Array
) -> jax.Array:
    """Return standardized conservative Kalman innovations after the first frame."""
    params = _persistent_lgssm(sequence, tau_p, diffusion_u)
    posterior = lgssm_filter(params, sequence.conservative)
    filtered_means = posterior.filtered_means[:-1]
    filtered_covariances = posterior.filtered_covariances[:-1]
    weights = params.dynamics.weights[:-1]
    process_covariances = params.dynamics.cov[:-1]
    emission = params.emissions.weights
    predicted_means = jax.vmap(jnp.matmul)(weights, filtered_means)
    predicted_covariances = jax.vmap(
        lambda weight, covariance, process: weight @ covariance @ weight.T + process
    )(weights, filtered_covariances, process_covariances)
    observation_means = jax.vmap(jnp.matmul)(
        jnp.broadcast_to(emission, (len(predicted_means), *emission.shape)),
        predicted_means,
    )
    observation_covariances = jax.vmap(
        lambda covariance: emission @ covariance @ emission.T + params.emissions.cov
    )(predicted_covariances)
    differences = sequence.conservative[1:] - observation_means
    return jax.vmap(
        lambda covariance, difference: jax.scipy.linalg.solve_triangular(
            jnp.linalg.cholesky(covariance), difference, lower=True
        )
    )(observation_covariances, differences)


def negative_log_likelihood(raw: jax.Array, data: TrainingTransitions) -> jax.Array:
    return parameter_negative_log_likelihood(positive_parameters(raw), data)
