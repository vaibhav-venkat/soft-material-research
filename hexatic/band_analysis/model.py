"""Define scaling, transitions, and likelihoods for the coupled area SDE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

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


class PersistentRateBlock(NamedTuple):
    initial: jax.Array
    initial_weight: jax.Array
    current: jax.Array
    following: jax.Array
    dt: jax.Array
    weight: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainingTransitions:
    scaling: Scaling
    blocks: tuple[TransitionBlock, ...]
    rate_blocks: tuple[PersistentRateBlock, ...]
    conservative_slopes: tuple[jax.Array, ...] = ()
    sigma_b_squared: jax.Array | float = 0.0


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
    slopes = conservative_segment_slopes(normalized)
    sequences = build_state_space_sequences(normalized, lag)
    return TrainingTransitions(
        scaling=scaling,
        blocks=build_transition_blocks(normalized, lag),
        rate_blocks=build_persistent_rate_blocks(sequences),
        conservative_slopes=tuple(jnp.asarray(slope) for slope in slopes),
        sigma_b_squared=jnp.asarray(
            estimate_slope_variance(slopes), dtype=jnp.float64
        ),
    )


def _helmert_basis(n_bands: int) -> np.ndarray:
    basis = np.zeros((n_bands, n_bands - 1), dtype=np.float64)
    for column in range(n_bands - 1):
        count = column + 1
        basis[:count, column] = 1.0 / np.sqrt(count * (count + 1))
        basis[count, column] = -count / np.sqrt(count * (count + 1))
    return basis


def conservative_segment_slope(segment: StableSegment) -> np.ndarray:
    """Estimate the constant, zero-sum conservative slope of one segment."""
    duration = float(segment.tau[-1] - segment.tau[0])
    if duration <= 0.0:
        raise ValueError("segment duration must be positive")
    delta = np.asarray(segment.areas[-1] - segment.areas[0], dtype=np.float64)
    return (delta - delta.mean()) / duration


def conservative_segment_slopes(
    segments: list[StableSegment],
) -> tuple[np.ndarray, ...]:
    """Return projected endpoint slopes for segments with positive duration."""
    return tuple(
        conservative_segment_slope(segment)
        for segment in segments
        if len(segment.tau) >= 2 and segment.tau[-1] > segment.tau[0]
    )


def estimate_slope_variance(slopes: tuple[np.ndarray, ...]) -> float:
    """Estimate sigma_b^2 under b_s ~ N(0, sigma_b^2 P_n)."""
    degrees_of_freedom = sum(slope.size - 1 for slope in slopes)
    if degrees_of_freedom == 0:
        return 0.0
    sum_squares = sum(float(np.dot(slope, slope)) for slope in slopes)
    return sum_squares / degrees_of_freedom


def build_state_space_sequences(
    segments: list[StableSegment], lag: int
) -> tuple[StateSpaceSequence, ...]:
    sequences = []
    for segment in segments:
        if (
            segment.n_bands == 1
            or len(segment.tau) < 2
            or segment.tau[-1] <= segment.tau[0]
        ):
            continue
        basis = _helmert_basis(segment.n_bands)
        slope = conservative_segment_slope(segment)
        slope_coordinates = slope @ basis
        for phase in range(lag):
            areas = segment.areas[phase::lag]
            tau = segment.tau[phase::lag]
            if len(tau) < 2:
                continue
            sequences.append(
                StateSpaceSequence(
                    conservative=jnp.asarray(
                        areas @ basis
                        - (tau - segment.tau[0])[:, None] * slope_coordinates,
                        dtype=jnp.float64,
                    ),
                    tau=jnp.asarray(tau, dtype=jnp.float64),
                    weight=jnp.asarray(1.0 / lag, dtype=jnp.float64),
                )
            )
    return tuple(sequences)


def build_persistent_rate_blocks(
    sequences: tuple[StateSpaceSequence, ...],
) -> tuple[PersistentRateBlock, ...]:
    """Batch observed conservative rates by dimension for a compact JAX graph."""
    grouped: dict[int, dict[str, list[np.ndarray | float]]] = {}
    for sequence in sequences:
        conservative = np.asarray(sequence.conservative)
        dt = np.diff(np.asarray(sequence.tau))
        rates = np.diff(conservative, axis=0) / dt[:, None]
        dimension = rates.shape[1]
        values = grouped.setdefault(
            dimension,
            {
                "initial": [],
                "initial_weight": [],
                "current": [],
                "following": [],
                "dt": [],
                "weight": [],
            },
        )
        sequence_weight = float(sequence.weight)
        values["initial"].append(rates[0])
        values["initial_weight"].append(sequence_weight)
        if len(rates) > 1:
            values["current"].append(rates[:-1])
            values["following"].append(rates[1:])
            values["dt"].append(dt[:-1])
            values["weight"].append(
                np.full(len(rates) - 1, sequence_weight, dtype=np.float64)
            )

    blocks = []
    for dimension, values in sorted(grouped.items()):
        current_parts = values["current"]
        following_parts = values["following"]
        dt_parts = values["dt"]
        weight_parts = values["weight"]
        blocks.append(
            PersistentRateBlock(
                initial=jnp.asarray(values["initial"], dtype=jnp.float64),
                initial_weight=jnp.asarray(
                    values["initial_weight"], dtype=jnp.float64
                ),
                current=jnp.asarray(
                    np.concatenate(current_parts)
                    if current_parts
                    else np.empty((0, dimension)),
                    dtype=jnp.float64,
                ),
                following=jnp.asarray(
                    np.concatenate(following_parts)
                    if following_parts
                    else np.empty((0, dimension)),
                    dtype=jnp.float64,
                ),
                dt=jnp.asarray(
                    np.concatenate(dt_parts) if dt_parts else np.empty(0),
                    dtype=jnp.float64,
                ),
                weight=jnp.asarray(
                    np.concatenate(weight_parts) if weight_parts else np.empty(0),
                    dtype=jnp.float64,
                ),
            )
        )
    return tuple(blocks)


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
    area: jax.Array,
    dt: jax.Array,
    parameters: jax.Array,
    sigma_b_squared: jax.Array | float = 0.0,
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
    conservative_variance = (sigma_b_squared + diffusion_u / tau_p) * dt**2
    covariance = conservative_variance * projection + 2.0 * dt * (
        diffusion_total * jnp.outer(allocation, allocation)
    )
    return mean, covariance + JITTER * jnp.eye(n_bands, dtype=jnp.float64)


def transition_log_density(
    following: jax.Array,
    area: jax.Array,
    dt: jax.Array,
    parameters: jax.Array,
    sigma_b_squared: jax.Array | float = 0.0,
) -> jax.Array:
    mean, covariance = transition(area, dt, parameters, sigma_b_squared)
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
    stationary_variance = diffusion_u / tau_p + JITTER
    for block in data.rate_blocks:
        dimension = block.initial.shape[1]
        initial_log_density = -0.5 * (
            dimension * jnp.log(2.0 * jnp.pi * stationary_variance)
            + jnp.sum(block.initial**2, axis=1) / stationary_variance
        )
        objective -= jnp.sum(block.initial_weight * initial_log_density)
        decay = jnp.exp(-block.dt / tau_p)
        variance = diffusion_u / tau_p * (1.0 - decay**2) + JITTER
        residual = block.following - decay[:, None] * block.current
        log_density = -0.5 * (
            dimension * jnp.log(2.0 * jnp.pi * variance)
            + jnp.sum(residual**2, axis=1) / variance
        )
        objective -= jnp.sum(block.weight * log_density)
    return objective


def persistent_innovations(
    sequence: StateSpaceSequence, tau_p: jax.Array, diffusion_u: jax.Array
) -> jax.Array:
    """Return standardized innovations of the observed conservative rates."""
    dt = jnp.diff(sequence.tau)
    rates = jnp.diff(sequence.conservative, axis=0) / dt[:, None]
    stationary_scale = jnp.sqrt(diffusion_u / tau_p + JITTER)
    initial = rates[:1] / stationary_scale
    decay = jnp.exp(-dt[:-1] / tau_p)
    variance = diffusion_u / tau_p * (1.0 - decay**2) + JITTER
    following = (rates[1:] - decay[:, None] * rates[:-1]) / jnp.sqrt(
        variance[:, None]
    )
    return jnp.concatenate((initial, following), axis=0)


def negative_log_likelihood(raw: jax.Array, data: TrainingTransitions) -> jax.Array:
    return parameter_negative_log_likelihood(positive_parameters(raw), data)
