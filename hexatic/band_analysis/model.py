from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .segments import StableSegment


jax.config.update("jax_enable_x64", True)

PARAMETER_NAMES = ("kappa_c", "kappa_T", "D_A", "D_T", "A_T_star")
JITTER = 1e-10


class TransitionBlock(NamedTuple):
    current: jax.Array
    following: jax.Array
    dt: jax.Array
    weight: jax.Array


@dataclass(frozen=True)
class TrainingTransitions:
    scaling: Scaling
    blocks: tuple[TransitionBlock, ...]


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
        kappa_c, kappa_T, diffusion_area, diffusion_total, area_star = normalized
        return np.asarray(
            [
                kappa_c / self.time,
                kappa_T / self.time,
                diffusion_area * self.area**2 / self.time,
                diffusion_total * self.area**2 / self.time,
                area_star * self.area,
            ],
            dtype=np.float64,
        )

    def parameters_to_normalized(self, physical: np.ndarray) -> np.ndarray:
        kappa_c, kappa_T, diffusion_area, diffusion_total, area_star = physical
        return np.asarray(
            [
                kappa_c * self.time,
                kappa_T * self.time,
                diffusion_area * self.time / self.area**2,
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
    )


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
    """Euler transition for the coupled total and conservative area modes."""
    kappa_c, kappa_total, diffusion_area, diffusion_total, area_star = parameters
    n_bands = area.shape[-1]
    projection = jnp.eye(n_bands, dtype=jnp.float64) - jnp.ones(
        (n_bands, n_bands), dtype=jnp.float64
    ) / n_bands
    total = jnp.sum(area)
    allocation = area / total
    drift = (
        -kappa_c * (projection @ area)
        - kappa_total * (total - area_star) * allocation
    )
    mean = area + drift * dt
    covariance = 2.0 * dt * (
        diffusion_area * projection
        + diffusion_total * jnp.outer(allocation, allocation)
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


def negative_log_likelihood(
    raw: jax.Array, blocks: tuple[TransitionBlock, ...]
) -> jax.Array:
    parameters = positive_parameters(raw)
    objective = jnp.asarray(0.0, dtype=jnp.float64)
    for block in blocks:
        log_density = jax.vmap(transition_log_density, in_axes=(0, 0, 0, None))(
            block.following, block.current, block.dt, parameters
        )
        objective -= jnp.sum(block.weight * log_density)
    return objective
