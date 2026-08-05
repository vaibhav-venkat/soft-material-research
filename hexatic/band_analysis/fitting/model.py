"""Clean-segment model for coupled band-area dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import linalg

from ..detection.segments import StableSegment


jax.config.update("jax_enable_x64", True)

PARAMETER_NAMES = (
    "gamma",
    "omega",
    "kappa_T",
    "D_u",
    "D_T",
    "A_T_star",
    "sigma_b",
)
EVENT_PARAMETER_NAMES = (
    "tau_p",
    "kappa_T",
    "D_u",
    "D_T",
    "A_T_star",
    "sigma_E",
)
JITTER = 1e-10


class TransitionBlock(NamedTuple):
    current: jax.Array
    following: jax.Array
    dt: jax.Array
    weight: jax.Array


class StateSpaceSequence(NamedTuple):
    """Redistribution observations belonging to one lagged sequence."""

    y: jax.Array
    dt: jax.Array
    tau: jax.Array
    weight: jax.Array
    segment_id: int

    @property
    def rates(self) -> jax.Array:
        return self.y / self.dt[:, None]


class SequenceBatch(NamedTuple):
    """Padded same-dimension sequences for one vectorized Kalman scan."""

    y: jax.Array
    dt: jax.Array
    mask: jax.Array
    weight: jax.Array
    segment_id: jax.Array
    segment_count: int


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainingTransitions:
    scaling: Scaling
    blocks: tuple[TransitionBlock, ...]
    sequence_batches: tuple[SequenceBatch, ...]


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

    @property
    def parameter_factors(self) -> np.ndarray:
        """Normalized-to-physical multipliers for the clean fit."""
        return np.asarray(
            [
                1.0 / self.time,
                1.0 / self.time,
                1.0 / self.time,
                self.area**2 / self.time**3,
                self.area**2 / self.time,
                self.area,
                self.area / self.time,
            ],
            dtype=np.float64,
        )

    @property
    def event_parameter_factors(self) -> np.ndarray:
        """Normalized-to-physical multipliers for the legacy event fit."""
        return np.asarray(
            [
                self.time,
                1.0 / self.time,
                self.area**2 / self.time,
                self.area**2 / self.time,
                self.area,
                self.area,
            ],
            dtype=np.float64,
        )


def prepare_training_transitions(
    segments: list[StableSegment], lag: int
) -> TrainingTransitions:
    """Normalize once, then construct total and redistribution observations."""
    scaling = Scaling.from_training(segments)
    normalized = [scaling.normalize_segment(segment) for segment in segments]
    sequences = build_state_space_sequences(normalized, lag)
    return TrainingTransitions(
        scaling=scaling,
        blocks=build_transition_blocks(normalized, lag),
        sequence_batches=build_sequence_batches(sequences, len(normalized)),
    )


@lru_cache(maxsize=None)
def helmert_basis(n_bands: int) -> np.ndarray:
    basis = linalg.helmert(n_bands, full=False).T.copy()
    basis.setflags(write=False)
    return basis


@lru_cache(maxsize=None)
def conservative_projection(n_bands: int) -> np.ndarray:
    """Return the zero-sum projector ``Q Q^T``."""
    basis = helmert_basis(n_bands)
    projection = basis @ basis.T
    projection.setflags(write=False)
    return projection


def area_fraction_allocation(area: jax.Array) -> jax.Array:
    """Allocate a total-area increment using the observed left endpoint."""
    return area / jnp.sum(area)


def build_state_space_sequences(
    segments: list[StableSegment], lag: int
) -> tuple[StateSpaceSequence, ...]:
    """Build non-overlapping redistribution-rate sequences for the Kalman filter."""
    sequences: list[StateSpaceSequence] = []
    for segment_id, segment in enumerate(segments):
        if (
            segment.n_bands <= 1
            or len(segment.tau) < 2
            or segment.tau[-1] <= segment.tau[0]
        ):
            continue
        basis = helmert_basis(segment.n_bands)
        for phase in range(lag):
            areas = np.asarray(segment.areas[phase::lag], dtype=np.float64)
            tau = np.asarray(segment.tau[phase::lag], dtype=np.float64)
            if len(tau) < 2:
                continue
            increments = areas[1:] - areas[:-1]
            total_increments = increments.sum(axis=1)
            weights = areas[:-1] / areas[:-1].sum(axis=1, keepdims=True)
            redistribution = increments - weights * total_increments[:, None]
            sequences.append(
                StateSpaceSequence(
                    y=jnp.asarray(redistribution @ basis, dtype=jnp.float64),
                    dt=jnp.asarray(np.diff(tau), dtype=jnp.float64),
                    tau=jnp.asarray(tau, dtype=jnp.float64),
                    weight=jnp.asarray(1.0 / lag, dtype=jnp.float64),
                    segment_id=segment_id,
                )
            )
    return tuple(sequences)


def build_sequence_batches(
    sequences: tuple[StateSpaceSequence, ...], segment_count: int
) -> tuple[SequenceBatch, ...]:
    """Pad sequences by redistribution dimension for one vectorized scan per group."""
    grouped: dict[int, list[StateSpaceSequence]] = {}
    for sequence in sequences:
        grouped.setdefault(int(sequence.y.shape[1]), []).append(sequence)

    batches = []
    for dimension, group in sorted(grouped.items()):
        maximum = max(sequence.y.shape[0] for sequence in group)
        y = np.zeros((len(group), maximum, dimension), dtype=np.float64)
        dt = np.ones((len(group), maximum), dtype=np.float64)
        mask = np.zeros((len(group), maximum), dtype=bool)
        weights = np.empty(len(group), dtype=np.float64)
        segment_ids = np.empty(len(group), dtype=np.int64)
        for index, sequence in enumerate(group):
            length = sequence.y.shape[0]
            y[index, :length] = np.asarray(sequence.y)
            dt[index, :length] = np.asarray(sequence.dt)
            mask[index, :length] = True
            weights[index] = float(sequence.weight)
            segment_ids[index] = sequence.segment_id
        batches.append(
            SequenceBatch(
                y=jnp.asarray(y, dtype=jnp.float64),
                dt=jnp.asarray(dt, dtype=jnp.float64),
                mask=jnp.asarray(mask),
                weight=jnp.asarray(weights, dtype=jnp.float64),
                segment_id=jnp.asarray(segment_ids, dtype=jnp.int64),
                segment_count=segment_count,
            )
        )
    return tuple(batches)


def build_transition_blocks(
    segments: list[StableSegment], lag: int
) -> tuple[TransitionBlock, ...]:
    """Use every lagged total-area transition wholly inside a clean segment."""
    grouped: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for segment in segments:
        count = len(segment.tau) - lag
        if count <= 0:
            continue
        grouped.setdefault(segment.n_bands, []).append(
            (
                segment.areas[:count],
                segment.areas[lag:],
                segment.tau[lag:] - segment.tau[:count],
            )
        )

    blocks = []
    for _, parts in sorted(grouped.items()):
        current, following, dt = (np.concatenate(field) for field in zip(*parts))
        blocks.append(
            TransitionBlock(
                current=jnp.asarray(current, dtype=jnp.float64),
                following=jnp.asarray(following, dtype=jnp.float64),
                dt=jnp.asarray(dt, dtype=jnp.float64),
                weight=jnp.full(len(dt), 1.0 / lag, dtype=jnp.float64),
            )
        )
    return tuple(blocks)


def positive_parameters(raw_parameters: jax.Array) -> jax.Array:
    return jax.nn.softplus(raw_parameters) + 1e-12


def raw_parameters(parameters: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(parameters, dtype=np.float64) - 1e-12, 1e-12)
    return values + np.log(-np.expm1(-values))


def oscillatory_transition(
    u: jax.Array,
    r: jax.Array,
    dt: jax.Array | float,
    gamma: jax.Array | float,
    omega: jax.Array | float,
) -> tuple[jax.Array, jax.Array]:
    """Apply the exact two-dimensional OU transition without a large matrix."""
    decay = jnp.exp(-gamma * dt)
    cosine = jnp.cos(omega * dt)
    sine = jnp.sin(omega * dt)
    return (
        decay * (cosine * u + sine * r),
        decay * (-sine * u + cosine * r),
    )


def oscillatory_transition_matrix(
    dt: jax.Array | float,
    gamma: jax.Array | float,
    omega: jax.Array | float,
) -> jax.Array:
    """Return the small two-by-two analytic transition matrix."""
    decay = jnp.exp(-gamma * dt)
    cosine = jnp.cos(omega * dt)
    sine = jnp.sin(omega * dt)
    return decay * jnp.stack(
        (
            jnp.stack((cosine, sine), axis=-1),
            jnp.stack((-sine, cosine), axis=-1),
        ),
        axis=-2,
    )


def oscillatory_process_variance(
    dt: jax.Array | float,
    gamma: jax.Array | float,
    diffusion_u: jax.Array | float,
) -> jax.Array:
    """Exact per-component innovation variance for the oscillatory OU state."""
    return diffusion_u / gamma * (1.0 - jnp.exp(-2.0 * gamma * dt))


def integrated_oscillatory_variance(
    dt: jax.Array | float,
    gamma: jax.Array | float,
    omega: jax.Array | float,
    diffusion_u: jax.Array | float,
) -> jax.Array:
    """Exact stationary variance of ``∫u(t)dt`` for one redistribution mode."""
    denominator = gamma**2 + omega**2
    exponential = jnp.exp(-gamma * dt)
    cosine = jnp.cos(omega * dt)
    sine = jnp.sin(omega * dt)
    real_second_term = (
        (1.0 - exponential * cosine) * (gamma**2 - omega**2)
        + 2.0 * gamma * omega * exponential * sine
    ) / denominator**2
    integral = gamma * dt / denominator - real_second_term
    return 2.0 * diffusion_u / gamma * integral


class _KalmanResult(NamedTuple):
    log_likelihood: jax.Array
    standardized_innovations: jax.Array
    innovation_variances: jax.Array


def _observe(
    mean: jax.Array, covariance: jax.Array, observation: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Update one independent vector of observations ``u + beta``."""
    observation_mean = mean[0] + mean[2]
    covariance_observation = covariance[:, 0] + covariance[:, 2]
    variance = covariance_observation[0] + covariance_observation[2] + JITTER
    innovation = observation - observation_mean
    gain = covariance_observation / variance
    updated_mean = mean + gain[:, None] * innovation[None, :]
    updated_covariance = covariance - jnp.outer(
        covariance_observation, covariance_observation
    ) / variance
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
    standardized = innovation / jnp.sqrt(variance)
    log_likelihood = -0.5 * (
        observation.shape[0] * jnp.log(2.0 * jnp.pi * variance)
        + jnp.sum(innovation**2) / variance
    )
    return updated_mean, updated_covariance, standardized, variance, log_likelihood


def _kalman_filter(
    rates: jax.Array, dt: jax.Array, parameters: jax.Array
) -> _KalmanResult:
    """Marginalize latent ``r`` and the segment-constant transfer with a scan."""
    gamma, omega, _, diffusion_u, _, _, sigma_b = parameters
    stationary_variance = diffusion_u / gamma
    mean = jnp.zeros((3, rates.shape[1]), dtype=jnp.float64)
    covariance = jnp.diag(
        jnp.asarray(
            [stationary_variance, stationary_variance, sigma_b**2],
            dtype=jnp.float64,
        )
    )
    mean, covariance, initial_innovation, initial_variance, initial_log_likelihood = _observe(
        mean, covariance, rates[0]
    )

    def step(
        carry: tuple[jax.Array, jax.Array],
        values: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
        previous_mean, previous_covariance = carry
        step_dt, observation = values
        u, r = oscillatory_transition(
            previous_mean[0], previous_mean[1], step_dt, gamma, omega
        )
        propagated_mean = jnp.stack((u, r, previous_mean[2]))
        two_state_transition = oscillatory_transition_matrix(
            step_dt, gamma, omega
        )
        transition_matrix = jnp.zeros((3, 3), dtype=jnp.float64)
        transition_matrix = transition_matrix.at[:2, :2].set(two_state_transition)
        transition_matrix = transition_matrix.at[2, 2].set(1.0)
        process_variance = oscillatory_process_variance(
            step_dt, gamma, diffusion_u
        )
        process_covariance = jnp.diag(
            jnp.asarray([process_variance, process_variance, 0.0])
        )
        propagated_covariance = (
            transition_matrix @ previous_covariance @ transition_matrix.T
            + process_covariance
        )
        updated_mean, updated_covariance, standardized, variance, log_likelihood = _observe(
            propagated_mean, propagated_covariance, observation
        )
        return (
            updated_mean,
            updated_covariance,
        ), (standardized, variance, log_likelihood)

    (_, _), (innovations, variances, log_likelihoods) = jax.lax.scan(
        step, (mean, covariance), (dt[:-1], rates[1:])
    )
    standardized = jnp.concatenate(
        (initial_innovation[None, :], innovations), axis=0
    )
    variances = jnp.concatenate(
        (
            jnp.asarray([initial_variance]),
            variances,
        )
    )
    return _KalmanResult(
        initial_log_likelihood + jnp.sum(log_likelihoods),
        standardized,
        variances,
    )


def kalman_log_likelihood(
    sequence: StateSpaceSequence, parameters: jax.Array
) -> jax.Array:
    """Return one sequence's marginalized redistribution log likelihood."""
    return _kalman_filter(sequence.rates, sequence.dt, parameters).log_likelihood


def persistent_innovations(
    sequence: StateSpaceSequence, parameters: jax.Array
) -> jax.Array:
    """Return the standardized innovations consumed by residual diagnostics."""
    return _kalman_filter(sequence.rates, sequence.dt, parameters).standardized_innovations


class _KalmanCoefficients(NamedTuple):
    normalizer: jax.Array
    squared: jax.Array
    linear: jax.Array
    curvature: jax.Array


def _kalman_coefficients_single(
    rates: jax.Array,
    dt: jax.Array,
    mask: jax.Array,
    parameters: jax.Array,
) -> _KalmanCoefficients:
    """Return the conditional likelihood coefficients for one padded sequence."""
    gamma, omega, _, diffusion_u, _, _, _ = parameters
    stationary_variance = diffusion_u / gamma
    dimension = rates.shape[1]
    mean = jnp.zeros((2, dimension), dtype=jnp.float64)
    sensitivity = jnp.zeros((2, dimension), dtype=jnp.float64)
    covariance = stationary_variance * jnp.eye(2, dtype=jnp.float64)

    def observe(
        mean: jax.Array,
        sensitivity: jax.Array,
        covariance: jax.Array,
        observation: jax.Array,
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        covariance_observation = covariance[:, 0]
        variance = covariance_observation[0] + JITTER
        innovation = observation - mean[0]
        derivative = sensitivity[0] + 1.0
        gain = covariance_observation / variance
        updated_mean = mean + gain[:, None] * innovation[None, :]
        updated_sensitivity = sensitivity - gain[:, None] * derivative[None, :]
        updated_covariance = covariance - jnp.outer(
            covariance_observation, covariance_observation
        ) / variance
        updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
        normalizer = jnp.full(
            (dimension,), jnp.log(2.0 * jnp.pi * variance), dtype=jnp.float64
        )
        squared = innovation**2 / variance
        linear = innovation * derivative / variance
        curvature = derivative**2 / variance
        return (
            updated_mean,
            updated_sensitivity,
            updated_covariance,
            normalizer,
            squared,
            linear,
            curvature,
        )

    (
        updated_mean,
        updated_sensitivity,
        updated_covariance,
        normalizer,
        squared,
        linear,
        curvature,
    ) = observe(mean, sensitivity, covariance, rates[0])
    valid = mask[0]
    mean = jnp.where(valid, updated_mean, mean)
    sensitivity = jnp.where(valid, updated_sensitivity, sensitivity)
    covariance = jnp.where(valid, updated_covariance, covariance)
    normalizer = jnp.where(valid, normalizer, 0.0)
    squared = jnp.where(valid, squared, 0.0)
    linear = jnp.where(valid, linear, 0.0)
    curvature = jnp.where(valid, curvature, 0.0)

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        values: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array],
        tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ]:
        previous_mean, previous_sensitivity, previous_covariance = carry
        step_dt, observation, valid = values
        transition_matrix = oscillatory_transition_matrix(
            step_dt, gamma, omega
        )
        propagated_mean = transition_matrix @ previous_mean
        propagated_sensitivity = transition_matrix @ previous_sensitivity
        process_variance = oscillatory_process_variance(
            step_dt, gamma, diffusion_u
        )
        propagated_covariance = (
            transition_matrix
            @ previous_covariance
            @ transition_matrix.T
            + process_variance * jnp.eye(2, dtype=jnp.float64)
        )
        (
            updated_mean,
            updated_sensitivity,
            updated_covariance,
            step_normalizer,
            step_squared,
            step_linear,
            step_curvature,
        ) = observe(
            propagated_mean,
            propagated_sensitivity,
            propagated_covariance,
            observation,
        )
        mean = jnp.where(valid, updated_mean, previous_mean)
        sensitivity = jnp.where(valid, updated_sensitivity, previous_sensitivity)
        covariance = jnp.where(valid, updated_covariance, previous_covariance)
        return (mean, sensitivity, covariance), (
            jnp.where(valid, step_normalizer, 0.0),
            jnp.where(valid, step_squared, 0.0),
            jnp.where(valid, step_linear, 0.0),
            jnp.where(valid, step_curvature, 0.0),
        )

    (_, _, _), (normalizers, squared_terms, linear_terms, curvature_terms) = (
        jax.lax.scan(
            step,
            (mean, sensitivity, covariance),
            (dt[:-1], rates[1:], mask[1:]),
        )
    )
    return _KalmanCoefficients(
        normalizer + jnp.sum(normalizers, axis=0),
        squared + jnp.sum(squared_terms, axis=0),
        linear + jnp.sum(linear_terms, axis=0),
        curvature + jnp.sum(curvature_terms, axis=0),
    )


def _kalman_coefficients_batch(
    rates: jax.Array,
    dt: jax.Array,
    mask: jax.Array,
    parameters: jax.Array,
) -> _KalmanCoefficients:
    """Run one padded scan over all sequences in a same-dimension batch."""
    return jax.vmap(
        _kalman_coefficients_single,
        in_axes=(0, 0, 0, None),
    )(rates, dt, mask, parameters)


def transition(
    area: jax.Array, dt: jax.Array, parameters: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Unconditional raw-area moments for posterior predictive checks."""
    gamma, omega, kappa_total, diffusion_u, diffusion_total, area_star, sigma_b = (
        parameters
    )
    total = jnp.sum(area)
    allocation = area_fraction_allocation(area)
    drift = -kappa_total * (total - area_star) * allocation
    mean = area + drift * dt
    conservative_variance = integrated_oscillatory_variance(
        dt, gamma, omega, diffusion_u
    ) + sigma_b**2 * dt**2
    projection = jnp.asarray(conservative_projection(area.shape[-1]))
    covariance = conservative_variance * projection + 2.0 * dt * (
        diffusion_total * jnp.outer(allocation, allocation)
    )
    return mean, covariance + JITTER * jnp.eye(area.shape[-1], dtype=jnp.float64)


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
    """Score total-area transitions and per-sequence Kalman innovations."""
    _, _, kappa_total, _, diffusion_total, area_star, _ = parameters
    objective = jnp.asarray(0.0, dtype=jnp.float64)
    for block in data.blocks:
        current_total = jnp.sum(block.current, axis=1)
        following_total = jnp.sum(block.following, axis=1)
        mean = current_total - kappa_total * (
            current_total - area_star
        ) * block.dt
        variance = 2.0 * diffusion_total * block.dt + JITTER
        log_density = -0.5 * (
            jnp.log(2.0 * jnp.pi * variance)
            + (following_total - mean) ** 2 / variance
        )
        objective -= jnp.sum(block.weight * log_density)
    for batch in data.sequence_batches:
        coefficients = _kalman_coefficients_batch(
            batch.y / batch.dt[:, :, None], batch.dt, batch.mask, parameters
        )
        weights = batch.weight
        segment_count = batch.segment_count
        weights = weights[:, None]
        weighted = (
            coefficients.normalizer * weights,
            coefficients.squared * weights,
            coefficients.linear * weights,
            coefficients.curvature * weights,
        )
        dimension = batch.y.shape[-1]
        segment_normalizer = jnp.zeros((segment_count, dimension)).at[
            batch.segment_id
        ].add(
            weighted[0]
        )
        segment_squared = jnp.zeros((segment_count, dimension)).at[
            batch.segment_id
        ].add(
            weighted[1]
        )
        segment_linear = jnp.zeros((segment_count, dimension)).at[
            batch.segment_id
        ].add(
            weighted[2]
        )
        segment_curvature = jnp.zeros((segment_count, dimension)).at[
            batch.segment_id
        ].add(
            weighted[3]
        )
        sigma_squared = parameters[6] ** 2
        precision = segment_curvature + 1.0 / sigma_squared
        objective -= jnp.sum(
            -0.5 * (segment_normalizer + segment_squared)
            - 0.5 * jnp.log(sigma_squared)
            - 0.5 * jnp.log(precision)
            + 0.5 * segment_linear**2 / precision
        )
    return objective


def negative_log_likelihood(raw: jax.Array, data: TrainingTransitions) -> jax.Array:
    return parameter_negative_log_likelihood(positive_parameters(raw), data)
