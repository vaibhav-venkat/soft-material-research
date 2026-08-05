"""Masked event likelihood without duplicating conservative increments.

Continuous steps factor into a scalar total-area density and the persistent-rate
density. Event steps use the mapped padded Gaussian instead and are omitted from
the rate density; their means condition on the fitted inter-event slope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..detection.chains import EventChain, build_knot_grids, event_transitions
from ..detection.events import EventStep, no_event_step
from .model import JITTER, Scaling, positive_parameters


class MaskedTransitionBlock(NamedTuple):
    current: jax.Array
    following: jax.Array
    mask: jax.Array
    mask_next: jax.Array
    H: jax.Array
    g: jax.Array
    C: jax.Array
    o: jax.Array
    sigma_e_rows: jax.Array
    y: jax.Array
    slope: jax.Array
    dt: jax.Array
    weight: jax.Array
    event: jax.Array


class MaskedPersistentRateBlock(NamedTuple):
    initial: jax.Array
    initial_mask: jax.Array
    initial_weight: jax.Array
    current: jax.Array
    following: jax.Array
    mask: jax.Array
    dt: jax.Array
    weight: jax.Array


@dataclass(frozen=True)
class MaskedSlope:
    value: np.ndarray
    mask: np.ndarray


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MaskedTrainingTransitions:
    scaling: Scaling
    blocks: tuple[MaskedTransitionBlock, ...]
    rate_blocks: tuple[MaskedPersistentRateBlock, ...]
    conservative_slopes: tuple[jax.Array, ...] = ()
    slope_masks: tuple[jax.Array, ...] = ()
    sigma_b_squared: jax.Array | float = 0.0


def conservative_projection(mask: jax.Array) -> jax.Array:
    """Return ``diag(m) - m m^T / sum(m)`` at the padded width."""
    active = jnp.asarray(mask, dtype=jnp.float64)
    return jnp.diag(active) - jnp.outer(active, active) / jnp.sum(active)


def allocation(mask: jax.Array) -> jax.Array:
    """Uniform allocation over active slots and zero over padding."""
    active = jnp.asarray(mask, dtype=jnp.float64)
    return active / jnp.sum(active)


def _moments(
    area: jax.Array,
    mask: jax.Array,
    dt: jax.Array,
    parameters: jax.Array,
    sigma_b_squared: jax.Array | float,
    conservative_slope: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    tau_p, kappa_total, diffusion_u, diffusion_total, area_star = parameters[:5]
    active = jnp.asarray(mask, dtype=jnp.float64)
    weights = allocation(active)
    total = jnp.dot(active, area)
    total_increment = -kappa_total * (total - area_star) * dt
    slope = (
        jnp.zeros_like(area)
        if conservative_slope is None
        else conservative_slope
    )
    mean = active * (area + weights * total_increment + slope * dt)
    conservative_variance = (sigma_b_squared + diffusion_u / tau_p) * dt**2
    covariance = conservative_variance * conservative_projection(active)
    covariance += 2.0 * diffusion_total * dt * jnp.outer(weights, weights)
    return mean, covariance


def transition(
    area: jax.Array,
    mask: jax.Array,
    dt: jax.Array,
    parameters: jax.Array,
    sigma_b_squared: jax.Array | float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Pre-event moments with Cholesky-safe inactive-slot padding."""
    mean, covariance = _moments(area, mask, dt, parameters, sigma_b_squared)
    inactive = 1.0 - jnp.asarray(mask, dtype=jnp.float64)
    padded = covariance + jnp.diag(inactive)
    padded += JITTER * jnp.eye(area.shape[-1], dtype=jnp.float64)
    return mean, padded


def propagate_event(
    area: jax.Array,
    mask: jax.Array,
    dt: jax.Array,
    H: jax.Array,
    g: jax.Array,
    parameters: jax.Array,
    sigma_b_squared: jax.Array | float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Propagate pre-event moments through ``H`` and ``g`` for simulation."""
    mean, covariance = _moments(area, mask, dt, parameters, sigma_b_squared)
    return H @ mean + g, H @ covariance @ H.T


def transition_log_density(
    y: jax.Array,
    area: jax.Array,
    mask: jax.Array,
    dt: jax.Array,
    C: jax.Array,
    g: jax.Array,
    o: jax.Array,
    sigma_e_rows: jax.Array,
    parameters: jax.Array,
    sigma_b_squared: jax.Array | float = 0.0,
) -> jax.Array:
    """Observation likelihood, marginalized by padding all silent rows."""
    mean, covariance = _moments(area, mask, dt, parameters, sigma_b_squared)
    sigma_event = parameters[5]
    observed = jnp.asarray(o, dtype=jnp.float64)
    event_variance = (
        observed
        * jnp.asarray(sigma_e_rows, dtype=jnp.float64)
        * sigma_event**2
    )
    covariance = C @ covariance @ C.T + jnp.diag(event_variance)
    covariance += jnp.diag(1.0 - observed)
    covariance += JITTER * jnp.eye(area.shape[-1], dtype=jnp.float64)
    residual = observed * (y - (C @ mean + g))
    cholesky = jnp.linalg.cholesky(covariance)
    whitened = jax.scipy.linalg.solve_triangular(cholesky, residual, lower=True)
    padded_log_density = -0.5 * (
        area.shape[-1] * jnp.log(2.0 * jnp.pi)
        + 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
        + jnp.dot(whitened, whitened)
    )
    padding = area.shape[-1] - jnp.sum(observed)
    padding_constant = 0.5 * padding * (
        jnp.log(2.0 * jnp.pi) + jnp.log1p(JITTER)
    )
    return padded_log_density + padding_constant


def event_transition_log_density(
    y: jax.Array,
    area: jax.Array,
    mask: jax.Array,
    dt: jax.Array,
    C: jax.Array,
    g: jax.Array,
    o: jax.Array,
    sigma_e_rows: jax.Array,
    slope: jax.Array,
    parameters: jax.Array,
) -> jax.Array:
    """Mapped event density conditioned on the fitted inter-event slope."""
    mean, covariance = _moments(
        area, mask, dt, parameters, 0.0, conservative_slope=slope
    )
    sigma_event = parameters[5]
    observed = jnp.asarray(o, dtype=jnp.float64)
    event_variance = (
        observed
        * jnp.asarray(sigma_e_rows, dtype=jnp.float64)
        * sigma_event**2
    )
    covariance = C @ covariance @ C.T + jnp.diag(event_variance)
    covariance += jnp.diag(1.0 - observed)
    covariance += JITTER * jnp.eye(area.shape[-1], dtype=jnp.float64)
    residual = observed * (y - (C @ mean + g))
    cholesky = jnp.linalg.cholesky(covariance)
    whitened = jax.scipy.linalg.solve_triangular(cholesky, residual, lower=True)
    padded = -0.5 * (
        area.shape[-1] * jnp.log(2.0 * jnp.pi)
        + 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
        + jnp.dot(whitened, whitened)
    )
    padding = area.shape[-1] - jnp.sum(observed)
    constant = 0.5 * padding * (
        jnp.log(2.0 * jnp.pi) + jnp.log1p(JITTER)
    )
    return padded + constant


def factorized_transition_log_density(
    y: jax.Array,
    following: jax.Array,
    area: jax.Array,
    mask: jax.Array,
    mask_next: jax.Array,
    dt: jax.Array,
    C: jax.Array,
    g: jax.Array,
    o: jax.Array,
    sigma_e_rows: jax.Array,
    slope: jax.Array,
    event: jax.Array,
    parameters: jax.Array,
) -> jax.Array:
    """One transition factor: total-only if continuous, mapped if an event."""
    event_density = event_transition_log_density(
        y, area, mask, dt, C, g, o, sigma_e_rows, slope, parameters
    )
    _, kappa_total, _, diffusion_total, area_star = parameters[:5]
    current_total = jnp.sum(mask * area)
    following_total = jnp.sum(mask_next * following)
    mean = current_total - kappa_total * (current_total - area_star) * dt
    variance = 2.0 * diffusion_total * dt + JITTER
    total_density = -0.5 * (
        jnp.log(2.0 * jnp.pi * variance)
        + (following_total - mean) ** 2 / variance
    )
    return jnp.where(event, event_density, total_density)


_factorized_density_rows = jax.vmap(
    factorized_transition_log_density,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None),
)


def _np_projection(mask: np.ndarray) -> np.ndarray:
    active = mask.astype(np.float64)
    return np.diag(active) - np.outer(active, active) / active.sum()


def scaling_from_chains(chains: list[EventChain]) -> Scaling:
    totals = np.concatenate(
        [(chain.areas * chain.mask).sum(axis=1) for chain in chains]
    )
    intervals = np.concatenate([np.diff(chain.tau) for chain in chains])
    return Scaling(area=float(np.median(totals)), time=float(np.median(intervals)))


def normalize_event_chain(chain: EventChain, scaling: Scaling) -> EventChain:
    events = tuple(
        EventStep(
            H=step.H,
            g=step.g / scaling.area,
            C=step.C,
            o=step.o,
            sigma_e_rows=step.sigma_e_rows,
            mask_next=step.mask_next,
            y=step.y / scaling.area,
        )
        for step in chain.events
    )
    return EventChain(
        seed_id=chain.seed_id,
        slot_ids=chain.slot_ids,
        frame_indices=chain.frame_indices,
        tau=chain.tau / scaling.time,
        areas=chain.areas / scaling.area,
        mask=chain.mask,
        events=events,
    )


def _runs(chain: EventChain) -> tuple[tuple[int, int], ...]:
    runs = []
    start = 0
    for boundary in np.flatnonzero(event_transitions(chain)):
        if boundary + 1 - start >= 2:
            runs.append((start, int(boundary) + 1))
        start = int(boundary) + 1
    if chain.n_steps - start >= 2:
        runs.append((start, chain.n_steps))
    return tuple(runs)


def conservative_run_slopes(chain: EventChain) -> tuple[MaskedSlope, ...]:
    """Endpoint slopes for constant-mask inter-event runs."""
    slopes = []
    for start, end in _runs(chain):
        duration = chain.tau[end - 1] - chain.tau[start]
        mask = chain.mask[start]
        delta = chain.areas[end - 1] - chain.areas[start]
        slopes.append(MaskedSlope(_np_projection(mask) @ delta / duration, mask))
    return tuple(slopes)


def estimate_slope_variance(slopes: tuple[MaskedSlope, ...]) -> float:
    """Estimate slope variance with ``n - 1`` degrees per inter-event run."""
    degrees = sum(int(slope.mask.sum()) - 1 for slope in slopes)
    if degrees == 0:
        return 0.0
    return sum(float(np.dot(slope.value, slope.value)) for slope in slopes) / degrees


def _slope_lookup(chain: EventChain) -> np.ndarray:
    values = np.zeros((chain.n_steps - 1, chain.n_max))
    for (start, end), slope in zip(
        _runs(chain), conservative_run_slopes(chain), strict=True
    ):
        values[start : min(end, len(values))] = slope.value
    return values


def _as_transition_block(
    values: dict[str, list[np.ndarray | float | bool]],
) -> MaskedTransitionBlock:
    return MaskedTransitionBlock(
        current=jnp.asarray(values["current"], dtype=jnp.float64),
        following=jnp.asarray(values["following"], dtype=jnp.float64),
        mask=jnp.asarray(values["mask"], dtype=jnp.float64),
        mask_next=jnp.asarray(values["mask_next"], dtype=jnp.float64),
        H=jnp.asarray(values["H"], dtype=jnp.float64),
        g=jnp.asarray(values["g"], dtype=jnp.float64),
        C=jnp.asarray(values["C"], dtype=jnp.float64),
        o=jnp.asarray(values["o"], dtype=jnp.float64),
        sigma_e_rows=jnp.asarray(values["sigma_e_rows"], dtype=jnp.float64),
        y=jnp.asarray(values["y"], dtype=jnp.float64),
        slope=jnp.asarray(values["slope"], dtype=jnp.float64),
        dt=jnp.asarray(values["dt"], dtype=jnp.float64),
        weight=jnp.asarray(values["weight"], dtype=jnp.float64),
        event=jnp.asarray(values["event"], dtype=bool),
    )


def _as_rate_block(
    values: dict[str, list[np.ndarray | float]], n: int
) -> MaskedPersistentRateBlock:
    def matrix(name: str) -> jax.Array:
        rows = values[name]
        return jnp.asarray(rows if rows else np.empty((0, n)), dtype=jnp.float64)

    return MaskedPersistentRateBlock(
        initial=matrix("initial"),
        initial_mask=matrix("initial_mask"),
        initial_weight=jnp.asarray(values["initial_weight"], dtype=jnp.float64),
        current=matrix("current"),
        following=matrix("following"),
        mask=matrix("mask"),
        dt=jnp.asarray(values["dt"], dtype=jnp.float64),
        weight=jnp.asarray(values["weight"], dtype=jnp.float64),
    )


def _build_blocks(
    chains: list[EventChain], lag: int
) -> tuple[MaskedTransitionBlock, MaskedPersistentRateBlock]:
    transition_values = {name: [] for name in MaskedTransitionBlock._fields}
    rate_values = {name: [] for name in MaskedPersistentRateBlock._fields}
    n_max = chains[0].n_max
    for chain in chains:
        slopes = _slope_lookup(chain)
        for grid in build_knot_grids(chain, (lag,)):
            rates = []
            rate_masks = []
            events = []
            for start, end, event_index in zip(
                grid.start, grid.end, grid.event_index, strict=True
            ):
                start, end, event_index = int(start), int(end), int(event_index)
                dt = chain.tau[end] - chain.tau[start]
                is_event = event_index >= 0
                step = (
                    chain.events[event_index]
                    if is_event
                    else no_event_step(chain.mask[start], chain.areas[end])
                )
                for name, value in (
                    ("current", chain.areas[start]),
                    ("following", chain.areas[end]),
                    ("mask", chain.mask[start]),
                    ("mask_next", chain.mask[end]),
                    ("H", step.H),
                    ("g", step.g),
                    ("C", step.C),
                    ("o", step.o),
                    ("sigma_e_rows", step.sigma_e_rows),
                    ("y", step.y),
                    ("slope", slopes[start]),
                    ("dt", dt),
                    ("weight", grid.weight),
                    ("event", is_event),
                ):
                    transition_values[name].append(value)
                mask = chain.mask[start]
                rate = _np_projection(mask) @ (
                    chain.areas[end] - chain.areas[start]
                ) / dt - slopes[start]
                rates.append(rate)
                rate_masks.append(mask)
                events.append(is_event)
            for index, (rate, mask) in enumerate(zip(rates, rate_masks, strict=True)):
                if events[index]:
                    continue
                if index == 0 or events[index - 1]:
                    rate_values["initial"].append(rate)
                    rate_values["initial_mask"].append(mask)
                    rate_values["initial_weight"].append(grid.weight)
                else:
                    rate_values["current"].append(rates[index - 1])
                    rate_values["following"].append(rate)
                    rate_values["mask"].append(mask)
                    previous_dt = (
                        chain.tau[grid.end[index - 1]]
                        - chain.tau[grid.start[index - 1]]
                    )
                    rate_values["dt"].append(previous_dt)
                    rate_values["weight"].append(grid.weight)
    return _as_transition_block(transition_values), _as_rate_block(
        rate_values, n_max
    )


def build_transition_blocks(
    chains: list[EventChain], lag: int
) -> tuple[MaskedTransitionBlock, ...]:
    return (_build_blocks(chains, lag)[0],)


def build_persistent_rate_blocks(
    chains: list[EventChain], lag: int
) -> tuple[MaskedPersistentRateBlock, ...]:
    return (_build_blocks(chains, lag)[1],)


def prepare_training_transitions(
    chains: list[EventChain], lag: int
) -> MaskedTrainingTransitions:
    """Normalize once, then cross the NumPy-to-JAX boundary in the block builder."""
    scaling = scaling_from_chains(chains)
    normalized = [normalize_event_chain(chain, scaling) for chain in chains]
    slopes = tuple(
        slope for chain in normalized for slope in conservative_run_slopes(chain)
    )
    transition_block, rate_block = _build_blocks(normalized, lag)
    return MaskedTrainingTransitions(
        scaling=scaling,
        blocks=(transition_block,),
        rate_blocks=(rate_block,),
        conservative_slopes=tuple(jnp.asarray(slope.value) for slope in slopes),
        slope_masks=tuple(jnp.asarray(slope.mask) for slope in slopes),
        sigma_b_squared=jnp.asarray(estimate_slope_variance(slopes)),
    )


def parameter_negative_log_likelihood(
    parameters: jax.Array, data: MaskedTrainingTransitions
) -> jax.Array:
    """Score each observed degree of freedom once under the factored model."""
    objective = jnp.asarray(0.0, dtype=jnp.float64)
    for block in data.blocks:
        log_density = _factorized_density_rows(
            block.y,
            block.following,
            block.current,
            block.mask,
            block.mask_next,
            block.dt,
            block.C,
            block.g,
            block.o,
            block.sigma_e_rows,
            block.slope,
            block.event,
            parameters,
        )
        objective -= jnp.sum(block.weight * log_density)
    tau_p, _, diffusion_u = parameters[:3]
    stationary_variance = diffusion_u / tau_p + JITTER
    for block in data.rate_blocks:
        initial_dof = jnp.sum(block.initial_mask, axis=1) - 1.0
        initial_log_density = -0.5 * (
            initial_dof * jnp.log(2.0 * jnp.pi * stationary_variance)
            + jnp.sum(block.initial**2, axis=1) / stationary_variance
        )
        objective -= jnp.sum(block.initial_weight * initial_log_density)
        decay = jnp.exp(-block.dt / tau_p)
        variance = diffusion_u / tau_p * (1.0 - decay**2) + JITTER
        residual = block.mask * (block.following - decay[:, None] * block.current)
        degrees = jnp.sum(block.mask, axis=1) - 1.0
        log_density = -0.5 * (
            degrees * jnp.log(2.0 * jnp.pi * variance)
            + jnp.sum(residual**2, axis=1) / variance
        )
        objective -= jnp.sum(block.weight * log_density)
    return objective


def negative_log_likelihood(
    raw: jax.Array, data: MaskedTrainingTransitions
) -> jax.Array:
    return parameter_negative_log_likelihood(positive_parameters(raw), data)
