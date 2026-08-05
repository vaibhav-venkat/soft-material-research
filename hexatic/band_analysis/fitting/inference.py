"""Fit normalized coupled area dynamics with deterministic optimization."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from .masked_model import (
    MaskedTrainingTransitions,
    negative_log_likelihood as masked_negative_log_likelihood,
)
from .model import (
    EVENT_PARAMETER_NAMES,
    PARAMETER_NAMES,
    TrainingTransitions,
    negative_log_likelihood as clean_negative_log_likelihood,
    positive_parameters,
    raw_parameters,
)


logger = logging.getLogger(__name__)

TrainingData = TrainingTransitions | MaskedTrainingTransitions


@dataclass(frozen=True)
class HessianDiagnostics:
    matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    condition_number: float
    nonpositive: bool
    weak: bool


@dataclass(frozen=True)
class OptimizationRun:
    raw_parameters: np.ndarray
    parameters: np.ndarray
    objective: float
    gradient_norm: float
    result: str
    hessian: HessianDiagnostics | None


@dataclass(frozen=True)
class OptimizationResult:
    best: OptimizationRun
    runs: tuple[OptimizationRun, ...]
    empirical_parameters: np.ndarray


def _total_transitions(
    data: TrainingData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(data, TrainingTransitions):
        return (
            np.concatenate(
                [np.asarray(block.current.sum(axis=1)) for block in data.blocks]
            ),
            np.concatenate(
                [np.asarray(block.following.sum(axis=1)) for block in data.blocks]
            ),
            np.concatenate([np.asarray(block.dt) for block in data.blocks]),
            np.concatenate([np.asarray(block.weight) for block in data.blocks]),
        )

    current_parts = []
    following_parts = []
    dt_parts = []
    weight_parts = []
    for block in data.blocks:
        current = np.asarray(block.current)
        mask = np.asarray(block.mask, dtype=bool)
        observed = np.asarray(block.o, dtype=bool)
        observation = np.asarray(block.y) - np.asarray(block.g)
        coverage = np.einsum("ti,tij->tj", observed, np.asarray(block.C))
        informative = np.all(np.isclose(coverage, mask), axis=1)
        current_parts.append(np.sum(current[informative] * mask[informative], axis=1))
        following_parts.append(
            np.sum(observation[informative] * observed[informative], axis=1)
        )
        dt_parts.append(np.asarray(block.dt)[informative])
        weight_parts.append(np.asarray(block.weight)[informative])
    return (
        np.concatenate(current_parts),
        np.concatenate(following_parts),
        np.concatenate(dt_parts),
        np.concatenate(weight_parts),
    )


def _rate_statistics(
    data: TrainingData,
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
    rates: list[np.ndarray] = []
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    intervals: list[np.ndarray] = []
    if isinstance(data, MaskedTrainingTransitions):
        for block in data.rate_blocks:
            initial_mask = np.asarray(block.initial_mask, dtype=bool)
            mask = np.asarray(block.mask, dtype=bool)
            rates.extend(
                (
                    np.asarray(block.initial)[initial_mask],
                    np.asarray(block.following)[mask],
                )
            )
            if len(block.current):
                pairs.append(
                    (
                        np.asarray(block.current)[mask],
                        np.asarray(block.following)[mask],
                    )
                )
                intervals.append(
                    np.repeat(np.asarray(block.dt), mask.sum(axis=1))
                )
    else:
        for batch in data.sequence_batches:
            batch_rates = np.asarray(batch.y) / np.asarray(batch.dt)[:, :, None]
            batch_mask = np.asarray(batch.mask, dtype=bool)
            for row_index, (sequence_rates, valid) in enumerate(
                zip(batch_rates, batch_mask, strict=True)
            ):
                sequence_rates = sequence_rates[valid]
                rates.append(sequence_rates.reshape(-1))
                if len(sequence_rates) <= 1:
                    continue
                pairs.append(
                    (
                        sequence_rates[:-1].reshape(-1),
                        sequence_rates[1:].reshape(-1),
                    )
                )
                intervals.append(
                    np.repeat(
                        np.asarray(batch.dt[row_index, : len(sequence_rates) - 1]),
                        sequence_rates.shape[1],
                    )
                )
    return rates, pairs, intervals


def _masked_rate_variance(data: MaskedTrainingTransitions) -> float:
    sum_squares = 0.0
    degrees = 0
    for block in data.rate_blocks:
        initial_mask = np.asarray(block.initial_mask, dtype=bool)
        mask = np.asarray(block.mask, dtype=bool)
        sum_squares += float(np.sum(np.asarray(block.initial)[initial_mask] ** 2))
        sum_squares += float(np.sum(np.asarray(block.following)[mask] ** 2))
        degrees += int(np.sum(initial_mask.sum(axis=1) - 1))
        degrees += int(np.sum(mask.sum(axis=1) - 1))
    return sum_squares / degrees if degrees else 0.0


def _event_noise_start(
    data: MaskedTrainingTransitions, parameters: np.ndarray
) -> float:
    tau_p, kappa_total, diffusion_u, diffusion_total, area_star = parameters
    excess_parts = []
    weight_parts = []
    for block in data.blocks:
        rows = np.asarray(block.sigma_e_rows, dtype=bool) & np.asarray(
            block.o, dtype=bool
        )
        if not np.any(rows):
            continue
        current = np.asarray(block.current)
        mask = np.asarray(block.mask, dtype=np.float64)
        dt = np.asarray(block.dt)
        counts = mask.sum(axis=1)
        allocation = mask / counts[:, None]
        totals = np.sum(mask * current, axis=1)
        increment = -kappa_total * (totals - area_star) * dt
        mean = mask * (
            current
            + allocation * increment[:, None]
            + np.asarray(block.slope) * dt[:, None]
        )
        projection = (
            np.eye(mask.shape[1])[None, :, :] * mask[:, :, None]
            - mask[:, :, None] * mask[:, None, :] / counts[:, None, None]
        )
        covariance = (
            (diffusion_u / tau_p)
            * dt[:, None, None] ** 2
            * projection
            + 2.0
            * diffusion_total
            * dt[:, None, None]
            * allocation[:, :, None]
            * allocation[:, None, :]
        )
        observation = np.asarray(block.C)
        predicted = np.einsum("tij,tj->ti", observation, mean) + np.asarray(block.g)
        residual_squared = (np.asarray(block.y) - predicted) ** 2
        continuous_variance = np.einsum(
            "tij,tjk,tik->ti", observation, covariance, observation
        )
        excess_parts.append(
            np.maximum(
                residual_squared[rows] - continuous_variance[rows], 0.0
            )
        )
        weight_parts.append(
            np.broadcast_to(np.asarray(block.weight)[:, None], rows.shape)[rows]
        )
    if not excess_parts:
        return 1e-2
    excess = np.concatenate(excess_parts)
    weights = np.concatenate(weight_parts)
    return max(float(np.sqrt(np.average(excess, weights=weights))), 1e-4)


def empirical_parameters(data: TrainingData) -> np.ndarray:
    """Return simple training-only starts for the selected fit path."""
    current_totals, following_totals, dt, weights = _total_transitions(data)
    total_weight = weights.sum()
    area_star = float(np.sum(weights * current_totals) / total_weight)
    centered = current_totals - area_star
    total_rate = max(
        1e-6,
        float(
            -np.sum(weights * centered * (following_totals - current_totals) / dt)
            / max(np.sum(weights * centered**2), 1e-12)
        ),
    )
    total_residual = (
        following_totals
        - current_totals
        + total_rate * centered * dt
    )
    diffusion_total = max(
        1e-8,
        float(np.sum(weights * total_residual**2 / (2.0 * dt)) / total_weight),
    )

    rates, rate_pairs, rate_intervals = _rate_statistics(data)

    def clean_omega_start(gamma: float, fallback_dt: float) -> float:
        """Estimate omega from the first pooled rate-ACF zero crossing."""
        candidates: list[float] = []
        if isinstance(data, TrainingTransitions):
            for batch in data.sequence_batches:
                batch_rates = np.asarray(batch.y) / np.asarray(batch.dt)[:, :, None]
                batch_mask = np.asarray(batch.mask, dtype=bool)
                for row_index, (sequence_rates, valid) in enumerate(
                    zip(batch_rates, batch_mask, strict=True)
                ):
                    values = sequence_rates[valid]
                    if len(values) < 3:
                        continue
                    intervals = np.asarray(
                        batch.dt[row_index, : len(values) - 1]
                    )
                    for component in range(values.shape[1]):
                        series = values[:, component]
                        centered = series - series.mean()
                        denominator = float(np.dot(centered, centered))
                        if denominator <= 1e-12:
                            continue
                        autocorrelation = np.correlate(
                            centered, centered, mode="full"
                        )[len(series) - 1 :] / denominator
                        crossing = np.flatnonzero(autocorrelation[1:] <= 0.0)
                        if len(crossing):
                            first = int(crossing[0])
                            zero_time = float(np.sum(intervals[: first + 1]))
                            if zero_time > 0.0:
                                candidates.append(np.pi / (2.0 * zero_time))
        if candidates:
            return max(float(np.median(candidates)), 1e-6)
        return max(0.1 * gamma, 0.25 / max(fallback_dt, 1e-6))

    if rates:
        all_rates = np.concatenate(rates)
        variance_rate = max(
            _masked_rate_variance(data)
            if isinstance(data, MaskedTrainingTransitions)
            else float(np.var(all_rates)),
            1e-8,
        )
        if rate_pairs:
            previous = np.concatenate([pair[0] for pair in rate_pairs])
            following = np.concatenate([pair[1] for pair in rate_pairs])
            correlation = float(
                np.dot(previous - previous.mean(), following - following.mean())
                / max(
                    np.linalg.norm(previous - previous.mean())
                    * np.linalg.norm(following - following.mean()),
                    1e-12,
                )
            )
            representative_dt = float(np.median(np.concatenate(rate_intervals)))
            if isinstance(data, MaskedTrainingTransitions):
                tau_p = (
                    -representative_dt / np.log(np.clip(correlation, 0.05, 0.99))
                    if correlation > 0.0
                    else representative_dt
                )
                gamma = 1.0 / max(tau_p, 1e-6)
                omega = 0.1 * gamma
            else:
                gamma = max(
                    1e-6,
                    -np.log(np.clip(abs(correlation), 0.05, 0.99))
                    / max(representative_dt, 1e-6),
                )
                omega = clean_omega_start(gamma, representative_dt)
        else:
            representative_dt = float(np.median(dt))
            if isinstance(data, MaskedTrainingTransitions):
                tau_p = representative_dt
                gamma = 1.0 / max(tau_p, 1e-6)
                omega = 0.1 * gamma
            else:
                gamma = 1.0 / max(representative_dt, 1e-6)
                omega = clean_omega_start(gamma, representative_dt)
        if isinstance(data, MaskedTrainingTransitions):
            diffusion_u = max(variance_rate * tau_p, 1e-8)
        else:
            diffusion_u = max(variance_rate * gamma, 1e-8)
    else:
        representative_dt = float(np.median(dt))
        gamma = 1.0 / max(representative_dt, 1e-6)
        omega = 0.25 * gamma
        diffusion_u = diffusion_total
        variance_rate = max(diffusion_u / gamma, 1e-8)

    if isinstance(data, MaskedTrainingTransitions):
        parameters = np.asarray(
            [
                max(1.0 / max(gamma, 1e-6), 1e-6),
                total_rate,
                diffusion_u,
                diffusion_total,
                max(area_star, 1e-6),
            ],
            dtype=np.float64,
        )
        parameters = np.append(parameters, _event_noise_start(data, parameters))
        return parameters

    segment_means: list[list[float]] = [
        [] for _ in range(data.sequence_batches[0].segment_count)
    ]
    for batch in data.sequence_batches:
        batch_rates = np.asarray(batch.y) / np.asarray(batch.dt)[:, :, None]
        batch_mask = np.asarray(batch.mask, dtype=bool)
        for rates_row, valid, segment_id in zip(
            batch_rates,
            batch_mask,
            np.asarray(batch.segment_id),
            strict=True,
        ):
            values = rates_row[valid]
            if len(values):
                segment_means[int(segment_id)].append(float(values.mean()))
    sequence_means = np.asarray(
        [np.mean(values) for values in segment_means if values], dtype=np.float64
    )
    sigma_b = (
        max(float(np.sqrt(max(np.var(sequence_means), 0.0))), 1e-6)
        if len(sequence_means) > 1
        else max(0.1 * np.sqrt(variance_rate), 1e-6)
    )
    # The lag-1 correlation sees the dominant fast mode; the slow mode is set an
    # order of magnitude below it, and the empirical zero crossing underestimates
    # omega because the fast tail delays it.
    lambda_f = max(gamma, 1e-6)
    slow_rate = lambda_f / 13.0
    return np.asarray(
        [
            slow_rate,
            12.0,
            max(1.5 * omega, 1e-6),
            max(0.9 * variance_rate, 1e-8),
            0.11,
            total_rate,
            diffusion_total,
            max(area_star, 1e-6),
            sigma_b,
        ],
        dtype=np.float64,
    )


def _negative_log_likelihood(raw: jax.Array, data: TrainingData) -> jax.Array:
    if isinstance(data, MaskedTrainingTransitions):
        return masked_negative_log_likelihood(raw, data)
    return clean_negative_log_likelihood(raw, data)


def _hessian_diagnostics(raw: np.ndarray, data: TrainingData) -> HessianDiagnostics:
    matrix = np.asarray(
        jax.hessian(_negative_log_likelihood)(jnp.asarray(raw), data)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    largest = float(np.max(eigenvalues))
    nonpositive = bool(np.any(eigenvalues <= 0.0))
    weak = bool(largest <= 0.0 or np.any(eigenvalues < 1e-6 * largest))
    condition = float(np.linalg.cond(matrix))
    return HessianDiagnostics(
        matrix=matrix,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        condition_number=condition,
        nonpositive=nonpositive,
        weak=weak,
    )


def _starts(
    empirical: np.ndarray,
    seed: int,
    count: int = 8,
    *,
    event: bool = False,
) -> tuple[np.ndarray, ...]:
    if count < 2:
        raise ValueError("optimizer starts must be at least two")
    empirical_raw = raw_parameters(empirical)
    generator = np.random.default_rng(seed)
    perturbed = tuple(
        empirical_raw + generator.normal(0.0, 0.75, len(empirical))
        for _ in range(count - 2)
    )
    if len(empirical) == len(PARAMETER_NAMES) and not event:
        generic_values = [0.08, 11.0, 0.25, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1]
    else:
        generic_values = [1.0, 0.1, 0.1, 0.1, 1.0, float(empirical[-1])]
    generic = raw_parameters(np.asarray(generic_values))
    return (empirical_raw, generic, *perturbed)


def _same_stationary_optimum(runs: list[OptimizationRun]) -> bool:
    """Require three finite, well-converged and parameter-consistent solutions."""
    if len(runs) < 3:
        return False
    recent = runs[-3:]
    # Gradient norms scale with the log likelihood, so the threshold must too.
    if not all(
        np.isfinite(run.objective)
        and run.gradient_norm < 1e-6 * max(1.0, abs(run.objective))
        and np.all(np.isfinite(run.parameters))
        for run in recent
    ):
        return False
    objectives = np.asarray([run.objective for run in recent])
    objective_tolerance = 1e-7 * max(1.0, abs(float(objectives.min())))
    if float(np.ptp(objectives)) > objective_tolerance:
        return False
    log_parameters = np.log(np.stack([run.parameters for run in recent]))
    return bool(np.max(np.ptp(log_parameters, axis=0)) < 0.05)


def optimize_parameters(
    data: TrainingData, *, seed: int = 0, starts: int = 8, max_steps: int = 10_000
) -> OptimizationResult:
    """Run reproducible empirical, perturbed, and generic BFGS fits."""
    if not data.blocks or (
        isinstance(data, MaskedTrainingTransitions)
        and not data.rate_blocks
    ) or (
        isinstance(data, TrainingTransitions)
        and not data.sequence_batches
    ):
        raise ValueError("at least one training transition is required")
    empirical = empirical_parameters(data)
    logger.info(
        "BFGS empirical start %s: %s",
        (
            EVENT_PARAMETER_NAMES
            if isinstance(data, MaskedTrainingTransitions)
            else PARAMETER_NAMES
        )[: len(empirical)],
        np.array2string(empirical, precision=4),
    )
    solver = optx.BFGS(rtol=1e-8, atol=1e-8)
    runs: list[OptimizationRun] = []
    for index, start in enumerate(
        _starts(
            empirical,
            seed,
            starts,
            event=isinstance(data, MaskedTrainingTransitions),
        ),
        start=1,
    ):
        logger.info("BFGS start %d/%d", index, starts)
        solution = optx.minimise(
            _negative_log_likelihood,
            solver,
            jnp.asarray(start),
            args=data,
            max_steps=max_steps,
            throw=False,
        )
        raw = np.asarray(solution.value)
        objective = float(
            _negative_log_likelihood(jnp.asarray(raw), data)
        )
        gradient = np.asarray(
            jax.grad(_negative_log_likelihood)(jnp.asarray(raw), data)
        )
        runs.append(
            OptimizationRun(
                raw_parameters=raw,
                parameters=np.asarray(positive_parameters(jnp.asarray(raw))),
                objective=objective,
                gradient_norm=float(np.linalg.norm(gradient)),
                result=str(solution.result),
                hessian=None,
            )
        )
        logger.info(
            "BFGS start %d/%d finished: objective=%.6g gradient_norm=%.3g result=%s",
            index,
            starts,
            objective,
            runs[-1].gradient_norm,
            runs[-1].result,
        )
        if index < starts and _same_stationary_optimum(runs):
            logger.info(
                "BFGS stopping after %d/%d starts: three starts reached the "
                "same stationary optimum",
                index,
                starts,
            )
            break
    finite_indices = [
        index for index, run in enumerate(runs) if np.isfinite(run.objective)
    ]
    if not finite_indices:
        raise RuntimeError("all BFGS starts produced nonfinite objectives")
    best_index = min(finite_indices, key=lambda index: runs[index].objective)
    logger.info("BFGS computing Hessian once at the selected optimum")
    best = replace(
        runs[best_index],
        hessian=_hessian_diagnostics(runs[best_index].raw_parameters, data),
    )
    runs[best_index] = best
    logger.info(
        "BFGS selected objective=%.6g gradient_norm=%.3g",
        best.objective,
        best.gradient_norm,
    )
    return OptimizationResult(
        best=best,
        runs=tuple(runs),
        empirical_parameters=empirical,
    )
