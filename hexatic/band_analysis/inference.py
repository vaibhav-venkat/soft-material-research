"""Fit normalized coupled area dynamics with deterministic optimization."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from .model import (
    PARAMETER_NAMES,
    TrainingTransitions,
    negative_log_likelihood,
    positive_parameters,
    raw_parameters,
)


logger = logging.getLogger(__name__)


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


def empirical_parameters(data: TrainingTransitions) -> np.ndarray:
    """Training-only total OU and persistent-rate starting estimates."""
    blocks = data.blocks
    current_totals = np.concatenate(
        [np.asarray(block.current.sum(axis=1)) for block in blocks]
    )
    following_totals = np.concatenate(
        [np.asarray(block.following.sum(axis=1)) for block in blocks]
    )
    dt = np.concatenate([np.asarray(block.dt) for block in blocks])
    weights = np.concatenate([np.asarray(block.weight) for block in blocks])
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

    rates: list[np.ndarray] = []
    rate_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    rate_intervals: list[np.ndarray] = []
    for block in data.rate_blocks:
        rates.extend(
            (np.asarray(block.initial).reshape(-1), np.asarray(block.following).reshape(-1))
        )
        if len(block.current):
            rate_pairs.append(
                (
                    np.asarray(block.current).reshape(-1),
                    np.asarray(block.following).reshape(-1),
                )
            )
            rate_intervals.append(
                np.repeat(np.asarray(block.dt), block.current.shape[1])
            )

    if rates:
        all_rates = np.concatenate(rates)
        variance_rate = max(float(np.var(all_rates)), 1e-8)
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
            tau_p = (
                -representative_dt / np.log(np.clip(correlation, 0.05, 0.99))
                if correlation > 0.0
                else representative_dt
            )
        else:
            tau_p = float(np.median(dt))
        diffusion_u = max(variance_rate * tau_p, 1e-8)
    else:
        tau_p = float(np.median(dt))
        diffusion_u = diffusion_total

    return np.asarray(
        [
            max(tau_p, 1e-6),
            total_rate,
            diffusion_u,
            diffusion_total,
            max(area_star, 1e-6),
        ],
        dtype=np.float64,
    )


def _hessian_diagnostics(raw: np.ndarray, data: TrainingTransitions) -> HessianDiagnostics:
    matrix = np.asarray(
        jax.hessian(negative_log_likelihood)(jnp.asarray(raw), data)
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
    empirical: np.ndarray, seed: int, count: int = 8
) -> tuple[np.ndarray, ...]:
    if count < 2:
        raise ValueError("optimizer starts must be at least two")
    empirical_raw = raw_parameters(empirical)
    generator = np.random.default_rng(seed)
    perturbed = tuple(
        empirical_raw + generator.normal(0.0, 0.75, len(PARAMETER_NAMES))
        for _ in range(count - 2)
    )
    generic = raw_parameters(np.asarray([1.0, 0.1, 0.1, 0.1, 1.0]))
    return (empirical_raw, generic, *perturbed)


def _same_stationary_optimum(runs: list[OptimizationRun]) -> bool:
    """Require three finite, well-converged and parameter-consistent solutions."""
    if len(runs) < 3:
        return False
    recent = runs[-3:]
    if not all(
        np.isfinite(run.objective)
        and run.gradient_norm < 1e-3
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
    data: TrainingTransitions, *, seed: int = 0, starts: int = 8
) -> OptimizationResult:
    """Run reproducible empirical, perturbed, and generic BFGS fits."""
    if not data.blocks or not data.rate_blocks:
        raise ValueError("at least one training transition is required")
    empirical = empirical_parameters(data)
    logger.info(
        "BFGS empirical start [tau_p, kappa_T, D_u, D_T, A_T_star]: %s",
        np.array2string(empirical, precision=4),
    )
    solver = optx.BFGS(rtol=1e-8, atol=1e-8)
    runs: list[OptimizationRun] = []
    for index, start in enumerate(_starts(empirical, seed, starts), start=1):
        logger.info("BFGS start %d/%d", index, starts)
        solution = optx.minimise(
            negative_log_likelihood,
            solver,
            jnp.asarray(start),
            args=data,
            max_steps=2_000,
            throw=False,
        )
        raw = np.asarray(solution.value)
        objective = float(
            negative_log_likelihood(jnp.asarray(raw), data)
        )
        gradient = np.asarray(
            jax.grad(negative_log_likelihood)(jnp.asarray(raw), data)
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
