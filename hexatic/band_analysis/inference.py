from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from .model import (
    PARAMETER_NAMES,
    TransitionBlock,
    negative_log_likelihood,
    positive_parameters,
    raw_parameters,
)


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


def empirical_parameters(blocks: tuple[TransitionBlock, ...]) -> np.ndarray:
    """Training-only OU and projected-increment starting estimates."""
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

    conservative_x: list[np.ndarray] = []
    conservative_y: list[np.ndarray] = []
    conservative_dt: list[np.ndarray] = []
    conservative_weight: list[np.ndarray] = []
    for block in blocks:
        current = np.asarray(block.current)
        following = np.asarray(block.following)
        n_bands = current.shape[1]
        if n_bands == 1:
            continue
        projection = np.eye(n_bands) - np.ones((n_bands, n_bands)) / n_bands
        allocation = current / current.sum(axis=1, keepdims=True)
        total_increment = (following - current).sum(axis=1, keepdims=True)
        exchange = (following - current - allocation * total_increment) @ projection
        conservative_x.append(current @ projection)
        conservative_y.append(exchange)
        conservative_dt.append(np.asarray(block.dt))
        conservative_weight.append(np.asarray(block.weight))

    if conservative_x:
        numerator = 0.0
        denominator = 0.0
        for area, increment, block_dt, block_weight in zip(
            conservative_x,
            conservative_y,
            conservative_dt,
            conservative_weight,
            strict=True,
        ):
            numerator += np.sum(block_weight[:, None] * area * increment / block_dt[:, None])
            denominator += np.sum(block_weight[:, None] * area**2)
        conservative_rate = max(1e-6, float(-numerator / max(denominator, 1e-12)))
        diffusion_numerator = 0.0
        diffusion_denominator = 0.0
        for area, increment, block_dt, block_weight in zip(
            conservative_x,
            conservative_y,
            conservative_dt,
            conservative_weight,
            strict=True,
        ):
            residual = increment + conservative_rate * area * block_dt[:, None]
            diffusion_numerator += np.sum(
                block_weight * np.sum(residual**2, axis=1) / (2.0 * block_dt)
            )
            diffusion_denominator += np.sum(block_weight * (area.shape[1] - 1))
        diffusion_area = max(
            1e-8, float(diffusion_numerator / diffusion_denominator)
        )
    else:
        conservative_rate = total_rate
        diffusion_area = diffusion_total

    return np.asarray(
        [
            conservative_rate,
            total_rate,
            diffusion_area,
            diffusion_total,
            max(area_star, 1e-6),
        ],
        dtype=np.float64,
    )


def _hessian_diagnostics(
    raw: np.ndarray, blocks: tuple[TransitionBlock, ...]
) -> HessianDiagnostics:
    matrix = np.asarray(jax.hessian(negative_log_likelihood)(jnp.asarray(raw), blocks))
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


def _starts(empirical: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    empirical_raw = raw_parameters(empirical)
    generator = np.random.default_rng(seed)
    perturbed = tuple(
        empirical_raw + generator.normal(0.0, 0.75, len(PARAMETER_NAMES))
        for _ in range(6)
    )
    generic = raw_parameters(np.asarray([1.0, 1.0, 0.1, 0.1, 1.0]))
    return (empirical_raw, *perturbed, generic)


def optimize_parameters(
    blocks: tuple[TransitionBlock, ...], *, seed: int = 0
) -> OptimizationResult:
    """Run exactly eight reproducible raw-space Optimistix BFGS fits."""
    if not blocks:
        raise ValueError("at least one training transition is required")
    empirical = empirical_parameters(blocks)
    solver = optx.BFGS(rtol=1e-8, atol=1e-8)
    runs: list[OptimizationRun] = []
    for start in _starts(empirical, seed):
        solution = optx.minimise(
            negative_log_likelihood,
            solver,
            jnp.asarray(start),
            args=blocks,
            max_steps=2_000,
            throw=False,
        )
        raw = np.asarray(solution.value)
        objective = float(negative_log_likelihood(jnp.asarray(raw), blocks))
        gradient = np.asarray(jax.grad(negative_log_likelihood)(jnp.asarray(raw), blocks))
        finite = bool(
            np.isfinite(objective)
            and np.all(np.isfinite(raw))
            and np.all(np.isfinite(gradient))
        )
        runs.append(
            OptimizationRun(
                raw_parameters=raw,
                parameters=np.asarray(positive_parameters(jnp.asarray(raw))),
                objective=objective,
                gradient_norm=float(np.linalg.norm(gradient)),
                result=str(solution.result),
                hessian=_hessian_diagnostics(raw, blocks) if finite else None,
            )
        )
    finite_runs = [run for run in runs if np.isfinite(run.objective)]
    if not finite_runs:
        raise RuntimeError("all eight BFGS starts produced nonfinite objectives")
    return OptimizationResult(
        best=min(finite_runs, key=lambda run: run.objective),
        runs=tuple(runs),
        empirical_parameters=empirical,
    )
