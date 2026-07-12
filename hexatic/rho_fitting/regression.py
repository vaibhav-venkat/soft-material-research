"""Regression utilities for rho fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _rho_fitting_core, _rho_fitting_core_import_error
from .constants import REGRESSION_TOLERANCE


@dataclass(frozen=True)
class SparseRegressionResult:
    """Single-lambda constrained sparse-regression result and evaluation metrics."""

    names: tuple[str, ...]
    labels: tuple[str, ...]
    coefficients: np.ndarray
    importance: np.ndarray
    raw_correlations: np.ndarray
    importance_samples: np.ndarray
    lambda_values: np.ndarray
    active: np.ndarray
    lambda_index: int | None
    y_pred: np.ndarray
    rmse: float
    r2: float
    auxiliary_rmse: float | None = None
    auxiliary_r2: float | None = None
    solver_status: str = "unknown"
    solver_iterations: int = 0
    objective: float = float("nan")


def fit_sparse_regression(
    X: np.ndarray,
    y: np.ndarray,
    names: tuple[str, ...],
    labels: tuple[str, ...],
    *,
    seed: int,
    tau_count: int,
    tau_eps: float,
    subsamples: int,
    importance_threshold: float,
    alpha: float,
    max_iter: int,
    evaluation_X: np.ndarray | None = None,
    evaluation_y: np.ndarray | None = None,
    auxiliary_X: np.ndarray | None = None,
    auxiliary_y: np.ndarray | None = None,
    non_positive_names: tuple[str, ...] = (),
    non_negative_names: tuple[str, ...] = (),
) -> SparseRegressionResult:
    """Fit one column-normalized, sign-constrained L1 problem.

    Parameters:
        X: Fit matrix with shape ``(rows, terms)``.
        y: Fit target with shape ``(rows,)``.
        names: Machine-readable term names aligned to columns of ``X``.
        labels: Human-readable labels aligned to columns of ``X``.
        seed: Reserved for a future subsampled selection implementation.
        tau_count: Number of configured candidate regularization strengths.
        tau_eps: Multiplicative endpoint scale for candidate strengths.
        subsamples: Reserved for a future subsampled selection implementation.
        importance_threshold: Minimum normalized coefficient magnitude to mark a term active.
        alpha: Center scale used to select the configured L1 regularization strength.
        max_iter: Clarabel iteration limit.
        evaluation_X: Optional matrix used for reported RMSE/R^2 instead of the fit rows.
        evaluation_y: Optional target used with ``evaluation_X``.
        auxiliary_X: Optional secondary matrix for extra metrics, usually flux rows.
        auxiliary_y: Optional secondary target for ``auxiliary_X``.
        non_positive_names: Term names constrained to have non-positive coefficients.
        non_negative_names: Term names constrained to have non-negative coefficients.

    Returns:
        ``SparseRegressionResult`` containing coefficients, diagnostics, predictions,
        and primary/auxiliary metrics.

    Examples:
        ``fit = fit_sparse_regression(X_div, y_div, names, labels, seed=0, tau_count=40, tau_eps=1e-3, subsamples=200, importance_threshold=0.6, alpha=1e-6, max_iter=20)``

    Edge cases:
        ``seed`` and ``subsamples`` are currently accepted but unused; evaluation rows
        control reported metrics and may differ from the rows used to fit coefficients.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    _validate_fit_inputs(X, y, names, labels)
    evaluation_X, evaluation_y = _evaluation_rows(X, y, evaluation_X, evaluation_y)
    auxiliary_rows = _auxiliary_rows(X, auxiliary_X, auxiliary_y)

    del seed, subsamples

    candidates = regularization_candidates(float(alpha), int(tau_count), float(tau_eps))
    configured_index = max(0, candidates.size - 10)
    selected_lambda = float(candidates[configured_index])
    lambda_values = np.asarray([selected_lambda], dtype=np.float64)
    lambda_index = 0
    _progress(
        f"running Rust Clarabel constrained L1 regression rows={X.shape[0]} terms={X.shape[1]} "
        f"lambda={selected_lambda:.6g} configured_lambda_index={configured_index}"
    )
    non_positive_indices = _constraint_indices(names, non_positive_names)
    non_negative_indices = _constraint_indices(names, non_negative_names)
    if non_positive_indices or non_negative_indices:
        _progress(
            "coefficient sign constraints: "
            f"non_positive={_constraint_labels(names, non_positive_indices)} "
            f"non_negative={_constraint_labels(names, non_negative_indices)}"
        )
    coefficient_samples, importance_samples, solver_diagnostics = _fit_single_lambda(
        X,
        y,
        selected_lambda,
        max_iter,
        non_positive_indices,
        non_negative_indices,
    )
    coefficients = coefficient_samples[lambda_index]

    raw_correlations = _raw_feature_correlations(evaluation_X, evaluation_y)
    _progress("raw feature correlations with target")
    for label, correlation in zip(labels, raw_correlations, strict=True):
        _progress(f"  {label}: {_format_correlation(correlation)}")

    y_pred = evaluation_X @ coefficients
    rmse, r2 = _regression_metrics(evaluation_y, y_pred)
    auxiliary_rmse = None
    auxiliary_r2 = None
    if auxiliary_rows is not None:
        auxiliary_X, auxiliary_y = auxiliary_rows
        auxiliary_pred = auxiliary_X @ coefficients
        auxiliary_rmse, auxiliary_r2 = _regression_metrics(auxiliary_y, auxiliary_pred)
    importance = importance_samples[lambda_index]
    active = (np.abs(coefficients) > 1.0e-12) & (importance >= float(importance_threshold))

    return SparseRegressionResult(
        names,
        labels,
        coefficients,
        importance,
        raw_correlations,
        importance_samples,
        lambda_values,
        active,
        lambda_index,
        y_pred,
        rmse,
        r2,
        auxiliary_rmse,
        auxiliary_r2,
        solver_diagnostics[0],
        solver_diagnostics[1],
        solver_diagnostics[2],
    )


def _validate_fit_inputs(
    X: np.ndarray,
    y: np.ndarray,
    names: tuple[str, ...],
    labels: tuple[str, ...],
) -> None:
    """Validate regression matrix, target vector, and term metadata dimensions."""
    assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.size, "X must be (N, terms) and y must be (N,)"
    assert X.shape[1] == len(names) == len(labels), "term metadata must match X columns"
    assert X.shape[0] > 0 and X.shape[1] > 0, "regression matrix must be non-empty"


def _evaluation_rows(
    X: np.ndarray,
    y: np.ndarray,
    evaluation_X: np.ndarray | None,
    evaluation_y: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return metric-evaluation rows, defaulting to the fit rows when none are supplied."""
    if evaluation_X is None:
        evaluation_X = X
    if evaluation_y is None:
        evaluation_y = y
    evaluation_X = np.asarray(evaluation_X, dtype=np.float64)
    evaluation_y = np.asarray(evaluation_y, dtype=np.float64)
    assert evaluation_X.ndim == 2 and evaluation_y.ndim == 1, "evaluation rows must be matrix/vector"
    assert evaluation_X.shape[1] == X.shape[1] and evaluation_X.shape[0] == evaluation_y.size, (
        "evaluation rows must match fit terms"
    )
    return evaluation_X, evaluation_y


def _auxiliary_rows(
    X: np.ndarray,
    auxiliary_X: np.ndarray | None,
    auxiliary_y: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Validate and return optional secondary metric rows."""
    if auxiliary_X is None and auxiliary_y is None:
        return None
    assert auxiliary_X is not None and auxiliary_y is not None, "auxiliary rows need both X and y"
    auxiliary_X = np.asarray(auxiliary_X, dtype=np.float64)
    auxiliary_y = np.asarray(auxiliary_y, dtype=np.float64)
    assert auxiliary_X.ndim == 2 and auxiliary_y.ndim == 1, "auxiliary rows must be matrix/vector"
    assert auxiliary_X.shape[1] == X.shape[1] and auxiliary_X.shape[0] == auxiliary_y.size, (
        "auxiliary rows must match fit terms"
    )
    return auxiliary_X, auxiliary_y


def _fit_single_lambda(
    X: np.ndarray,
    y: np.ndarray,
    regularization: float,
    max_iter: int,
    non_positive_indices: tuple[int, ...],
    non_negative_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, tuple[str, int, float]]:
    """Fit constrained L1 coefficients at the selected regularization strength."""
    coefficient_samples = np.zeros((1, X.shape[1]), dtype=np.float64)
    importance_samples = np.zeros_like(coefficient_samples)
    _progress(f"  lambda: {regularization:.6g}")
    coefficients, diagnostics = _constrained_sr3_coefficients(
        X,
        y,
        reg_weight_lam=float(regularization),
        max_iter=max_iter,
        non_positive_indices=non_positive_indices,
        non_negative_indices=non_negative_indices,
    )
    coefficient_samples[0] = coefficients
    importance_samples[0] = _coefficient_importance(coefficients)
    return coefficient_samples, importance_samples, diagnostics


def _regression_metrics(y: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Return RMSE and coefficient of determination for one prediction vector."""
    residual = y - y_pred
    rmse = float(np.sqrt(np.mean(residual * residual)))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residual * residual)) / total if total > 0.0 else float("nan")
    return rmse, r2


def _progress(message: str) -> None:
    """Print a flushed rho-fitting progress message."""
    print(f"[rho_fitting] {message}", flush=True)


def _format_correlation(value: float) -> str:
    """Format finite correlations compactly and preserve non-finite values as ``nan``."""
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def _raw_feature_correlations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Pearson correlations between each raw feature column and the target."""
    y_centered = y - np.mean(y)
    y_norm = float(np.linalg.norm(y_centered))
    correlations = np.full(X.shape[1], np.nan, dtype=np.float64)
    for feature in range(X.shape[1]):
        x_centered = X[:, feature] - np.mean(X[:, feature])
        denominator = float(np.linalg.norm(x_centered) * y_norm)
        if denominator > 0.0:
            correlations[feature] = float(np.dot(x_centered, y_centered) / denominator)
    return correlations


def regularization_candidates(alpha: float, count: int = 40, eps: float = 1e-3) -> np.ndarray:
    """Return candidate L1 regularization strengths centered around ``alpha``.

    Parameters:
        alpha: Central regularization scale.
        count: Number of candidate values.
        eps: Multiplicative endpoint factor, producing ``alpha/eps`` down to ``alpha*eps``.

    Returns:
        One-dimensional descending array of candidate lambda values.
    """
    assert alpha > 0.0, "alpha must be positive"
    assert count > 0, "candidate count must be positive"
    assert eps > 0.0, "candidate endpoint scale must be positive"
    return alpha * np.logspace(np.log10(1.0 / eps), np.log10(eps), count)


def _constrained_sr3_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    *,
    reg_weight_lam: float,
    max_iter: int,
    non_positive_indices: tuple[int, ...],
    non_negative_indices: tuple[int, ...],
) -> tuple[np.ndarray, tuple[str, int, float]]:
    """Fit the current single-lambda constrained L1 problem in native Rust."""
    if _rho_fitting_core is None:
        raise ImportError(f"rho-fitting Rust core is unavailable: {_rho_fitting_core_import_error}")
    result = _rho_fitting_core.fit_constrained_lasso(
        np.ascontiguousarray(X, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        float(reg_weight_lam),
        REGRESSION_TOLERANCE,
        int(max_iter),
        np.asarray(non_positive_indices, dtype=np.int64),
        np.asarray(non_negative_indices, dtype=np.int64),
    )
    coefficients = np.asarray(result["coefficients"], dtype=np.float64)
    assert coefficients.shape == (X.shape[1],), "Rust solver returned an unexpected coefficient shape"
    diagnostics = (
        str(result["status"]),
        int(result["iterations"]),
        float(result["objective"]),
    )
    return coefficients, diagnostics


def _constraint_indices(names: tuple[str, ...], constrained_names: tuple[str, ...]) -> tuple[int, ...]:
    """Return coefficient indices whose names should receive inequality constraints."""
    constrained = set(constrained_names)
    return tuple(index for index, name in enumerate(names) if name in constrained)


def _constraint_labels(names: tuple[str, ...], indices: tuple[int, ...]) -> str:
    """Format constrained coefficient names for progress logs."""
    if not indices:
        return "none"
    return ",".join(names[index] for index in indices)


def _coefficient_importance(coefficients: np.ndarray) -> np.ndarray:
    """Normalize absolute coefficients by the largest magnitude in the vector."""
    magnitudes = np.abs(coefficients)
    maximum = float(np.max(magnitudes)) if magnitudes.size else 0.0
    if maximum == 0.0:
        return np.zeros_like(magnitudes)
    return magnitudes / maximum
