"""Regression utilities for rho fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pysindy as ps


@dataclass(frozen=True)
class StabilityResult:
    names: tuple[str, ...]
    labels: tuple[str, ...]
    coefficients: np.ndarray
    importance: np.ndarray
    raw_correlations: np.ndarray
    importance_path: np.ndarray
    tau_values: np.ndarray
    active: np.ndarray
    tau_index: int | None
    y_pred: np.ndarray
    rmse: float
    r2: float
    auxiliary_rmse: float | None = None
    auxiliary_r2: float | None = None


def stability_selection(
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
) -> StabilityResult:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    _validate_fit_inputs(X, y, names, labels)
    evaluation_X, evaluation_y = _evaluation_rows(X, y, evaluation_X, evaluation_y)
    auxiliary_rows = _auxiliary_rows(X, auxiliary_X, auxiliary_y)

    del seed, subsamples

    tau_values = tau_path(float(alpha), int(tau_count), float(tau_eps))
    tau_index = max(0, tau_values.size - 10)
    _progress(
        f"running PySINDy SR3 L1 regression rows={X.shape[0]} terms={X.shape[1]} "
        f"tau_count={tau_values.size}"
    )
    coefficients_path, importance_path = _fit_tau_path(X, y, tau_values, max_iter)
    coefficients = coefficients_path[tau_index]

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
    importance = importance_path[tau_index]
    active = (np.abs(coefficients) > 1.0e-12) & (importance >= float(importance_threshold))

    return StabilityResult(
        names,
        labels,
        coefficients,
        importance,
        raw_correlations,
        importance_path,
        tau_values,
        active,
        tau_index,
        y_pred,
        rmse,
        r2,
        auxiliary_rmse,
        auxiliary_r2,
    )


def _validate_fit_inputs(
    X: np.ndarray,
    y: np.ndarray,
    names: tuple[str, ...],
    labels: tuple[str, ...],
) -> None:
    assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.size, "X must be (N, terms) and y must be (N,)"
    assert X.shape[1] == len(names) == len(labels), "term metadata must match X columns"
    assert X.shape[0] > 0 and X.shape[1] > 0, "regression matrix must be non-empty"


def _evaluation_rows(
    X: np.ndarray,
    y: np.ndarray,
    evaluation_X: np.ndarray | None,
    evaluation_y: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
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


def _fit_tau_path(
    X: np.ndarray,
    y: np.ndarray,
    tau_values: np.ndarray,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients_path = np.zeros((tau_values.size, X.shape[1]), dtype=np.float64)
    importance_path = np.zeros_like(coefficients_path)
    for index, tau in enumerate(tau_values):
        _progress(f"  tau {index + 1}/{tau_values.size}: {tau:.6g}")
        coefficients_path[index] = _sr3_coefficients(X, y, reg_weight_lam=float(tau), max_iter=max_iter)
        importance_path[index] = _coefficient_importance(coefficients_path[index])
    return coefficients_path, importance_path


def _regression_metrics(y: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    residual = y - y_pred
    rmse = float(np.sqrt(np.mean(residual * residual)))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residual * residual)) / total if total > 0.0 else float("nan")
    return rmse, r2


def _progress(message: str) -> None:
    print(f"[rho_fitting] {message}", flush=True)


def _format_correlation(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def _raw_feature_correlations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    y_centered = y - np.mean(y)
    y_norm = float(np.linalg.norm(y_centered))
    correlations = np.full(X.shape[1], np.nan, dtype=np.float64)
    for feature in range(X.shape[1]):
        x_centered = X[:, feature] - np.mean(X[:, feature])
        denominator = float(np.linalg.norm(x_centered) * y_norm)
        if denominator > 0.0:
            correlations[feature] = float(np.dot(x_centered, y_centered) / denominator)
    return correlations


def tau_path(alpha: float, count: int = 40, eps: float = 1e-3) -> np.ndarray:
    assert alpha > 0.0, "alpha must be positive"
    assert count > 0, "tau count must be positive"
    assert eps > 0.0, "tau eps must be positive"
    return alpha * np.logspace(np.log10(1.0 / eps), np.log10(eps), count)


def _sr3_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    *,
    reg_weight_lam: float,
    max_iter: int,
) -> np.ndarray:
    optimizer = ps.SR3(
        reg_weight_lam=float(reg_weight_lam),
        regularizer="L1",
        max_iter=int(max_iter),
        normalize_columns=True,
        unbias=False,
    )
    optimizer.fit_intercept = False
    assert optimizer.fit_intercept is False, "PySINDy SR3 must not fit an intercept"
    optimizer.fit(np.ascontiguousarray(X, dtype=np.float64), np.ascontiguousarray(y, dtype=np.float64))
    coefficients = np.asarray(optimizer.coef_, dtype=np.float64).reshape(-1)
    assert coefficients.shape == (X.shape[1],), "PySINDy returned an unexpected coefficient shape"
    return coefficients


def _coefficient_importance(coefficients: np.ndarray) -> np.ndarray:
    magnitudes = np.abs(coefficients)
    maximum = float(np.max(magnitudes)) if magnitudes.size else 0.0
    if maximum == 0.0:
        return np.zeros_like(magnitudes)
    return magnitudes / maximum
