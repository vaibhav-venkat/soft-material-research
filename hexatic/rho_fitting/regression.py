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


def stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    names: tuple[str, ...],
    labels: tuple[str, ...],
    *,
    seed: int,
    tau_eps: float,
    subsamples: int,
    importance_threshold: float,
    alpha: float,
    max_iter: int,
) -> StabilityResult:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.size, "X must be (N, terms) and y must be (N,)"
    assert X.shape[1] == len(names) == len(labels), "term metadata must match X columns"
    assert X.shape[0] > 0 and X.shape[1] > 0, "regression matrix must be non-empty"

    del seed, tau_eps, subsamples

    _progress(f"running PySINDy SR3 L1 regression rows={X.shape[0]} terms={X.shape[1]}")
    optimizer = ps.SR3(
        reg_weight_lam=float(alpha),
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

    raw_correlations = _raw_feature_correlations(X, y)
    _progress("raw feature correlations with target")
    for label, correlation in zip(labels, raw_correlations, strict=True):
        _progress(f"  {label}: {_format_correlation(correlation)}")

    y_pred = X @ coefficients
    residual = y - y_pred
    rmse = float(np.sqrt(np.mean(residual * residual)))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residual * residual)) / total if total > 0.0 else float("nan")
    importance = _coefficient_importance(coefficients)
    active = (np.abs(coefficients) > 1.0e-12) & (importance >= float(importance_threshold))

    return StabilityResult(
        names,
        labels,
        coefficients,
        importance,
        raw_correlations,
        importance.reshape(1, -1),
        np.asarray([], dtype=np.float64),
        active,
        None,
        y_pred,
        rmse,
        r2,
    )


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


def _coefficient_importance(coefficients: np.ndarray) -> np.ndarray:
    magnitudes = np.abs(coefficients)
    maximum = float(np.max(magnitudes)) if magnitudes.size else 0.0
    if maximum == 0.0:
        return np.zeros_like(magnitudes)
    return magnitudes / maximum
