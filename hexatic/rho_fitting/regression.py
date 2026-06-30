"""Regression utilities for rho fitting."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pysindy as ps


@dataclass(frozen=True)
class StabilityResult:
    names: tuple[str, ...]
    labels: tuple[str, ...]
    coefficients: np.ndarray
    importance: np.ndarray
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
    tau_count: int,
    tau_eps: float,
    subsamples: int,
    importance_threshold: float,
    alpha: float,
    max_iter: int,
) -> StabilityResult:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
        raise ValueError("X must be (N, terms) and y must be (N,)")
    if X.shape[1] != len(names) or len(names) != len(labels):
        raise ValueError("term metadata must match X columns")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("regression matrix must be non-empty")

    Xn, yn = _normalize_for_path(X, y)
    coef0 = _pysindy_coefficients(Xn, yn, threshold=0.0, alpha=alpha, max_iter=max_iter)
    tau_max = float(np.max(np.abs(coef0)) * 1.01) if coef0.size else 0.0
    tau_values = tau_path(tau_max, tau_count, tau_eps)
    if tau_max == 0.0:
        active = np.zeros(X.shape[1], dtype=bool)
        coefficients = np.zeros(X.shape[1], dtype=np.float64)
        y_pred = np.zeros_like(y)
        importance_path = np.zeros((tau_count, X.shape[1]))
        importance = np.zeros(X.shape[1], dtype=np.float64)
        return _result(
            names,
            labels,
            coefficients,
            importance,
            active,
            importance_path,
            tau_values,
            None,
            y,
            y_pred,
        )

    rng = np.random.default_rng(seed)
    importance_path = np.zeros((tau_values.size, X.shape[1]), dtype=np.float64)
    n_sub = max(1, X.shape[0] // 2)
    for tau_i, tau in enumerate(tau_values):
        _progress(f"stability tau {tau_i + 1}/{tau_values.size}: tau={tau:.6g}")
        kept = np.zeros(X.shape[1], dtype=np.float64)
        for sub_i in range(subsamples):
            if _should_report_subsample(sub_i, subsamples):
                _progress(f"  subsample {sub_i + 1}/{subsamples}")
            rows = rng.choice(X.shape[0], size=n_sub, replace=False)
            active_terms = _pysindy_active_terms(
                Xn[rows],
                yn[rows],
                threshold=float(tau),
                alpha=alpha,
                max_iter=max_iter,
            )
            kept += active_terms
        importance_path[tau_i] = kept / float(subsamples)

    tau_index = tau_values.size - 1
    importance = importance_path[tau_index]
    active = importance >= importance_threshold
    coefficients = final_refit(X, y, active, alpha)
    y_pred = X @ coefficients
    return _result(
        names,
        labels,
        coefficients,
        importance,
        active,
        importance_path,
        tau_values,
        tau_index,
        y,
        y_pred,
    )


def final_refit(X: np.ndarray, y: np.ndarray, active: np.ndarray, alpha: float) -> np.ndarray:
    coefficients = np.zeros(X.shape[1], dtype=np.float64)
    if np.any(active):
        coefficients[active] = _pysindy_coefficients(
            X[:, active],
            y,
            threshold=0.0,
            alpha=alpha,
            max_iter=1,
        )
    return coefficients


def tau_path(tau_max: float, count: int = 40, eps: float = 1e-2) -> np.ndarray:
    if tau_max < 0.0:
        raise ValueError("tau_max must be nonnegative")
    if count < 1 or eps <= 0.0:
        raise ValueError("invalid tau path settings")
    if tau_max == 0.0:
        return np.zeros(count, dtype=np.float64)
    return tau_max * np.logspace(0.0, np.log10(eps), count)


def _normalize_for_path(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_centered = X - np.mean(X, axis=0)
    scale = np.std(X_centered, axis=0)
    scale[scale == 0.0] = 1.0
    y_centered = y - np.mean(y)
    y_scale = float(np.std(y_centered))
    if y_scale == 0.0:
        y_scale = 1.0
    return X_centered / scale, y_centered / y_scale


def _pysindy_active_terms(
    X: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float,
    alpha: float,
    max_iter: int,
) -> np.ndarray:
    coefficients = _pysindy_coefficients(
        X,
        y,
        threshold=threshold,
        alpha=alpha,
        max_iter=max_iter,
    )
    return np.abs(coefficients) > 0.0


def _pysindy_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float,
    alpha: float,
    max_iter: int,
) -> np.ndarray:
    if X.shape[1] == 0:
        return np.empty(0, dtype=np.float64)
    optimizer = ps.STLSQ(
        threshold=threshold,
        alpha=alpha,
        max_iter=max_iter,
        normalize_columns=False,
        unbias=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparsity parameter is too big.*",
            category=UserWarning,
        )
        optimizer.fit(X, y)
    return np.asarray(optimizer.coef_, dtype=np.float64).reshape(-1)


def _choose_tau_index(importance_path: np.ndarray, threshold: float) -> int | None:
    hits = np.where(np.any(importance_path >= threshold, axis=1))[0]
    return int(hits[0]) if hits.size else None


def _result(
    names: tuple[str, ...],
    labels: tuple[str, ...],
    coefficients: np.ndarray,
    importance: np.ndarray,
    active: np.ndarray,
    importance_path: np.ndarray,
    tau_values: np.ndarray,
    tau_index: int | None,
    y: np.ndarray,
    y_pred: np.ndarray,
) -> StabilityResult:
    residual = y - y_pred
    rmse = float(np.sqrt(np.mean(residual**2)))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denom if denom > 0.0 else np.nan
    return StabilityResult(
        names=names,
        labels=labels,
        coefficients=coefficients,
        importance=importance,
        importance_path=importance_path,
        tau_values=tau_values,
        active=active,
        tau_index=tau_index,
        y_pred=y_pred,
        rmse=rmse,
        r2=r2,
    )


def _should_report_subsample(index: int, total: int) -> bool:
    if total <= 4:
        return True
    interval = max(1, total // 4)
    return index == 0 or index + 1 == total or (index + 1) % interval == 0


def _progress(message: str) -> None:
    print(f"[rho_fitting] {message}", flush=True)
