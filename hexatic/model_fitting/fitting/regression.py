"""Regression utilities for hydrodynamic model fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from .library import ScalarLibrary, VectorLibrary


@dataclass(frozen=True)
class RegressionResult:
    names: tuple[str, ...]
    labels: tuple[str, ...]
    coefficients: np.ndarray
    prediction: np.ndarray
    residual: np.ndarray
    metrics: dict[str, float]
    scales: np.ndarray
    active: np.ndarray
    rows_used: int


def fit_scalar_library(
    library: ScalarLibrary,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    ridge_alpha: float,
    stlsq_threshold: float,
    stlsq_max_iter: int,
) -> RegressionResult:
    design = np.asarray(library.values, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    assert design.shape[:-1] == target.shape == mask.shape

    coefficients, scales, active, rows_used = _fit_rows(
        design[mask].reshape(-1, design.shape[-1]),
        target[mask].reshape(-1),
        ridge_alpha=ridge_alpha,
        stlsq_threshold=stlsq_threshold,
        stlsq_max_iter=stlsq_max_iter,
    )
    prediction = np.tensordot(design, coefficients, axes=([-1], [0]))
    residual = target - prediction
    metrics = _scalar_metrics(target, prediction, mask)
    return RegressionResult(
        library.names, library.labels, coefficients, prediction, residual,
        metrics, scales, active, rows_used,
    )


def fit_vector_library(
    library: VectorLibrary,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    ridge_alpha: float,
    stlsq_threshold: float,
    stlsq_max_iter: int,
) -> RegressionResult:
    features = np.asarray(library.values, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    assert features.shape[:-2] == target.shape[:-1] == mask.shape
    assert features.shape[-1] == target.shape[-1] == 2

    valid_features = features[mask]  # (samples, terms, 2)
    valid_target = target[mask]      # (samples, 2)
    rows = valid_features.transpose(0, 2, 1).reshape(-1, features.shape[-2])
    measured = valid_target.reshape(-1)

    coefficients, scales, active, rows_used = _fit_rows(
        rows, measured,
        ridge_alpha=ridge_alpha,
        stlsq_threshold=stlsq_threshold,
        stlsq_max_iter=stlsq_max_iter,
    )
    prediction = np.einsum("...tc,t->...c", features, coefficients)
    residual = target - prediction
    metrics = _vector_metrics(target, prediction, mask)
    return RegressionResult(
        library.names, library.labels, coefficients, prediction, residual,
        metrics, scales, active, rows_used,
    )


def _fit_rows(
    design: np.ndarray,
    target: np.ndarray,
    *,
    ridge_alpha: float,
    stlsq_threshold: float,
    stlsq_max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    n_features = design.shape[1] if design.ndim == 2 else 0
    coefficients = np.zeros(n_features, dtype=float)
    scales = np.ones(n_features, dtype=float)
    active = np.zeros(n_features, dtype=bool)
    if design.ndim != 2 or target.ndim != 1 or target.shape[0] != design.shape[0]:
        raise ValueError("design/target shape mismatch")
    if n_features == 0 or design.shape[0] == 0:
        return coefficients, scales, active, 0

    finite = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
    X = design[finite]
    y = target[finite]
    if X.shape[0] == 0:
        return coefficients, scales, active, 0

    rms = np.sqrt(np.mean(X * X, axis=0))
    good_scale = np.isfinite(rms) & (rms > 0.0)
    scales[good_scale] = rms[good_scale]
    Xn = X / scales[None, :]
    nonzero = np.any(np.abs(Xn) > 0.0, axis=0)
    if not np.any(nonzero):
        return coefficients, scales, active, int(X.shape[0])

    coef_n = _ridge(Xn, y, nonzero, ridge_alpha)
    active = nonzero & (np.abs(coef_n) >= stlsq_threshold)
    if not np.any(active):
        return coefficients, scales, active, int(X.shape[0])

    for _ in range(stlsq_max_iter):
        next_coef_n = _ridge(Xn, y, active, ridge_alpha)
        next_active = nonzero & (np.abs(next_coef_n) >= stlsq_threshold)
        if np.array_equal(next_active, active):
            coef_n = next_coef_n
            break
        coef_n = next_coef_n
        active = next_active
        if not np.any(active):
            coef_n[:] = 0.0
            break

    coefficients = coef_n / scales
    coefficients[~np.isfinite(coefficients)] = 0.0
    return coefficients, scales, active, int(X.shape[0])


def _ridge(X: np.ndarray, y: np.ndarray, active: np.ndarray, alpha: float) -> np.ndarray:
    coef = np.zeros(X.shape[1], dtype=float)
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return coef
    Xa = X[:, idx]
    lhs = Xa.T @ Xa
    if alpha > 0.0:
        lhs = lhs + float(alpha) * np.eye(idx.size)
    rhs = Xa.T @ y
    try:
        coef[idx] = linalg.solve(lhs, rhs, assume_a="pos", check_finite=False)
    except linalg.LinAlgError:
        coef[idx], *_ = linalg.lstsq(Xa, y, check_finite=False)
    return coef


def _scalar_metrics(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.isfinite(target) & np.isfinite(prediction)
    return {
        "correlation": _correlation(target[valid], prediction[valid]),
        "r2": _r2(target[valid], prediction[valid]),
        "normalized_mae": _normalized_mae(target[valid], prediction[valid]),
    }


def _vector_metrics(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.all(np.isfinite(target), axis=-1) & np.all(np.isfinite(prediction), axis=-1)
    tx = target[..., 0][valid]
    ty = target[..., 1][valid]
    px = prediction[..., 0][valid]
    py = prediction[..., 1][valid]
    return {
        "correlation": _correlation(np.concatenate((tx, ty)), np.concatenate((px, py))),
        "r2_x": _r2(tx, px),
        "r2_y": _r2(ty, py),
        "normalized_mae_x": _normalized_mae(tx, px),
        "normalized_mae_y": _normalized_mae(ty, py),
    }


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    if target.size == 0:
        return float("nan")
    ss_res = float(np.sum((target - prediction) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _normalized_mae(target: np.ndarray, prediction: np.ndarray) -> float:
    if target.size == 0:
        return float("nan")
    scale = float(np.mean(np.abs(target)))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    return float(np.mean(np.abs(target - prediction)) / scale)
