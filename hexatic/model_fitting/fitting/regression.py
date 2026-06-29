"""Regression utilities for hydrodynamic model fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pysindy as ps

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

    coef_n = _pysindy_coefficients(
        Xn,
        y,
        ridge_alpha=ridge_alpha,
        stlsq_threshold=stlsq_threshold,
        stlsq_max_iter=stlsq_max_iter,
    )
    coef_n[~nonzero] = 0.0
    active = nonzero & (np.abs(coef_n) >= stlsq_threshold)

    coefficients = coef_n / scales
    coefficients[~np.isfinite(coefficients)] = 0.0
    return coefficients, scales, active, int(X.shape[0])


def _pysindy_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ridge_alpha: float,
    stlsq_threshold: float,
    stlsq_max_iter: int,
) -> np.ndarray:
    optimizer = ps.STLSQ(
        threshold=stlsq_threshold,
        alpha=ridge_alpha,
        max_iter=stlsq_max_iter,
        normalize_columns=False,
    )
    # optimizer = ps.SR3(
    #     reg_weight_lam=stlsq_threshold,
    #     regularizer="L0",
    #     max_iter=stlsq_max_iter,
    #     normalize_columns=False,
    # )
    optimizer.fit(X, y)
    coefficients = np.ravel(np.asarray(optimizer.coef_, dtype=float))
    if coefficients.size != X.shape[1]:
        raise ValueError("PySINDy optimizer returned unexpected coefficient shape.")
    coefficients[~np.isfinite(coefficients)] = 0.0
    return coefficients


def _scalar_metrics(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.isfinite(target) & np.isfinite(prediction)
    return {
        "correlation": _correlation(target[valid], prediction[valid]),
        "r2": _r2(target[valid], prediction[valid]),
        "mae": _mae(target[valid], prediction[valid]),
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


def _mae(target: np.ndarray, prediction: np.ndarray) -> float:
    if target.size == 0:
        return float("nan")
    return float(np.mean(np.abs(target - prediction)))


def _normalized_mae(target: np.ndarray, prediction: np.ndarray) -> float:
    if target.size == 0:
        return float("nan")
    scale = float(np.mean(np.abs(target)))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    return float(np.mean(np.abs(target - prediction)) / scale)
