"""Regression utilities for rho fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _rho_fitting_core


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

    assert _rho_fitting_core is not None, "rho_fitting extension is not built"
    _progress(f"running Rust STLSQ stability selection rows={X.shape[0]} terms={X.shape[1]} subsamples={subsamples}")
    result = _rho_fitting_core.stability_selection(
        np.ascontiguousarray(X, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        int(seed),
        float(tau_eps),
        int(subsamples),
        float(importance_threshold),
        float(alpha),
        int(max_iter),
    )

    raw_correlations = np.asarray(result["raw_correlations"], dtype=np.float64)
    _progress("raw feature correlations with target")
    for label, correlation in zip(labels, raw_correlations, strict=True):
        _progress(f"  {label}: {_format_correlation(correlation)}")
    tau_index = int(result["tau_index"])
    if tau_index >= 0:
        tau_values = np.asarray(result["tau_values"], dtype=np.float64)
        _progress(f"stability tau {tau_index + 1}/{tau_values.size}: tau={tau_values[tau_index]:.6g}")
    return StabilityResult(
        names,
        labels,
        np.asarray(result["coefficients"], dtype=np.float64),
        np.asarray(result["importance"], dtype=np.float64),
        raw_correlations,
        np.asarray(result["importance_path"], dtype=np.float64),
        np.asarray(result["tau_values"], dtype=np.float64),
        np.asarray(result["active"], dtype=bool),
        None if tau_index < 0 else tau_index,
        np.asarray(result["y_pred"], dtype=np.float64),
        float(result["rmse"]),
        float(result["r2"]),
    )


def _progress(message: str) -> None:
    print(f"[rho_fitting] {message}", flush=True)


def _format_correlation(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6g}"
