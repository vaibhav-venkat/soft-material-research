"""Regression utilities for rho fitting."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pysindy as ps
from sklearn.exceptions import ConvergenceWarning


@dataclass(frozen=True)
class StabilityResult:
    """Sparse-regression result with coefficient path diagnostics and evaluation metrics."""

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
    non_positive_names: tuple[str, ...] = (),
    non_negative_names: tuple[str, ...] = (),
) -> StabilityResult:
    """Fit a constrained SR3 path and select active terms by relative coefficient size.

    Parameters:
        X: Fit matrix with shape ``(rows, terms)``.
        y: Fit target with shape ``(rows,)``.
        names: Machine-readable term names aligned to columns of ``X``.
        labels: Human-readable labels aligned to columns of ``X``.
        seed: Reserved for compatibility with earlier subsampled stability selection.
        tau_count: Number of regularization strengths to scan.
        tau_eps: Log-path endpoint scale around ``alpha``.
        subsamples: Reserved for compatibility with earlier subsampled stability selection.
        importance_threshold: Minimum normalized coefficient magnitude to mark a term active.
        alpha: Center scale for the SR3 regularization path.
        max_iter: CVXPY/SR3 iteration limit.
        evaluation_X: Optional matrix used for reported RMSE/R^2 instead of the fit rows.
        evaluation_y: Optional target used with ``evaluation_X``.
        auxiliary_X: Optional secondary matrix for extra metrics, usually flux rows.
        auxiliary_y: Optional secondary target for ``auxiliary_X``.
        non_positive_names: Term names constrained to have non-positive coefficients.
        non_negative_names: Term names constrained to have non-negative coefficients.

    Returns:
        ``StabilityResult`` containing selected coefficients, path diagnostics, predictions,
        and primary/auxiliary metrics.

    Examples:
        ``fit = stability_selection(X_div, y_div, names, labels, seed=0, tau_count=40, tau_eps=1e-3, subsamples=200, importance_threshold=0.6, alpha=1e-6, max_iter=20)``

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

    full_tau_values = tau_path(float(alpha), int(tau_count), float(tau_eps))
    selected_tau_index = max(0, full_tau_values.size - 10)
    selected_tau = float(full_tau_values[selected_tau_index])
    tau_values = np.asarray([selected_tau], dtype=np.float64)
    tau_index = 0
    _progress(
        f"running PySINDy ConstrainedSR3 L1 regression rows={X.shape[0]} terms={X.shape[1]} "
        f"tau={selected_tau:.6g} original_tau_index={selected_tau_index}"
    )
    non_positive_indices = _constraint_indices(names, non_positive_names)
    non_negative_indices = _constraint_indices(names, non_negative_names)
    if non_positive_indices or non_negative_indices:
        _progress(
            "coefficient sign constraints: "
            f"non_positive={_constraint_labels(names, non_positive_indices)} "
            f"non_negative={_constraint_labels(names, non_negative_indices)}"
        )
    coefficients_path, importance_path = _fit_single_tau(
        X,
        y,
        selected_tau,
        max_iter,
        non_positive_indices,
        non_negative_indices,
    )
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


def _fit_single_tau(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    max_iter: int,
    non_positive_indices: tuple[int, ...],
    non_negative_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit constrained SR3 coefficients at one selected tau value."""
    coefficients_path = np.zeros((1, X.shape[1]), dtype=np.float64)
    importance_path = np.zeros_like(coefficients_path)
    _progress(f"  tau 1/1: {tau:.6g}")
    coefficients_path[0] = _constrained_sr3_coefficients(
        X,
        y,
        reg_weight_lam=float(tau),
        max_iter=max_iter,
        non_positive_indices=non_positive_indices,
        non_negative_indices=non_negative_indices,
    )
    importance_path[0] = _coefficient_importance(coefficients_path[0])
    return coefficients_path, importance_path


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


def tau_path(alpha: float, count: int = 40, eps: float = 1e-3) -> np.ndarray:
    """Return a log-spaced SR3 regularization path centered around ``alpha``.

    Parameters:
        alpha: Central regularization scale.
        count: Number of path values.
        eps: Multiplicative endpoint factor, producing ``alpha/eps`` down to ``alpha*eps``.

    Returns:
        One-dimensional descending path of tau values.
    """
    assert alpha > 0.0, "alpha must be positive"
    assert count > 0, "tau count must be positive"
    assert eps > 0.0, "tau eps must be positive"
    return alpha * np.logspace(np.log10(1.0 / eps), np.log10(eps), count)


def _constrained_sr3_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    *,
    reg_weight_lam: float,
    max_iter: int,
    non_positive_indices: tuple[int, ...],
    non_negative_indices: tuple[int, ...],
) -> np.ndarray:
    """Fit one PySINDy ConstrainedSR3 model and return a flat coefficient vector."""
    constraint_lhs = _sign_constraint_lhs(X.shape[1], 1, non_positive_indices, non_negative_indices)
    constraint_rhs = np.zeros(constraint_lhs.shape[0], dtype=np.float64) if constraint_lhs is not None else None
    optimizer = _ConstrainedSR3(
        reg_weight_lam=float(reg_weight_lam),
        regularizer="L1",
        max_iter=int(max_iter),
        normalize_columns=True,
        unbias=False,
        constraint_lhs=constraint_lhs,
        constraint_rhs=constraint_rhs,
        constraint_order="feature",
        inequality_constraints=constraint_lhs is not None,
    )
    optimizer.fit_intercept = False
    assert optimizer.fit_intercept is False, "PySINDy ConstrainedSR3 must not fit an intercept"
    optimizer.fit(np.ascontiguousarray(X, dtype=np.float64), np.ascontiguousarray(y, dtype=np.float64))
    coefficients = np.asarray(optimizer.coef_, dtype=np.float64).reshape(-1)
    assert coefficients.shape == (X.shape[1],), "PySINDy returned an unexpected coefficient shape"
    return coefficients


def _constraint_indices(names: tuple[str, ...], constrained_names: tuple[str, ...]) -> tuple[int, ...]:
    """Return coefficient indices whose names should receive inequality constraints."""
    constrained = set(constrained_names)
    return tuple(index for index, name in enumerate(names) if name in constrained)


def _constraint_labels(names: tuple[str, ...], indices: tuple[int, ...]) -> str:
    """Format constrained coefficient names for progress logs."""
    if not indices:
        return "none"
    return ",".join(names[index] for index in indices)


class _ConstrainedSR3(ps.ConstrainedSR3):
    """ConstrainedSR3 variant that tries installed CVXPY solvers and falls back to zeros."""

    def _update_coef_cvxpy(self, xi, cost, var_len, coef_prev, tol):  # type: ignore[no-untyped-def]
        """Solve the SR3 CVXPY subproblem with local solver-specific options."""
        import cvxpy as cp

        if self.use_constraints:
            assert self.constraint_lhs is not None
            assert self.constraint_rhs is not None
            assert self.constraint_separation_index is not None
            constraints = []
            if self.equality_constraints:
                constraints.append(
                    self.constraint_lhs[self.constraint_separation_index :, :] @ xi
                    == self.constraint_rhs[self.constraint_separation_index :],
                )
            if self.inequality_constraints:
                constraints.append(
                    self.constraint_lhs[: self.constraint_separation_index, :] @ xi
                    <= self.constraint_rhs[: self.constraint_separation_index]
                )
            problem = cp.Problem(cp.Minimize(cost), constraints)
        else:
            problem = cp.Problem(cp.Minimize(cost))

        solvers = tuple(solver for solver in ("CLARABEL", "SCS") if solver in cp.installed_solvers())
        for solver in solvers:
            try:
                problem.solve(
                    solver=solver,
                    verbose=self.verbose_cvxpy,
                    **_cvxpy_solver_options(solver, self.max_iter, tol),
                )
                if xi.value is not None:
                    return np.asarray(xi.value, dtype=np.float64).reshape(coef_prev.shape)
            except cp.error.SolverError:
                continue
        warnings.warn(
            "ConstrainedSR3 CVXPY solve failed or was infeasible; setting coefs to zeros",
            ConvergenceWarning,
        )
        return np.zeros((var_len,), dtype=np.float64).reshape(coef_prev.shape)


def _cvxpy_solver_options(solver: str, max_iter: int, tol: float) -> dict[str, float | int]:
    """Return option names expected by the selected CVXPY solver."""
    if solver == "CLARABEL":
        return {
            "max_iter": int(max_iter),
            "tol_gap_abs": float(tol),
            "tol_gap_rel": float(tol),
            "tol_feas": float(tol),
        }
    if solver == "SCS":
        return {
            "max_iters": int(max_iter),
            "eps_abs": float(tol),
            "eps_rel": float(tol),
        }
    return {}


def _sign_constraint_lhs(
    n_features: int,
    n_targets: int,
    non_positive_indices: tuple[int, ...],
    non_negative_indices: tuple[int, ...],
) -> np.ndarray | None:
    """Build inequality rows enforcing selected coefficient signs independently."""
    rows: list[np.ndarray] = []
    for index in non_positive_indices:
        rows.extend(_constraint_rows(n_features, n_targets, index, sign=1.0))
    for index in non_negative_indices:
        rows.extend(_constraint_rows(n_features, n_targets, index, sign=-1.0))
    if not rows:
        return None
    return np.vstack(rows)


def _constraint_rows(
    n_features: int,
    n_targets: int,
    index: int,
    *,
    sign: float,
) -> list[np.ndarray]:
    """Return one sign-constraint row per target component for a feature."""
    rows = []
    offset = index * n_targets
    for target in range(n_targets):
        row = np.zeros(n_features * n_targets, dtype=np.float64)
        row[offset + target] = float(sign)
        rows.append(row)
    return rows


def _coefficient_importance(coefficients: np.ndarray) -> np.ndarray:
    """Normalize absolute coefficients by the largest magnitude in the vector."""
    magnitudes = np.abs(coefficients)
    maximum = float(np.max(magnitudes)) if magnitudes.size else 0.0
    if maximum == 0.0:
        return np.zeros_like(magnitudes)
    return magnitudes / maximum
