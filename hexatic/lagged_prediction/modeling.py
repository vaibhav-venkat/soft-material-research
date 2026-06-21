from typing import Dict
from pathlib import Path
import warnings

import numpy as np

from hexatic.constants import cylinder

from .features import (
    load_active_families,
    load_ccm_family,
    load_hexatic_families,
    load_stress_families,
)
from .types import (
    CYLINDER_PATHS,
    LaggedPredictionConfig,
    LaggedPredictionResult, FeatureFamily,
)


def sklearn_imports():
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import ElasticNetCV
        from sklearn.metrics import r2_score
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "Lagged predictive decomposition requires scikit-learn. "
            "Install scikit-learn in the active environment and rerun."
        ) from exc
    return ConvergenceWarning, ElasticNetCV, r2_score, TimeSeriesSplit, make_pipeline, StandardScaler


def align_families(
    families,
    vx_steps: np.ndarray,
    vx_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    common = set(int(step) for step in vx_steps)
    for family in families.values():
        common &= set(int(step) for step in family.steps)
    common_steps = np.asarray(sorted(common), dtype=np.int64)
    if common_steps.size < 12:
        raise ValueError(
            "Too few common simulation steps across saved observables for lagged "
            f"prediction: {common_steps.size}."
        )

    vx_by_step = {int(step): float(value) for step, value in zip(vx_steps, vx_values)}
    y = np.asarray([vx_by_step[int(step)] for step in common_steps], dtype=np.float64)

    columns = []
    feature_names: list[str] = []
    feature_families: list[str] = []
    for family_name, family in families.items():
        index_by_step = {int(step): idx for idx, step in enumerate(family.steps)}
        values = np.vstack(
            [family.values[index_by_step[int(step)]] for step in common_steps]
        )
        columns.append(values)
        feature_names.extend(family.feature_names)
        feature_families.extend([family_name] * len(family.feature_names))

    x = np.column_stack(columns)
    finite_feature = np.isfinite(x)
    column_means = np.nanmean(np.where(finite_feature, x, np.nan), axis=0)
    column_means = np.where(np.isfinite(column_means), column_means, 0.0)
    x = np.where(finite_feature, x, column_means)

    finite_rows = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    return (
        common_steps[finite_rows],
        x[finite_rows],
        y[finite_rows],
        np.asarray(feature_names),
        np.asarray(feature_families),
    )


def time_series_splitter(n_samples: int, preferred_splits: int):
    _, _, _, TimeSeriesSplit, _, _ = sklearn_imports()
    n_splits = min(preferred_splits, n_samples - 2)
    if n_splits < 2:
        return None
    return TimeSeriesSplit(n_splits=n_splits)


def split_indices(splitter, x: np.ndarray):
    if splitter is None:
        n_samples = x.shape[0]
        return [(np.arange(max(1, n_samples - 1)), np.asarray([n_samples - 1]))]
    return list(splitter.split(x))


def fit_predict_elastic_net(
    x: np.ndarray,
    y: np.ndarray,
    config: LaggedPredictionConfig,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    (
        ConvergenceWarning,
        ElasticNetCV,
        r2_score,
        _TimeSeriesSplit,
        make_pipeline,
        StandardScaler,
    ) = sklearn_imports()
    n_samples, n_features = x.shape
    predictions = np.full(n_samples, np.nan, dtype=np.float64)

    if n_features == 0:
        splitter = time_series_splitter(n_samples, config.cv_splits)
        if splitter is None:
            predictions[:] = np.nanmean(y)
        else:
            for train_idx, test_idx in splitter.split(x):
                predictions[test_idx] = np.mean(y[train_idx])
            predictions[~np.isfinite(predictions)] = np.nanmean(y)
        rmse = float(np.sqrt(np.mean((y - predictions) ** 2)))
        return predictions, np.empty(0, dtype=np.float64), float(r2_score(y, predictions)), rmse

    splitter = time_series_splitter(n_samples, config.cv_splits)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for train_idx, test_idx in split_indices(splitter, x):
            inner = time_series_splitter(len(train_idx), config.inner_cv_splits)
            model = make_pipeline(
                StandardScaler(),
                ElasticNetCV(
                    l1_ratio=config.elastic_l1_ratios,
                    cv=inner if inner is not None else 2,
                    max_iter=100000,
                    random_state=config.random_state,
                ),
            )
            model.fit(x[train_idx], y[train_idx])
            predictions[test_idx] = model.predict(x[test_idx])

        predictions[~np.isfinite(predictions)] = np.nanmean(y)
        final_inner = time_series_splitter(n_samples, config.inner_cv_splits)
        final_model = make_pipeline(
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=config.elastic_l1_ratios,
                cv=final_inner if final_inner is not None else 2,
                max_iter=100000,
                random_state=config.random_state,
            ),
        )
        final_model.fit(x, y)
        coefficients = np.asarray(final_model[-1].coef_, dtype=np.float64)

    rmse = float(np.sqrt(np.mean((y - predictions) ** 2)))
    return predictions, coefficients, float(r2_score(y, predictions)), rmse


def lagged_design(
    steps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lag < 0:
        raise ValueError("lag must be non-negative.")
    if lag == 0:
        return steps, x, y
    return steps[:-lag], x[:-lag], y[lag:]


def compute_lagged_predictive_decomposition(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    active_fields_file: str | Path = Path(CYLINDER_PATHS.in_gsd).parent
    / "active_matter_fields.npz",
    geometric_chirality_file: str | Path = Path(CYLINDER_PATHS.in_gsd).parent
    / "geometric_chirality_fields.npz",
    shear_decomposition_file: str | Path = Path(CYLINDER_PATHS.in_gsd).parent
    / "shear_flux_decomposition_series.npz",
    hexatic_gsd: str | Path = CYLINDER_PATHS.out_gsd,
    config: LaggedPredictionConfig = LaggedPredictionConfig(),
) -> LaggedPredictionResult:
    from hexatic.cylinder_dynamics.series import x_center_of_mass_velocity_series

    active_families, active_context = load_active_families(active_fields_file, config)
    hexatic_families = load_hexatic_families(hexatic_gsd, active_context, config)
    ccm_family = load_ccm_family(geometric_chirality_file, config)
    stress_families = load_stress_families(shear_decomposition_file, config)
    families = {
        "px": active_families["px"],
        "ccm": ccm_family,
        "defects": hexatic_families["defects"],
        "rho": active_families["rho"],
        "psi6": hexatic_families["psi6"],
        **stress_families,
    }

    return compute_lagged_prediction_from_families(
        families=families,
        input_gsd=input_gsd,
        target_name="Vx",
        config=config,
    )


def compute_lagged_prediction_from_families(
    families: Dict[str, FeatureFamily],
    input_gsd: str | Path,
    target_name: str,
    config: LaggedPredictionConfig,
    target_values: np.ndarray | None = None,
    target_steps: np.ndarray | None = None,
) -> LaggedPredictionResult:
    from hexatic.cylinder_dynamics.series import x_center_of_mass_velocity_series

    if target_values is None or target_steps is None:
        vx_series = x_center_of_mass_velocity_series(input_gsd, shell_only=True)
        target_steps = vx_series.steps
        target_values = vx_series.x_velocities
    steps, x, y, feature_names, feature_families = align_families(
        families,
        target_steps,
        target_values,
    )
    family_names = np.asarray(tuple(families.keys()))
    lag_frames = np.asarray(config.lag_frames, dtype=np.int64)
    lag_times = (
        lag_frames
        * float(cylinder.TIMESTEP)
        * float(cylinder.SIMULATION.trajectory_write_period)
    )
    n_lags = len(lag_frames)
    n_features = x.shape[1]

    full_r2 = np.full(n_lags, np.nan, dtype=np.float64)
    full_rmse = np.full(n_lags, np.nan, dtype=np.float64)
    full_coefficients = np.full((n_lags, n_features), np.nan, dtype=np.float64)
    family_delta_r2 = np.full((n_lags, len(family_names)), np.nan, dtype=np.float64)
    family_coefficient_norms = np.full_like(family_delta_r2, np.nan)
    mediation_without_px = np.full(n_lags, np.nan, dtype=np.float64)
    mediation_with_px = np.full(n_lags, np.nan, dtype=np.float64)
    predictions = np.full((n_lags, len(steps)), np.nan, dtype=np.float64)
    actual = np.full((n_lags, len(steps)), np.nan, dtype=np.float64)

    for lag_idx, lag in enumerate(lag_frames):
        _lag_steps, lag_x, lag_y = lagged_design(steps, x, y, int(lag))
        if lag_y.size < 8:
            continue
        pred, coef, r2, rmse = fit_predict_elastic_net(lag_x, lag_y, config)
        full_r2[lag_idx] = r2
        full_rmse[lag_idx] = rmse
        full_coefficients[lag_idx] = coef
        predictions[lag_idx, : pred.size] = pred
        actual[lag_idx, : lag_y.size] = lag_y

        for family_idx, family_name in enumerate(family_names):
            mask = feature_families == family_name
            family_coefficient_norms[lag_idx, family_idx] = float(
                np.linalg.norm(coef[mask])
            )
            _, _, drop_r2, _ = fit_predict_elastic_net(
                lag_x[:, ~mask],
                lag_y,
                config,
            )
            family_delta_r2[lag_idx, family_idx] = r2 - drop_r2

        direct_mask = np.isin(feature_families, ("defects", "ccm"))
        mediated_mask = np.isin(feature_families, ("defects", "ccm", "px"))
        if np.any(direct_mask):
            _, _, mediation_without_px[lag_idx], _ = fit_predict_elastic_net(
                lag_x[:, direct_mask],
                lag_y,
                config,
            )
        if np.any(mediated_mask):
            _, _, mediation_with_px[lag_idx], _ = fit_predict_elastic_net(
                lag_x[:, mediated_mask],
                lag_y,
                config,
            )

    return LaggedPredictionResult(
        target_name=target_name,
        steps=steps,
        lag_frames=lag_frames,
        lag_times=lag_times,
        feature_names=feature_names,
        feature_families=feature_families,
        full_r2=full_r2,
        full_rmse=full_rmse,
        full_coefficients=full_coefficients,
        family_names=family_names,
        family_delta_r2=family_delta_r2,
        family_coefficient_norms=family_coefficient_norms,
        mediation_r2_without_px=mediation_without_px,
        mediation_r2_with_px=mediation_with_px,
        predictions=predictions,
        actual=actual,
    )
