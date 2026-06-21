from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hexatic.constants import cylinder


CYLINDER_PATHS = cylinder.PATHS
LAGGED_PREDICTION_DATA = (
    Path(CYLINDER_PATHS.in_gsd).parent / "lagged_predictive_decomposition.npz"
)
LAGGED_PREDICTION_IMAGE_DIR = (
    Path(CYLINDER_PATHS.com_plot).parent / "lagged_prediction"
)
@dataclass(frozen=True)
class LaggedPredictionConfig:
    lag_frames: tuple[int, ...] = (0, 1, 2, 3, 5, 10)
    max_modes: int = 5
    variance_threshold: float = 0.95
    elastic_l1_ratios: tuple[float, ...] = (0.1, 0.5, 0.9, 1.0)
    cv_splits: int = 5
    inner_cv_splits: int = 3
    random_state: int = 1


@dataclass(frozen=True)
class FeatureFamily:
    name: str
    steps: np.ndarray
    values: np.ndarray
    feature_names: tuple[str, ...]
    mode_variance: np.ndarray


@dataclass(frozen=True)
class LaggedPredictionResult:
    target_name: str
    steps: np.ndarray
    lag_frames: np.ndarray
    lag_times: np.ndarray
    feature_names: np.ndarray
    feature_families: np.ndarray
    full_r2: np.ndarray
    full_rmse: np.ndarray
    full_coefficients: np.ndarray
    family_names: np.ndarray
    family_delta_r2: np.ndarray
    family_coefficient_norms: np.ndarray
    mediation_r2_without_px: np.ndarray
    mediation_r2_with_px: np.ndarray
    predictions: np.ndarray
    actual: np.ndarray
