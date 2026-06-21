from pathlib import Path

import numpy as np

from .modeling import (
    compute_lagged_predictive_decomposition,
)
from .plotting import plot_lagged_predictive_decomposition
from .types import (
    CYLINDER_PATHS,
    LAGGED_PREDICTION_DATA,
    LAGGED_PREDICTION_IMAGE_DIR,
    LaggedPredictionConfig,
    LaggedPredictionResult,
)


def save_lagged_predictive_decomposition(
    result: LaggedPredictionResult,
    filename: str | Path = LAGGED_PREDICTION_DATA,
) -> None:
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        target_name=np.asarray(result.target_name),
        steps=result.steps,
        lag_frames=result.lag_frames,
        lag_times=result.lag_times,
        feature_names=result.feature_names,
        feature_families=result.feature_families,
        full_r2=result.full_r2,
        full_rmse=result.full_rmse,
        full_coefficients=result.full_coefficients,
        family_names=result.family_names,
        family_delta_r2=result.family_delta_r2,
        family_coefficient_norms=result.family_coefficient_norms,
        mediation_r2_without_px=result.mediation_r2_without_px,
        mediation_r2_with_px=result.mediation_r2_with_px,
        predictions=result.predictions,
        actual=result.actual,
    )


def write_lagged_predictive_decomposition_outputs(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    output_npz: str | Path = LAGGED_PREDICTION_DATA,
    image_dir: str | Path = LAGGED_PREDICTION_IMAGE_DIR,
    config: LaggedPredictionConfig = LaggedPredictionConfig(),
) -> LaggedPredictionResult:
    result = compute_lagged_predictive_decomposition(input_gsd=input_gsd, config=config)
    save_lagged_predictive_decomposition(result, output_npz)
    plot_lagged_predictive_decomposition(result, image_dir=image_dir)
    return result
