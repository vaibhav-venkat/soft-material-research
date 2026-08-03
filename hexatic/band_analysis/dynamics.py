"""Public facade and orchestration for stochastic band analysis."""

from __future__ import annotations

import numpy as np

from .stochastic.area import (
    add_area_diffusion_fit,
    add_area_drift_fits,
    add_area_geometry_fits,
    add_area_increment_conservation,
)
from .stochastic.common import (
    DYNAMIC_PROPERTIES,
    MAX_FRAME_LAG,
    MIN_BIN_SAMPLES,
    PREFERRED_BIN_SAMPLES,
    TARGET_BINS,
    DynamicProperty,
    ProgressCallback,
    area_diffusion_model,
    area_drift_constant_model,
    area_drift_cubic_model,
)
from .stochastic.conditional import (
    add_area_cv_squared_drift,
    add_conditional_dynamics,
    add_total_area_drift,
)
from .stochastic.correlations import add_area_coupling, add_position_msd
from .stochastic.markov import add_area_chapman_kolmogorov
from .stochastic.neighbors import (
    add_neighbor_relative_area_change_drift,
    add_neighbor_relative_area_change_fit,
    add_neighbor_relative_area_drifts,
    add_neighbor_relative_area_fit,
    neighbor_relative_area,
    normalized_neighbor_relative_area,
)


def add_stochastic_statistics(
    tensors: dict[str, np.ndarray],
    *,
    max_frame_lag: int = MAX_FRAME_LAG,
    progress: ProgressCallback | None = None,
) -> None:
    """Run all stochastic reductions and fitted models."""
    if progress is not None:
        progress("stage=stochastic conditional_start")
    add_conditional_dynamics(
        tensors, max_frame_lag=max_frame_lag, progress=progress
    )
    if progress is not None:
        progress("stage=stochastic area_cv_squared_fixed_n start")
    add_area_cv_squared_drift(tensors, max_frame_lag=max_frame_lag)
    if progress is not None:
        progress("stage=stochastic area_cv_squared_fixed_n complete")
        progress("stage=stochastic total_area_fixed_n start")
    add_total_area_drift(tensors, max_frame_lag=max_frame_lag)
    if progress is not None:
        progress("stage=stochastic total_area_fixed_n complete")
        progress("stage=stochastic neighbor_relative_area start")
    add_neighbor_relative_area_drifts(tensors, max_frame_lag=max_frame_lag)
    if progress is not None:
        progress("stage=stochastic neighbor_relative_area change_start")
    add_neighbor_relative_area_change_drift(
        tensors, max_frame_lag=max_frame_lag
    )
    if progress is not None:
        progress("stage=stochastic neighbor_relative_area change_fit")
    add_neighbor_relative_area_change_fit(tensors)
    if progress is not None:
        progress("stage=stochastic neighbor_relative_area fits_start")
    add_neighbor_relative_area_fit(tensors)
    if progress is not None:
        progress("stage=stochastic neighbor_relative_area complete")
        progress("stage=stochastic area_drift_fits start")
    add_area_drift_fits(tensors)
    if progress is not None:
        progress("stage=stochastic area_drift_fits complete")
        progress("stage=stochastic area_diffusion_fit start")
    add_area_diffusion_fit(tensors)
    if progress is not None:
        progress("stage=stochastic area_diffusion_fit complete")
        progress("stage=stochastic area_geometry_fits start")
    add_area_geometry_fits(tensors)
    if progress is not None:
        progress("stage=stochastic area_geometry_fits complete")
        progress("stage=stochastic chapman_kolmogorov start")
    add_area_chapman_kolmogorov(tensors, max_frame_lag=max_frame_lag)
    if progress is not None:
        progress("stage=stochastic chapman_kolmogorov complete")
        progress("stage=stochastic msd_start")
    add_position_msd(tensors, progress=progress)
    if progress is not None:
        progress("stage=stochastic area_coupling_start")
    add_area_coupling(tensors, progress=progress)
    if progress is not None:
        progress("stage=stochastic area_increment_conservation start")
    add_area_increment_conservation(tensors)
    if progress is not None:
        progress("stage=stochastic area_increment_conservation complete")
        progress("stage=stochastic complete")


__all__ = [
    "DYNAMIC_PROPERTIES",
    "MAX_FRAME_LAG",
    "MIN_BIN_SAMPLES",
    "PREFERRED_BIN_SAMPLES",
    "TARGET_BINS",
    "DynamicProperty",
    "add_stochastic_statistics",
    "area_diffusion_model",
    "area_drift_constant_model",
    "area_drift_cubic_model",
    "neighbor_relative_area",
    "normalized_neighbor_relative_area",
]
