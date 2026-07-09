"""Shared validation types and constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray


Array = NDArray[Any]
D = 3
COMPONENTS = 13
P_RELAXATION_COEFFICIENT = 2.0
Q_RELAXATION_COEFFICIENT = 0.6642702159746572
VALIDATION_BOUND_SCALE = 10.0
FILTERED_EULER_SIGMA_SCALE = 0.1
BILATERAL_SPATIAL_SIGMA_CELLS = 0.6
BILATERAL_RANGE_SCALE = 0.75
BILATERAL_RADIUS_CELLS = 1


@dataclass(frozen=True)
class ValidationResult:
    """Predicted and reference rollout fields with per-frame validation metrics."""

    rho_fit: Array
    p_fit: Array
    q_fit: Array
    rho_true: Array
    p_true: Array
    q_true: Array
    times: Array
    rmse_t: Array
    r2_t: Array


@dataclass(frozen=True)
class ValidationOptions:
    """Solver, filtering, frame-count, and validation-mode controls for PDE rollouts."""

    max_frames: int | None = None
    solver: str = "imex-rk"
    dt: float | None = None
    filter_sigma: float | None = None
    mode: str = "full"
    ubar_source: str = "cached"
