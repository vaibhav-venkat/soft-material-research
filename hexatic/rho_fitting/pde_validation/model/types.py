"""Shared validation types and constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray


Array = NDArray[Any]
D = 3
COMPONENTS = 13
RELAXATION_COEFFICIENT = -0.01
VALIDATION_BOUND_SCALE = 10.0


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
    backend: str = "jax"
    jax_device: str = "mps"
    solver: str = "filtered-euler"
    dt: float | None = None
    filter_sigma: float | None = None
    mode: str = "full"
