"""Regression utilities for rho fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StabilityResult:
    names: tuple[str, ...]
    coefficients: np.ndarray
    importance: np.ndarray
    active: np.ndarray


def tau_path(tau_max: float, count: int = 40, eps: float = 1e-2) -> np.ndarray:
    if tau_max < 0.0:
        raise ValueError("tau_max must be nonnegative")
    if count < 1 or eps <= 0.0:
        raise ValueError("invalid tau path settings")
    return tau_max * np.logspace(0.0, np.log10(eps), count)
