"""Chebyshev time basis helpers."""

from __future__ import annotations

import numpy as np


def temporal_power_spectrum(rho_hat: np.ndarray, px_hat: np.ndarray, py_hat: np.ndarray) -> np.ndarray:
    return (
        np.sum(np.abs(rho_hat) ** 2, axis=(1, 2))
        + np.sum(np.abs(px_hat) ** 2, axis=(1, 2))
        + np.sum(np.abs(py_hat) ** 2, axis=(1, 2))
    )


def validate_cheb_cutoff(cutoff: int, frame_count: int) -> int:
    if cutoff < 1:
        raise ValueError("cheb_cutoff must be positive")
    return min(cutoff, frame_count)
