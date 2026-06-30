"""Cylinder geometry helpers."""

from __future__ import annotations

import numpy as np


def theta_to_y(theta: np.ndarray, radius: float) -> np.ndarray:
    return radius * theta


def minimum_image(delta: np.ndarray, period: float) -> np.ndarray:
    return delta - period * np.round(delta / period)


def grid_centers(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def surface_lengths(x_edges: np.ndarray, theta_edges: np.ndarray, radius: float) -> tuple[float, float]:
    lx = float(x_edges[-1] - x_edges[0])
    ly = float(radius * (theta_edges[-1] - theta_edges[0]))
    if lx <= 0.0 or ly <= 0.0:
        raise ValueError("surface lengths must be positive")
    return lx, ly
