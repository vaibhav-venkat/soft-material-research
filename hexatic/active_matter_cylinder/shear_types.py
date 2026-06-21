from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShearFluxDecomposition:
    step: int
    frame_index: int
    pocket_radius: float
    x_edges: np.ndarray
    x_centers: np.ndarray
    r_edges: np.ndarray
    r_centers: np.ndarray
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    grid_coords: np.ndarray
    grid_points: np.ndarray
    rho_density: np.ndarray
    sigma_full: np.ndarray
    deriv_sigma_full: np.ndarray
    sigma_normal: np.ndarray
    sigma_shear: np.ndarray
    div_sigma_full: np.ndarray
    div_sigma_normal: np.ndarray
    div_sigma_shear: np.ndarray
    div_split_residual: np.ndarray
    polar_density: np.ndarray
    pair_force_density: np.ndarray
    wall_force_density: np.ndarray
    logged_force_density: np.ndarray
    j_active: np.ndarray
    j_normal: np.ndarray
    j_shear: np.ndarray
    j_wall: np.ndarray
    j_total: np.ndarray
    j_total_with_wall: np.ndarray
    j_force_baseline: np.ndarray
    j_split_residual: np.ndarray
    near_wall_mask: np.ndarray
    pair_count: int
    pair_force_correlation: np.ndarray
    pair_force_slope: np.ndarray
    stress_force_correlation: np.ndarray
    stress_force_slope: np.ndarray
