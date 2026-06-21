from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .types import ShearFluxDecomposition


def _stack_field(decompositions: Sequence[ShearFluxDecomposition], name: str) -> np.ndarray:
    return np.stack([getattr(decomposition, name) for decomposition in decompositions], axis=0)


def _array_field(decompositions: Sequence[ShearFluxDecomposition], name: str, dtype=None) -> np.ndarray:
    return np.asarray([getattr(decomposition, name) for decomposition in decompositions], dtype=dtype)


def save_shear_flux_decomposition(
    decomposition: ShearFluxDecomposition,
    filename: str | Path,
) -> None:
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        step=np.asarray(decomposition.step, dtype=np.int64),
        frame_index=np.asarray(decomposition.frame_index, dtype=np.int64),
        pocket_radius=np.asarray(decomposition.pocket_radius, dtype=np.float64),
        x_edges=decomposition.x_edges,
        x_centers=decomposition.x_centers,
        r_edges=decomposition.r_edges,
        r_centers=decomposition.r_centers,
        theta_edges=decomposition.theta_edges,
        theta_centers=decomposition.theta_centers,
        grid_coords=decomposition.grid_coords,
        grid_points=decomposition.grid_points,
        rho_density=decomposition.rho_density,
        sigma_full=decomposition.sigma_full,
        deriv_sigma_full=decomposition.deriv_sigma_full,
        sigma_normal=decomposition.sigma_normal,
        sigma_shear=decomposition.sigma_shear,
        div_sigma_full=decomposition.div_sigma_full,
        div_sigma_normal=decomposition.div_sigma_normal,
        div_sigma_shear=decomposition.div_sigma_shear,
        div_split_residual=decomposition.div_split_residual,
        polar_density=decomposition.polar_density,
        pair_force_density=decomposition.pair_force_density,
        wall_force_density=decomposition.wall_force_density,
        logged_force_density=decomposition.logged_force_density,
        j_active=decomposition.j_active,
        j_normal=decomposition.j_normal,
        j_shear=decomposition.j_shear,
        j_wall=decomposition.j_wall,
        j_total=decomposition.j_total,
        j_total_with_wall=decomposition.j_total_with_wall,
        j_force_baseline=decomposition.j_force_baseline,
        j_split_residual=decomposition.j_split_residual,
        near_wall_mask=decomposition.near_wall_mask,
        pair_count=np.asarray(decomposition.pair_count, dtype=np.int64),
        pair_force_correlation=decomposition.pair_force_correlation,
        pair_force_slope=decomposition.pair_force_slope,
        stress_force_correlation=decomposition.stress_force_correlation,
        stress_force_slope=decomposition.stress_force_slope,
    )


def save_shear_flux_decomposition_series(
    decompositions: Sequence[ShearFluxDecomposition],
    filename: str | Path,
) -> None:
    """Save a time series of stress decomposition fields with simulation steps."""
    if not decompositions:
        raise ValueError("At least one shear flux decomposition is required.")

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = decompositions[0]
    stack_names = (
        "grid_coords",
        "grid_points",
        "rho_density",
        "sigma_full",
        "deriv_sigma_full",
        "sigma_normal",
        "sigma_shear",
        "div_sigma_full",
        "div_sigma_normal",
        "div_sigma_shear",
        "div_split_residual",
        "polar_density",
        "pair_force_density",
        "wall_force_density",
        "logged_force_density",
        "j_active",
        "j_normal",
        "j_shear",
        "j_wall",
        "j_total",
        "j_total_with_wall",
        "j_force_baseline",
        "j_split_residual",
        "near_wall_mask",
        "pair_force_correlation",
        "pair_force_slope",
        "stress_force_correlation",
        "stress_force_slope",
    )
    payload = {name: _stack_field(decompositions, name) for name in stack_names}
    payload.update(
        steps=_array_field(decompositions, "step", dtype=np.int64),
        frame_indices=_array_field(decompositions, "frame_index", dtype=np.int64),
        pocket_radius=np.asarray(first.pocket_radius, dtype=np.float64),
        x_edges=first.x_edges,
        x_centers=first.x_centers,
        r_edges=first.r_edges,
        r_centers=first.r_centers,
        theta_edges=first.theta_edges,
        theta_centers=first.theta_centers,
        pair_count=_array_field(decompositions, "pair_count", dtype=np.int64),
    )
    np.savez_compressed(output_path, **payload)
