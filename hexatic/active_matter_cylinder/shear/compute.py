from collections.abc import Iterable
from pathlib import Path

import gsd.hoomd
import numpy as np

try:
    from scipy.special import erf as _erf
except ImportError:
    import math

    _erf = np.vectorize(math.erf, otypes=[np.float64])

from ..common import (
    _active_direction_from_quaternion,
    _axis_edges_and_centers,
    _cartesian_vector_to_cylindrical_components,
    _cylindrical_basis,
    _density_sum,
    _logged_particle_array,
    _minimum_image_delta,
    _theta_edges_and_centers,
)
from ..config import (
    ACTIVE_FLUX_PLOT_THETA_BINS,
    ACTIVE_GRID_DX,
    ACTIVE_GRID_DY,
    CYLINDER,
    CYLINDER_PATHS,
    CYLINDER_SIM,
    LOCAL_POCKET_RADIUS,
)
from .types import ShearFluxDecomposition


def _xrtheta_grid(
    box_length_x: float,
    dx: float,
    dr: float,
    n_theta_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_edges, x_centers = _axis_edges_and_centers(-0.5 * box_length_x, 0.5 * box_length_x, dx)
    r_edges, r_centers = _axis_edges_and_centers(0.0, CYLINDER.cylinder_radius, dr)
    theta_edges, theta_centers = _theta_edges_and_centers(n_theta_bins)

    x_grid, r_grid, theta_grid = np.meshgrid(
        x_centers,
        r_centers,
        theta_centers,
        indexing="ij",
    )
    y_grid = r_grid * np.sin(theta_grid)
    z_grid = r_grid * np.cos(theta_grid)
    grid_coords = np.column_stack((x_grid.ravel(), r_grid.ravel(), theta_grid.ravel()))
    grid_points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
    return (
        x_edges,
        x_centers,
        r_edges,
        r_centers,
        theta_edges,
        theta_centers,
        grid_coords,
        grid_points,
    )


def _component_stats(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    correlations = np.full(3, np.nan, dtype=np.float64)
    slopes = np.full(3, np.nan, dtype=np.float64)
    for component in range(3):
        x_values = reference[:, component]
        y_values = candidate[:, component]
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[valid]
        y_values = y_values[valid]
        if len(x_values) > 1 and not np.isclose(np.std(x_values), 0.0) and not np.isclose(np.std(y_values), 0.0):
            correlations[component] = np.corrcoef(x_values, y_values)[0, 1]
        denominator = float(np.dot(x_values, x_values))
        if not np.isclose(denominator, 0.0):
            slopes[component] = float(np.dot(x_values, y_values) / denominator)
    return correlations, slopes


def _lj_pair_forces(
    positions: np.ndarray,
    box_length_x: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sigma = float(CYLINDER.sigma)
    epsilon = float(50.0 * CYLINDER_SIM.gamma * CYLINDER_SIM.u0 * sigma)
    r_cut = float(CYLINDER.wall_cutoff)
    r_cut_sq = r_cut * r_cut

    shifted = positions.copy()
    shifted[:, 0] += 0.5 * box_length_x
    shifted[:, 1] += CYLINDER.cylinder_radius
    shifted[:, 2] += CYLINDER.cylinder_radius
    cell_size = r_cut
    n_x = max(1, int(np.floor(box_length_x / cell_size)))
    n_y = max(1, int(np.ceil(2.0 * CYLINDER.cylinder_radius / cell_size)))
    n_z = max(1, int(np.ceil(2.0 * CYLINDER.cylinder_radius / cell_size)))
    cell_indices = np.column_stack(
        (
            np.floor(shifted[:, 0] / box_length_x * n_x).astype(np.int64) % n_x,
            np.clip(np.floor(shifted[:, 1] / cell_size).astype(np.int64), 0, n_y - 1),
            np.clip(np.floor(shifted[:, 2] / cell_size).astype(np.int64), 0, n_z - 1),
        )
    )

    cells: dict[tuple[int, int, int], list[int]] = {}
    for particle_index, key_array in enumerate(cell_indices):
        key = tuple(int(value) for value in key_array)
        cells.setdefault(key, []).append(particle_index)

    pair_i: list[int] = []
    pair_j: list[int] = []
    pair_delta: list[np.ndarray] = []
    pair_force: list[np.ndarray] = []
    offsets = tuple((dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1))

    for key, indices_i in cells.items():
        cx, cy, cz = key
        for ox, oy, oz in offsets:
            neighbor_key = ((cx + ox) % n_x, cy + oy, cz + oz)
            if neighbor_key[1] < 0 or neighbor_key[1] >= n_y or neighbor_key[2] < 0 or neighbor_key[2] >= n_z:
                continue
            indices_j = cells.get(neighbor_key)
            if not indices_j:
                continue
            if key > neighbor_key:
                continue
            for i in indices_i:
                for j in indices_j:
                    if key == neighbor_key and i >= j:
                        continue
                    delta = positions[i] - positions[j]
                    delta[0] = _minimum_image_delta(delta[0], box_length_x)
                    r_sq = float(np.dot(delta, delta))
                    if r_sq <= 0.0 or r_sq >= r_cut_sq:
                        continue
                    inv_r_sq = 1.0 / r_sq
                    sigma_over_r_sq = sigma * sigma * inv_r_sq
                    sr6 = sigma_over_r_sq**3
                    sr12 = sr6 * sr6
                    force = 24.0 * epsilon * (2.0 * sr12 - sr6) * inv_r_sq * delta
                    pair_i.append(i)
                    pair_j.append(j)
                    pair_delta.append(delta.copy())
                    pair_force.append(force.copy())

    if not pair_i:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )
    return (
        np.asarray(pair_i, dtype=np.int64),
        np.asarray(pair_j, dtype=np.int64),
        np.asarray(pair_delta, dtype=np.float64),
        np.asarray(pair_force, dtype=np.float64),
    )


def _particle_pair_force_sums(
    n_particles: int,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_force: np.ndarray,
) -> np.ndarray:
    particle_forces = np.zeros((n_particles, 3), dtype=np.float64)
    np.add.at(particle_forces, pair_i, pair_force)
    np.add.at(particle_forces, pair_j, -pair_force)
    return particle_forces


def _logged_particle_array_filtered(
    frame,
    quantity: str,
    n_particles: int,
    include_tokens: tuple[str, ...],
    exclude_tokens: tuple[str, ...] = (),
    required: bool = False,
) -> np.ndarray:
    log = getattr(frame, "log", None)
    if not log:
        raise ValueError(f"GSD frame has no logger data; expected {quantity}.")

    quantity_lower = quantity.lower()
    candidates: list[np.ndarray] = []
    for key, value in log.items():
        key_lower = str(key).lower()
        if quantity_lower not in key_lower:
            continue
        if include_tokens and not all(token in key_lower for token in include_tokens):
            continue
        if any(token in key_lower for token in exclude_tokens):
            continue
        array = np.asarray(value)
        if array.shape[:1] == (n_particles,):
            candidates.append(array)

    if not candidates and required:
        available = ", ".join(str(key) for key in log)
        raise ValueError(
            f"Could not find logged per-particle {quantity} matching "
            f"{include_tokens}. Available logger keys: {available}"
        )
    if not candidates:
        return np.full((n_particles, 3), np.nan, dtype=np.float64)

    total = np.zeros_like(np.asarray(candidates[0], dtype=np.float64))
    for array in candidates:
        total += np.asarray(array, dtype=np.float64)
    return total[:, :3]


def _hardy_bond_stress_and_gradient(
    grid_points: np.ndarray,
    positions: np.ndarray,
    pair_j: np.ndarray,
    pair_delta: np.ndarray,
    pair_force: np.ndarray,
    box_length_x: float,
    radius: float,
    grid_block_size: int = 192,
    pair_block_size: int = 192,
) -> tuple[np.ndarray, np.ndarray]:
    n_grid = len(grid_points)
    stress = np.zeros((n_grid, 3, 3), dtype=np.float64)
    gradient = np.zeros((n_grid, 3, 3, 3), dtype=np.float64)
    if len(pair_delta) == 0:
        return stress, gradient

    norm = (2.0 * np.pi) ** 1.5 * radius**3
    sqrt_half_pi = np.sqrt(0.5 * np.pi)
    sqrt_two = np.sqrt(2.0)
    radius_sq = radius * radius

    for pair_start in range(0, len(pair_delta), pair_block_size):
        pair_stop = min(pair_start + pair_block_size, len(pair_delta))
        delta_block = pair_delta[pair_start:pair_stop]
        force_block = pair_force[pair_start:pair_stop]
        start_points = positions[pair_j[pair_start:pair_stop]]
        a = np.sum(delta_block * delta_block, axis=1)
        sqrt_a = np.sqrt(a)
        bond_tensor = -np.einsum("pa,pb->pab", force_block, delta_block)

        for grid_start in range(0, n_grid, grid_block_size):
            grid_stop = min(grid_start + grid_block_size, n_grid)
            relative = grid_points[grid_start:grid_stop, np.newaxis, :] - start_points[np.newaxis, :, :]
            relative[..., 0] = _minimum_image_delta(relative[..., 0], box_length_x)
            b = np.einsum("gpa,pa->gp", relative, delta_block)
            c = np.sum(relative * relative, axis=2)
            m = b / a[np.newaxis, :]
            perp_sq = np.maximum(c - b * b / a[np.newaxis, :], 0.0)
            perpendicular = relative - m[..., np.newaxis] * delta_block[np.newaxis, :, :]

            prefactor = np.exp(-0.5 * perp_sq / radius_sq) / norm
            upper = sqrt_a[np.newaxis, :] * (1.0 - m) / (sqrt_two * radius)
            lower = -sqrt_a[np.newaxis, :] * m / (sqrt_two * radius)
            bond_kernel = (
                prefactor
                * sqrt_half_pi
                * radius
                / sqrt_a[np.newaxis, :]
                * (_erf(upper) - _erf(lower))
            )

            exp_lower = np.exp(-0.5 * a[np.newaxis, :] * m * m / radius_sq)
            exp_upper = np.exp(-0.5 * a[np.newaxis, :] * (1.0 - m) ** 2 / radius_sq)
            moment_parallel = prefactor * radius_sq / a[np.newaxis, :] * (exp_lower - exp_upper)
            first_moment = (
                perpendicular * bond_kernel[..., np.newaxis]
                - delta_block[np.newaxis, :, :] * moment_parallel[..., np.newaxis]
            )
            bond_kernel_gradient = -first_moment / radius_sq

            stress[grid_start:grid_stop] += np.einsum(
                "gp,pab->gab",
                bond_kernel,
                bond_tensor,
            )
            gradient[grid_start:grid_stop] += np.einsum(
                "gpk,pab->gkab",
                bond_kernel_gradient,
                bond_tensor,
            )

    return stress, gradient


def _cylindrical_basis_derivative(points: np.ndarray) -> np.ndarray:
    basis = _cylindrical_basis(points)
    derivative = np.zeros_like(basis)
    derivative[:, 1, :] = basis[:, 2, :]
    derivative[:, 2, :] = -basis[:, 1, :]
    return derivative


def _cartesian_stress_derivatives_to_cylindrical(
    points: np.ndarray,
    stress_cart: np.ndarray,
    grad_stress_cart: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    basis = _cylindrical_basis(points)
    basis_theta = _cylindrical_basis_derivative(points)
    radii = np.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2)
    theta = np.mod(np.arctan2(points[:, 1], points[:, 2]), 2.0 * np.pi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    stress_cyl = np.einsum("gac,gcd,gbd->gab", basis, stress_cart, basis)
    grad_x = grad_stress_cart[:, 0]
    grad_r = sin_theta[:, np.newaxis, np.newaxis] * grad_stress_cart[:, 1] + (
        cos_theta[:, np.newaxis, np.newaxis] * grad_stress_cart[:, 2]
    )
    grad_theta_direction = radii[:, np.newaxis, np.newaxis] * (
        cos_theta[:, np.newaxis, np.newaxis] * grad_stress_cart[:, 1]
        - sin_theta[:, np.newaxis, np.newaxis] * grad_stress_cart[:, 2]
    )

    deriv_cyl = np.zeros((len(points), 3, 3, 3), dtype=np.float64)
    deriv_cyl[:, 0] = np.einsum("gac,gcd,gbd->gab", basis, grad_x, basis)
    deriv_cyl[:, 1] = np.einsum("gac,gcd,gbd->gab", basis, grad_r, basis)
    deriv_cyl[:, 2] = (
        np.einsum("gac,gcd,gbd->gab", basis_theta, stress_cart, basis)
        + np.einsum("gac,gcd,gbd->gab", basis, grad_theta_direction, basis)
        + np.einsum("gac,gcd,gbd->gab", basis, stress_cart, basis_theta)
    )
    return stress_cyl, deriv_cyl


def _cylindrical_divergence_split(
    coords: np.ndarray,
    stress_cyl: np.ndarray,
    deriv_cyl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radii = coords[:, 1]
    inv_r = np.divide(
        1.0,
        radii,
        out=np.full_like(radii, np.nan, dtype=np.float64),
        where=radii > 0.0,
    )

    div_full = np.zeros((len(coords), 3), dtype=np.float64)
    div_full[:, 0] = deriv_cyl[:, 0, 0, 0] + deriv_cyl[:, 1, 0, 1] + inv_r * deriv_cyl[:, 2, 0, 2] + inv_r * stress_cyl[:, 0, 1]
    div_full[:, 1] = deriv_cyl[:, 0, 1, 0] + deriv_cyl[:, 1, 1, 1] + inv_r * deriv_cyl[:, 2, 1, 2] + inv_r * (stress_cyl[:, 1, 1] - stress_cyl[:, 2, 2])
    div_full[:, 2] = deriv_cyl[:, 0, 2, 0] + deriv_cyl[:, 1, 2, 1] + inv_r * deriv_cyl[:, 2, 2, 2] + 2.0 * inv_r * stress_cyl[:, 1, 2]

    sigma_normal = np.zeros_like(stress_cyl)
    sigma_normal[:, 0, 0] = stress_cyl[:, 0, 0]
    sigma_normal[:, 1, 1] = stress_cyl[:, 1, 1]
    sigma_normal[:, 2, 2] = stress_cyl[:, 2, 2]
    sigma_shear = stress_cyl - sigma_normal

    div_normal = np.zeros_like(div_full)
    div_normal[:, 0] = deriv_cyl[:, 0, 0, 0]
    div_normal[:, 1] = deriv_cyl[:, 1, 1, 1] + inv_r * (stress_cyl[:, 1, 1] - stress_cyl[:, 2, 2])
    div_normal[:, 2] = inv_r * deriv_cyl[:, 2, 2, 2]
    div_shear = div_full - div_normal

    invalid = ~np.isfinite(inv_r)
    if np.any(invalid):
        div_full[invalid] = np.nan
        div_normal[invalid] = np.nan
        div_shear[invalid] = np.nan
    return sigma_normal, sigma_shear, div_full, div_normal, div_shear


def compute_shear_flux_decomposition(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    frame_index: int = -2,
    pocket_radius: float = LOCAL_POCKET_RADIUS,
    dx: float = ACTIVE_GRID_DX,
    dr: float = ACTIVE_GRID_DY,
    n_theta_bins: int = ACTIVE_FLUX_PLOT_THETA_BINS,
) -> ShearFluxDecomposition:
    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        assert len(source) >= 1
        if frame_index < 0:
            frame_index += len(source)
        assert 0 <= frame_index < len(source)
        frame = source[frame_index]
        particles = frame.particles
        assert particles.position is not None
        assert particles.orientation is not None
        positions = np.asarray(particles.position, dtype=np.float64)
        box_length_x = float(frame.configuration.box[0])
        step = int(frame.configuration.step)

        (
            x_edges,
            x_centers,
            r_edges,
            r_centers,
            theta_edges,
            theta_centers,
            grid_coords,
            grid_points,
        ) = _xrtheta_grid(box_length_x, dx, dr, n_theta_bins)

        pair_i, pair_j, pair_delta, pair_force = _lj_pair_forces(positions, box_length_x)
        reconstructed_particle_pair_forces = _particle_pair_force_sums(
            len(positions),
            pair_i,
            pair_j,
            pair_force,
        )
        logged_pair_forces = _logged_particle_array_filtered(
            frame,
            "forces",
            int(particles.N),
            include_tokens=("pair", "lj"),
            exclude_tokens=("wall", "external"),
        )
        all_logged_forces = _logged_particle_array(frame, "forces", int(particles.N))[:, :3]
        wall_particle_forces = _logged_particle_array_filtered(
            frame,
            "forces",
            int(particles.N),
            include_tokens=("wall",),
            required=True,
        )

        sigma_cart, grad_sigma_cart = _hardy_bond_stress_and_gradient(
            grid_points,
            positions,
            pair_j,
            pair_delta,
            pair_force,
            box_length_x,
            pocket_radius,
        )
        sigma_full, deriv_sigma = _cartesian_stress_derivatives_to_cylindrical(
            grid_points,
            sigma_cart,
            grad_sigma_cart,
        )
        sigma_normal, sigma_shear, div_full, div_normal, div_shear = _cylindrical_divergence_split(
            grid_coords,
            sigma_full,
            deriv_sigma,
        )

        directions = _active_direction_from_quaternion(particles.orientation)
        ones = np.ones(len(positions), dtype=np.float64)
        rho_density = _density_sum(
            grid_points,
            positions,
            ones,
            box_length_x,
            pocket_radius,
        )
        polar_cart = _density_sum(
            grid_points,
            positions,
            directions,
            box_length_x,
            pocket_radius,
        )
        pair_force_density_cart = _density_sum(
            grid_points,
            positions,
            reconstructed_particle_pair_forces,
            box_length_x,
            pocket_radius,
        )
        wall_force_density_cart = _density_sum(
            grid_points,
            positions,
            wall_particle_forces,
            box_length_x,
            pocket_radius,
        )
        logged_force_density_cart = _density_sum(
            grid_points,
            positions,
            all_logged_forces,
            box_length_x,
            pocket_radius,
        )

    polar_density = _cartesian_vector_to_cylindrical_components(grid_points, polar_cart)
    pair_force_density = _cartesian_vector_to_cylindrical_components(
        grid_points,
        pair_force_density_cart,
    )
    wall_force_density = _cartesian_vector_to_cylindrical_components(
        grid_points,
        wall_force_density_cart,
    )
    logged_force_density = _cartesian_vector_to_cylindrical_components(
        grid_points,
        logged_force_density_cart,
    )
    j_active = CYLINDER_SIM.u0 * polar_density
    j_normal = div_normal / CYLINDER_SIM.gamma
    j_shear = div_shear / CYLINDER_SIM.gamma
    j_wall = wall_force_density / CYLINDER_SIM.gamma
    j_total = j_active + j_normal + j_shear
    j_total_with_wall = j_total + j_wall
    j_force_baseline = j_active + logged_force_density / CYLINDER_SIM.gamma
    div_split_residual = div_normal + div_shear - div_full
    j_split_residual = j_active + j_normal + j_shear - j_total
    near_wall_mask = grid_coords[:, 1] >= (CYLINDER.cylinder_radius - pocket_radius)

    pair_force_correlation, pair_force_slope = _component_stats(
        logged_pair_forces,
        reconstructed_particle_pair_forces,
    )
    stress_force_correlation, stress_force_slope = _component_stats(
        pair_force_density,
        div_full,
    )

    return ShearFluxDecomposition(
        step=step,
        frame_index=frame_index,
        pocket_radius=pocket_radius,
        x_edges=x_edges,
        x_centers=x_centers,
        r_edges=r_edges,
        r_centers=r_centers,
        theta_edges=theta_edges,
        theta_centers=theta_centers,
        grid_coords=grid_coords,
        grid_points=grid_points,
        rho_density=rho_density,
        sigma_full=sigma_full,
        deriv_sigma_full=deriv_sigma,
        sigma_normal=sigma_normal,
        sigma_shear=sigma_shear,
        div_sigma_full=div_full,
        div_sigma_normal=div_normal,
        div_sigma_shear=div_shear,
        div_split_residual=div_split_residual,
        polar_density=polar_density,
        pair_force_density=pair_force_density,
        wall_force_density=wall_force_density,
        logged_force_density=logged_force_density,
        j_active=j_active,
        j_normal=j_normal,
        j_shear=j_shear,
        j_wall=j_wall,
        j_total=j_total,
        j_total_with_wall=j_total_with_wall,
        j_force_baseline=j_force_baseline,
        j_split_residual=j_split_residual,
        near_wall_mask=near_wall_mask,
        pair_count=len(pair_delta),
        pair_force_correlation=pair_force_correlation,
        pair_force_slope=pair_force_slope,
        stress_force_correlation=stress_force_correlation,
        stress_force_slope=stress_force_slope,
    )


def compute_shear_flux_decomposition_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    frame_indices: Iterable[int] | None = None,
    pocket_radius: float = LOCAL_POCKET_RADIUS,
    dx: float = ACTIVE_GRID_DX,
    dr: float = ACTIVE_GRID_DY,
    n_theta_bins: int = ACTIVE_FLUX_PLOT_THETA_BINS,
) -> list[ShearFluxDecomposition]:
    """Compute the shear/stress decomposition for a sequence of frames."""
    input_gsd = Path(input_gsd)
    if frame_indices is None:
        with gsd.hoomd.open(input_gsd, mode="r") as source:
            frame_indices = range(len(source))

    return [
        compute_shear_flux_decomposition(
            input_gsd=input_gsd,
            frame_index=int(frame_index),
            pocket_radius=pocket_radius,
            dx=dx,
            dr=dr,
            n_theta_bins=n_theta_bins,
        )
        for frame_index in frame_indices
    ]
