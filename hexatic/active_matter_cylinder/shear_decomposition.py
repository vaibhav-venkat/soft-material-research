from dataclasses import dataclass
from pathlib import Path

import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import TwoSlopeNorm

try:
    from scipy.integrate import trapezoid
except ImportError:
    trapezoid = np.trapz

try:
    from scipy.special import erf as _erf
except ImportError:
    import math

    _erf = np.vectorize(math.erf, otypes=[np.float64])

from .common import (
    _active_direction_from_quaternion,
    _axis_edges_and_centers,
    _cartesian_vector_to_cylindrical_components,
    _cylindrical_basis,
    _density_sum,
    _logged_particle_array,
    _minimum_image_delta,
    _theta_edges_and_centers,
    _write_full_view_html,
)
from .config import (
    ACTIVE_FLUX_PLOT_THETA_BINS,
    ACTIVE_GRID_DX,
    ACTIVE_GRID_DY,
    ACTIVE_IMAGE_DIR,
    CYLINDER,
    CYLINDER_PATHS,
    CYLINDER_SIM,
    LOCAL_POCKET_RADIUS,
)


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


def _field_component_values(field: np.ndarray, component: str) -> np.ndarray:
    if component == "magnitude":
        return np.linalg.norm(field, axis=1)
    component_index = {"x": 0, "r": 1, "theta": 2}[component]
    return field[:, component_index]


def _valid_plot_indices(
    decomposition: ShearFluxDecomposition,
    max_points: int,
) -> np.ndarray:
    valid = np.isfinite(decomposition.grid_coords).all(axis=1) & np.isfinite(decomposition.rho_density)
    valid &= decomposition.rho_density > 0.0
    indices = np.flatnonzero(valid)
    if len(indices) > max_points:
        indices = indices[np.linspace(0, len(indices) - 1, max_points).astype(np.int64)]
    return indices


def _trace_map_literal(trace_indices: dict[str, int]) -> str:
    return "{" + ",".join(f'"{key}": {value}' for key, value in trace_indices.items()) + "}"


def _select_options(values: tuple[str, ...], selected: str) -> str:
    return "\n".join(
        f'<option value="{value}"{" selected" if value == selected else ""}>{value}</option>'
        for value in values
    )


def _symmetric_color_limit(values: np.ndarray, percentile: float | None = None) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(np.abs(finite), percentile)) if percentile else float(np.max(np.abs(finite)))
    return 1.0 if np.isclose(limit, 0.0) else limit


def _shear_scene_layout(coords: np.ndarray) -> dict:
    return {
        "xaxis": {"title": "x", "range": [float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))]},
        "yaxis": {"title": "r", "range": [0.0, CYLINDER.cylinder_radius]},
        "zaxis": {"title": "theta", "range": [0.0, 2.0 * np.pi]},
        "aspectmode": "data",
    }


def _grid_marker_trace(
    coords: np.ndarray,
    values: np.ndarray,
    name: str,
    colorbar_title: str,
    visible: bool,
    cmin: float,
    cmax: float,
    colorscale: str,
    hover_label: str = "value",
) -> go.Scatter3d:
    return go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker={
            "size": 3,
            "color": values,
            "colorscale": colorscale,
            "cmin": cmin,
            "cmax": cmax,
            "opacity": 0.8,
            "colorbar": {"title": colorbar_title},
        },
        name=name,
        visible=visible,
        customdata=values,
        hovertemplate=(
            "x=%{x:.3g}<br>r=%{y:.3g}<br>theta=%{z:.3g}"
            f"<br>{hover_label}=%{{customdata:.3g}}<extra></extra>"
        ),
    )


def _grid_shape(decomposition: ShearFluxDecomposition) -> tuple[int, int, int]:
    return (
        len(decomposition.x_centers),
        len(decomposition.r_centers),
        len(decomposition.theta_centers),
    )


def _reshape_grid_field(decomposition: ShearFluxDecomposition, values: np.ndarray) -> np.ndarray:
    return np.asarray(values).reshape(_grid_shape(decomposition) + np.asarray(values).shape[1:])


def _radial_trapezoid(values: np.ndarray, radii: np.ndarray, include_jacobian: bool) -> np.ndarray:
    integrand = values
    if include_jacobian:
        shape = (1, len(radii)) + (1,) * (values.ndim - 2)
        integrand = values * radii.reshape(shape)
    return trapezoid(integrand, x=radii, axis=1)


def _direct_radial_j_integral(decomposition: ShearFluxDecomposition) -> np.ndarray:
    values = _reshape_grid_field(decomposition, decomposition.j_total_with_wall)
    return _radial_trapezoid(values, decomposition.r_centers, include_jacobian=True)


def _formula_radial_j_integral(decomposition: ShearFluxDecomposition) -> np.ndarray:
    radii = decomposition.r_centers
    sigma = _reshape_grid_field(decomposition, decomposition.sigma_full)
    deriv_sigma = _reshape_grid_field(decomposition, decomposition.deriv_sigma_full)
    nonstress_j = _reshape_grid_field(
        decomposition,
        decomposition.j_active + decomposition.j_wall,
    )

    def sigma_int(a: int, b: int, include_jacobian: bool) -> np.ndarray:
        return _radial_trapezoid(sigma[..., a, b], radii, include_jacobian)

    def deriv_int(direction: int, a: int, b: int, include_jacobian: bool) -> np.ndarray:
        return _radial_trapezoid(deriv_sigma[..., direction, a, b], radii, include_jacobian)

    def radial_boundary(a: int, b: int) -> np.ndarray:
        return radii[-1] * sigma[:, -1, :, a, b] - radii[0] * sigma[:, 0, :, a, b]

    nx, _, ntheta = _grid_shape(decomposition)
    div_integral = np.empty((nx, ntheta, 3), dtype=np.float64)
    div_integral[..., 0] = (
        deriv_int(0, 0, 0, True)
        + deriv_int(2, 0, 2, False)
        + radial_boundary(0, 1)
    )
    div_integral[..., 1] = (
        deriv_int(0, 1, 0, True)
        + deriv_int(2, 1, 2, False)
        + radial_boundary(1, 1)
        - sigma_int(2, 2, False)
    )
    div_integral[..., 2] = (
        deriv_int(0, 2, 0, True)
        + deriv_int(2, 2, 2, False)
        + radial_boundary(2, 1)
        + sigma_int(1, 2, False)
    )

    return (
        _radial_trapezoid(nonstress_j, radii, include_jacobian=True)
        + div_integral / CYLINDER_SIM.gamma
    )


def _plot_radial_j_integral_components(
    decomposition: ShearFluxDecomposition,
    values: np.ndarray,
    filename: Path,
    title: str,
) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    components = (("x", 0), ("theta", 2), ("r", 1))
    limit = _symmetric_color_limit(values)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, (label, component_index) in zip(axes, components):
        mesh = axis.pcolormesh(
            decomposition.x_edges,
            decomposition.theta_edges,
            values[..., component_index].T,
            shading="auto",
            cmap="coolwarm",
            norm=norm,
        )
        fig.colorbar(mesh, ax=axis, label=rf"$\int r J_{label}\,dr$")
        axis.set_xlabel("x")
        axis.set_title(f"{label} component")
    axes[0].set_ylabel("theta")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def plot_radial_j_integral_comparison(
    decomposition: ShearFluxDecomposition,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
) -> None:
    image_path = Path(image_dir) / "shear"
    _plot_radial_j_integral_components(
        decomposition,
        _direct_radial_j_integral(decomposition),
        image_path / "hardy_j_radial_integral_direct.png",
        (
            "Direct radial integral of "
            r"$J_\mathrm{total}=U_0P+\gamma^{-1}\nabla\cdot\sigma+J_\mathrm{wall}$"
        ),
    )
    _plot_radial_j_integral_components(
        decomposition,
        _formula_radial_j_integral(decomposition),
        image_path / "hardy_j_radial_integral_formula.png",
        "Radial integral from integrated stress-divergence identities plus active and wall terms",
    )


def plot_shear_flux_decomposition(
    decomposition: ShearFluxDecomposition,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
    max_points: int = 9000,
) -> None:
    output_path = Path(image_dir) / "shear" / "hardy_shear_flux_decomposition.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = {
        "J_active": decomposition.j_active,
        "J_normal": decomposition.j_normal,
        "J_shear": decomposition.j_shear,
        "J_wall": decomposition.j_wall,
        "J_pair_total": decomposition.j_total,
        "J_total_with_wall": decomposition.j_total_with_wall,
        "J_force_baseline": decomposition.j_force_baseline,
    }
    components = ("x", "r", "theta", "magnitude")
    indices = _valid_plot_indices(decomposition, max_points)

    if len(indices) == 0:
        fig = go.Figure()
        fig.update_layout(title="flux decomposition: no occupied grid points")
        fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
        return

    coords = decomposition.grid_coords[indices]
    traces = []
    trace_indices: dict[str, int] = {}
    for field_name, field_values in fields.items():
        for component in components:
            key = f"{field_name}:{component}"
            trace_indices[key] = len(traces)
            values = _field_component_values(field_values[indices], component)
            finite = values[np.isfinite(values)]
            if component == "magnitude":
                cmin = 0.0
                cmax = float(np.max(finite)) if finite.size else 1.0
                colorscale = "Plasma"
            else:
                limit = _symmetric_color_limit(values)
                cmin = -limit
                cmax = limit
                colorscale = "RdBu"
            traces.append(
                _grid_marker_trace(
                    coords,
                    values,
                    name=key,
                    colorbar_title=component,
                    visible=(field_name == "J_total_with_wall" and component == "magnitude"),
                    cmin=cmin,
                    cmax=cmax,
                    colorscale=colorscale,
                )
            )

    labels = {
        "J_active": "active",
        "J_normal": "normal stress",
        "J_shear": "shear stress",
        "J_wall": "wall force",
        "J_pair_total": "active + normal + shear",
        "J_total_with_wall": "active + normal + shear + wall",
        "J_force_baseline": "force-density baseline",
    }
    field_options = "\n".join(
        f'<option value="{name}"{" selected" if name == "J_total_with_wall" else ""}>{label}</option>'
        for name, label in labels.items()
    )
    component_options = _select_options(components, "magnitude")
    trace_map = _trace_map_literal(trace_indices)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Flux decomposition: J_total_with_wall magnitude, step {decomposition.step}",
        scene=_shear_scene_layout(coords),
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
    )

    div_id = "hardy-shear-flux-decomposition-plot"
    controls = f"""
<div class="controls">
  <label for="shear-field">Field:&nbsp;</label>
  <select id="shear-field">{field_options}</select>
  <label for="shear-component" style="margin-left: 18px;">Component:&nbsp;</label>
  <select id="shear-component">{component_options}</select>
</div>
"""
    script = f"""
<script>
const shearTraceMap = {trace_map};
const shearTraceCount = {len(traces)};
const shearStep = {decomposition.step};
function updateShearTrace() {{
  const field = document.getElementById("shear-field").value;
  const component = document.getElementById("shear-component").value;
  const key = field + ":" + component;
  const visible = Array(shearTraceCount).fill(false);
  visible[shearTraceMap[key]] = true;
  Plotly.restyle("{div_id}", {{"visible": visible}});
  Plotly.relayout(
    "{div_id}",
    {{"title.text": "Flux decomposition: "
      + field + " " + component + ", step " + shearStep}}
  );
}}
document.getElementById("shear-field").addEventListener("change", updateShearTrace);
document.getElementById("shear-component").addEventListener("change", updateShearTrace);
</script>
"""
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id=div_id,
        default_width="100%",
        default_height="100%",
    )
    _write_full_view_html(output_path, div_id, controls, plot_html, script)


def plot_shear_stress_tensor_components(
    decomposition: ShearFluxDecomposition,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
    max_points: int = 9000,
) -> None:
    output_path = Path(image_dir) / "shear" / "hardy_sigma_shear_tensor.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ("x", "r", "theta")
    indices = _valid_plot_indices(decomposition, max_points)
    if len(indices) == 0:
        fig = go.Figure()
        fig.update_layout(title="Hardy stress tensor: no occupied grid points")
        fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
        return

    coords = decomposition.grid_coords[indices]
    traces = []
    trace_indices: dict[str, int] = {}
    for component_index, component_label in enumerate(labels):
        for normal_index, normal_label in enumerate(labels):
            key = f"{component_label}:{normal_label}"
            trace_indices[key] = len(traces)
            values = decomposition.sigma_full[indices, component_index, normal_index]
            limit = _symmetric_color_limit(values)
            traces.append(
                _grid_marker_trace(
                    coords,
                    values,
                    name=f"sigma_{component_label}{normal_label}",
                    visible=(component_label == "x" and normal_label == "r"),
                    colorbar_title="sigma",
                    cmin=-limit,
                    cmax=limit,
                    colorscale="RdBu",
                    hover_label="sigma",
                )
            )

    options = _select_options(labels, "x")
    normal_options = _select_options(labels, "r")
    trace_map = _trace_map_literal(trace_indices)
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Hardy stress tensor: sigma_xr, step {decomposition.step}",
        scene=_shear_scene_layout(coords),
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
    )

    div_id = "hardy-sigma-shear-tensor-plot"
    controls = f"""
<div class="controls">
  <label for="sigma-component">Component a:&nbsp;</label>
  <select id="sigma-component">{options}</select>
  <label for="sigma-normal" style="margin-left: 18px;">Normal b:&nbsp;</label>
  <select id="sigma-normal">{normal_options}</select>
</div>
"""
    script = f"""
<script>
const sigmaTraceMap = {trace_map};
const sigmaTraceCount = {len(traces)};
const sigmaStep = {decomposition.step};
function updateSigmaTrace() {{
  const component = document.getElementById("sigma-component").value;
  const normal = document.getElementById("sigma-normal").value;
  const key = component + ":" + normal;
  const visible = Array(sigmaTraceCount).fill(false);
  visible[sigmaTraceMap[key]] = true;
  Plotly.restyle("{div_id}", {{"visible": visible}});
  Plotly.relayout(
    "{div_id}",
    {{"title.text": "Hardy stress tensor: sigma_" + component + normal + ", step " + sigmaStep}}
  );
}}
document.getElementById("sigma-component").addEventListener("change", updateSigmaTrace);
document.getElementById("sigma-normal").addEventListener("change", updateSigmaTrace);
</script>
"""
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id=div_id,
        default_width="100%",
        default_height="100%",
    )
    _write_full_view_html(output_path, div_id, controls, plot_html, script)


def plot_shear_flux_fraction(
    decomposition: ShearFluxDecomposition,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
    max_points: int = 9000,
) -> None:
    output_path = Path(image_dir) / "shear" / "hardy_j_shear_fraction.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    components = ("x", "r", "theta", "magnitude")
    indices = _valid_plot_indices(decomposition, max_points)
    if len(indices) == 0:
        fig = go.Figure()
        fig.update_layout(title="J_shear/J_total: no occupied grid points")
        fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
        return

    coords = decomposition.grid_coords[indices]
    shear = decomposition.j_shear[indices]
    total = decomposition.j_total[indices]
    traces = []
    trace_indices: dict[str, int] = {}
    fraction_means: dict[str, float] = {}
    for component in components:
        if component == "magnitude":
            numerator = np.linalg.norm(shear, axis=1)
            denominator = np.linalg.norm(total, axis=1)
        else:
            component_index = {"x": 0, "r": 1, "theta": 2}[component]
            numerator = shear[:, component_index]
            denominator = total[:, component_index]
        values = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=np.float64),
            where=np.abs(denominator) > 1e-12,
        )
        finite = values[np.isfinite(values)]
        fraction_means[component] = float(np.mean(finite)) if finite.size else float("nan")
        if component == "magnitude":
            cmin = 0.0
            cmax = float(np.nanpercentile(finite, 99.0)) if finite.size else 1.0
            if np.isclose(cmax, 0.0):
                cmax = 1.0
            colorscale = "Plasma"
        else:
            limit = _symmetric_color_limit(values, percentile=99.0)
            cmin = -limit
            cmax = limit
            colorscale = "RdBu"
        trace_indices[component] = len(traces)
        traces.append(
            _grid_marker_trace(
                coords,
                values,
                name=f"J_shear/J_total {component}",
                visible=(component == "magnitude"),
                colorbar_title="J_shear/J_total",
                cmin=cmin,
                cmax=cmax,
                colorscale=colorscale,
                hover_label="fraction",
            )
        )

    component_options = _select_options(components, "magnitude")
    trace_map = _trace_map_literal(trace_indices)
    mean_map = "{" + ",".join(
        f'"{key}": {value:.17g}' if np.isfinite(value) else f'"{key}": NaN'
        for key, value in fraction_means.items()
    ) + "}"
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Hardy J_shear / (J_active + J_normal + J_shear): magnitude, step {decomposition.step}",
        scene=_shear_scene_layout(coords),
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
    )

    div_id = "hardy-j-shear-fraction-plot"
    controls = f"""
<div class="controls">
  <label for="fraction-component">Component:&nbsp;</label>
  <select id="fraction-component">{component_options}</select>
  <span id="fraction-mean" style="margin-left: 18px;">mean = {fraction_means["magnitude"]:.6g}</span>
</div>
"""
    script = f"""
<script>
const fractionTraceMap = {trace_map};
const fractionMeanMap = {mean_map};
const fractionTraceCount = {len(traces)};
const fractionStep = {decomposition.step};
function updateFractionTrace() {{
  const component = document.getElementById("fraction-component").value;
  const visible = Array(fractionTraceCount).fill(false);
  visible[fractionTraceMap[component]] = true;
  Plotly.restyle("{div_id}", {{"visible": visible}});
  Plotly.relayout(
    "{div_id}",
    {{"title.text": "Hardy J_shear / (J_active + J_normal + J_shear): " + component + ", step " + fractionStep}}
  );
  const meanValue = fractionMeanMap[component];
  document.getElementById("fraction-mean").textContent =
    Number.isFinite(meanValue) ? "mean = " + meanValue.toPrecision(6) : "mean = NaN";
}}
document.getElementById("fraction-component").addEventListener("change", updateFractionTrace);
</script>
"""
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id=div_id,
        default_width="100%",
        default_height="100%",
    )
    _write_full_view_html(output_path, div_id, controls, plot_html, script)
