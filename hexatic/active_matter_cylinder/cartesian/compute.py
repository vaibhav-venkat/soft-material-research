from pathlib import Path

import gsd.hoomd
import numpy as np

from ..common import (
    _active_direction_from_quaternion,
    _axis_edges_and_centers,
    _cartesian_tensor_to_cylindrical,
    _density_sum,
    _gaussian_kernel_volume,
    _logged_particle_array,
    _minimum_image_delta,
)
from ..config import (
    ACTIVE_GRID_DX,
    ACTIVE_GRID_DY,
    ACTIVE_GRID_DZ,
    CYLINDER,
    CYLINDER_SIM,
    LOCAL_POCKET_RADIUS,
    VIRIAL_STRESS_SIGN,
    CartesianFluxComparison,
)

def _delta_volume(radius: float) -> float:
    return _gaussian_kernel_volume(radius)


def _cartesian_grid_points(
    box_length_x: float,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    _, x_centers = _axis_edges_and_centers(-0.5 * box_length_x, 0.5 * box_length_x, dx)
    _, y_centers = _axis_edges_and_centers(-CYLINDER.cylinder_radius, CYLINDER.cylinder_radius, dy)
    _, z_centers = _axis_edges_and_centers(-CYLINDER.cylinder_radius, CYLINDER.cylinder_radius, dz)
    x_grid, y_grid, z_grid = np.meshgrid(
        x_centers,
        y_centers,
        z_centers,
        indexing="ij",
    )
    points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
    inside_cylinder = points[:, 1] ** 2 + points[:, 2] ** 2 <= CYLINDER.cylinder_radius**2
    return points[inside_cylinder]


def _virial_tensor_from_components(virials: np.ndarray) -> np.ndarray:
    virials = np.asarray(virials, dtype=np.float64)
    if virials.ndim == 3 and virials.shape[1:] == (3, 3):
        return virials
    assert virials.ndim == 2 and virials.shape[1] >= 6
    tensor = np.zeros((virials.shape[0], 3, 3), dtype=np.float64)
    tensor[:, 0, 0] = virials[:, 0]
    tensor[:, 0, 1] = tensor[:, 1, 0] = virials[:, 1]
    tensor[:, 0, 2] = tensor[:, 2, 0] = virials[:, 2]
    tensor[:, 1, 1] = virials[:, 3]
    tensor[:, 1, 2] = tensor[:, 2, 1] = virials[:, 4]
    tensor[:, 2, 2] = virials[:, 5]
    return tensor


def _regular_grid_indices(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]:
    x_values = np.unique(points[:, 0])
    y_values = np.unique(points[:, 1])
    z_values = np.unique(points[:, 2])
    x_indices = np.searchsorted(x_values, points[:, 0])
    y_indices = np.searchsorted(y_values, points[:, 1])
    z_indices = np.searchsorted(z_values, points[:, 2])
    return x_indices, y_indices, z_indices, (len(x_values), len(y_values), len(z_values))


def _central_derivative(values: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    forward = np.roll(values, -1, axis=axis)
    backward = np.roll(values, 1, axis=axis)
    valid = np.isfinite(forward) & np.isfinite(backward)
    derivative = np.full_like(values, np.nan, dtype=np.float64)
    derivative[valid] = (forward[valid] - backward[valid]) / (2.0 * spacing)

    first = [slice(None)] * values.ndim
    first[axis] = 0
    derivative[tuple(first)] = np.nan

    last = [slice(None)] * values.ndim
    last[axis] = -1
    derivative[tuple(last)] = np.nan
    return derivative


def _cartesian_stress_divergence(
    points: np.ndarray,
    stress: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    x_indices, y_indices, z_indices, shape = _regular_grid_indices(points)
    stress_grid = np.full(shape + (3, 3), np.nan, dtype=np.float64)
    stress_grid[x_indices, y_indices, z_indices] = stress

    div_grid = np.zeros(shape + (3,), dtype=np.float64)
    valid = np.ones(shape, dtype=bool)
    for component_index in range(3):
        component_terms = []
        for normal_index in range(3):
            derivative = _central_derivative(
                stress_grid[..., component_index, normal_index],
                float(spacing[normal_index]),
                axis=normal_index,
            )
            component_terms.append(derivative)
            valid &= np.isfinite(derivative)
        div_grid[..., component_index] = np.sum(component_terms, axis=0)

    divergence = div_grid[x_indices, y_indices, z_indices]
    divergence[~valid[x_indices, y_indices, z_indices]] = np.nan
    return divergence


def _finite_difference_velocity(
    positions: np.ndarray,
    next_positions: np.ndarray,
    box_length_x: float,
    dt: float,
) -> np.ndarray:
    assert dt > 0.0
    displacement = next_positions - positions
    displacement[:, 0] = _minimum_image_delta(displacement[:, 0], box_length_x)
    return displacement / dt


def compute_cartesian_flux_comparison(
    input_gsd: str | Path,
    pocket_radius: float = LOCAL_POCKET_RADIUS,
    dx: float = ACTIVE_GRID_DX,
    dy: float = ACTIVE_GRID_DY,
    dz: float = ACTIVE_GRID_DZ,
    frame_index: int = -2,
) -> CartesianFluxComparison:
    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        assert len(source) >= 2
        if frame_index < 0:
            frame_index += len(source)
        assert 0 <= frame_index < len(source) - 1

        frame = source[frame_index]
        next_frame = source[frame_index + 1]
        particles = frame.particles
        next_particles = next_frame.particles
        assert particles.position is not None
        assert particles.orientation is not None
        assert next_particles.position is not None
        assert next_particles.orientation is not None
        assert int(particles.N) == int(next_particles.N)

        positions = np.asarray(particles.position, dtype=np.float64)
        next_positions = np.asarray(next_particles.position, dtype=np.float64)
        box_length_x = float(frame.configuration.box[0])
        step = int(frame.configuration.step)
        next_step = int(next_frame.configuration.step)
        dt = float(next_step - step) * CYLINDER_SIM.timestep

        directions = _active_direction_from_quaternion(particles.orientation)
        next_directions = _active_direction_from_quaternion(next_particles.orientation)
        forces = _logged_particle_array(frame, "forces", int(particles.N))
        next_forces = _logged_particle_array(next_frame, "forces", int(particles.N))
        virials = _logged_particle_array(frame, "virials", int(particles.N))
        next_virials = _logged_particle_array(next_frame, "virials", int(particles.N))
        force_divergence_values = forces[:, :3]
        force_velocity = force_divergence_values / CYLINDER_SIM.gamma
        instantaneous_velocity = CYLINDER_SIM.u0 * directions + force_velocity
        finite_difference_velocity = _finite_difference_velocity(
            positions,
            next_positions,
            box_length_x,
            dt,
        )

        grid_points = _cartesian_grid_points(box_length_x, dx, dy, dz)
        ones = np.ones(len(positions), dtype=np.float64)
        rho_density = _density_sum(
            grid_points,
            positions,
            ones,
            box_length_x,
            pocket_radius,
        )
        polar_density = _density_sum(
            grid_points,
            positions,
            directions,
            box_length_x,
            pocket_radius,
        )
        next_polar_density = _density_sum(
            grid_points,
            next_positions,
            next_directions,
            box_length_x,
            pocket_radius,
        )
        finite_time_polar_density = 0.5 * (polar_density + next_polar_density)
        virial_divergence_density = _density_sum(
            grid_points,
            positions,
            force_divergence_values,
            box_length_x,
            pocket_radius,
        )
        next_virial_divergence_density = _density_sum(
            grid_points,
            next_positions,
            next_forces[:, :3],
            box_length_x,
            pocket_radius,
        )
        finite_time_virial_divergence_density = 0.5 * (
            virial_divergence_density + next_virial_divergence_density
        )
        force_density = _density_sum(
            grid_points,
            positions,
            force_velocity,
            box_length_x,
            pocket_radius,
        )
        instantaneous_flux_density = _density_sum(
            grid_points,
            positions,
            instantaneous_velocity,
            box_length_x,
            pocket_radius,
        )
        finite_difference_flux_density = _density_sum(
            grid_points,
            positions,
            finite_difference_velocity,
            box_length_x,
            pocket_radius,
        )
        virial_stress_density = _density_sum(
            grid_points,
            positions,
            _virial_tensor_from_components(virials),
            box_length_x,
            pocket_radius,
        )
        next_virial_stress_density = _density_sum(
            grid_points,
            next_positions,
            _virial_tensor_from_components(next_virials),
            box_length_x,
            pocket_radius,
        )
        finite_time_virial_stress_density = 0.5 * (
            virial_stress_density + next_virial_stress_density
        )
        virial_stress_cylindrical = VIRIAL_STRESS_SIGN * _cartesian_tensor_to_cylindrical(
            grid_points,
            virial_stress_density,
        )
        instantaneous_stress_flux_density = (
            CYLINDER_SIM.u0 * polar_density
            + virial_divergence_density / CYLINDER_SIM.gamma
        )
        finite_time_stress_flux_density = (
            CYLINDER_SIM.u0 * finite_time_polar_density
            + finite_time_virial_divergence_density / CYLINDER_SIM.gamma
        )

    return CartesianFluxComparison(
        step=step,
        next_step=next_step,
        frame_index=frame_index,
        dt=dt,
        pocket_radius=pocket_radius,
        delta_volume=_delta_volume(pocket_radius),
        grid_spacing=np.asarray([dx, dy, dz], dtype=np.float64),
        grid_points=grid_points,
        rho_density=rho_density,
        polar_density=polar_density,
        force_density=force_density,
        instantaneous_flux_density=instantaneous_flux_density,
        finite_difference_flux_density=finite_difference_flux_density,
        virial_stress_density=virial_stress_density,
        virial_divergence_density=virial_divergence_density,
        instantaneous_stress_flux_density=instantaneous_stress_flux_density,
        finite_time_polar_density=finite_time_polar_density,
        finite_time_virial_stress_density=finite_time_virial_stress_density,
        finite_time_virial_divergence_density=finite_time_virial_divergence_density,
        finite_time_stress_flux_density=finite_time_stress_flux_density,
        virial_stress_cylindrical=virial_stress_cylindrical,
    )


def save_cartesian_flux_comparison(
    comparison: CartesianFluxComparison,
    filename: str | Path,
) -> None:
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        step=np.asarray(comparison.step, dtype=np.int64),
        next_step=np.asarray(comparison.next_step, dtype=np.int64),
        frame_index=np.asarray(comparison.frame_index, dtype=np.int64),
        dt=np.asarray(comparison.dt, dtype=np.float64),
        pocket_radius=np.asarray(comparison.pocket_radius, dtype=np.float64),
        delta_volume=np.asarray(comparison.delta_volume, dtype=np.float64),
        grid_spacing=comparison.grid_spacing,
        grid_points=comparison.grid_points,
        rho_density=comparison.rho_density,
        polar_density=comparison.polar_density,
        force_density=comparison.force_density,
        instantaneous_flux_density=comparison.instantaneous_flux_density,
        finite_difference_flux_density=comparison.finite_difference_flux_density,
        virial_stress_density=comparison.virial_stress_density,
        virial_divergence_density=comparison.virial_divergence_density,
        instantaneous_stress_flux_density=comparison.instantaneous_stress_flux_density,
        finite_time_polar_density=comparison.finite_time_polar_density,
        finite_time_virial_stress_density=comparison.finite_time_virial_stress_density,
        finite_time_virial_divergence_density=comparison.finite_time_virial_divergence_density,
        finite_time_stress_flux_density=comparison.finite_time_stress_flux_density,
        virial_stress_cylindrical=comparison.virial_stress_cylindrical,
    )

