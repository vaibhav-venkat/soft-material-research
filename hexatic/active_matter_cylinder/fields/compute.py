from pathlib import Path

import gsd.hoomd
import numpy as np

try:
    from hexatic import analysis as hx
except ImportError:
    import analysis as hx

from ..common import (
    _active_direction_from_quaternion,
    _cylindrical_components,
    _logged_particle_array,
    _pocket_fields,
    _pocket_vector_density,
    _theta_edges_and_centers,
    _x_edges_and_centers,
)
from ..config import (
    ACTIVE_FIELD_THETA_BINS,
    ACTIVE_FIELD_X_BINS,
    CYLINDER,
    CYLINDER_SIM,
    LOCAL_POCKET_RADIUS,
    ActiveMatterFields,
)


def active_matter_field_series(
    input_gsd: str | Path,
    pocket_radius: float = LOCAL_POCKET_RADIUS,
    n_x_bins: int = ACTIVE_FIELD_X_BINS,
    n_theta_bins: int = ACTIVE_FIELD_THETA_BINS,
    cylinder_radius: float = CYLINDER.cylinder_radius,
    wall_cutoff: float = CYLINDER.wall_cutoff,
) -> ActiveMatterFields:
    steps: list[int] = []
    x_edges: np.ndarray | None = None
    x_centers: np.ndarray | None = None
    theta_edges, theta_centers = _theta_edges_and_centers(n_theta_bins)

    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        assert len(source) > 0
        n_frames = len(source)
        n_particles = int(source[0].particles.N)

        coords = np.full((n_frames, n_particles, 3), np.nan, dtype=np.float64)
        shell_masks = np.zeros((n_frames, n_particles), dtype=np.bool_)
        rho = np.zeros((n_frames, n_particles), dtype=np.float64)
        active_direction = np.zeros((n_frames, n_particles, 3), dtype=np.float64)
        direction_cylindrical = np.zeros((n_frames, n_particles, 3), dtype=np.float64)
        polar_mean = np.zeros((n_frames, n_particles, 3), dtype=np.float64)
        polar_cylindrical = np.zeros((n_frames, n_particles, 3), dtype=np.float64)
        flux_cylindrical = np.zeros((n_frames, n_particles, 3), dtype=np.float64)
        force_density_values = np.zeros(
            (n_frames, n_particles, 3),
            dtype=np.float64,
        )
        force_density_cylindrical = np.zeros(
            (n_frames, n_particles, 3),
            dtype=np.float64,
        )

        for frame_idx, frame in enumerate(source):
            particles = frame.particles
            assert particles.position is not None
            assert particles.orientation is not None
            positions = np.asarray(particles.position, dtype=np.float64)
            assert positions.shape == (n_particles, 3)

            box_length_x = float(frame.configuration.box[0])
            if x_edges is None or x_centers is None:
                x_edges, x_centers = _x_edges_and_centers(box_length_x, n_x_bins)

            directions = _active_direction_from_quaternion(particles.orientation)
            forces = _logged_particle_array(frame, "forces", n_particles)
            assert forces.ndim == 2 and forces.shape[1] >= 3

            frame_coords = hx.get_new_coords(positions)
            dynamic_values = hx.get_dynamic_values(
                positions,
                contain_all=False,
                cylinder_radius=cylinder_radius,
                cutoff=wall_cutoff,
            )
            pocket_rho, _, pocket_polar_density = _pocket_fields(
                positions,
                directions,
                box_length_x,
                pocket_radius,
            )
            force_velocity = forces[:, :3] / CYLINDER_SIM.gamma
            velocities = CYLINDER_SIM.u0 * directions + force_velocity
            pocket_force_density = _pocket_vector_density(
                positions,
                force_velocity,
                box_length_x,
                pocket_radius,
            )
            pocket_flux_density = _pocket_vector_density(
                positions,
                velocities,
                box_length_x,
                pocket_radius,
            )

            coords[frame_idx] = frame_coords
            shell_masks[frame_idx] = dynamic_values.shell_mask
            rho[frame_idx] = pocket_rho.astype(np.float64)
            active_direction[frame_idx] = directions
            polar_mean[frame_idx] = np.nan_to_num(pocket_polar_density, nan=0.0)
            force_density_values[frame_idx] = pocket_force_density
            direction_cylindrical[frame_idx] = _cylindrical_components(
                directions,
                frame_coords[:, 1],
            )
            polar_cylindrical[frame_idx] = _cylindrical_components(
                polar_mean[frame_idx],
                frame_coords[:, 1],
            )
            flux_cylindrical[frame_idx] = _cylindrical_components(
                pocket_flux_density,
                frame_coords[:, 1],
            )
            force_density_cylindrical[frame_idx] = _cylindrical_components(
                pocket_force_density,
                frame_coords[:, 1],
            )
            steps.append(int(frame.configuration.step))

    assert x_edges is not None and x_centers is not None
    return ActiveMatterFields(
        steps=np.asarray(steps, dtype=np.int64),
        x_edges=x_edges,
        x_centers=x_centers,
        theta_edges=theta_edges,
        theta_centers=theta_centers,
        coords=coords,
        shell_mask=shell_masks,
        rho=rho,
        active_direction=active_direction,
        direction_cylindrical=direction_cylindrical,
        polar_mean=polar_mean,
        polar_cylindrical=polar_cylindrical,
        flux_cylindrical=flux_cylindrical,
        force_density=force_density_values,
        force_density_cylindrical=force_density_cylindrical,
    )


def save_active_matter_fields(
    fields: ActiveMatterFields,
    filename: str | Path,
    pocket_radius: float = LOCAL_POCKET_RADIUS,
) -> None:
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        pocket_radius=np.asarray(pocket_radius, dtype=np.float64),
        steps=fields.steps,
        x_edges=fields.x_edges,
        x_centers=fields.x_centers,
        theta_edges=fields.theta_edges,
        theta_centers=fields.theta_centers,
        coords=fields.coords,
        shell_mask=fields.shell_mask,
        rho=fields.rho,
        active_direction=fields.active_direction,
        direction_cylindrical=fields.direction_cylindrical,
        polar_mean=fields.polar_mean,
        polar_cylindrical=fields.polar_cylindrical,
        flux_cylindrical=fields.flux_cylindrical,
        force_density=fields.force_density,
        force_density_cylindrical=fields.force_density_cylindrical,
    )
