from pathlib import Path

import gsd.hoomd
import numpy as np
import plotly.graph_objects as go

from .common import (
    _active_direction_from_quaternion,
    _logged_particle_array,
    _minimum_image_delta,
)
from .config import (
    ACTIVE_GRID_DX,
    ACTIVE_GRID_DY,
    ACTIVE_GRID_DZ,
    ACTIVE_IMAGE_DIR,
    CYLINDER,
    CYLINDER_SIM,
    LOCAL_POCKET_RADIUS,
    CartesianFluxComparison,
)


def _delta_volume(radius: float) -> float:
    assert radius > 0.0
    return 4.0 * np.pi * radius**3 / 3.0


def _axis_edges_and_centers(low: float, high: float, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    assert high > low
    assert spacing > 0.0
    n_bins = int(np.ceil((high - low) / spacing))
    edges = low + spacing * np.arange(n_bins + 1)
    edges[-1] = high
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


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


def _cartesian_density_sum(
    grid_points: np.ndarray,
    positions: np.ndarray,
    values: np.ndarray,
    box_length_x: float,
    radius: float,
    block_size: int = 512,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    is_scalar = values.ndim == 1
    if is_scalar:
        density = np.zeros(len(grid_points), dtype=np.float64)
    else:
        density = np.zeros((len(grid_points), values.shape[1]), dtype=np.float64)

    cutoff_sq = radius * radius
    volume = _delta_volume(radius)
    for start in range(0, len(grid_points), block_size):
        stop = min(start + block_size, len(grid_points))
        deltas = grid_points[start:stop, np.newaxis, :] - positions[np.newaxis, :, :]
        deltas[..., 0] = _minimum_image_delta(deltas[..., 0], box_length_x)
        pocket_mask = np.sum(deltas * deltas, axis=2) <= cutoff_sq
        if is_scalar:
            density[start:stop] = pocket_mask.astype(np.float64) @ values
        else:
            density[start:stop] = pocket_mask.astype(np.float64) @ values

    return density / volume


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
        assert int(particles.N) == int(next_particles.N)

        positions = np.asarray(particles.position, dtype=np.float64)
        next_positions = np.asarray(next_particles.position, dtype=np.float64)
        box_length_x = float(frame.configuration.box[0])
        step = int(frame.configuration.step)
        next_step = int(next_frame.configuration.step)
        dt = float(next_step - step) * CYLINDER_SIM.timestep

        directions = _active_direction_from_quaternion(particles.orientation)
        forces = _logged_particle_array(frame, "forces", int(particles.N))
        force_velocity = forces[:, :3] / CYLINDER_SIM.gamma
        instantaneous_velocity = CYLINDER_SIM.u0 * directions + force_velocity
        finite_difference_velocity = _finite_difference_velocity(
            positions,
            next_positions,
            box_length_x,
            dt,
        )

        grid_points = _cartesian_grid_points(box_length_x, dx, dy, dz)
        ones = np.ones(len(positions), dtype=np.float64)
        rho_density = _cartesian_density_sum(
            grid_points,
            positions,
            ones,
            box_length_x,
            pocket_radius,
        )
        polar_density = _cartesian_density_sum(
            grid_points,
            positions,
            directions,
            box_length_x,
            pocket_radius,
        )
        force_density = _cartesian_density_sum(
            grid_points,
            positions,
            force_velocity,
            box_length_x,
            pocket_radius,
        )
        instantaneous_flux_density = _cartesian_density_sum(
            grid_points,
            positions,
            instantaneous_velocity,
            box_length_x,
            pocket_radius,
        )
        finite_difference_flux_density = _cartesian_density_sum(
            grid_points,
            positions,
            finite_difference_velocity,
            box_length_x,
            pocket_radius,
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
    )


def _plot_flux_density_3d(
    comparison: CartesianFluxComparison,
    vectors: np.ndarray,
    filename: str | Path,
    title: str,
    max_vectors: int = 900,
) -> None:
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points = comparison.grid_points
    magnitudes = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(magnitudes) & (comparison.rho_density > 0.0)
    plot_points = points[valid]
    plot_vectors = vectors[valid]
    plot_magnitudes = magnitudes[valid]

    if len(plot_points) > max_vectors:
        indices = np.linspace(0, len(plot_points) - 1, max_vectors).astype(np.int64)
        vector_points = plot_points[indices]
        vector_values = plot_vectors[indices]
        vector_magnitudes = plot_magnitudes[indices]
    else:
        vector_points = plot_points
        vector_values = plot_vectors
        vector_magnitudes = plot_magnitudes

    traces = []
    if len(plot_points) == 0:
        traces.append(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={"size": 2, "color": "lightgray", "opacity": 0.25},
                name="empty cylinder grid",
            )
        )

    if len(vector_points) > 0:
        finite_magnitudes = vector_magnitudes[np.isfinite(vector_magnitudes)]
        scale = float(np.max(finite_magnitudes)) if finite_magnitudes.size else 1.0
        if np.isclose(scale, 0.0):
            scale = 1.0
        unit_vectors = np.divide(
            vector_values,
            vector_magnitudes[:, np.newaxis],
            out=np.zeros_like(vector_values, dtype=np.float64),
            where=vector_magnitudes[:, np.newaxis] > 0.0,
        )
        length_scale = 2.0 * float(np.min(comparison.grid_spacing))
        scaled_lengths = np.clip(vector_magnitudes / scale, 0.0, 1.0) * length_scale
        cone_vectors = unit_vectors * scaled_lengths[:, np.newaxis]
        color_min = 0.0
        color_max = length_scale
        tick_values = np.linspace(color_min, color_max, 6)
        tick_text = [f"{value / length_scale * scale:.3g}" for value in tick_values]
        traces.append(
            go.Cone(
                x=vector_points[:, 0],
                y=vector_points[:, 1],
                z=vector_points[:, 2],
                u=cone_vectors[:, 0],
                v=cone_vectors[:, 1],
                w=cone_vectors[:, 2],
                sizemode="absolute",
                sizeref=1.0,
                anchor="tail",
                colorscale="Plasma",
                cmin=color_min,
                cmax=color_max,
                colorbar={
                    "title": "|J|",
                    "tickvals": tick_values,
                    "ticktext": tick_text,
                },
                showscale=True,
                opacity=0.85,
                name="flux density vectors",
                customdata=np.column_stack((vector_magnitudes, vector_values)),
                hovertemplate=(
                    "x=%{x:.3g}<br>y=%{y:.3g}<br>z=%{z:.3g}"
                    "<br>|J|=%{customdata[0]:.3g}"
                    "<br>Jx=%{customdata[1]:.3g}"
                    "<br>Jy=%{customdata[2]:.3g}"
                    "<br>Jz=%{customdata[3]:.3g}<extra></extra>"
                ),
            )
        )

    radius = CYLINDER.cylinder_radius
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene={
            "xaxis": {"title": "x", "range": [float(np.min(points[:, 0])), float(np.max(points[:, 0]))]},
            "yaxis": {"title": "y", "range": [-radius, radius]},
            "zaxis": {"title": "z", "range": [-radius, radius]},
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
        legend={"x": 0.02, "y": 0.98},
    )
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)


def plot_cartesian_flux_comparison(
    comparison: CartesianFluxComparison,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
) -> None:
    image_path = Path(image_dir) / "flux" / "cartesian"
    _plot_flux_density_3d(
        comparison,
        comparison.instantaneous_flux_density,
        image_path / "active_flux_density_instantaneous_xyz.html",
        f"Instantaneous flux density, step {comparison.step}",
    )
    _plot_flux_density_3d(
        comparison,
        comparison.finite_difference_flux_density,
        image_path / "active_flux_density_finite_difference_xyz.html",
        f"Finite-difference flux density, step {comparison.step} to {comparison.next_step}",
    )
