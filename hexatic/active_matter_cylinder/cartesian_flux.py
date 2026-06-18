from pathlib import Path

import gsd.hoomd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .common import (
    _active_direction_from_quaternion,
    _axis_edges_and_centers,
    _cartesian_tensor_to_cylindrical,
    _cartesian_vector_to_cylindrical_components,
    _cylindrical_plot_points,
    _cylindrical_plot_vectors,
    _density_sum,
    _gaussian_kernel_volume,
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


def _plot_coordinates(
    points: np.ndarray,
    vectors: np.ndarray,
    coordinate_system: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, str, str], tuple[str, str, str]]:
    if coordinate_system == "xyz":
        return points, vectors, ("x", "y", "z"), ("Jx", "Jy", "Jz")
    if coordinate_system == "xrtheta":
        return (
            _cylindrical_plot_points(points),
            _cylindrical_plot_vectors(points, vectors),
            ("x", "r", "theta"),
            ("Jx", "Jr", "Jtheta/r"),
        )
    raise ValueError("coordinate_system must be 'xyz' or 'xrtheta'.")


def _normalize_coordinate_system(coordinate_system: str) -> str:
    normalized = (
        coordinate_system.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace(",", "")
        .replace("-", "")
    )
    if normalized == "xyz":
        return "xyz"
    if normalized in {"xrtheta", "xrthetacoordinates"}:
        return "xrtheta"
    raise ValueError("coordinate_system must be 'xyz' or 'xrtheta'.")


def _plot_flux_density_3d(
    comparison: CartesianFluxComparison,
    vectors: np.ndarray,
    filename: str | Path,
    title: str,
    coordinate_system: str,
    max_vectors: int = 900,
) -> None:
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points = comparison.grid_points
    magnitudes = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(magnitudes) & (comparison.rho_density > 0.0)
    plot_points, plot_vectors, axis_labels, component_labels = _plot_coordinates(
        points[valid],
        vectors[valid],
        coordinate_system,
    )
    plot_magnitudes = magnitudes[valid]
    all_points = _plot_coordinates(points, vectors, coordinate_system)[0]

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
                x=all_points[:, 0],
                y=all_points[:, 1],
                z=all_points[:, 2],
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
                    f"{axis_labels[0]}=%{{x:.3g}}"
                    f"<br>{axis_labels[1]}=%{{y:.3g}}"
                    f"<br>{axis_labels[2]}=%{{z:.3g}}"
                    "<br>|J|=%{customdata[0]:.3g}"
                    f"<br>{component_labels[0]}=%{{customdata[1]:.3g}}"
                    f"<br>{component_labels[1]}=%{{customdata[2]:.3g}}"
                    f"<br>{component_labels[2]}=%{{customdata[3]:.3g}}<extra></extra>"
                ),
            )
        )

    radius = CYLINDER.cylinder_radius
    axis_ranges = [
        [float(np.min(all_points[:, 0])), float(np.max(all_points[:, 0]))],
        [float(np.min(all_points[:, 1])), float(np.max(all_points[:, 1]))],
        [float(np.min(all_points[:, 2])), float(np.max(all_points[:, 2]))],
    ]
    if coordinate_system == "xyz":
        axis_ranges[1] = [-radius, radius]
        axis_ranges[2] = [-radius, radius]
    else:
        axis_ranges[1] = [0.0, radius]
        axis_ranges[2] = [0.0, 2.0 * np.pi]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene={
            "xaxis": {"title": axis_labels[0], "range": axis_ranges[0]},
            "yaxis": {"title": axis_labels[1], "range": axis_ranges[1]},
            "zaxis": {"title": axis_labels[2], "range": axis_ranges[2]},
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
        legend={"x": 0.02, "y": 0.98},
    )
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)


def _plot_virial_shear_stress_dropdown(
    comparison: CartesianFluxComparison,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
) -> None:
    output_path = Path(image_dir) / "shear" / "smoothed_virial_shear_stress.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ("x", "r", "theta")
    points = comparison.grid_points
    values = comparison.virial_stress_cylindrical
    valid = np.isfinite(values).all(axis=(1, 2)) & (comparison.rho_density > 0.0)
    plot_points = _cylindrical_plot_points(points[valid])
    plot_values = values[valid]

    if len(plot_points) == 0:
        fig = go.Figure()
        fig.update_layout(title="Smoothed virial shear stress: no occupied grid points")
        fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
        return

    finite_values = plot_values[np.isfinite(plot_values)]
    max_abs = float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
    if np.isclose(max_abs, 0.0):
        max_abs = 1.0

    traces = []
    trace_indices: dict[str, int] = {}
    for component_index, component_label in enumerate(labels):
        for normal_index, normal_label in enumerate(labels):
            trace_indices[f"{component_label}_{normal_label}"] = len(traces)
            field = plot_values[:, component_index, normal_index]
            traces.append(
                go.Scatter3d(
                    x=plot_points[:, 0],
                    y=plot_points[:, 1],
                    z=plot_points[:, 2],
                    mode="markers",
                    marker={
                        "size": 3,
                        "color": field,
                        "colorscale": "RdBu",
                        "cmin": -max_abs,
                        "cmax": max_abs,
                        "colorbar": {"title": "stress"},
                        "opacity": 0.85,
                    },
                    name=f"sigma_{component_label}{normal_label}",
                    visible=(component_index == 0 and normal_index == 1),
                    customdata=field,
                    hovertemplate=(
                        "x=%{x:.3g}<br>r=%{y:.3g}<br>theta=%{z:.3g}"
                        "<br>stress=%{customdata:.3g}<extra></extra>"
                    ),
                )
            )

    radius = CYLINDER.cylinder_radius
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Smoothed virial stress: component x, normal r, step {comparison.step}",
        scene={
            "xaxis": {
                "title": "x",
                "range": [float(np.min(plot_points[:, 0])), float(np.max(plot_points[:, 0]))],
            },
            "yaxis": {"title": "r", "range": [0.0, radius]},
            "zaxis": {"title": "theta", "range": [0.0, 2.0 * np.pi]},
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
    )

    div_id = "smoothed-virial-shear-stress-plot"
    plot_html = fig.to_html(full_html=False, include_plotlyjs=True, div_id=div_id)
    options = "\n".join(f'<option value="{label}">{label}</option>' for label in labels)
    trace_map = "{" + ",".join(
        f'"{key}": {value}' for key, value in trace_indices.items()
    ) + "}"
    controls = f"""
<div style="font-family: sans-serif; margin: 12px 0;">
  <label for="stress-component">Component:&nbsp;</label>
  <select id="stress-component">{options}</select>
  <label for="stress-normal" style="margin-left: 18px;">Normal:&nbsp;</label>
  <select id="stress-normal">
    <option value="x">x</option>
    <option value="r" selected>r</option>
    <option value="theta">theta</option>
  </select>
</div>
"""
    script = f"""
<script>
const stressTraceMap = {trace_map};
const stressTraceCount = {len(traces)};
const stressStep = {comparison.step};
function updateStressTrace() {{
  const component = document.getElementById("stress-component").value;
  const normal = document.getElementById("stress-normal").value;
  const key = component + "_" + normal;
  const visible = Array(stressTraceCount).fill(false);
  visible[stressTraceMap[key]] = true;
  Plotly.restyle("{div_id}", {{"visible": visible}});
  Plotly.relayout(
    "{div_id}",
    {{"title.text": "Smoothed virial stress: component "
      + component + ", normal " + normal + ", step " + stressStep}}
  );
}}
document.getElementById("stress-component").addEventListener("change", updateStressTrace);
document.getElementById("stress-normal").addEventListener("change", updateStressTrace);
</script>
"""
    output_path.write_text(f"<!doctype html><html><body>{controls}{plot_html}{script}</body></html>")


def _plot_div_sigma_force_density_check(
    comparison: CartesianFluxComparison,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
    component_system: str = "xyz",
    max_points: int = 7000,
) -> None:
    component_system = _normalize_coordinate_system(component_system)
    filename_suffix = "xyz" if component_system == "xyz" else "xrtheta"
    output_path = (
        Path(image_dir)
        / "stress"
        / f"div_sigma_vs_force_density_{filename_suffix}.html"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    current_div_sigma = _cartesian_stress_divergence(
        comparison.grid_points,
        VIRIAL_STRESS_SIGN * comparison.virial_stress_density,
        comparison.grid_spacing,
    )
    opposite_div_sigma = -current_div_sigma
    net_force_density = comparison.virial_divergence_density
    labels = ("x", "y", "z")
    if component_system == "xrtheta":
        current_div_sigma = _cartesian_vector_to_cylindrical_components(
            comparison.grid_points,
            current_div_sigma,
        )
        opposite_div_sigma = -current_div_sigma
        net_force_density = _cartesian_vector_to_cylindrical_components(
            comparison.grid_points,
            net_force_density,
        )
        labels = ("x", "r", "theta")

    valid = (
        (comparison.rho_density > 0.0)
        & np.isfinite(current_div_sigma).all(axis=1)
        & np.isfinite(net_force_density).all(axis=1)
    )

    if not np.any(valid):
        fig = go.Figure()
        fig.update_layout(
            title=(
                f"div(sigma) vs net force density ({component_system}): "
                "no interior grid points with finite central differences"
            )
        )
        fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
        return

    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) > max_points:
        valid_indices = valid_indices[np.linspace(0, len(valid_indices) - 1, max_points).astype(np.int64)]

    def component_correlation(x_values: np.ndarray, y_values: np.ndarray) -> float:
        if len(x_values) < 2 or np.isclose(np.std(x_values), 0.0) or np.isclose(np.std(y_values), 0.0):
            return float("nan")
        return float(np.corrcoef(x_values, y_values)[0, 1])

    def component_slope(x_values: np.ndarray, y_values: np.ndarray) -> float:
        denominator = float(np.dot(x_values, x_values))
        if np.isclose(denominator, 0.0):
            return float("nan")
        return float(np.dot(x_values, y_values) / denominator)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[f"{label} component" for label in labels],
        horizontal_spacing=0.08,
    )
    annotations = []
    for component_index, label in enumerate(labels):
        x_force = net_force_density[valid_indices, component_index]
        y_current = current_div_sigma[valid_indices, component_index]
        y_opposite = opposite_div_sigma[valid_indices, component_index]
        finite_values = np.concatenate((x_force, y_current, y_opposite))
        finite_values = finite_values[np.isfinite(finite_values)]
        limit = float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
        if np.isclose(limit, 0.0):
            limit = 1.0

        corr_current = component_correlation(x_force, y_current)
        corr_opposite = component_correlation(x_force, y_opposite)
        slope_current = component_slope(x_force, y_current)
        slope_opposite = -slope_current

        fig.add_trace(
            go.Scattergl(
                x=x_force,
                y=y_current,
                mode="markers",
                marker={"size": 4, "opacity": 0.45, "color": "#2563eb"},
                name=f"configured sign ({VIRIAL_STRESS_SIGN:+.0f})",
                legendgroup="current",
                showlegend=component_index == 0,
                hovertemplate=(
                    f"F_{label} density=%{{x:.3g}}"
                    f"<br>div(sigma)_{label}=%{{y:.3g}}<extra></extra>"
                ),
            ),
            row=1,
            col=component_index + 1,
        )
        fig.add_trace(
            go.Scattergl(
                x=x_force,
                y=y_opposite,
                mode="markers",
                marker={"size": 4, "opacity": 0.35, "color": "#f97316"},
                name=f"opposite sign ({-VIRIAL_STRESS_SIGN:+.0f})",
                legendgroup="opposite",
                showlegend=component_index == 0,
                hovertemplate=(
                    f"F_{label} density=%{{x:.3g}}"
                    f"<br>-div(sigma)_{label}=%{{y:.3g}}<extra></extra>"
                ),
            ),
            row=1,
            col=component_index + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=[-limit, limit],
                y=[-limit, limit],
                mode="lines",
                line={"color": "black", "dash": "dash", "width": 1},
                name="y = x",
                legendgroup="identity",
                showlegend=component_index == 0,
                hoverinfo="skip",
            ),
            row=1,
            col=component_index + 1,
        )
        fig.update_xaxes(
            title_text=f"net force density F_{label}",
            range=[-limit, limit],
            row=1,
            col=component_index + 1,
        )
        fig.update_yaxes(
            title_text=f"div(sigma)_{label}",
            range=[-limit, limit],
            row=1,
            col=component_index + 1,
        )
        annotations.append(
            f"{label}: corr sign={corr_current:.3g}, corr opposite={corr_opposite:.3g}, "
            f"slope sign={slope_current:.3g}, slope opposite={slope_opposite:.3g}"
        )

    fig.update_layout(
        title=(
            "Numerical div(sigma) vs smoothed net force density "
            f"({component_system}, step {comparison.step})"
        ),
        margin={"l": 40, "r": 20, "b": 70, "t": 70},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25, "x": 0.0},
        annotations=list(fig.layout.annotations)
        + [
            {
                "x": 0.0,
                "y": -0.18 - 0.05 * idx,
                "xref": "paper",
                "yref": "paper",
                "text": text,
                "showarrow": False,
                "align": "left",
            }
            for idx, text in enumerate(annotations)
        ],
    )
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)


def plot_cartesian_flux_comparison(
    comparison: CartesianFluxComparison,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
    coordinate_system: str = "xyz",
) -> None:
    coordinate_system = _normalize_coordinate_system(coordinate_system)
    filename_suffix = "xyz" if coordinate_system == "xyz" else "xrtheta"
    image_path = Path(image_dir) / "flux" / filename_suffix
    _plot_virial_shear_stress_dropdown(comparison, image_dir=image_dir)
    _plot_div_sigma_force_density_check(
        comparison,
        image_dir=image_dir,
        component_system="xyz",
    )
    _plot_div_sigma_force_density_check(
        comparison,
        image_dir=image_dir,
        component_system="xrtheta",
    )
    _plot_flux_density_3d(
        comparison,
        comparison.instantaneous_stress_flux_density,
        image_path / f"active_flux_density_force_density_{filename_suffix}.html",
        (
            f"Force-density J = U0 P + F/gamma "
            f"({coordinate_system}), step {comparison.step}"
        ),
        coordinate_system,
    )
