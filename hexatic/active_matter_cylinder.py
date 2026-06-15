from dataclasses import dataclass
from pathlib import Path

import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

if __package__:
    from hexatic import analysis as hx
    from hexatic.constants import cylinder
else:
    import analysis as hx
    from constants import cylinder

CYLINDER = cylinder.ANALYSIS
CYLINDER_PATHS = cylinder.PATHS
CYLINDER_SIM = cylinder.SIMULATION
LOCAL_POCKET_RADIUS = 2.0 * CYLINDER.particle_diameter
ACTIVE_FIELD_X_BINS = 100
ACTIVE_FIELD_THETA_BINS = 72
ACTIVE_FLUX_PLOT_X_BINS = 32
ACTIVE_FLUX_PLOT_THETA_BINS = 18
ACTIVE_MOVIE_FPS = 8
ACTIVE_DATA_DIR = Path(CYLINDER_PATHS.in_gsd).parent
ACTIVE_IMAGE_DIR = Path(CYLINDER_PATHS.com_plot).parent / "active"


@dataclass(frozen=True)
class ActiveMatterFields:
    steps: np.ndarray
    x_edges: np.ndarray
    x_centers: np.ndarray
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    coords: np.ndarray
    shell_mask: np.ndarray
    rho: np.ndarray
    polar_mean: np.ndarray
    polar_cylindrical: np.ndarray
    flux_cylindrical: np.ndarray
    force_density_cylindrical: np.ndarray


def _active_direction_from_quaternion(orientation: np.ndarray) -> np.ndarray:
    orientation = np.asarray(orientation, dtype=np.float64)
    assert orientation.ndim == 2 and orientation.shape[1] == 4

    norms = np.linalg.norm(orientation, axis=1)
    assert np.all(norms > 0.0)
    quat = orientation / norms[:, np.newaxis]
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    return np.column_stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        )
    )


def _logged_particle_array(frame, quantity: str, n_particles: int) -> np.ndarray:
    log = getattr(frame, "log", None)
    if not log:
        raise ValueError(f"GSD frame has no logger data; expected LJ {quantity}.")

    quantity_lower = quantity.lower()
    candidates: list[tuple[str, np.ndarray]] = []
    for key, value in log.items():
        key_lower = str(key).lower()
        array = np.asarray(value)
        if quantity_lower in key_lower and array.shape[:1] == (n_particles,):
            candidates.append((str(key), array))

    if not candidates:
        available = ", ".join(str(key) for key in log)
        raise ValueError(
            f"Could not find logged per-particle LJ {quantity}. "
            f"Available logger keys: {available}"
        )

    candidates.sort(key=lambda item: ("lj" not in item[0].lower(), item[0]))
    return np.asarray(candidates[0][1], dtype=np.float64)


def _minimum_image_delta(values: np.ndarray, period: float) -> np.ndarray:
    assert period > 0.0
    return values - period * np.round(values / period)


def _x_edges_and_centers(box_length_x: float, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    assert box_length_x > 0.0
    assert n_bins > 0
    edges = np.linspace(-0.5 * box_length_x, 0.5 * box_length_x, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _theta_edges_and_centers(n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    assert n_bins > 0
    edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _x_bin_indices(x_positions: np.ndarray, box_length_x: float, n_bins: int) -> np.ndarray:
    wrapped = np.mod(x_positions + 0.5 * box_length_x, box_length_x)
    indices = np.floor(wrapped / box_length_x * n_bins).astype(np.int64)
    return np.clip(indices, 0, n_bins - 1)


def _theta_bin_indices(theta: np.ndarray, n_bins: int) -> np.ndarray:
    wrapped = np.mod(theta, 2.0 * np.pi)
    indices = np.floor(wrapped / (2.0 * np.pi) * n_bins).astype(np.int64)
    return np.clip(indices, 0, n_bins - 1)


def _pocket_fields(
    positions: np.ndarray,
    directions: np.ndarray,
    box_length_x: float,
    pocket_radius: float,
    block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_particles = len(positions)
    rho = np.zeros(n_particles, dtype=np.int64)
    polar_sum = np.zeros((n_particles, 3), dtype=np.float64)
    cutoff_sq = pocket_radius * pocket_radius

    for start in range(0, n_particles, block_size):
        stop = min(start + block_size, n_particles)
        deltas = positions[start:stop, np.newaxis, :] - positions[np.newaxis, :, :]
        deltas[..., 0] = _minimum_image_delta(deltas[..., 0], box_length_x)
        pocket_mask = np.sum(deltas * deltas, axis=2) <= cutoff_sq
        rho[start:stop] = np.count_nonzero(pocket_mask, axis=1)
        polar_sum[start:stop] = pocket_mask.astype(np.float64) @ directions

    polar_mean = np.divide(
        polar_sum,
        rho[:, np.newaxis],
        out=np.zeros_like(polar_sum),
        where=rho[:, np.newaxis] > 0,
    )
    return rho, polar_sum, polar_mean


def _cylindrical_components(vectors: np.ndarray, theta: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    assert vectors.ndim == 2 and vectors.shape[1] == 3
    assert theta.shape == (vectors.shape[0],)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    radial = vectors[:, 1] * sin_theta + vectors[:, 2] * cos_theta
    azimuthal = vectors[:, 1] * cos_theta - vectors[:, 2] * sin_theta
    return np.column_stack((vectors[:, 0], radial, azimuthal))


def _radial_integral_mean(
    coords: np.ndarray,
    values: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    box_length_x: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    x_bins = len(x_edges) - 1
    theta_bins = len(theta_edges) - 1
    x_indices = _x_bin_indices(coords[:, 0], box_length_x, x_bins)
    theta_indices = _theta_bin_indices(coords[:, 1], theta_bins)
    counts = np.zeros((x_bins, theta_bins), dtype=np.float64)
    np.add.at(counts, (x_indices, theta_indices), 1.0)

    if values.ndim == 1:
        sums = np.zeros((x_bins, theta_bins), dtype=np.float64)
        np.add.at(sums, (x_indices, theta_indices), values)
        return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    sums = np.zeros((x_bins, theta_bins, values.shape[1]), dtype=np.float64)
    for component in range(values.shape[1]):
        np.add.at(sums[..., component], (x_indices, theta_indices), values[:, component])
    return np.divide(
        sums,
        counts[..., np.newaxis],
        out=np.zeros_like(sums),
        where=counts[..., np.newaxis] > 0,
    )


def _radial_integral_sum(
    coords: np.ndarray,
    values: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    box_length_x: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    x_bins = len(x_edges) - 1
    theta_bins = len(theta_edges) - 1
    x_indices = _x_bin_indices(coords[:, 0], box_length_x, x_bins)
    theta_indices = _theta_bin_indices(coords[:, 1], theta_bins)

    if values.ndim == 1:
        sums = np.zeros((x_bins, theta_bins), dtype=np.float64)
        np.add.at(sums, (x_indices, theta_indices), values)
        return sums

    sums = np.zeros((x_bins, theta_bins, values.shape[1]), dtype=np.float64)
    for component in range(values.shape[1]):
        np.add.at(sums[..., component], (x_indices, theta_indices), values[:, component])
    return sums


def _color_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0

    vmin, vmax = np.percentile(finite, [2.0, 98.0])
    if np.isclose(vmin, vmax):
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def _format_theta_axis(axis) -> None:
    axis.set_ylabel("theta")
    axis.set_ylim(0.0, 2.0 * np.pi)
    axis.set_yticks(
        [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi]
    )
    axis.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: {
                0.0: "0",
                0.5: r"$\pi/2$",
                1.0: r"$\pi$",
                1.5: r"$3\pi/2$",
                2.0: r"$2\pi$",
            }.get(round(value / np.pi, 1), "")
        )
    )


def _fixed_length_quiver_components(
    x_components: np.ndarray,
    theta_components: np.ndarray,
    x_span: float,
    n_x_bins: int,
    n_theta_bins: int,
    theta_span: float = 2.0 * np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    x_fraction = x_components / x_span
    theta_fraction = theta_components / theta_span
    lengths = np.hypot(x_fraction, theta_fraction)
    arrow_fraction = 0.35 * min(1.0 / n_x_bins, 1.0 / n_theta_bins)

    unit_x = np.divide(
        x_fraction,
        lengths,
        out=np.zeros_like(x_fraction, dtype=np.float64),
        where=lengths > 0.0,
    )
    unit_theta = np.divide(
        theta_fraction,
        lengths,
        out=np.zeros_like(theta_fraction, dtype=np.float64),
        where=lengths > 0.0,
    )
    return unit_x * x_span * arrow_fraction, unit_theta * theta_span * arrow_fraction


def _frame_index(index: int, n_frames: int) -> int:
    if index < 0:
        index += n_frames
    assert 0 <= index < n_frames
    return index


def _coarse_vector_density_grid(
    fields: ActiveMatterFields,
    vectors: np.ndarray,
    frame_idx: int,
    shell_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x_span = fields.x_edges[-1] - fields.x_edges[0]
    x_edges, x_centers = _x_edges_and_centers(x_span, ACTIVE_FLUX_PLOT_X_BINS)
    theta_edges, theta_centers = _theta_edges_and_centers(ACTIVE_FLUX_PLOT_THETA_BINS)
    mask = fields.shell_mask[frame_idx] if shell_only else np.ones(
        fields.coords.shape[1],
        dtype=np.bool_,
    )
    vector_grid = _radial_integral_sum(
        fields.coords[frame_idx, mask],
        vectors[frame_idx, mask],
        x_edges,
        theta_edges,
        x_span,
    )
    x_grid, theta_grid = np.meshgrid(x_centers, theta_centers, indexing="ij")
    return x_grid, theta_grid, vector_grid, x_span


def active_matter_field_series(
    input_gsd: str | Path,
    pocket_radius: float = LOCAL_POCKET_RADIUS,
    n_x_bins: int = ACTIVE_FIELD_X_BINS,
    n_theta_bins: int = ACTIVE_FIELD_THETA_BINS,
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
        polar_mean = np.zeros((n_frames, n_particles, 3), dtype=np.float64)
        polar_cylindrical = np.zeros((n_frames, n_particles, 3), dtype=np.float64)
        flux_cylindrical = np.zeros((n_frames, n_particles, 3), dtype=np.float64)
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
                cylinder_radius=CYLINDER.cylinder_radius,
                cutoff=CYLINDER.wall_cutoff,
            )
            pocket_rho, _, pocket_polar_mean = _pocket_fields(
                positions,
                directions,
                box_length_x,
                pocket_radius,
            )
            force_density = forces[:, :3] / CYLINDER_SIM.gamma
            velocities = CYLINDER_SIM.u0 * directions + force_density

            coords[frame_idx] = frame_coords
            shell_masks[frame_idx] = dynamic_values.shell_mask
            rho[frame_idx] = pocket_rho.astype(np.float64)
            polar_mean[frame_idx] = np.nan_to_num(pocket_polar_mean, nan=0.0)
            polar_cylindrical[frame_idx] = _cylindrical_components(
                polar_mean[frame_idx],
                frame_coords[:, 1],
            )
            flux_cylindrical[frame_idx] = _cylindrical_components(
                velocities,
                frame_coords[:, 1],
            )
            force_density_cylindrical[frame_idx] = _cylindrical_components(
                force_density,
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
        polar_mean=polar_mean,
        polar_cylindrical=polar_cylindrical,
        flux_cylindrical=flux_cylindrical,
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
        polar_mean=fields.polar_mean,
        polar_cylindrical=fields.polar_cylindrical,
        flux_cylindrical=fields.flux_cylindrical,
        force_density_cylindrical=fields.force_density_cylindrical,
    )


def plot_rho_shell(
    fields: ActiveMatterFields,
    filename: str | Path,
    frame_index: int = -1,
) -> None:
    frame_idx = _frame_index(frame_index, len(fields.steps))
    mask = fields.shell_mask[frame_idx]
    coords = fields.coords[frame_idx, mask]
    values = fields.rho[frame_idx, mask]
    vmin, vmax = _color_limits(values)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 5))
    scatter = axis.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        s=9,
        cmap="magma",
        norm=Normalize(vmin=vmin, vmax=vmax),
        linewidths=0,
    )
    fig.colorbar(scatter, ax=axis, label="local rho")
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(f"Outer-shell local rho, step {fields.steps[frame_idx]}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_rho_radial_integral(
    fields: ActiveMatterFields,
    filename: str | Path,
    frame_index: int = -1,
) -> None:
    frame_idx = _frame_index(frame_index, len(fields.steps))
    values = _radial_integral_mean(
        fields.coords[frame_idx],
        fields.rho[frame_idx],
        fields.x_edges,
        fields.theta_edges,
        fields.x_edges[-1] - fields.x_edges[0],
    )
    x_grid, theta_grid = np.meshgrid(
        fields.x_centers,
        fields.theta_centers,
        indexing="ij",
    )
    vmin, vmax = _color_limits(values)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 5))
    scatter = axis.scatter(
        x_grid.ravel(),
        theta_grid.ravel(),
        c=values.ravel(),
        s=9,
        marker="o",
        cmap="magma",
        norm=Normalize(vmin=vmin, vmax=vmax),
        linewidths=0,
    )
    fig.colorbar(scatter, ax=axis, label="r-averaged local rho")
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(f"Integrated local rho, step {fields.steps[frame_idx]}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_polar_shell(
    fields: ActiveMatterFields,
    filename: str | Path,
    frame_index: int = -1,
) -> None:
    frame_idx = _frame_index(frame_index, len(fields.steps))
    mask = fields.shell_mask[frame_idx]
    coords = fields.coords[frame_idx, mask]
    polar = fields.polar_cylindrical[frame_idx, mask]
    projected_theta = polar[:, 2] / CYLINDER.cylinder_radius
    magnitude = np.hypot(polar[:, 0], polar[:, 2])
    vmin, vmax = _color_limits(magnitude)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 5))
    quiver = axis.quiver(
        coords[:, 0],
        coords[:, 1],
        polar[:, 0],
        projected_theta,
        magnitude,
        cmap="viridis",
        norm=Normalize(vmin=vmin, vmax=vmax),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
    )
    fig.colorbar(quiver, ax=axis, label="|polar mean in x-theta|")
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(f"Outer-shell polar mean, step {fields.steps[frame_idx]}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_polar_radial_integral(
    fields: ActiveMatterFields,
    filename: str | Path,
    frame_index: int = -1,
) -> None:
    frame_idx = _frame_index(frame_index, len(fields.steps))
    polar = _radial_integral_mean(
        fields.coords[frame_idx],
        fields.polar_cylindrical[frame_idx],
        fields.x_edges,
        fields.theta_edges,
        fields.x_edges[-1] - fields.x_edges[0],
    )
    x_grid, theta_grid = np.meshgrid(
        fields.x_centers,
        fields.theta_centers,
        indexing="ij",
    )
    projected_theta = polar[..., 2] / CYLINDER.cylinder_radius
    magnitude = np.hypot(polar[..., 0], polar[..., 2])
    vmin, vmax = _color_limits(magnitude)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 5))
    quiver = axis.quiver(
        x_grid,
        theta_grid,
        polar[..., 0],
        projected_theta,
        magnitude,
        cmap="viridis",
        norm=Normalize(vmin=vmin, vmax=vmax),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
    )
    fig.colorbar(quiver, ax=axis, label="r-averaged |polar mean in x-theta|")
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(f"Integrated averaged polar mean, step {fields.steps[frame_idx]}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_vector_density(
    fields: ActiveMatterFields,
    vectors: np.ndarray,
    filename: str | Path,
    title_prefix: str,
    colorbar_label: str,
    shell_only: bool,
    frame_index: int = -1,
    cmap: str = "plasma",
) -> None:
    frame_idx = _frame_index(frame_index, len(fields.steps))
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 5))
    _draw_vector_density(
        fields,
        vectors,
        fig,
        axis,
        frame_idx,
        title_prefix,
        colorbar_label,
        shell_only=shell_only,
        cmap=cmap,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _draw_vector_density(
    fields: ActiveMatterFields,
    vectors: np.ndarray,
    fig,
    axis,
    frame_idx: int,
    title_prefix: str,
    colorbar_label: str,
    shell_only: bool,
    cmap: str = "plasma",
) -> None:
    x_grid, theta_grid, vector_grid, x_span = _coarse_vector_density_grid(
        fields,
        vectors,
        frame_idx,
        shell_only=shell_only,
    )
    projected_theta = vector_grid[..., 2] / CYLINDER.cylinder_radius
    magnitude = np.linalg.norm(vector_grid, axis=2)
    vmin, vmax = _color_limits(magnitude)
    arrow_x, arrow_theta = _fixed_length_quiver_components(
        vector_grid[..., 0],
        projected_theta,
        x_span,
        ACTIVE_FLUX_PLOT_X_BINS,
        ACTIVE_FLUX_PLOT_THETA_BINS,
    )
    quiver = axis.quiver(
        x_grid,
        theta_grid,
        arrow_x,
        arrow_theta,
        magnitude,
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
    )
    fig.colorbar(quiver, ax=axis, label=colorbar_label)
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(f"{title_prefix}, step {fields.steps[frame_idx]}")


def plot_flux_shell(
    fields: ActiveMatterFields,
    filename: str | Path,
    frame_index: int = -1,
) -> None:
    _plot_vector_density(
        fields,
        fields.flux_cylindrical,
        filename,
        "Outer-shell flux density",
        r"$|\sum_i \dot{\mathbf{r}}_i|$",
        shell_only=True,
        frame_index=frame_index,
    )


def plot_flux_radial_integral(
    fields: ActiveMatterFields,
    filename: str | Path,
    frame_index: int = -1,
) -> None:
    _plot_vector_density(
        fields,
        fields.flux_cylindrical,
        filename,
        "Radially integrated flux density",
        r"$|\int dr\sum_i \dot{\mathbf{r}}_i|$",
        shell_only=False,
        frame_index=frame_index,
    )


def plot_force_density_shell(
    fields: ActiveMatterFields,
    filename: str | Path,
    frame_index: int = -1,
) -> None:
    _plot_vector_density(
        fields,
        fields.force_density_cylindrical,
        filename,
        "Outer-shell force density",
        r"$|\gamma^{-1}\sum_i \mathbf{F}_i|$",
        shell_only=True,
        frame_index=frame_index,
        cmap="cividis",
    )


def plot_force_density_radial_integral(
    fields: ActiveMatterFields,
    filename: str | Path,
    frame_index: int = -1,
) -> None:
    _plot_vector_density(
        fields,
        fields.force_density_cylindrical,
        filename,
        "Radially integrated force density",
        r"$|\int dr\,\gamma^{-1}\sum_i \mathbf{F}_i|$",
        shell_only=False,
        frame_index=frame_index,
        cmap="cividis",
    )


def _write_movie(
    fields: ActiveMatterFields,
    filename: str | Path,
    draw_frame,
    fps: int = ACTIVE_MOVIE_FPS,
) -> None:
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 5))
    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, str(output_path), dpi=160):
        for frame_idx in range(len(fields.steps)):
            fig.clear()
            axis = fig.add_subplot(111)
            draw_frame(fig, axis, frame_idx)
            fig.tight_layout()
            writer.grab_frame()
    plt.close(fig)


def _draw_rho_shell(fields: ActiveMatterFields, fig, axis, frame_idx: int) -> None:
    mask = fields.shell_mask[frame_idx]
    coords = fields.coords[frame_idx, mask]
    values = fields.rho[frame_idx, mask]
    vmin, vmax = _color_limits(values)
    scatter = axis.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        s=9,
        cmap="magma",
        norm=Normalize(vmin=vmin, vmax=vmax),
        linewidths=0,
    )
    fig.colorbar(scatter, ax=axis, label="local rho")
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(f"Outer-shell local rho, step {fields.steps[frame_idx]}")


def _draw_rho_radial_integral(
    fields: ActiveMatterFields,
    fig,
    axis,
    frame_idx: int,
) -> None:
    values = _radial_integral_mean(
        fields.coords[frame_idx],
        fields.rho[frame_idx],
        fields.x_edges,
        fields.theta_edges,
        fields.x_edges[-1] - fields.x_edges[0],
    )
    x_grid, theta_grid = np.meshgrid(
        fields.x_centers,
        fields.theta_centers,
        indexing="ij",
    )
    vmin, vmax = _color_limits(values)
    scatter = axis.scatter(
        x_grid.ravel(),
        theta_grid.ravel(),
        c=values.ravel(),
        s=9,
        marker="o",
        cmap="magma",
        norm=Normalize(vmin=vmin, vmax=vmax),
        linewidths=0,
    )
    fig.colorbar(scatter, ax=axis, label="r-averaged local rho")
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(f"Integrated local rho, step {fields.steps[frame_idx]}")


def _draw_polar_shell(fields: ActiveMatterFields, fig, axis, frame_idx: int) -> None:
    mask = fields.shell_mask[frame_idx]
    coords = fields.coords[frame_idx, mask]
    polar = fields.polar_cylindrical[frame_idx, mask]
    projected_theta = polar[:, 2] / CYLINDER.cylinder_radius
    magnitude = np.hypot(polar[:, 0], polar[:, 2])
    vmin, vmax = _color_limits(magnitude)
    quiver = axis.quiver(
        coords[:, 0],
        coords[:, 1],
        polar[:, 0],
        projected_theta,
        magnitude,
        cmap="viridis",
        norm=Normalize(vmin=vmin, vmax=vmax),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
    )
    fig.colorbar(quiver, ax=axis, label="|polar mean in x-theta|")
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(f"Outer-shell polar mean, step {fields.steps[frame_idx]}")


def _draw_polar_radial_integral(
    fields: ActiveMatterFields,
    fig,
    axis,
    frame_idx: int,
) -> None:
    polar = _radial_integral_mean(
        fields.coords[frame_idx],
        fields.polar_cylindrical[frame_idx],
        fields.x_edges,
        fields.theta_edges,
        fields.x_edges[-1] - fields.x_edges[0],
    )
    x_grid, theta_grid = np.meshgrid(
        fields.x_centers,
        fields.theta_centers,
        indexing="ij",
    )
    projected_theta = polar[..., 2] / CYLINDER.cylinder_radius
    magnitude = np.hypot(polar[..., 0], polar[..., 2])
    vmin, vmax = _color_limits(magnitude)
    quiver = axis.quiver(
        x_grid,
        theta_grid,
        polar[..., 0],
        projected_theta,
        magnitude,
        cmap="viridis",
        norm=Normalize(vmin=vmin, vmax=vmax),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
    )
    fig.colorbar(quiver, ax=axis, label="r-averaged |polar mean in x-theta|")
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(f"Integrated averaged polar mean, step {fields.steps[frame_idx]}")


def plot_active_matter_movies(
    fields: ActiveMatterFields,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
    fps: int = ACTIVE_MOVIE_FPS,
) -> None:
    image_path = Path(image_dir)
    _write_movie(
        fields,
        image_path / "active_rho_shell.mp4",
        lambda fig, axis, frame_idx: _draw_rho_shell(fields, fig, axis, frame_idx),
        fps=fps,
    )
    _write_movie(
        fields,
        image_path / "active_rho_radial_integral.mp4",
        lambda fig, axis, frame_idx: _draw_rho_radial_integral(
            fields,
            fig,
            axis,
            frame_idx,
        ),
        fps=fps,
    )
    _write_movie(
        fields,
        image_path / "active_polar_shell.mp4",
        lambda fig, axis, frame_idx: _draw_polar_shell(fields, fig, axis, frame_idx),
        fps=fps,
    )
    _write_movie(
        fields,
        image_path / "active_polar_radial_integral.mp4",
        lambda fig, axis, frame_idx: _draw_polar_radial_integral(
            fields,
            fig,
            axis,
            frame_idx,
        ),
        fps=fps,
    )
    _write_movie(
        fields,
        image_path / "active_flux_shell.mp4",
        lambda fig, axis, frame_idx: _draw_vector_density(
            fields,
            fields.flux_cylindrical,
            fig,
            axis,
            frame_idx,
            "Outer-shell flux density",
            r"$|\sum_i \dot{\mathbf{r}}_i|$",
            shell_only=True,
            cmap="plasma",
        ),
        fps=fps,
    )
    _write_movie(
        fields,
        image_path / "active_flux_radial_integral.mp4",
        lambda fig, axis, frame_idx: _draw_vector_density(
            fields,
            fields.flux_cylindrical,
            fig,
            axis,
            frame_idx,
            "Radially integrated flux density",
            r"$|\int dr\sum_i \dot{\mathbf{r}}_i|$",
            shell_only=False,
            cmap="plasma",
        ),
        fps=fps,
    )
    _write_movie(
        fields,
        image_path / "active_force_density_shell.mp4",
        lambda fig, axis, frame_idx: _draw_vector_density(
            fields,
            fields.force_density_cylindrical,
            fig,
            axis,
            frame_idx,
            "Outer-shell force density",
            r"$|\gamma^{-1}\sum_i \mathbf{F}_i|$",
            shell_only=True,
            cmap="cividis",
        ),
        fps=fps,
    )
    _write_movie(
        fields,
        image_path / "active_force_density_radial_integral.mp4",
        lambda fig, axis, frame_idx: _draw_vector_density(
            fields,
            fields.force_density_cylindrical,
            fig,
            axis,
            frame_idx,
            "Radially integrated force density",
            r"$|\int dr\,\gamma^{-1}\sum_i \mathbf{F}_i|$",
            shell_only=False,
            cmap="cividis",
        ),
        fps=fps,
    )


def plot_active_matter_fields(
    fields: ActiveMatterFields,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
    frame_index: int = -1,
) -> None:
    image_path = Path(image_dir)
    plot_rho_shell(fields, image_path / "active_rho_shell.png", frame_index)
    plot_rho_radial_integral(
        fields,
        image_path / "active_rho_radial_integral.png",
        frame_index,
    )
    plot_polar_shell(fields, image_path / "active_polar_shell.png", frame_index)
    plot_polar_radial_integral(
        fields,
        image_path / "active_polar_radial_integral.png",
        frame_index,
    )
    plot_flux_shell(fields, image_path / "active_flux_shell.png", frame_index)
    plot_flux_radial_integral(
        fields,
        image_path / "active_flux_radial_integral.png",
        frame_index,
    )
    plot_force_density_shell(
        fields,
        image_path / "active_force_density_shell.png",
        frame_index,
    )
    plot_force_density_radial_integral(
        fields,
        image_path / "active_force_density_radial_integral.png",
        frame_index,
    )


def write_active_matter_field_outputs(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    data_dir: str | Path = ACTIVE_DATA_DIR,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
    pocket_radius: float = LOCAL_POCKET_RADIUS,
    n_x_bins: int = ACTIVE_FIELD_X_BINS,
    n_theta_bins: int = ACTIVE_FIELD_THETA_BINS,
    frame_index: int = -1,
    write_movies: bool = True,
    movie_fps: int = ACTIVE_MOVIE_FPS,
) -> ActiveMatterFields:
    fields = active_matter_field_series(
        input_gsd,
        pocket_radius=pocket_radius,
        n_x_bins=n_x_bins,
        n_theta_bins=n_theta_bins,
    )
    save_active_matter_fields(
        fields,
        Path(data_dir) / "active_matter_fields.npz",
        pocket_radius=pocket_radius,
    )
    plot_active_matter_fields(fields, image_dir=image_dir, frame_index=frame_index)
    if write_movies:
        plot_active_matter_movies(fields, image_dir=image_dir, fps=movie_fps)
    return fields
