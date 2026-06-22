from pathlib import Path

import gsd.hoomd
import numpy as np

from hexatic import analysis as hx
from hexatic.constants import cylinder

from .common import (
    CYLINDER,
    CYLINDER_PATHS,
    CenterOfMassSeries,
    DisclinationCenterOfMassSeries,
    DislocationSummarySeries,
    NeighborCountMatrix,
    ThetaCOMVelocitySeries,
    XCOMVelocitySeries,
    _center_of_mass_or_nan,
    _minimum_image_delta,
    _particle_masses,
    _theta_bin_indices,
    _theta_from_positions,
    _unwrapped_x_positions,
    _x_bin_indices,
)


def load_neighbor_count_matrix(
    filename: str | Path,
) -> NeighborCountMatrix:
    table = np.loadtxt(filename, dtype=np.int64)
    if table.ndim == 1:
        table = table[np.newaxis, :]
    assert table.ndim == 2 and table.shape[1] >= 4

    frame_indices = table[:, 0]
    step_values = table[:, 1]
    particle_indices = table[:, 2]
    neighbor_counts = table[:, 3]

    n_frames = int(frame_indices.max()) + 1
    n_particles = int(particle_indices.max()) + 1
    flat_indices = frame_indices * n_particles + particle_indices
    assert np.unique(flat_indices).size == flat_indices.size

    steps = np.full(n_frames, -1, dtype=np.int64)
    counts = np.full((n_frames, n_particles), -1, dtype=np.int64)
    counts[frame_indices, particle_indices] = neighbor_counts

    for frame_idx in range(n_frames):
        frame_steps = np.unique(step_values[frame_indices == frame_idx])
        assert frame_steps.size == 1
        steps[frame_idx] = frame_steps[0]

    assert np.all(counts >= 0)
    return NeighborCountMatrix(steps=steps, counts=counts)


def first_trajectory_step(input_gsd: str | Path) -> int:
    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        if len(source) == 0:
            raise ValueError(f"No frames found in {input_gsd}")
        return int(source[0].configuration.step)


def center_of_mass_series(
    input_gsd: str | Path,
) -> CenterOfMassSeries:
    """Computer the center of mass of the cylinder system w.r.t x and theta"""
    steps: list[int] = []
    x_centers: list[float] = []
    theta_centers: list[float] = []

    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        for frame in source:
            particles = frame.particles
            assert particles.position is not None
            dynamic_values = hx.get_dynamic_values(
                particles.position,
                contain_all=False,
                cylinder_radius=CYLINDER.cylinder_radius,
                cutoff=CYLINDER.wall_cutoff,
            )
            box_length_x = float(frame.configuration.box[0])
            center = _center_of_mass_or_nan(
                dynamic_values.coords,
                box_length_x=box_length_x,
            )
            steps.append(int(frame.configuration.step))
            x_centers.append(center.x)
            theta_centers.append(center.theta)

    return CenterOfMassSeries(
        steps=np.asarray(steps, dtype=np.int64),
        x_centers=np.asarray(x_centers, dtype=np.float64),
        theta_centers=np.asarray(theta_centers, dtype=np.float64),
    )


def x_center_of_mass_velocity_series(
    input_gsd: str | Path,
    shell_only: bool = False,
) -> XCOMVelocitySeries:
    steps: list[int] = []
    x_velocities: list[float] = []
    previous_x: np.ndarray | None = None
    previous_step: int | None = None
    previous_mask: np.ndarray | None = None
    previous_masses: np.ndarray | None = None
    previous_box_length_x: float | None = None

    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        for frame in source:
            particles = frame.particles
            assert particles.position is not None

            positions = np.asarray(particles.position, dtype=np.float64)
            box_length_x = float(frame.configuration.box[0])
            current_x = _unwrapped_x_positions(particles, box_length_x)
            current_step = int(frame.configuration.step)
            current_masses = _particle_masses(particles, positions.shape[0])
            current_mask = np.ones(positions.shape[0], dtype=bool)
            if shell_only:
                current_mask = hx.get_dynamic_values(
                    positions,
                    contain_all=False,
                    cylinder_radius=CYLINDER.cylinder_radius,
                    cutoff=CYLINDER.wall_cutoff,
                ).shell_mask

            if (
                previous_x is not None
                and previous_step is not None
                and previous_mask is not None
                and previous_masses is not None
                and previous_box_length_x is not None
            ):
                assert current_x.shape == previous_x.shape
                delta_t = (current_step - previous_step) * float(cylinder.TIMESTEP)
                mask = previous_mask & current_mask
                if delta_t > 0.0 and np.any(mask):
                    delta_x = _minimum_image_delta(
                        current_x - previous_x,
                        previous_box_length_x,
                    )
                    weights = previous_masses[mask]
                    x_velocity = float(np.average(delta_x[mask], weights=weights) / delta_t)
                else:
                    x_velocity = np.nan
                steps.append(current_step)
                x_velocities.append(x_velocity)

            previous_x = current_x
            previous_step = current_step
            previous_mask = current_mask
            previous_masses = current_masses
            previous_box_length_x = box_length_x

    return XCOMVelocitySeries(
        steps=np.asarray(steps, dtype=np.int64),
        x_velocities=np.asarray(x_velocities, dtype=np.float64),
    )


def theta_com_velocity_series(
    input_gsd: str | Path,
    shell_only: bool = False,
) -> ThetaCOMVelocitySeries:
    steps: list[int] = []
    theta_velocities: list[float] = []
    previous_theta: np.ndarray | None = None
    previous_step: int | None = None
    previous_mask: np.ndarray | None = None
    previous_masses: np.ndarray | None = None

    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        for frame in source:
            particles = frame.particles
            assert particles.position is not None

            positions = np.asarray(particles.position, dtype=np.float64)
            current_step = int(frame.configuration.step)
            current_theta = _theta_from_positions(positions)
            current_masses = _particle_masses(particles, positions.shape[0])
            current_mask = np.ones(positions.shape[0], dtype=bool)
            if shell_only:
                current_mask = hx.get_dynamic_values(
                    positions,
                    contain_all=False,
                    cylinder_radius=CYLINDER.cylinder_radius,
                    cutoff=CYLINDER.wall_cutoff,
                ).shell_mask

            if (
                previous_theta is not None
                and previous_step is not None
                and previous_mask is not None
                and previous_masses is not None
            ):
                assert current_theta.shape == previous_theta.shape
                delta_t = (current_step - previous_step) * float(cylinder.TIMESTEP)
                mask = previous_mask & current_mask
                if delta_t > 0.0 and np.any(mask):
                    delta_theta = _minimum_image_delta(
                        current_theta - previous_theta,
                        2.0 * np.pi,
                    )
                    weights = previous_masses[mask]
                    theta_velocity = float(
                        np.average(delta_theta[mask], weights=weights) / delta_t
                    )
                else:
                    theta_velocity = np.nan
                steps.append(current_step)
                theta_velocities.append(theta_velocity)

            previous_theta = current_theta
            previous_step = current_step
            previous_mask = current_mask
            previous_masses = current_masses

    return ThetaCOMVelocitySeries(
        steps=np.asarray(steps, dtype=np.int64),
        theta_velocities=np.asarray(theta_velocities, dtype=np.float64),
    )


def _outer_shell_xtheta_velocity_frames(
    input_gsd: str | Path,
    start_frame: int,
    n_x_bins: int,
    n_theta_bins: int,
    velocity_component: str,
) -> list[dict[str, np.ndarray | int | float]]:
    if velocity_component not in {"x", "theta"}:
        raise ValueError("velocity_component must be 'x' or 'theta'.")

    frames: list[dict[str, np.ndarray | int | float]] = []
    previous_x: np.ndarray | None = None
    previous_theta: np.ndarray | None = None
    previous_step: int | None = None
    previous_shell_mask: np.ndarray | None = None
    previous_box_length_x: float | None = None

    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        for frame_idx, frame in enumerate(source):
            particles = frame.particles
            assert particles.position is not None

            positions = np.asarray(particles.position, dtype=np.float64)
            box_length_x = float(frame.configuration.box[0])
            step = int(frame.configuration.step)
            current_x = _unwrapped_x_positions(particles, box_length_x)
            current_theta = _theta_from_positions(positions)
            dynamic_values = hx.get_dynamic_values(
                positions,
                contain_all=False,
                cylinder_radius=CYLINDER.cylinder_radius,
                cutoff=CYLINDER.wall_cutoff,
            )
            shell_mask = dynamic_values.shell_mask

            if (
                frame_idx >= start_frame
                and previous_x is not None
                and previous_theta is not None
                and previous_step is not None
                and previous_shell_mask is not None
                and previous_box_length_x is not None
            ):
                assert current_x.shape == previous_x.shape
                delta_t = (step - previous_step) * float(cylinder.TIMESTEP)
                mask = shell_mask & previous_shell_mask
                if delta_t > 0.0 and np.any(mask):
                    if velocity_component == "x":
                        delta = _minimum_image_delta(
                            current_x - previous_x,
                            previous_box_length_x,
                        )
                    else:
                        delta = _minimum_image_delta(
                            current_theta - previous_theta,
                            2.0 * np.pi,
                        )
                    velocity = delta[mask] / delta_t
                    shell_positions = positions[mask]
                    x_indices = _x_bin_indices(
                        shell_positions[:, 0],
                        box_length_x,
                        n_x_bins,
                    )
                    theta = _theta_from_positions(shell_positions)
                    theta_indices = _theta_bin_indices(theta, n_theta_bins)
                    flat_indices = x_indices * n_theta_bins + theta_indices
                    n_bins = n_x_bins * n_theta_bins
                    counts = np.bincount(flat_indices, minlength=n_bins)
                    velocity_sums = np.bincount(
                        flat_indices,
                        weights=velocity,
                        minlength=n_bins,
                    )
                    occupied = counts > 0
                    bin_velocity = np.divide(
                        velocity_sums,
                        counts,
                        out=np.full(n_bins, np.nan, dtype=np.float64),
                        where=occupied,
                    )
                    occupied_indices = np.flatnonzero(occupied)
                    x_edges = np.linspace(
                        -0.5 * box_length_x,
                        0.5 * box_length_x,
                        n_x_bins + 1,
                    )
                    theta_edges = np.linspace(0.0, 2.0 * np.pi, n_theta_bins + 1)
                    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
                    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
                    occupied_x = occupied_indices // n_theta_bins
                    occupied_theta = occupied_indices % n_theta_bins
                    frames.append(
                        {
                            "frame_idx": frame_idx,
                            "step": step,
                            "x": x_centers[occupied_x],
                            "theta": theta_centers[occupied_theta],
                            "velocity": bin_velocity[occupied_indices],
                            "counts": counts[occupied_indices],
                            "box_length_x": box_length_x,
                        }
                    )
                else:
                    frames.append(
                        {
                            "frame_idx": frame_idx,
                            "step": step,
                            "x": np.asarray([], dtype=np.float64),
                            "theta": np.asarray([], dtype=np.float64),
                            "velocity": np.asarray([], dtype=np.float64),
                            "counts": np.asarray([], dtype=np.int64),
                            "box_length_x": box_length_x,
                        }
                    )

            previous_x = current_x
            previous_theta = current_theta
            previous_step = step
            previous_shell_mask = shell_mask
            previous_box_length_x = box_length_x

    return frames

def disclination_center_of_mass_series(
    input_gsd: str | Path,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
) -> DisclinationCenterOfMassSeries:
    neighbor_data = load_neighbor_count_matrix(neighbor_count_txt)
    steps: list[int] = []
    plus_x_centers: list[float] = []
    plus_theta_centers: list[float] = []
    minus_x_centers: list[float] = []
    minus_theta_centers: list[float] = []

    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        for frame_idx, frame in enumerate(source):
            particles = frame.particles
            assert particles.position is not None
            assert frame_idx < neighbor_data.counts.shape[0]
            assert int(frame.configuration.step) == int(neighbor_data.steps[frame_idx])

            dynamic_values = hx.get_dynamic_values(
                particles.position,
                contain_all=False,
                cylinder_radius=CYLINDER.cylinder_radius,
                cutoff=CYLINDER.wall_cutoff,
            )
            shell_charges = (
                CYLINDER.neighbors
                - neighbor_data.counts[frame_idx, dynamic_values.shell_mask]
            )
            box_length_x = float(frame.configuration.box[0])

            plus_center = _center_of_mass_or_nan(
                dynamic_values.coords[shell_charges == 1],
                box_length_x=box_length_x,
            )
            minus_center = _center_of_mass_or_nan(
                dynamic_values.coords[shell_charges == -1],
                box_length_x=box_length_x,
            )

            steps.append(int(frame.configuration.step))
            plus_x_centers.append(plus_center.x)
            plus_theta_centers.append(plus_center.theta)
            minus_x_centers.append(minus_center.x)
            minus_theta_centers.append(minus_center.theta)

    return DisclinationCenterOfMassSeries(
        steps=np.asarray(steps, dtype=np.int64),
        plus_x_centers=np.asarray(plus_x_centers, dtype=np.float64),
        plus_theta_centers=np.asarray(plus_theta_centers, dtype=np.float64),
        minus_x_centers=np.asarray(minus_x_centers, dtype=np.float64),
        minus_theta_centers=np.asarray(minus_theta_centers, dtype=np.float64),
    )


def plot_disclination_center_of_mass_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
    filename: str | Path | None = CYLINDER_PATHS.disclination_com_plot,
) -> None:
    series = disclination_center_of_mass_series(input_gsd, neighbor_count_txt)

    fig, x_axis = plt.subplots(figsize=(12, 6))
    plus_x_line = x_axis.plot(
        series.steps,
        series.plus_x_centers,
        color="#1f77b4",
        linewidth=1.8,
        alpha=0.9,
        label="+1 x COM",
    )
    minus_x_line = x_axis.plot(
        series.steps,
        series.minus_x_centers,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.8,
        alpha=0.65,
        label="-1 x COM",
    )
    x_axis.set_xlabel("Simulation step")
    x_axis.set_ylabel("Center of mass x", color="tab:blue")
    x_axis.tick_params(axis="y", labelcolor="tab:blue")
    x_axis.margins(x=0.02, y=0.15)

    theta_axis = x_axis.twinx()
    plus_theta_line = theta_axis.plot(
        series.steps,
        series.plus_theta_centers,
        color="#ff7f0e",
        linewidth=1.8,
        alpha=0.9,
        label="+1 theta COM",
    )
    minus_theta_line = theta_axis.plot(
        series.steps,
        series.minus_theta_centers,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.8,
        alpha=0.65,
        label="-1 theta COM",
    )
    _format_theta_axis(theta_axis)

    lines = plus_x_line + plus_theta_line + minus_x_line + minus_theta_line
    labels = [line.get_label() for line in lines]
    x_axis.legend(
        lines,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
    )
    x_axis.set_title("Cylinder disclination center of mass")
    x_axis.grid(True, ls="--", alpha=0.28)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def dislocation_summary_series(
    input_gsd: str | Path,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
) -> DislocationSummarySeries:
    neighbor_data = load_neighbor_count_matrix(neighbor_count_txt)
    steps: list[int] = []
    dislocation_x_centers: list[float] = []
    dislocation_theta_centers: list[float] = []
    dislocation_counts: list[int] = []
    plus_disclination_counts: list[int] = []
    minus_disclination_counts: list[int] = []
    net_disclination_charges: list[int] = []

    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        for frame_idx, frame in enumerate(source):
            particles = frame.particles
            assert particles.position is not None
            assert frame_idx < neighbor_data.counts.shape[0]
            assert int(frame.configuration.step) == int(neighbor_data.steps[frame_idx])

            dynamic_values = hx.get_dynamic_values(
                particles.position,
                contain_all=False,
                cylinder_radius=CYLINDER.cylinder_radius,
                cutoff=CYLINDER.wall_cutoff,
            )
            shell_charges = (
                CYLINDER.neighbors
                - neighbor_data.counts[frame_idx, dynamic_values.shell_mask]
            )

            charges = np.zeros(particles.position.shape[0], dtype=np.int64)
            charges[dynamic_values.shell_mask] = shell_charges
            dislocation_particles = hx.identify_dislocation_particles_frame(
                particles.position,
                charges,
                pair_distance=CYLINDER.dislocation_pair_distance,
                box_length_x=float(frame.configuration.box[0]),
            )
            shell_dislocations = dislocation_particles[dynamic_values.shell_mask] == 1
            center = _center_of_mass_or_nan(
                dynamic_values.coords[shell_dislocations],
                box_length_x=float(frame.configuration.box[0]),
            )

            steps.append(int(frame.configuration.step))
            dislocation_x_centers.append(center.x)
            dislocation_theta_centers.append(center.theta)
            dislocation_counts.append(int(np.count_nonzero(shell_dislocations)))
            plus_disclination_counts.append(int(np.count_nonzero(shell_charges == 1)))
            minus_disclination_counts.append(int(np.count_nonzero(shell_charges == -1)))
            net_disclination_charges.append(int(np.sum(shell_charges)))

    return DislocationSummarySeries(
        steps=np.asarray(steps, dtype=np.int64),
        x_centers=np.asarray(dislocation_x_centers, dtype=np.float64),
        theta_centers=np.asarray(dislocation_theta_centers, dtype=np.float64),
        dislocation_counts=np.asarray(dislocation_counts, dtype=np.int64),
        plus_disclination_counts=np.asarray(plus_disclination_counts, dtype=np.int64),
        minus_disclination_counts=np.asarray(minus_disclination_counts, dtype=np.int64),
        net_disclination_charges=np.asarray(net_disclination_charges, dtype=np.int64),
    )
