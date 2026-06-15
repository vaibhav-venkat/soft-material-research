from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import gsd.hoomd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

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
ACTIVE_DATA_DIR = Path(CYLINDER_PATHS.in_gsd).parent
ACTIVE_IMAGE_DIR = Path(CYLINDER_PATHS.com_plot).parent / "active"


@dataclass(frozen=True)
class NeighborCountMatrix:
    steps: np.ndarray
    counts: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.steps
        yield self.counts


@dataclass(frozen=True)
class CenterOfMassSeries:
    steps: np.ndarray
    x_centers: np.ndarray
    theta_centers: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.steps
        yield self.x_centers
        yield self.theta_centers


@dataclass(frozen=True)
class DisclinationCenterOfMassSeries:
    steps: np.ndarray
    plus_x_centers: np.ndarray
    plus_theta_centers: np.ndarray
    minus_x_centers: np.ndarray
    minus_theta_centers: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.steps
        yield self.plus_x_centers
        yield self.plus_theta_centers
        yield self.minus_x_centers
        yield self.minus_theta_centers


@dataclass(frozen=True)
class DislocationSummarySeries:
    steps: np.ndarray
    x_centers: np.ndarray
    theta_centers: np.ndarray
    dislocation_counts: np.ndarray
    plus_disclination_counts: np.ndarray
    minus_disclination_counts: np.ndarray
    net_disclination_charges: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.steps
        yield self.x_centers
        yield self.theta_centers
        yield self.dislocation_counts
        yield self.plus_disclination_counts
        yield self.minus_disclination_counts
        yield self.net_disclination_charges


@dataclass(frozen=True)
class ActiveMatterFields:
    domain: str
    steps: np.ndarray
    x_edges: np.ndarray
    x_centers: np.ndarray
    domain_mask: np.ndarray
    coords: np.ndarray
    rho_count: np.ndarray
    polar_sum: np.ndarray
    polar_mean: np.ndarray
    density_binned: np.ndarray
    polar_magnitude_binned: np.ndarray
    polar_components_binned: np.ndarray
    j_x_sum: np.ndarray
    j_x_density: np.ndarray
    rho_dot: np.ndarray


def _copy_masked_particle_property(source, destination, name: str, mask: np.ndarray) -> None:
    values = getattr(source, name, None)
    if values is None:
        return

    values = np.asarray(values)
    if values.shape[:1] != mask.shape:
        return

    setattr(destination, name, values[mask].copy())


def _format_theta_axis(theta_axis) -> None:
    theta_axis.set_ylabel("Circular mean theta (rad)", color="tab:orange")
    theta_axis.tick_params(axis="y", labelcolor="tab:orange")
    theta_axis.set_ylim(0.0, 2.0 * np.pi)
    theta_axis.set_yticks(
        [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi]
    )
    theta_axis.yaxis.set_major_formatter(
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


def _center_of_mass_or_nan(
    coords: np.ndarray,
    box_length_x: float | None = cylinder.LX,
) -> hx.CenterOfMass:
    if coords.size == 0:
        return hx.CenterOfMass(x=np.nan, theta=np.nan)
    return hx.get_center_of_mass_x_theta(
        coords,
        periodic_x=box_length_x is not None,
        box_length_x=box_length_x,
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


def write_dynamic_values_gsd(
    input_gsd: str | Path,
    output_gsd: str | Path,
    cylinder_radius: float = CYLINDER.cylinder_radius,
    wall_cutoff: float = CYLINDER.wall_cutoff,
) -> None:
    output_path = Path(output_gsd)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        with gsd.hoomd.open(name=str(output_path), mode="w") as destination:
            for frame in source:
                particles = frame.particles
                assert particles.position is not None

                dynamic_values = hx.get_dynamic_values(
                    particles.position,
                    contain_all=True,
                    cylinder_radius=cylinder_radius,
                    cutoff=wall_cutoff,
                )
                new_frame = gsd.hoomd.Frame()
                new_frame.configuration.step = frame.configuration.step
                new_frame.configuration.box = frame.configuration.box
                new_frame.particles.N = dynamic_values.coords.shape[0]
                new_frame.particles.position = dynamic_values.coords.astype(np.float32)
                new_frame.particles.velocity = dynamic_values.coords.astype(np.float32)
                if particles.types is None:
                    new_frame.particles.types = ["A"]
                else:
                    new_frame.particles.types = list(particles.types)

                if particles.typeid is None:
                    new_frame.particles.typeid = np.zeros(
                        dynamic_values.coords.shape[0],
                        dtype=np.uint32,
                    )
                else:
                    new_frame.particles.typeid = np.asarray(particles.typeid)[
                        dynamic_values.shell_mask
                    ].copy()

                for name in (
                    "diameter",
                    "mass",
                    "charge",
                    "body",
                    "image",
                    "orientation",
                    "moment_inertia",
                    "angular_momentum",
                ):
                    _copy_masked_particle_property(
                        particles,
                        new_frame.particles,
                        name,
                        dynamic_values.shell_mask,
                    )

                destination.append(new_frame)


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


def plot_center_of_mass_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    filename: str | Path | None = CYLINDER_PATHS.com_plot,
) -> None:
    series = center_of_mass_series(input_gsd)

    fig, x_axis = plt.subplots(figsize=(9, 5))
    x_line = x_axis.plot(
        series.steps,
        series.x_centers,
        color="tab:blue",
        label="x center of mass",
    )
    x_axis.set_xlabel("Simulation step")
    x_axis.set_ylabel("Center of mass x", color="tab:blue")
    x_axis.tick_params(axis="y", labelcolor="tab:blue")

    theta_axis = x_axis.twinx()
    theta_line = theta_axis.plot(
        series.steps,
        series.theta_centers,
        color="tab:orange",
        label="theta center of mass",
    )
    _format_theta_axis(theta_axis)

    lines = x_line + theta_line
    labels = [line.get_label() for line in lines]
    x_axis.legend(lines, labels, loc="best")
    x_axis.set_title("Cylinder center of mass")
    x_axis.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


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


def plot_dislocation_center_of_mass_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
    filename: str | Path | None = CYLINDER_PATHS.dislocation_com_plot,
) -> None:
    series = dislocation_summary_series(
        input_gsd,
        neighbor_count_txt,
    )

    fig, x_axis = plt.subplots(figsize=(10, 5))
    x_line = x_axis.plot(
        series.steps,
        series.x_centers,
        color="tab:blue",
        label="dislocation x COM",
    )
    x_axis.set_xlabel("Simulation step")
    x_axis.set_ylabel("Center of mass x", color="tab:blue")
    x_axis.tick_params(axis="y", labelcolor="tab:blue")
    x_axis.margins(x=0.02, y=0.15)

    theta_axis = x_axis.twinx()
    theta_line = theta_axis.plot(
        series.steps,
        series.theta_centers,
        color="tab:orange",
        label="dislocation theta COM",
    )
    _format_theta_axis(theta_axis)

    lines = x_line + theta_line
    labels = [line.get_label() for line in lines]
    x_axis.legend(lines, labels, loc="best")
    x_axis.set_title("Cylinder dislocation center of mass")
    x_axis.grid(True, ls="--", alpha=0.35)
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_dislocation_count_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
    filename: str | Path | None = CYLINDER_PATHS.dislocation_count_plot,
) -> None:
    series = dislocation_summary_series(
        input_gsd,
        neighbor_count_txt,
    )

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        series.steps,
        series.dislocation_counts,
        color="tab:purple",
        label="dislocation particles",
    )
    axis.set_xlabel("Simulation step")
    axis.set_ylabel("Number of dislocation particles")
    axis.set_title("Cylinder dislocation count")
    axis.grid(True, ls="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_disclination_count_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
    filename: str | Path | None = CYLINDER_PATHS.disclination_count_plot,
) -> None:
    series = dislocation_summary_series(
        input_gsd,
        neighbor_count_txt,
    )

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        series.steps,
        series.plus_disclination_counts,
        color="tab:orange",
        label="+1 disclinations",
    )
    axis.plot(
        series.steps,
        series.minus_disclination_counts,
        color="tab:red",
        linestyle="--",
        label="-1 disclinations",
    )
    axis.set_xlabel("Simulation step")
    axis.set_ylabel("Number of disclinations")
    axis.set_title("Cylinder disclination counts")
    axis.grid(True, ls="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_net_disclination_charge_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
    filename: str | Path | None = CYLINDER_PATHS.net_charge_plot,
) -> None:
    series = dislocation_summary_series(
        input_gsd,
        neighbor_count_txt,
    )

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        series.steps,
        series.net_disclination_charges,
        color="tab:green",
        label=r"net charge $\sum_i(6 - n_i)$",
    )
    axis.axhline(0, color="black", linewidth=1.0, alpha=0.45)
    axis.set_xlabel("Simulation step")
    axis.set_ylabel("Net disclination charge")
    axis.set_title("Cylinder net disclination charge")
    axis.grid(True, ls="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


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


def _x_bin_indices(x_positions: np.ndarray, box_length_x: float, n_bins: int) -> np.ndarray:
    wrapped = np.mod(x_positions + 0.5 * box_length_x, box_length_x)
    indices = np.floor(wrapped / box_length_x * n_bins).astype(np.int64)
    return np.clip(indices, 0, n_bins - 1)


def _time_edges(steps: np.ndarray) -> np.ndarray:
    steps = np.asarray(steps, dtype=np.float64)
    if len(steps) == 1:
        return np.asarray([steps[0] - 0.5, steps[0] + 0.5], dtype=np.float64)

    mids = 0.5 * (steps[:-1] + steps[1:])
    edges = np.empty(len(steps) + 1, dtype=np.float64)
    edges[1:-1] = mids
    edges[0] = steps[0] - (mids[0] - steps[0])
    edges[-1] = steps[-1] + (steps[-1] - mids[-1])
    return edges


def _binned_mean(
    values: np.ndarray,
    bin_indices: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    counts = np.bincount(bin_indices, minlength=n_bins).astype(np.float64)

    if values.ndim == 1:
        sums = np.bincount(bin_indices, weights=values, minlength=n_bins)
        return np.divide(sums, counts, out=np.full(n_bins, np.nan), where=counts > 0)

    means = np.full((n_bins, values.shape[1]), np.nan, dtype=np.float64)
    for component in range(values.shape[1]):
        sums = np.bincount(
            bin_indices,
            weights=values[:, component],
            minlength=n_bins,
        )
        means[:, component] = np.divide(
            sums,
            counts,
            out=np.full(n_bins, np.nan),
            where=counts > 0,
        )
    return means


def _pocket_fields_for_domain(
    positions: np.ndarray,
    coords: np.ndarray,
    directions: np.ndarray,
    domain_indices: np.ndarray,
    domain: str,
    box_length_x: float,
    pocket_radius: float,
    block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    domain_positions = positions[domain_indices]
    domain_coords = coords[domain_indices]
    domain_directions = directions[domain_indices]

    rho_count = np.zeros(len(domain_indices), dtype=np.int64)
    polar_sum = np.zeros((len(domain_indices), 3), dtype=np.float64)
    cutoff_sq = pocket_radius * pocket_radius

    for start in range(0, len(domain_indices), block_size):
        stop = min(start + block_size, len(domain_indices))
        if domain == "shell":
            dx = _minimum_image_delta(
                domain_coords[start:stop, 0, np.newaxis]
                - domain_coords[np.newaxis, :, 0],
                box_length_x,
            )
            dtheta = _minimum_image_delta(
                domain_coords[start:stop, 1, np.newaxis]
                - domain_coords[np.newaxis, :, 1],
                2.0 * np.pi,
            )
            distances_sq = dx * dx + (CYLINDER.cylinder_radius * dtheta) ** 2
        else:
            deltas = (
                domain_positions[start:stop, np.newaxis, :]
                - domain_positions[np.newaxis, :, :]
            )
            deltas[..., 0] = _minimum_image_delta(deltas[..., 0], box_length_x)
            distances_sq = np.sum(deltas * deltas, axis=2)

        pocket_mask = distances_sq <= cutoff_sq
        rho_count[start:stop] = np.count_nonzero(pocket_mask, axis=1)
        polar_sum[start:stop] = pocket_mask.astype(np.float64) @ domain_directions

    polar_mean = np.divide(
        polar_sum,
        rho_count[:, np.newaxis],
        out=np.full_like(polar_sum, np.nan),
        where=rho_count[:, np.newaxis] > 0,
    )
    return rho_count, polar_sum, polar_mean


def active_matter_field_series(
    input_gsd: str | Path,
    domain: str,
    pocket_radius: float = LOCAL_POCKET_RADIUS,
    n_x_bins: int = ACTIVE_FIELD_X_BINS,
) -> ActiveMatterFields:
    assert domain in ("all", "shell")
    steps: list[int] = []
    x_edges: np.ndarray | None = None
    x_centers: np.ndarray | None = None

    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        assert len(source) > 0
        n_frames = len(source)
        n_particles = int(source[0].particles.N)

        domain_masks = np.zeros((n_frames, n_particles), dtype=np.bool_)
        coords = np.full((n_frames, n_particles, 3), np.nan, dtype=np.float64)
        rho_count = np.zeros((n_frames, n_particles), dtype=np.int64)
        polar_sum = np.zeros((n_frames, n_particles, 3), dtype=np.float64)
        polar_mean = np.full((n_frames, n_particles, 3), np.nan, dtype=np.float64)
        density_binned = np.full((n_frames, n_x_bins), np.nan, dtype=np.float64)
        polar_magnitude_binned = np.full((n_frames, n_x_bins), np.nan, dtype=np.float64)
        polar_components_binned = np.full(
            (n_frames, n_x_bins, 3),
            np.nan,
            dtype=np.float64,
        )
        j_x_sum = np.zeros((n_frames, n_x_bins), dtype=np.float64)
        j_x_density = np.zeros((n_frames, n_x_bins), dtype=np.float64)
        rho_dot = np.zeros((n_frames, n_x_bins), dtype=np.float64)

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
            virials = _logged_particle_array(frame, "virials", n_particles)
            assert forces.ndim == 2 and forces.shape[1] >= 3
            assert virials.ndim == 2 and virials.shape[1] >= 6

            frame_coords = hx.get_new_coords(positions)
            coords[frame_idx] = frame_coords
            if domain == "shell":
                dynamic_values = hx.get_dynamic_values(
                    positions,
                    contain_all=False,
                    cylinder_radius=CYLINDER.cylinder_radius,
                    cutoff=CYLINDER.wall_cutoff,
                )
                domain_mask = dynamic_values.shell_mask
            else:
                domain_mask = np.ones(n_particles, dtype=np.bool_)

            domain_indices = np.flatnonzero(domain_mask)
            domain_masks[frame_idx] = domain_mask
            steps.append(int(frame.configuration.step))
            if len(domain_indices) == 0:
                continue

            pocket_rho, pocket_polar_sum, pocket_polar_mean = _pocket_fields_for_domain(
                positions,
                frame_coords,
                directions,
                domain_indices,
                domain=domain,
                box_length_x=box_length_x,
                pocket_radius=pocket_radius,
            )
            rho_count[frame_idx, domain_indices] = pocket_rho
            polar_sum[frame_idx, domain_indices] = pocket_polar_sum
            polar_mean[frame_idx, domain_indices] = pocket_polar_mean

            bin_indices = _x_bin_indices(
                positions[domain_indices, 0],
                box_length_x,
                n_x_bins,
            )
            density_binned[frame_idx] = _binned_mean(
                pocket_rho.astype(np.float64),
                bin_indices,
                n_x_bins,
            )
            polar_magnitude_binned[frame_idx] = _binned_mean(
                np.linalg.norm(pocket_polar_mean, axis=1),
                bin_indices,
                n_x_bins,
            )
            polar_components_binned[frame_idx] = _binned_mean(
                pocket_polar_mean,
                bin_indices,
                n_x_bins,
            )

            rdot_x = (
                CYLINDER_SIM.u0 * directions[domain_indices, 0]
                + forces[domain_indices, 0] / CYLINDER_SIM.gamma
            )
            np.add.at(j_x_sum[frame_idx], bin_indices, rdot_x)
            dx = box_length_x / n_x_bins
            j_x_density[frame_idx] = j_x_sum[frame_idx] / dx
            rho_dot[frame_idx] = -(
                np.roll(j_x_density[frame_idx], -1)
                - np.roll(j_x_density[frame_idx], 1)
            ) / (2.0 * dx)

    assert x_edges is not None and x_centers is not None
    return ActiveMatterFields(
        domain=domain,
        steps=np.asarray(steps, dtype=np.int64),
        x_edges=x_edges,
        x_centers=x_centers,
        domain_mask=domain_masks,
        coords=coords,
        rho_count=rho_count,
        polar_sum=polar_sum,
        polar_mean=polar_mean,
        density_binned=density_binned,
        polar_magnitude_binned=polar_magnitude_binned,
        polar_components_binned=polar_components_binned,
        j_x_sum=j_x_sum,
        j_x_density=j_x_density,
        rho_dot=rho_dot,
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
        domain=fields.domain,
        pocket_radius=np.asarray(pocket_radius, dtype=np.float64),
        steps=fields.steps,
        x_edges=fields.x_edges,
        x_centers=fields.x_centers,
        domain_mask=fields.domain_mask,
        coords=fields.coords,
        rho_count=fields.rho_count,
        polar_sum=fields.polar_sum,
        polar_mean=fields.polar_mean,
        density_binned=fields.density_binned,
        polar_magnitude_binned=fields.polar_magnitude_binned,
        polar_components_binned=fields.polar_components_binned,
        j_x_sum=fields.j_x_sum,
        j_x_density=fields.j_x_density,
        rho_dot=fields.rho_dot,
    )


def plot_active_heatmap(
    steps: np.ndarray,
    x_edges: np.ndarray,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    filename: str | Path,
    cmap: str = "viridis",
) -> None:
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 5))
    mesh = axis.pcolormesh(
        x_edges,
        _time_edges(steps),
        values,
        shading="auto",
        cmap=cmap,
    )
    fig.colorbar(mesh, ax=axis, label=colorbar_label)
    axis.set_xlabel("x")
    axis.set_ylabel("Simulation step")
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_active_matter_fields(
    fields: ActiveMatterFields,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
) -> None:
    image_path = Path(image_dir)
    prefix = f"active_{fields.domain}"
    plot_active_heatmap(
        fields.steps,
        fields.x_edges,
        fields.density_binned,
        f"{fields.domain} local density",
        "Binned mean local density",
        image_path / f"{prefix}_local_density.png",
    )
    plot_active_heatmap(
        fields.steps,
        fields.x_edges,
        fields.polar_magnitude_binned,
        f"{fields.domain} polar magnitude",
        "Binned mean |polar mean|",
        image_path / f"{prefix}_polar_magnitude.png",
    )
    plot_active_heatmap(
        fields.steps,
        fields.x_edges,
        fields.j_x_density,
        f"{fields.domain} axial flux",
        r"$J_x$ density",
        image_path / f"{prefix}_j_x_density.png",
    )
    plot_active_heatmap(
        fields.steps,
        fields.x_edges,
        fields.rho_dot,
        f"{fields.domain} density time derivative",
        r"$\dot{\rho}$",
        image_path / f"{prefix}_rho_dot.png",
        cmap="coolwarm",
    )


def write_active_matter_field_outputs(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    data_dir: str | Path = ACTIVE_DATA_DIR,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
    pocket_radius: float = LOCAL_POCKET_RADIUS,
    n_x_bins: int = ACTIVE_FIELD_X_BINS,
) -> tuple[ActiveMatterFields, ActiveMatterFields]:
    data_path = Path(data_dir)
    all_fields = active_matter_field_series(
        input_gsd,
        domain="all",
        pocket_radius=pocket_radius,
        n_x_bins=n_x_bins,
    )
    shell_fields = active_matter_field_series(
        input_gsd,
        domain="shell",
        pocket_radius=pocket_radius,
        n_x_bins=n_x_bins,
    )
    save_active_matter_fields(
        all_fields,
        data_path / "active_matter_all_fields.npz",
        pocket_radius=pocket_radius,
    )
    save_active_matter_fields(
        shell_fields,
        data_path / "active_matter_shell_fields.npz",
        pocket_radius=pocket_radius,
    )
    plot_active_matter_fields(all_fields, image_dir=image_dir)
    plot_active_matter_fields(shell_fields, image_dir=image_dir)
    return all_fields, shell_fields


def main() -> None:
    write_dynamic_values_gsd(CYLINDER_PATHS.in_gsd, CYLINDER_PATHS.dynamic_values_gsd)
    plot_center_of_mass_series(CYLINDER_PATHS.in_gsd, CYLINDER_PATHS.com_plot)
    plot_disclination_center_of_mass_series(
        CYLINDER_PATHS.in_gsd,
        CYLINDER_PATHS.neighbor_count_txt,
        CYLINDER_PATHS.disclination_com_plot,
    )
    plot_dislocation_center_of_mass_series(
        CYLINDER_PATHS.in_gsd,
        CYLINDER_PATHS.neighbor_count_txt,
        CYLINDER_PATHS.dislocation_com_plot,
    )
    plot_dislocation_count_series(
        CYLINDER_PATHS.in_gsd,
        CYLINDER_PATHS.neighbor_count_txt,
        CYLINDER_PATHS.dislocation_count_plot,
    )
    plot_disclination_count_series(
        CYLINDER_PATHS.in_gsd,
        CYLINDER_PATHS.neighbor_count_txt,
        CYLINDER_PATHS.disclination_count_plot,
    )
    plot_net_disclination_charge_series(
        CYLINDER_PATHS.in_gsd,
        CYLINDER_PATHS.neighbor_count_txt,
        CYLINDER_PATHS.net_charge_plot,
    )
    write_active_matter_field_outputs(
        CYLINDER_PATHS.in_gsd,
        ACTIVE_DATA_DIR,
        ACTIVE_IMAGE_DIR,
        pocket_radius=LOCAL_POCKET_RADIUS,
        n_x_bins=ACTIVE_FIELD_X_BINS,
    )
    print(f"Wrote OVITO dynamic values file to {CYLINDER_PATHS.dynamic_values_gsd}.")
    print(f"Wrote center-of-mass plot to {CYLINDER_PATHS.com_plot}.")
    print(
        "Wrote disclination center-of-mass plot to "
        f"{CYLINDER_PATHS.disclination_com_plot}."
    )
    print(
        "Wrote dislocation center-of-mass plot to "
        f"{CYLINDER_PATHS.dislocation_com_plot}."
    )
    print(f"Wrote dislocation count plot to {CYLINDER_PATHS.dislocation_count_plot}.")
    print(f"Wrote disclination count plot to {CYLINDER_PATHS.disclination_count_plot}.")
    print(f"Wrote net disclination charge plot to {CYLINDER_PATHS.net_charge_plot}.")
    print(
        "Wrote active matter fields to "
        f"{ACTIVE_DATA_DIR / 'active_matter_all_fields.npz'} and "
        f"{ACTIVE_DATA_DIR / 'active_matter_shell_fields.npz'}."
    )
    print(f"Wrote active matter plots to {ACTIVE_IMAGE_DIR}.")
    print("OVITO position.x stores x")
    print("OVITO position.y stores theta")
    print("OVITO position.z stores r")
    print("OVITO velocity duplicates position")


if __name__ == "__main__":
    main()
