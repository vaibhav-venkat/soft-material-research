from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import gsd.hoomd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

if __package__:
    from hexatic import analysis as hx
    from hexatic.active_matter_cylinder import (
        ACTIVE_DATA_DIR,
        write_active_matter_field_outputs,
    )
    from hexatic.chirality import CHIRALITY_DATA_DIR, write_chirality_outputs, ChiralityConfig
    from hexatic.constants import cylinder
else:
    import analysis as hx
    from active_matter_cylinder import ACTIVE_DATA_DIR, write_active_matter_field_outputs
    from chirality import CHIRALITY_DATA_DIR, write_chirality_outputs, ChiralityConfig
    from constants import cylinder

CYLINDER = cylinder.ANALYSIS
CYLINDER_PATHS = cylinder.PATHS


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


def main() -> None:
    write_dynamic_values_gsd(CYLINDER_PATHS.in_gsd, CYLINDER_PATHS.dynamic_values_gsd)
    print(f"Wrote OVITO dynamic values file to {CYLINDER_PATHS.dynamic_values_gsd}.")
    # plot_center_of_mass_series(CYLINDER_PATHS.in_gsd, CYLINDER_PATHS.com_plot)
    # print(f"Wrote center-of-mass plot to {CYLINDER_PATHS.com_plot}.")
    # plot_disclination_center_of_mass_series(
    #     CYLINDER_PATHS.in_gsd,
    #     CYLINDER_PATHS.neighbor_count_txt,
    #     CYLINDER_PATHS.disclination_com_plot,
    # )
    # print(
    #     "Wrote disclination center-of-mass plot to "
    #     f"{CYLINDER_PATHS.disclination_com_plot}."
    # )
    # plot_dislocation_center_of_mass_series(
    #     CYLINDER_PATHS.in_gsd,
    #     CYLINDER_PATHS.neighbor_count_txt,
    #     CYLINDER_PATHS.dislocation_com_plot,
    # )
    # print(
    #     "Wrote dislocation center-of-mass plot to "
    #     f"{CYLINDER_PATHS.dislocation_com_plot}."
    # )
    # plot_dislocation_count_series(
    #     CYLINDER_PATHS.in_gsd,
    #     CYLINDER_PATHS.neighbor_count_txt,
    #     CYLINDER_PATHS.dislocation_count_plot,
    # )
    # print(f"Wrote dislocation count plot to {CYLINDER_PATHS.dislocation_count_plot}.")
    # plot_disclination_count_series(
    #     CYLINDER_PATHS.in_gsd,
    #     CYLINDER_PATHS.neighbor_count_txt,
    #     CYLINDER_PATHS.disclination_count_plot,
    # )
    # print(f"Wrote disclination count plot to {CYLINDER_PATHS.disclination_count_plot}.")
    # plot_net_disclination_charge_series(
    #     CYLINDER_PATHS.in_gsd,
    #     CYLINDER_PATHS.neighbor_count_txt,
    #     CYLINDER_PATHS.net_charge_plot,
    # )
    # print(f"Wrote net disclination charge plot to {CYLINDER_PATHS.net_charge_plot}.")
    write_active_matter_field_outputs(
        CYLINDER_PATHS.in_gsd,
    )
    print(
        "Wrote active matter fields to "
        f"{ACTIVE_DATA_DIR / 'active_matter_fields.npz'}."
    )
    write_chirality_outputs(
        CYLINDER_PATHS.in_gsd,
        config = ChiralityConfig(limit_disclination=True)
    )
    print(
        "Wrote disclination chirality fields to "
        f"{CHIRALITY_DATA_DIR / 'chirality_disclinations'}."
    )
    print("OVITO position.x stores x")
    print("OVITO position.y stores theta")
    print("OVITO position.z stores r")
    print("OVITO velocity duplicates position")

if __name__ == "__main__":
    main()
