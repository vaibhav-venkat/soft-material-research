from pathlib import Path

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
    box_length_x: float = cylinder.LX
) -> tuple[float, float]:
    if coords.size == 0:
        return np.nan, np.nan
    return hx.get_center_of_mass_x_theta(
        coords,
        # periodic_x=box_length_x is not None,
        # box_length_x=box_length_x,
    )


def load_neighbor_count_matrix(
    filename: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
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
    return steps, counts


def write_dynamic_values_gsd(
    input_gsd: str | Path,
    output_gsd: str | Path,
    cylinder_radius: float = cylinder.CYLINDER_RADIUS,
    wall_cutoff: float = cylinder.WALL_CUTOFF,
) -> None:
    output_path = Path(output_gsd)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        with gsd.hoomd.open(name=str(output_path), mode="w") as destination:
            for frame in source:
                particles = frame.particles
                assert particles.position is not None

                dynamic_values, shell_mask = hx.get_dynamic_values(
                    particles.position,
                    contain_all=True,
                    cylinder_radius=cylinder_radius,
                    cutoff=wall_cutoff,
                )
                new_frame = gsd.hoomd.Frame()
                new_frame.configuration.step = frame.configuration.step
                new_frame.configuration.box = frame.configuration.box
                new_frame.particles.N = dynamic_values.shape[0]
                new_frame.particles.position = dynamic_values.astype(np.float32)
                new_frame.particles.velocity = dynamic_values.astype(np.float32)
                if particles.types is None:
                    new_frame.particles.types = ["A"]
                else:
                    new_frame.particles.types = list(particles.types)

                if particles.typeid is None:
                    new_frame.particles.typeid = np.zeros(
                        dynamic_values.shape[0],
                        dtype=np.uint32,
                    )
                else:
                    new_frame.particles.typeid = np.asarray(particles.typeid)[
                        shell_mask
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
                        shell_mask,
                    )

                destination.append(new_frame)


def center_of_mass_series(
    input_gsd: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computer the center of mass of the cylinder system w.r.t x and theta"""
    steps: list[int] = []
    x_centers: list[float] = []
    theta_centers: list[float] = []

    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        for frame in source:
            particles = frame.particles
            assert particles.position is not None
            dynamic_values, shell_mask = hx.get_dynamic_values(
                particles.position,
                contain_all=False,
                cylinder_radius=cylinder.CYLINDER_RADIUS,
                cutoff=cylinder.WALL_CUTOFF,
            )
            box_length_x = float(frame.configuration.box[0])
            x_center, theta_center = _center_of_mass_or_nan(
                dynamic_values,
                box_length_x=box_length_x,
            )
            steps.append(int(frame.configuration.step))
            x_centers.append(x_center)
            theta_centers.append(theta_center)

    return (
        np.asarray(steps, dtype=np.int64),
        np.asarray(x_centers, dtype=np.float64),
        np.asarray(theta_centers, dtype=np.float64),
    )


def plot_center_of_mass_series(
    input_gsd: str | Path = cylinder.IN_GSD,
    filename: str | Path | None = cylinder.COM_PLOT,
) -> None:
    steps, x_centers, theta_centers = center_of_mass_series(input_gsd)

    fig, x_axis = plt.subplots(figsize=(9, 5))
    x_line = x_axis.plot(
        steps,
        x_centers,
        color="tab:blue",
        label="x center of mass",
    )
    x_axis.set_xlabel("Simulation step")
    x_axis.set_ylabel("Center of mass x", color="tab:blue")
    x_axis.tick_params(axis="y", labelcolor="tab:blue")

    theta_axis = x_axis.twinx()
    theta_line = theta_axis.plot(
        steps,
        theta_centers,
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
    neighbor_count_txt: str | Path = cylinder.NEIGHBOR_COUNT_TXT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    neighbor_steps, neighbor_counts = load_neighbor_count_matrix(neighbor_count_txt)
    steps: list[int] = []
    plus_x_centers: list[float] = []
    plus_theta_centers: list[float] = []
    minus_x_centers: list[float] = []
    minus_theta_centers: list[float] = []

    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        for frame_idx, frame in enumerate(source):
            particles = frame.particles
            assert particles.position is not None
            assert frame_idx < neighbor_counts.shape[0]
            assert int(frame.configuration.step) == int(neighbor_steps[frame_idx])

            dynamic_values, shell_mask = hx.get_dynamic_values(
                particles.position,
                contain_all=False,
                cylinder_radius=cylinder.CYLINDER_RADIUS,
                cutoff=cylinder.WALL_CUTOFF,
            )
            shell_charges = cylinder.NEIGHBORS - neighbor_counts[frame_idx, shell_mask]
            box_length_x = float(frame.configuration.box[0])

            plus_x, plus_theta = _center_of_mass_or_nan(
                dynamic_values[shell_charges == 1],
                box_length_x=box_length_x,
            )
            minus_x, minus_theta = _center_of_mass_or_nan(
                dynamic_values[shell_charges == -1],
                box_length_x=box_length_x,
            )

            steps.append(int(frame.configuration.step))
            plus_x_centers.append(plus_x)
            plus_theta_centers.append(plus_theta)
            minus_x_centers.append(minus_x)
            minus_theta_centers.append(minus_theta)

    return (
        np.asarray(steps, dtype=np.int64),
        np.asarray(plus_x_centers, dtype=np.float64),
        np.asarray(plus_theta_centers, dtype=np.float64),
        np.asarray(minus_x_centers, dtype=np.float64),
        np.asarray(minus_theta_centers, dtype=np.float64),
    )


def plot_disclination_center_of_mass_series(
    input_gsd: str | Path = cylinder.IN_GSD,
    neighbor_count_txt: str | Path = cylinder.NEIGHBOR_COUNT_TXT,
    filename: str | Path | None = cylinder.DISCLINATION_COM_PLOT,
) -> None:
    (
        steps,
        plus_x_centers,
        plus_theta_centers,
        minus_x_centers,
        minus_theta_centers,
    ) = disclination_center_of_mass_series(input_gsd, neighbor_count_txt)

    fig, x_axis = plt.subplots(figsize=(12, 6))
    plus_x_line = x_axis.plot(
        steps,
        plus_x_centers,
        color="#1f77b4",
        linewidth=1.8,
        alpha=0.9,
        label="+1 x COM",
    )
    minus_x_line = x_axis.plot(
        steps,
        minus_x_centers,
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
        steps,
        plus_theta_centers,
        color="#ff7f0e",
        linewidth=1.8,
        alpha=0.9,
        label="+1 theta COM",
    )
    minus_theta_line = theta_axis.plot(
        steps,
        minus_theta_centers,
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


def main() -> None:
    write_dynamic_values_gsd(cylinder.IN_GSD, cylinder.DYNAMIC_VALUES_GSD)
    plot_center_of_mass_series(cylinder.IN_GSD, cylinder.COM_PLOT)
    plot_disclination_center_of_mass_series(
        cylinder.IN_GSD,
        cylinder.NEIGHBOR_COUNT_TXT,
        cylinder.DISCLINATION_COM_PLOT,
    )
    print(f"Wrote OVITO dynamic values file to {cylinder.DYNAMIC_VALUES_GSD}.")
    print(f"Wrote center-of-mass plot to {cylinder.COM_PLOT}.")
    print(f"Wrote disclination center-of-mass plot to {cylinder.DISCLINATION_COM_PLOT}.")
    print("OVITO position.x stores x")
    print("OVITO position.y stores theta")
    print("OVITO position.z stores r")
    print("OVITO velocity duplicates position")


if __name__ == "__main__":
    main()
