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
            x_center, theta_center = hx.get_center_of_mass_x_theta(
                dynamic_values
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


def main() -> None:
    write_dynamic_values_gsd(cylinder.IN_GSD, cylinder.DYNAMIC_VALUES_GSD)
    plot_center_of_mass_series(cylinder.IN_GSD, cylinder.COM_PLOT)
    print(f"Wrote OVITO dynamic values file to {cylinder.DYNAMIC_VALUES_GSD}.")
    print(f"Wrote center-of-mass plot to {cylinder.COM_PLOT}.")
    print("OVITO position.x stores x")
    print("OVITO position.y stores theta")
    print("OVITO position.z stores r")
    print("OVITO velocity duplicates position")


if __name__ == "__main__":
    main()
