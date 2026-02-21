from typing import cast

import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# Constants
FILENAME_ACTIVE = "perimeter-conserved-2D-vesicle_CPU_harmonic_bonds-active.gsd"
FILENAME_PASSIVE = "perimeter-conserved-2D-vesicle_CPU_harmonic_bonds-passive.gsd"
SIGMA_VERTEX = 0.05
EQUILIBRIUM_FRAMES = 10
VERTEX_TYPE = 0


# Calculate the structure factor
def calculate(
    filename: str, sigma_vertex: float, equilibrium_frames: int, vertex_type: int
) -> tuple[npt.NDArray[np.float64] | None, np.ndarray | None]:
    file: gsd.hoomd.HOOMDTrajectory = gsd.hoomd.open(name=filename, mode="r")
    a = 4 * sigma_vertex

    N_vertex: int | None = None
    k: np.ndarray | None = None
    total_structure_factor: np.ndarray | None = None
    averaged_frames = 0

    frame: gsd.hoomd.Frame
    for frame_idx, frame in enumerate(file):
        particles = frame.particles

        # typing issues
        assert particles is not None and particles.position is not None

        vertex_indices = np.where(particles.typeid == vertex_type)[0]
        positions: np.ndarray = cast(np.ndarray, particles.position[vertex_indices])

        # the first iteration
        if N_vertex is None or k is None:
            N_vertex = cast(int, np.sum(particles.typeid == vertex_type))
            k = np.fft.fftfreq(N_vertex, a)
            total_structure_factor = np.zeros(N_vertex)

        x_pos: np.ndarray = positions[:, 0]
        y_pos: np.ndarray = positions[:, 1]

        centered_x = x_pos - np.mean(x_pos)
        centered_y = y_pos - np.mean(y_pos)

        r = np.sqrt(centered_x**2 + centered_y**2)
        theta = np.arctan2(centered_y, centered_x)

        theta_sort_indices = np.argsort(theta)
        r_sorted = r[theta_sort_indices]
        theta_sorted = theta[theta_sort_indices]

        # Interpolate radial profile onto uniform angular grid
        r_interp, _ = interpolate_radial_profile_on_grid(r_sorted, theta_sorted)

        h_k = np.fft.fft(r_interp)
        structure_factor = np.abs(h_k) ** 2

        if frame_idx >= equilibrium_frames:
            if total_structure_factor is not None:
                total_structure_factor += structure_factor
                averaged_frames += 1

    if averaged_frames <= 0 or total_structure_factor is None or N_vertex is None:
        print("No frames averaged")
        return None, k

    avg_structure_factor: npt.NDArray[np.float64] = np.zeros(N_vertex)
    avg_structure_factor = total_structure_factor / averaged_frames
    return avg_structure_factor, k


def plot(
    avg_structure_factor: npt.NDArray[np.float64],
    k: np.ndarray,
    label: str,
    title: str = "Averaged Structure Factor",
    filename: str | None = None,
):
    positive_k_indices = np.where(k > 0)
    k_pos = k[positive_k_indices]
    structure_positive = avg_structure_factor[positive_k_indices]

    sort_k_indices = np.argsort(k_pos)
    k_plot = k_pos[sort_k_indices]
    plot = structure_positive[sort_k_indices]

    plt.plot(k_plot, np.log(plot), label=label)
    plt.xscale("log")
    plt.xlabel("Wavenumber, k")
    plt.ylabel("log(|h(k)|^2)")
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.legend()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_two(
    avg_structure_factor1: npt.NDArray[np.float64],
    k1: np.ndarray,
    label1: str,
    avg_structure_factor2: npt.NDArray[np.float64],
    k2: np.ndarray,
    label2: str,
    title: str = "Averaged Structure Factor",
    filename: str | None = None,
):
    plt.figure(figsize=(10, 6))

    # make sure all vars are positive
    positive_k_indices1 = np.where(k1 > 0)
    k_pos1 = k1[positive_k_indices1]
    structure_positive1 = avg_structure_factor1[positive_k_indices1]
    sort_k_indices1 = np.argsort(k_pos1)
    k_plot1 = k_pos1[sort_k_indices1]
    plot1 = structure_positive1[sort_k_indices1]

    plt.plot(k_plot1, np.log(plot1), label=label1)  # plot 1

    positive_k_indices2 = np.where(k2 > 0)
    k_pos2 = k2[positive_k_indices2]
    structure_positive2 = avg_structure_factor2[positive_k_indices2]
    sort_k_indices2 = np.argsort(k_pos2)
    k_plot2 = k_pos2[sort_k_indices2]
    plot2 = structure_positive2[sort_k_indices2]

    plt.plot(k_plot2, np.log(plot2), label=label2)  # plot 2

    plt.xscale("log")
    plt.xlabel("Wavenumber, k")
    plt.ylabel("log(|h(k)|^2)")
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.legend()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def extract_scaling_factor(
    avg_structure_factor: npt.NDArray[np.float64], k: np.ndarray
) -> float:
    intermediate_k_indices = np.where((k >= 0.2) & (k <= 1.0))
    k_intermediate = k[intermediate_k_indices]
    structure_intermediate = avg_structure_factor[intermediate_k_indices]

    log_structure_factor = np.log(structure_intermediate)
    log_k = np.log(k_intermediate)
    slope, _ = np.polyfit(log_k, log_structure_factor, 1)

    return slope


def interpolate_radial_profile_on_grid(
    radial_profile: npt.NDArray[np.float64],
    angular_coordinates: npt.NDArray[np.float64],
    n_points: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if n_points is None:
        n_points = len(radial_profile)

    new_angle_grid = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    extended_angular_coordinates = np.append(
        angular_coordinates, angular_coordinates[0] + 2 * np.pi
    )
    extended_radial_profile = np.append(radial_profile, radial_profile[0])

    interpolated_radial_profile = np.interp(
        new_angle_grid, extended_angular_coordinates, extended_radial_profile
    )

    return interpolated_radial_profile, new_angle_grid


# sample code.

s1, k1 = calculate(FILENAME_ACTIVE, SIGMA_VERTEX, EQUILIBRIUM_FRAMES, VERTEX_TYPE)
s2, k2 = calculate(FILENAME_PASSIVE, SIGMA_VERTEX, EQUILIBRIUM_FRAMES, VERTEX_TYPE)

if s1 is not None and k1 is not None and s2 is not None and k2 is not None:
    plot_two(s1, k1, "Active", s2, k2, "Passive", filename="comparison.png")

    scaling_factor_s1 = extract_scaling_factor(s1, k1)
    print(f"Scaling factor for s1 (Active): {scaling_factor_s1}")
    scaling_factor_s2 = extract_scaling_factor(s2, k2)
    print(f"Scaling factor for s2 (Passive): {scaling_factor_s2}")

else:
    print("Error during calculation")
