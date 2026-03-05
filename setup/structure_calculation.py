from abc import ABC, abstractmethod
from typing import cast

import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

FILENAME_ACTIVE_VESICLE = "perimeter-conserved-2D-vesicle_CPU_harmonic_bonds-active.gsd"
FILENAME_PASSIVE_VESICLE = (
    "perimeter-conserved-2D-vesicle_CPU_harmonic_bonds-passive.gsd"
)
FILENAME_ACTIVE_MEMBRANE = (
    "perimeter-conserved-2D-membrane_CPU_harmonic_bonds-active.gsd"
)
FILENAME_PASSIVE_MEMBRANE = (
    "perimeter-conserved-2D-membrane_CPU_harmonic_bonds-passive.gsd"
)
SIGMA_VERTEX = 0.05
EQUILIBRIUM_FRAMES = 10
VERTEX_TYPE = 0


def interpolate_periodic(
    values: npt.NDArray[np.float64],
    coords: npt.NDArray[np.float64],
    period: float,
    n_points: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if n_points is None:
        n_points = len(values)

    start = coords[0]
    new_grid = np.linspace(start, start + period, n_points, endpoint=False)

    extended_coords = np.append(coords, coords[0] + period)
    extended_values = np.append(values, values[0])

    interpolated_values = np.interp(new_grid, extended_coords, extended_values)

    return interpolated_values, new_grid


class ModeAnalyzer(ABC):
    @abstractmethod
    def extract(
        self,
        positions: np.ndarray,
        box: np.ndarray,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        pass

    @abstractmethod
    def compute_n_points(
        self, coords_sorted: npt.NDArray[np.float64], period: float
    ) -> int:
        pass

    @abstractmethod
    def compute_k(self, n_points: int, period: float) -> np.ndarray:
        pass


class VesicleAnalyzer(ModeAnalyzer):
    def __init__(self, sigma_vertex: float):
        self.sigma_vertex = sigma_vertex

    def extract(
        self,
        positions: np.ndarray,
        box: np.ndarray,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        x_pos = positions[:, 0]
        y_pos = positions[:, 1]

        centered_x = x_pos - np.mean(x_pos)
        centered_y = y_pos - np.mean(y_pos)

        r = np.sqrt(centered_x**2 + centered_y**2)

        vals_sorted = r
        coords_sorted = np.linspace(0, 2 * np.pi, len(r), endpoint=False)
        period = 2 * np.pi

        return vals_sorted, coords_sorted, period

    def compute_n_points(
        self, coords_sorted: npt.NDArray[np.float64], period: float
    ) -> int:
        return len(coords_sorted)

    def compute_k(self, n_points: int, period: float) -> np.ndarray:
        a = 4 * self.sigma_vertex
        return np.fft.fftfreq(n_points, a)


class MembraneAnalyzer(ModeAnalyzer):
    def extract(
        self,
        positions: np.ndarray,
        box: np.ndarray,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        x_pos = positions[:, 0]
        period = float(box[1])

        vals_sorted = x_pos
        coords_sorted = np.linspace(0, period, len(x_pos), endpoint=False)

        return vals_sorted, coords_sorted, period

    def compute_n_points(
        self, coords_sorted: npt.NDArray[np.float64], period: float
    ) -> int:
        dy = np.diff(coords_sorted)
        positive_dy = dy[dy > 1e-6]
        if len(positive_dy) == 0:
            return len(coords_sorted)
        min_dy = np.min(positive_dy)
        return int(np.ceil(period / min_dy))

    def compute_k(self, n_points: int, period: float) -> np.ndarray:
        spacing = period / n_points
        return np.fft.fftfreq(n_points, spacing)


ANALYZERS: dict[str, type[ModeAnalyzer]] = {
    "vesicle": VesicleAnalyzer,
    "membrane": MembraneAnalyzer,
}


def get_analyzer(mode: str, sigma_vertex: float) -> ModeAnalyzer:
    if mode == "vesicle":
        return VesicleAnalyzer(sigma_vertex)
    if mode == "membrane":
        return MembraneAnalyzer()
    raise ValueError(f"Unknown mode: {mode}. Available: {list(ANALYZERS.keys())}")


def calculate(
    filename: str,
    sigma_vertex: float,
    equilibrium_frames: int,
    vertex_type: int,
    mode: str = "vesicle",
) -> tuple[npt.NDArray[np.float64] | None, np.ndarray | None]:
    analyzer = get_analyzer(mode, sigma_vertex)
    file: gsd.hoomd.HOOMDTrajectory = gsd.hoomd.open(name=filename, mode="r")

    n_grid: int | None = None
    k: np.ndarray | None = None
    total_structure_factor: np.ndarray | None = None
    averaged_frames = 0

    for frame_idx, frame in enumerate(file):
        particles = frame.particles
        box = frame.configuration.box

        assert particles is not None and particles.position is not None

        vertex_indices = np.where(particles.typeid == vertex_type)[0]
        positions: np.ndarray = cast(np.ndarray, particles.position[vertex_indices])

        vals_sorted, coords_sorted, period = analyzer.extract(positions, box)

        if n_grid is None or k is None:
            n_grid = analyzer.compute_n_points(coords_sorted, period)
            k = analyzer.compute_k(n_grid, period)
            total_structure_factor = np.zeros(n_grid)

        vals_interp, _ = interpolate_periodic(
            vals_sorted, coords_sorted, period, n_grid
        )
        vals_interp -= np.mean(vals_interp)

        h_k = np.fft.fft(vals_interp)
        structure_factor = np.abs(h_k) ** 2

        if frame_idx >= equilibrium_frames:
            if total_structure_factor is not None:
                total_structure_factor += structure_factor
                averaged_frames += 1

    if averaged_frames <= 0 or total_structure_factor is None:
        print("No frames averaged")
        return None, k

    avg_structure_factor: npt.NDArray[np.float64] = (
        total_structure_factor / averaged_frames
    )
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
    s_plot = structure_positive[sort_k_indices]

    plt.plot(k_plot, np.log(s_plot), label=label)
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

    positive_k_indices1 = np.where(k1 > 0)
    k_pos1 = k1[positive_k_indices1]
    structure_positive1 = avg_structure_factor1[positive_k_indices1]
    sort_k_indices1 = np.argsort(k_pos1)
    k_plot1 = k_pos1[sort_k_indices1]
    s_plot1 = structure_positive1[sort_k_indices1]
    plt.plot(k_plot1, np.log(s_plot1), label=label1)

    positive_k_indices2 = np.where(k2 > 0)
    k_pos2 = k2[positive_k_indices2]
    structure_positive2 = avg_structure_factor2[positive_k_indices2]
    sort_k_indices2 = np.argsort(k_pos2)
    k_plot2 = k_pos2[sort_k_indices2]
    s_plot2 = structure_positive2[sort_k_indices2]
    plt.plot(k_plot2, np.log(s_plot2), label=label2)

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
    avg_structure_factor: npt.NDArray[np.float64], k: np.ndarray, mode: str = "vesicle"
) -> float:
    if mode == "membrane":
        intermediate_k_indices = np.where((k >= 1) & (k <= 10))
    else:
        intermediate_k_indices = np.where((k >= 0.2) & (k <= 1))
    k_intermediate = k[intermediate_k_indices]
    structure_intermediate = avg_structure_factor[intermediate_k_indices]

    log_structure_factor = np.log(structure_intermediate)
    log_k = np.log(k_intermediate)
    slope, _ = np.polyfit(log_k, log_structure_factor, 1)

    return slope


if __name__ == "__main__":
    MODE = "vesicle"

    if MODE == "vesicle":
        fn_active = FILENAME_ACTIVE_VESICLE
        fn_passive = FILENAME_PASSIVE_VESICLE
        sigma_vertex = 0.05
    elif MODE == "membrane":
        fn_active = FILENAME_ACTIVE_MEMBRANE
        fn_passive = FILENAME_PASSIVE_MEMBRANE
        sigma_vertex = 0.05
    else:
        raise ValueError(f"Unknown mode: {MODE}")

    s1, k1 = calculate(fn_active, sigma_vertex, EQUILIBRIUM_FRAMES, VERTEX_TYPE, MODE)
    s2, k2 = calculate(fn_passive, sigma_vertex, EQUILIBRIUM_FRAMES, VERTEX_TYPE, MODE)

    if s1 is not None and k1 is not None and s2 is not None and k2 is not None:
        plot_two(s1, k1, "Active", s2, k2, "Passive", filename="comparison.png")

        scaling_factor_s1 = extract_scaling_factor(s1, k1, MODE)
        print(f"Scaling factor for s1 (Active): {scaling_factor_s1}")
        scaling_factor_s2 = extract_scaling_factor(s2, k2, MODE)
        print(f"Scaling factor for s2 (Passive): {scaling_factor_s2}")
    else:
        print("Error during calculation")
