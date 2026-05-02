from typing import cast
from pathlib import Path

import gsd.hoomd
import numpy as np
import numpy.typing as npt

from .analyzers import get_analyzer, interpolate_periodic


def calculate(
    filename: str | Path,
    sigma_vertex: float,
    equilibrium_frames: int,
    vertex_type: int,
    mode: str = "vesicle",
) -> tuple[npt.NDArray[np.float64] | None, np.ndarray | None]:
    analyzer = get_analyzer(mode, sigma_vertex)
    file: gsd.hoomd.HOOMDTrajectory = gsd.hoomd.open(name=str(filename), mode="r")

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
