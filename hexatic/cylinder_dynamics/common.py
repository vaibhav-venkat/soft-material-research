from dataclasses import dataclass
from typing import Iterator

from matplotlib.ticker import FuncFormatter
import numpy as np

from hexatic import analysis as hx
from hexatic.constants import cylinder

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
class XCOMVelocitySeries:
    steps: np.ndarray
    x_velocities: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.steps
        yield self.x_velocities

@dataclass(frozen = True)
class ThetaCOMVelocitySeries:
    steps: np.ndarray
    theta_velocities: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.steps
        yield self.theta_velocities


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


def _minimum_image_delta(delta: np.ndarray, box_length: float) -> np.ndarray:
    return delta - box_length * np.round(delta / box_length)


def _theta_from_positions(positions: np.ndarray) -> np.ndarray:
    return np.mod(np.arctan2(positions[:, 1], positions[:, 2]), 2.0 * np.pi)


def _x_bin_indices(x_positions: np.ndarray, box_length_x: float, n_bins: int) -> np.ndarray:
    wrapped = np.mod(x_positions + 0.5 * box_length_x, box_length_x)
    indices = np.floor(wrapped / box_length_x * n_bins).astype(np.int64)
    return np.clip(indices, 0, n_bins - 1)


def _theta_bin_indices(theta: np.ndarray, n_bins: int) -> np.ndarray:
    wrapped = np.mod(theta, 2.0 * np.pi)
    indices = np.floor(wrapped / (2.0 * np.pi) * n_bins).astype(np.int64)
    return np.clip(indices, 0, n_bins - 1)


def _particle_masses(particles, n_particles: int) -> np.ndarray:
    masses = getattr(particles, "mass", None)
    if masses is None:
        return np.ones(n_particles, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if masses.shape != (n_particles,):
        return np.ones(n_particles, dtype=np.float64)
    return masses


def _unwrapped_x_positions(particles, box_length_x: float) -> np.ndarray:
    positions = np.asarray(particles.position, dtype=np.float64)
    x_positions = positions[:, 0].copy()
    images = getattr(particles, "image", None)
    if images is not None:
        images = np.asarray(images)
        if images.shape == positions.shape:
            x_positions += images[:, 0] * box_length_x
    return x_positions
