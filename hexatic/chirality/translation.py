from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import gsd.hoomd
import numpy as np
from scipy.spatial import cKDTree

try:
    from hexatic.analysis.base import _validate_positions
except ImportError:
    from analysis.base import _validate_positions


@dataclass(frozen=True)
class TranslationChiralityTrajectory:
    steps: np.ndarray
    chirality: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.steps
        yield self.chirality


def compute_translation_chirality_frame(
    positions: np.ndarray,
    neighborhood_radius: float,
    box_length_x: float | None = None,
) -> np.ndarray:
    positions = _validate_positions(positions)
    assert neighborhood_radius > 0.0

    translation_chirality = np.zeros(positions.shape[0], dtype=np.float64)
    for particle_idx, _, bond, bond_length in iter_neighbor_bonds(
        positions,
        neighborhood_radius=neighborhood_radius,
        box_length_x=box_length_x,
    ):
        translation_chirality[particle_idx] += bond[0] / bond_length
    return translation_chirality


def _x_image_shifts(
    neighborhood_radius: float,
    box_length_x: float | None,
) -> np.ndarray:
    if box_length_x is None:
        return np.asarray([0.0], dtype=np.float64)
    assert box_length_x > 0.0
    n_images = max(1, int(np.ceil(neighborhood_radius / box_length_x)))
    return box_length_x * np.arange(-n_images, n_images + 1, dtype=np.float64)


def _periodic_x_search_points(
    positions: np.ndarray,
    neighborhood_radius: float,
    box_length_x: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    shifts = _x_image_shifts(neighborhood_radius, box_length_x)
    search_points = np.repeat(positions, len(shifts), axis=0)
    search_points[:, 0] += np.tile(shifts, positions.shape[0])
    source_indices = np.repeat(np.arange(positions.shape[0], dtype=np.int64), len(shifts))
    return search_points, source_indices


def iter_neighbor_bonds(
    positions: np.ndarray,
    neighborhood_radius: float,
    box_length_x: float | None = None,
):
    """Yield directed neighbor bonds within a radius with periodic x wrapping."""
    positions = _validate_positions(positions)
    assert neighborhood_radius > 0.0
    search_points, source_indices = _periodic_x_search_points(
        positions,
        neighborhood_radius,
        box_length_x,
    )
    tree = cKDTree(search_points)
    neighbor_hits = tree.query_ball_point(positions, neighborhood_radius)
    radius_sq = neighborhood_radius * neighborhood_radius

    for particle_idx, hits in enumerate(neighbor_hits):
        best_by_source: dict[int, tuple[float, np.ndarray]] = {}
        for hit in hits:
            source_idx = int(source_indices[hit])
            if source_idx == particle_idx:
                continue
            bond = search_points[hit] - positions[particle_idx]
            distance_sq = float(np.dot(bond, bond))
            if distance_sq <= 0.0 or distance_sq > radius_sq:
                continue
            previous = best_by_source.get(source_idx)
            if previous is None or distance_sq < previous[0]:
                best_by_source[source_idx] = (distance_sq, bond)

        for source_idx, (distance_sq, bond) in best_by_source.items():
            yield particle_idx, source_idx, bond, float(np.sqrt(distance_sq))


def compute_translation_chirality_trajectory(
    input_gsd: str | Path,
    neighborhood_radius: float,
) -> TranslationChiralityTrajectory:
    assert neighborhood_radius > 0.0

    steps: list[int] = []
    chirality_frames: list[np.ndarray] = []
    n_particles: int | None = None
    with gsd.hoomd.open(name=str(input_gsd), mode="r") as trajectory:
        for frame in trajectory:
            positions = _validate_positions(frame.particles.position)
            n_particles = positions.shape[0] if n_particles is None else n_particles
            chirality_frames.append(
                compute_translation_chirality_frame(
                    positions,
                    neighborhood_radius=neighborhood_radius,
                    box_length_x=float(frame.configuration.box[0]),
                )
            )
            steps.append(int(frame.configuration.step))

    return TranslationChiralityTrajectory(
        steps=np.asarray(steps, dtype=np.int64),
        chirality=np.vstack(chirality_frames),
    )
