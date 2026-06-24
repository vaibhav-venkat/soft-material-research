from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import gsd.hoomd
import numpy as np

try:
    from hexatic.analysis.base import _minimum_image_x, _validate_positions
except ImportError:
    from analysis.base import _minimum_image_x, _validate_positions


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

    bonds = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    bonds = _minimum_image_x(bonds, box_length_x=box_length_x)
    bond_lengths = np.linalg.norm(bonds, axis=2)
    valid_bonds = (
        np.isfinite(bond_lengths)
        & (bond_lengths > 0.0)
        & (bond_lengths <= neighborhood_radius)
    )
    translation_chirality = np.divide(
        bonds[:, :, 0],
        bond_lengths,
        out=np.zeros_like(bond_lengths, dtype=np.float64),
        where=valid_bonds,
    )
    return np.sum(translation_chirality, axis=1)


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
