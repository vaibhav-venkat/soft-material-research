from abc import ABC as _ABC, abstractmethod as _abstractmethod
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path

import gsd.hoomd as _gsd_hoomd
import numpy as _np
import numpy.typing as _npt

from .types import (
    ComplexArray,
    FloatArray,
    HexaticFrame,
    HexaticTrajectory,
    IntArray,
    NeighborCountTrajectory,
)


def _validate_positions(positions: _np.ndarray) -> FloatArray:
    arr = _np.asarray(positions, dtype=_np.float64)
    assert arr.ndim == 2 and arr.shape[1] == 3
    return arr


def _validate_center(center: _np.ndarray | None) -> FloatArray:
    if center is None:
        return _np.zeros(3, dtype=_np.float64)
    arr = _np.asarray(center, dtype=_np.float64)
    assert arr.shape == (3,)
    return arr


def _minimum_image_x(
    vectors: _np.ndarray,
    box_length_x: float | None = None,
) -> FloatArray:
    arr = _np.asarray(vectors, dtype=_np.float64)
    if box_length_x is None:
        return arr
    assert box_length_x > 0.0
    wrapped = arr.copy()
    wrapped[..., 0] -= box_length_x * _np.round(wrapped[..., 0] / box_length_x)
    return wrapped


def _pairwise_distances_sq(
    positions: _np.ndarray,
    box_length_x: float | None = None,
) -> FloatArray:
    positions = _validate_positions(positions)
    dx = positions[:, 0, _np.newaxis] - positions[_np.newaxis, :, 0]
    if box_length_x is not None:
        assert box_length_x > 0.0
        dx -= box_length_x * _np.round(dx / box_length_x)

    dy = positions[:, 1, _np.newaxis] - positions[_np.newaxis, :, 1]
    dz = positions[:, 2, _np.newaxis] - positions[_np.newaxis, :, 2]
    distances_sq = dx * dx + dy * dy + dz * dz
    _np.maximum(distances_sq, 0.0, out=distances_sq)
    _np.fill_diagonal(distances_sq, _np.inf)
    return distances_sq


def local_tangent_basis(normal: _np.ndarray) -> tuple[FloatArray, FloatArray]:
    normal_arr = _np.asarray(normal, dtype=_np.float64)
    assert normal_arr.shape == (3,)
    normal_norm = _np.linalg.norm(normal_arr)
    assert normal_norm > 0.0

    unit_normal = normal_arr / normal_norm
    reference = _np.array([0.0, 0.0, 1.0])
    if abs(float(_np.dot(unit_normal, reference))) > 0.9:
        reference = _np.array([0.0, 1.0, 0.0])

    e1 = _np.cross(reference, unit_normal)
    e1 /= _np.linalg.norm(e1)
    e2 = _np.cross(unit_normal, e1)
    e2 /= _np.linalg.norm(e2)
    return e1, e2


def _hexatic_from_neighbors(
    positions: _np.ndarray,
    neighbors: _np.ndarray,
    normals: _np.ndarray,
    box_length_x: float | None = None,
    tangent_basis=local_tangent_basis,
) -> ComplexArray:
    positions = _validate_positions(positions)
    neighbors = _np.asarray(neighbors, dtype=_np.int64)
    normals = _validate_positions(normals)
    assert neighbors.ndim == 2 and neighbors.shape[0] == positions.shape[0]
    assert normals.shape == positions.shape

    psi = _np.empty(positions.shape[0], dtype=_np.complex128)
    for particle_idx, neighbor_ids in enumerate(neighbors):
        normal = normals[particle_idx]
        e1, e2 = tangent_basis(normal)
        bonds = _minimum_image_x(
            positions[neighbor_ids] - positions[particle_idx],
            box_length_x=box_length_x,
        )
        tangent_bonds = bonds - (bonds @ normal)[:, _np.newaxis] * normal
        theta = _np.arctan2(tangent_bonds @ e2, tangent_bonds @ e1)
        psi[particle_idx] = _np.mean(_np.exp(6j * theta))
    return psi


def nearest_neighbors(
    positions: _np.ndarray,
    n_neighbors: int = 6,
    box_length_x: float | None = None,
) -> IntArray:
    positions = _validate_positions(positions)
    assert positions.shape[0] > n_neighbors

    distances_sq = _pairwise_distances_sq(positions, box_length_x=box_length_x)
    neighbor_indices = _np.argpartition(
        distances_sq,
        kth=n_neighbors - 1,
        axis=1,
    )[:, :n_neighbors]
    neighbor_distances = _np.take_along_axis(distances_sq, neighbor_indices, axis=1)
    distance_order = _np.argsort(neighbor_distances, axis=1)
    return _np.take_along_axis(
        neighbor_indices,
        distance_order,
        axis=1,
    ).astype(_np.int64, copy=False)


def count_neighbors_within_radius(
    positions: _np.ndarray,
    radius: float,
    box_length_x: float | None = None,
) -> IntArray:
    positions = _validate_positions(positions)
    assert radius > 0.0
    counts = _np.count_nonzero(
        _pairwise_distances_sq(positions, box_length_x=box_length_x) <= radius**2,
        axis=1,
    )
    return counts.astype(_np.int64, copy=False)


@_dataclass
class SurfaceHexaticCalculator(_ABC):
    surface_radius: float
    shell_delta: float
    n_neighbors: int = 6
    center: _np.ndarray | None = None
    tangent_basis = staticmethod(local_tangent_basis)

    def __post_init__(self) -> None:
        assert self.surface_radius > 0.0
        assert self.shell_delta > 0.0
        assert self.n_neighbors > 0
        self.surface_radius = float(self.surface_radius)
        self.shell_delta = float(self.shell_delta)
        self.n_neighbors = int(self.n_neighbors)
        self.center = _validate_center(self.center)

    @_abstractmethod
    def shell_mask(self, positions: _np.ndarray) -> _npt.NDArray[_np.bool_]:
        raise NotImplementedError

    @_abstractmethod
    def surface_normals(self, positions: _np.ndarray) -> FloatArray:
        raise NotImplementedError

    def frame_box_length_x(self, frame) -> float | None:
        return None

    def compute_hexatic_order_frame(
        self,
        positions: _np.ndarray,
        box_length_x: float | None = None,
    ) -> HexaticFrame:
        positions = _validate_positions(positions)
        shell_indices = _np.flatnonzero(self.shell_mask(positions))
        psi = _np.zeros(positions.shape[0], dtype=_np.complex128)
        neighbors = _np.full(
            (positions.shape[0], self.n_neighbors),
            -1,
            dtype=_np.int64,
        )
        if len(shell_indices) <= self.n_neighbors:
            return HexaticFrame(psi=psi, neighbors=neighbors)

        shell_positions = positions[shell_indices]
        shell_neighbors = nearest_neighbors(
            shell_positions,
            n_neighbors=self.n_neighbors,
            box_length_x=box_length_x,
        )
        psi[shell_indices] = _hexatic_from_neighbors(
            shell_positions,
            shell_neighbors,
            self.surface_normals(shell_positions),
            box_length_x=box_length_x,
            tangent_basis=self.tangent_basis,
        )
        neighbors[shell_indices] = shell_indices[shell_neighbors]
        return HexaticFrame(psi=psi, neighbors=neighbors)

    def compute_neighbor_counts_frame(
        self,
        positions: _np.ndarray,
        neighbor_radius: float,
        box_length_x: float | None = None,
    ) -> IntArray:
        positions = _validate_positions(positions)
        shell_indices = _np.flatnonzero(self.shell_mask(positions))
        counts = _np.zeros(positions.shape[0], dtype=_np.int64)
        if len(shell_indices):
            counts[shell_indices] = count_neighbors_within_radius(
                positions[shell_indices],
                radius=neighbor_radius,
                box_length_x=box_length_x,
            )
        return counts

    def compute_hexatic_order_trajectory(
        self,
        filename: str | _Path,
    ) -> HexaticTrajectory:
        steps: list[int] = []
        psi_frames: list[ComplexArray] = []
        n_particles: int | None = None
        with _gsd_hoomd.open(name=str(filename), mode="r") as trajectory:
            for frame in trajectory:
                positions = _validate_positions(frame.particles.position)
                n_particles = positions.shape[0] if n_particles is None else n_particles
                assert positions.shape[0] == n_particles
                psi, _ = self.compute_hexatic_order_frame(
                    positions,
                    box_length_x=self.frame_box_length_x(frame),
                )
                psi_frames.append(psi)
                steps.append(int(frame.configuration.step))
        return HexaticTrajectory(
            steps=_np.asarray(steps, dtype=_np.int64),
            psi=_np.vstack(psi_frames),
        )

    def compute_neighbor_counts_trajectory(
        self,
        filename: str | _Path,
        neighbor_radius: float,
    ) -> NeighborCountTrajectory:
        steps: list[int] = []
        count_frames: list[IntArray] = []
        n_particles: int | None = None
        with _gsd_hoomd.open(name=str(filename), mode="r") as trajectory:
            for frame in trajectory:
                positions = _validate_positions(frame.particles.position)
                n_particles = positions.shape[0] if n_particles is None else n_particles
                assert positions.shape[0] == n_particles
                count_frames.append(
                    self.compute_neighbor_counts_frame(
                        positions,
                        neighbor_radius=neighbor_radius,
                        box_length_x=self.frame_box_length_x(frame),
                    )
                )
                steps.append(int(frame.configuration.step))
        return NeighborCountTrajectory(
            steps=_np.asarray(steps, dtype=_np.int64),
            counts=_np.vstack(count_frames),
        )
