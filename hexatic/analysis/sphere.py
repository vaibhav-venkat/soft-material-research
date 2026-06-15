from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path

import gsd.hoomd as _gsd_hoomd
import numpy as _np
import numpy.typing as _npt

from .base import (
    SurfaceHexaticCalculator,
    _hexatic_from_neighbors,
    _validate_center,
    _validate_positions,
    nearest_neighbors,
)
from .types import (
    FloatArray,
    HexaticFrame,
    HexaticTrajectory,
    IntArray,
    NeighborCountTrajectory,
)


def sphere_normals(
    positions: _np.ndarray,
    center: _np.ndarray | None = None,
) -> FloatArray:
    positions = _validate_positions(positions)
    radial_vectors = positions - _validate_center(center)
    radii = _np.linalg.norm(radial_vectors, axis=1)
    assert not _np.any(radii <= 0.0)
    return radial_vectors / radii[:, _np.newaxis]


def cavity_shell_mask(
    positions: _np.ndarray,
    cavity_radius: float,
    shell_delta: float,
    center: _np.ndarray | None = None,
) -> _npt.NDArray[_np.bool_]:
    positions = _validate_positions(positions)
    assert cavity_radius > 0.0
    assert shell_delta > 0.0
    radii = _np.linalg.norm(positions - _validate_center(center), axis=1)
    return radii > cavity_radius - shell_delta


@_dataclass(init=False)
class SphereHexaticCalculator(SurfaceHexaticCalculator):
    def __init__(
        self,
        cavity_radius: float,
        shell_delta: float,
        n_neighbors: int = 6,
        center: _np.ndarray | None = None,
    ) -> None:
        super().__init__(cavity_radius, shell_delta, n_neighbors, center)
        self.cavity_radius = self.surface_radius

    def shell_mask(self, positions: _np.ndarray) -> _npt.NDArray[_np.bool_]:
        return cavity_shell_mask(
            positions,
            cavity_radius=self.cavity_radius,
            shell_delta=self.shell_delta,
            center=self.center,
        )

    def surface_normals(self, positions: _np.ndarray) -> FloatArray:
        return sphere_normals(positions, center=self.center)


def compute_hexatic_order_frame(
    positions: _np.ndarray,
    n_neighbors: int = 6,
    center: _np.ndarray | None = None,
    cavity_radius: float | None = None,
    shell_delta: float | None = None,
) -> HexaticFrame:
    positions = _validate_positions(positions)
    if cavity_radius is not None or shell_delta is not None:
        assert cavity_radius is not None and shell_delta is not None
        return compute_hexatic_order_frame_near_cavity(
            positions,
            cavity_radius=cavity_radius,
            shell_delta=shell_delta,
            n_neighbors=n_neighbors,
            center=center,
        )
    neighbors = nearest_neighbors(positions, n_neighbors)
    return HexaticFrame(
        psi=_hexatic_from_neighbors(
            positions,
            neighbors,
            sphere_normals(positions, center),
        ),
        neighbors=neighbors,
    )


def compute_hexatic_order_frame_near_cavity(
    positions: _np.ndarray,
    cavity_radius: float,
    shell_delta: float,
    n_neighbors: int = 6,
    center: _np.ndarray | None = None,
) -> HexaticFrame:
    return SphereHexaticCalculator(
        cavity_radius=cavity_radius,
        shell_delta=shell_delta,
        n_neighbors=n_neighbors,
        center=center,
    ).compute_hexatic_order_frame(positions)


def compute_neighbor_counts_frame_near_cavity(
    positions: _np.ndarray,
    cavity_radius: float,
    shell_delta: float,
    neighbor_radius: float,
    center: _np.ndarray | None = None,
) -> IntArray:
    return SphereHexaticCalculator(
        cavity_radius=cavity_radius,
        shell_delta=shell_delta,
        center=center,
    ).compute_neighbor_counts_frame(positions, neighbor_radius=neighbor_radius)


def compute_hexatic_order_trajectory(
    filename: str | _Path,
    n_neighbors: int = 6,
    center: _np.ndarray | None = None,
    cavity_radius: float | None = None,
    shell_delta: float | None = None,
) -> HexaticTrajectory:
    if cavity_radius is not None or shell_delta is not None:
        assert cavity_radius is not None and shell_delta is not None
        return SphereHexaticCalculator(
            cavity_radius=cavity_radius,
            shell_delta=shell_delta,
            n_neighbors=n_neighbors,
            center=center,
        ).compute_hexatic_order_trajectory(filename)

    steps: list[int] = []
    psi_frames: list[_np.ndarray] = []
    n_particles: int | None = None
    with _gsd_hoomd.open(name=str(filename), mode="r") as trajectory:
        for frame in trajectory:
            positions = _validate_positions(frame.particles.position)
            n_particles = positions.shape[0] if n_particles is None else n_particles
            assert positions.shape[0] == n_particles
            psi, _ = compute_hexatic_order_frame(
                positions,
                n_neighbors=n_neighbors,
                center=center,
            )
            psi_frames.append(psi)
            steps.append(int(frame.configuration.step))
    return HexaticTrajectory(
        steps=_np.asarray(steps, dtype=_np.int64),
        psi=_np.vstack(psi_frames),
    )


def compute_neighbor_counts_trajectory(
    filename: str | _Path,
    neighbor_radius: float,
    cavity_radius: float,
    shell_delta: float,
    center: _np.ndarray | None = None,
) -> NeighborCountTrajectory:
    return SphereHexaticCalculator(
        cavity_radius=cavity_radius,
        shell_delta=shell_delta,
        center=center,
    ).compute_neighbor_counts_trajectory(filename, neighbor_radius=neighbor_radius)
