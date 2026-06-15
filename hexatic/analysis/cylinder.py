from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path

import gsd.hoomd as _gsd_hoomd
import numpy as _np
import numpy.typing as _npt

from .base import (
    SurfaceHexaticCalculator,
    _minimum_image_x,
    _validate_center,
    _validate_positions,
)
from .types import (
    CenterOfMass,
    DynamicValues,
    FloatArray,
    HexaticFrame,
    HexaticTrajectory,
    IntArray,
    NeighborCountTrajectory,
)


def cylinder_radial_distances(
    positions: _np.ndarray,
    center: _np.ndarray | None = None,
) -> FloatArray:
    positions = _validate_positions(positions)
    yz_vectors = positions[:, 1:3] - _validate_center(center)[1:3]
    return _np.linalg.norm(yz_vectors, axis=1)


def cylinder_normals(
    positions: _np.ndarray,
    center: _np.ndarray | None = None,
) -> FloatArray:
    positions = _validate_positions(positions)
    yz_vectors = positions[:, 1:3] - _validate_center(center)[1:3]
    radii = _np.linalg.norm(yz_vectors, axis=1)
    assert _np.all(radii > 0.0)
    normals = _np.zeros_like(positions, dtype=_np.float64)
    normals[:, 1:3] = yz_vectors / radii[:, _np.newaxis]
    return normals


def local_cylinder_tangent_basis(normal: _np.ndarray) -> tuple[FloatArray, FloatArray]:
    normal_arr = _np.asarray(normal, dtype=_np.float64)
    assert normal_arr.shape == (3,)
    unit_normal = normal_arr / _np.linalg.norm(normal_arr)
    axis = _np.array([1.0, 0.0, 0.0])
    assert abs(float(_np.dot(unit_normal, axis))) < 1e-12
    e2 = _np.cross(unit_normal, axis)
    e2 /= _np.linalg.norm(e2)
    return axis, e2


def cylinder_shell_mask(
    positions: _np.ndarray,
    cylinder_radius: float,
    shell_delta: float,
    center: _np.ndarray | None = None,
) -> _npt.NDArray[_np.bool_]:
    assert cylinder_radius > 0.0
    assert shell_delta > 0.0
    return (
        cylinder_radial_distances(positions, center=center)
        > cylinder_radius - shell_delta
    )


@_dataclass(init=False)
class CylinderHexaticCalculator(SurfaceHexaticCalculator):
    tangent_basis = staticmethod(local_cylinder_tangent_basis)

    def __init__(
        self,
        cylinder_radius: float,
        shell_delta: float,
        n_neighbors: int = 6,
        center: _np.ndarray | None = None,
    ) -> None:
        super().__init__(cylinder_radius, shell_delta, n_neighbors, center)
        self.cylinder_radius = self.surface_radius

    def shell_mask(self, positions: _np.ndarray) -> _npt.NDArray[_np.bool_]:
        return cylinder_shell_mask(
            positions,
            cylinder_radius=self.cylinder_radius,
            shell_delta=self.shell_delta,
            center=self.center,
        )

    def surface_normals(self, positions: _np.ndarray) -> FloatArray:
        return cylinder_normals(positions, center=self.center)

    def frame_box_length_x(self, frame) -> float | None:
        return float(frame.configuration.box[0])


def compute_hexatic_order_frame_on_cylinder(
    positions: _np.ndarray,
    cylinder_radius: float,
    shell_delta: float,
    n_neighbors: int = 6,
    center: _np.ndarray | None = None,
    box_length_x: float | None = None,
) -> HexaticFrame:
    return CylinderHexaticCalculator(
        cylinder_radius=cylinder_radius,
        shell_delta=shell_delta,
        n_neighbors=n_neighbors,
        center=center,
    ).compute_hexatic_order_frame(positions, box_length_x=box_length_x)


def compute_neighbor_counts_frame_on_cylinder(
    positions: _np.ndarray,
    cylinder_radius: float,
    shell_delta: float,
    neighbor_radius: float,
    center: _np.ndarray | None = None,
    box_length_x: float | None = None,
) -> IntArray:
    return CylinderHexaticCalculator(
        cylinder_radius=cylinder_radius,
        shell_delta=shell_delta,
        center=center,
    ).compute_neighbor_counts_frame(
        positions,
        neighbor_radius=neighbor_radius,
        box_length_x=box_length_x,
    )


def compute_hexatic_order_cylinder_trajectory(
    filename: str | _Path,
    cylinder_radius: float,
    shell_delta: float,
    n_neighbors: int = 6,
    center: _np.ndarray | None = None,
) -> HexaticTrajectory:
    return CylinderHexaticCalculator(
        cylinder_radius=cylinder_radius,
        shell_delta=shell_delta,
        n_neighbors=n_neighbors,
        center=center,
    ).compute_hexatic_order_trajectory(filename)


def compute_neighbor_counts_cylinder_trajectory(
    filename: str | _Path,
    neighbor_radius: float,
    cylinder_radius: float,
    shell_delta: float,
    center: _np.ndarray | None = None,
) -> NeighborCountTrajectory:
    return CylinderHexaticCalculator(
        cylinder_radius=cylinder_radius,
        shell_delta=shell_delta,
        center=center,
    ).compute_neighbor_counts_trajectory(filename, neighbor_radius=neighbor_radius)


def identify_dislocation_particles_frame(
    positions: _np.ndarray,
    disclination_charges: _np.ndarray,
    pair_distance: float,
    box_length_x: float | None = None,
) -> IntArray:
    positions = _validate_positions(positions)
    charges = _np.asarray(disclination_charges, dtype=_np.int64)
    assert charges.shape == (positions.shape[0],)
    assert pair_distance > 0.0

    plus_indices = _np.flatnonzero(charges == 1)
    minus_indices = _np.flatnonzero(charges == -1)
    dislocation_particles = _np.zeros(positions.shape[0], dtype=_np.int64)
    if len(plus_indices) == 0 or len(minus_indices) == 0:
        return dislocation_particles

    vectors = positions[minus_indices][_np.newaxis, :, :] - positions[
        plus_indices
    ][:, _np.newaxis, :]
    distances_sq = _np.sum(
        _minimum_image_x(vectors, box_length_x=box_length_x) ** 2,
        axis=2,
    )
    paired_plus, paired_minus = _np.nonzero(distances_sq <= pair_distance**2)
    dislocation_particles[plus_indices[paired_plus]] = 1
    dislocation_particles[minus_indices[paired_minus]] = 1
    return dislocation_particles


def identify_dislocation_particles_trajectory(
    input_gsd: str | _Path,
    disclination_charges: _np.ndarray,
    pair_distance: float,
) -> IntArray:
    charges = _np.asarray(disclination_charges, dtype=_np.int64)
    assert charges.ndim == 2
    assert pair_distance > 0.0

    dislocation_frames: list[IntArray] = []
    with _gsd_hoomd.open(name=str(input_gsd), mode="r") as trajectory:
        assert len(trajectory) == charges.shape[0]
        for frame_idx, frame in enumerate(trajectory):
            positions = _validate_positions(frame.particles.position)
            assert positions.shape[0] == charges.shape[1]
            dislocation_frames.append(
                identify_dislocation_particles_frame(
                    positions,
                    charges[frame_idx],
                    pair_distance=pair_distance,
                    box_length_x=float(frame.configuration.box[0]),
                )
            )
    return _np.vstack(dislocation_frames)


def get_new_coords(positions: _np.ndarray) -> FloatArray:
    positions = _validate_positions(positions)
    radii = _np.sqrt(positions[:, 1] ** 2 + positions[:, 2] ** 2)
    theta = _np.mod(_np.arctan2(positions[:, 1], positions[:, 2]), 2 * _np.pi)
    return _np.column_stack((positions[:, 0], theta, radii))


def get_dynamic_values(
    positions: _np.ndarray,
    contain_all: bool,
    cylinder_radius: float,
    cutoff: float,
) -> DynamicValues:
    positions = _validate_positions(positions)
    assert cylinder_radius > 0.0
    assert cutoff > 0.0

    coords = get_new_coords(positions)
    radii = coords[:, 2]
    shell_mask = _np.ones_like(radii, dtype=_np.bool_)
    if not contain_all:
        shell_mask = (radii > cylinder_radius - cutoff) & (radii < cylinder_radius)
    return DynamicValues(coords=coords[shell_mask], shell_mask=shell_mask)


def get_center_of_mass_x_theta(
    coords: _np.ndarray,
    circular: bool = True,
    periodic_x: bool = False,
    box_length_x: float | None = None,
) -> CenterOfMass:
    coords = _np.asarray(coords, dtype=_np.float64)
    assert coords.ndim == 2 and coords.shape[1] >= 2

    if periodic_x:
        assert box_length_x is not None and box_length_x > 0.0
        x_angle = 2.0 * _np.pi * coords[:, 0] / box_length_x
        mean_x_angle = _np.arctan2(
            _np.mean(_np.sin(x_angle)),
            _np.mean(_np.cos(x_angle)),
        )
        x_center = float(mean_x_angle * box_length_x / (2.0 * _np.pi))
    else:
        x_center = float(_np.mean(coords[:, 0]))

    theta = coords[:, 1]
    if circular:
        theta_center = float(
            _np.mod(
                _np.arctan2(_np.mean(_np.sin(theta)), _np.mean(_np.cos(theta))),
                2 * _np.pi,
            )
        )
    else:
        theta_center = float(_np.mean(theta))
    return CenterOfMass(x=x_center, theta=theta_center)
