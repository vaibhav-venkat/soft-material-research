from dataclasses import dataclass as _dataclass
from typing import Iterator as _Iterator

import numpy as _np
import numpy.typing as _npt

FloatArray = _npt.NDArray[_np.float64]
ComplexArray = _npt.NDArray[_np.complex128]
IntArray = _npt.NDArray[_np.int64]


@_dataclass(frozen=True)
class HexaticFrame:
    psi: ComplexArray
    neighbors: IntArray

    def __iter__(self) -> _Iterator[_np.ndarray]:
        yield self.psi
        yield self.neighbors


@_dataclass(frozen=True)
class HexaticTrajectory:
    steps: IntArray
    psi: ComplexArray

    def __iter__(self) -> _Iterator[_np.ndarray]:
        yield self.steps
        yield self.psi


@_dataclass(frozen=True)
class NeighborCountTrajectory:
    steps: IntArray
    counts: IntArray

    def __iter__(self) -> _Iterator[_np.ndarray]:
        yield self.steps
        yield self.counts


@_dataclass(frozen=True)
class ProbabilityDistribution:
    bin_centers: FloatArray
    probability_density: FloatArray
    counts: IntArray

    def __iter__(self) -> _Iterator[_np.ndarray]:
        yield self.bin_centers
        yield self.probability_density
        yield self.counts


@_dataclass(frozen=True)
class DynamicValues:
    coords: FloatArray
    shell_mask: _npt.NDArray[_np.bool_]

    def __iter__(self) -> _Iterator[_np.ndarray]:
        yield self.coords
        yield self.shell_mask


@_dataclass(frozen=True)
class CenterOfMass:
    x: float
    theta: float

    def __iter__(self) -> _Iterator[float]:
        yield self.x
        yield self.theta


@_dataclass(frozen=True)
class HexaticVelocityFields:
    component: int = 0
    neighbor_counts: _np.ndarray | None = None
    neighbor_component: int = 1
    disclination_charges: _np.ndarray | None = None
    charge_component: int = 2
    dislocation_particles: _np.ndarray | None = None

    def validate_components(self) -> None:
        assert self.component in (0, 1, 2)
        assert self.neighbor_component in (0, 1, 2)
        assert self.charge_component in (0, 1, 2)
        assert self.component != self.neighbor_component or self.neighbor_counts is None
        assert self.component != self.charge_component or self.disclination_charges is None
        assert self.neighbor_component != self.charge_component or (
            self.neighbor_counts is None or self.disclination_charges is None
        )
