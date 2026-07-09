"""Shared Shenfun cylindrical spectral operators for rho fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, idct


Array = np.ndarray


@dataclass
class CylindricalSpectralOperators:
    """Spectral operators on physical ``(x, theta, r)`` component arrays.

    Cached moments are expressed in the orthonormal ``(x, e_theta, e_r)`` frame.
    Shenfun internally uses the equivalent curvilinear map ``(r, theta, x)``.
    """

    lx: float
    theta_period: float
    r_min: float
    r_max: float
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        assert self.r_min > 0.0 and self.r_max > self.r_min
        nx, ntheta, nr = self.shape
        assert nx > 1 and ntheta > 1 and nr > 1
        import sympy as sp
        from shenfun import FunctionSpace, TensorProductSpace, comm

        # Shenfun maps coordinate symbols by x/y/z position, so retain these names.
        radial, theta, axial = sp.symbols("x y z", real=True, positive=True)
        radial_space = FunctionSpace(nr, "Chebyshev", domain=(self.r_min, self.r_max))
        theta_space = FunctionSpace(ntheta, "F", dtype="D", domain=(0.0, self.theta_period))
        axial_space = FunctionSpace(nx, "F", dtype="d", domain=(0.0, self.lx))
        self.space = TensorProductSpace(
            comm,
            (radial_space, theta_space, axial_space),
            coordinates=((radial, theta, axial), (axial, radial * sp.cos(theta), radial * sp.sin(theta))),
        )
        mesh = self.space.local_mesh(True)
        self.r = np.asarray(mesh[0][:, 0, 0], dtype=np.float64)
        self._r3 = self.r[None, None, :]

    @property
    def spectral_shape(self) -> tuple[int, int, int]:
        return (self.shape[2], self.shape[1], self.shape[0])

    def radial_nodes(self) -> Array:
        return self.r.copy()

    def derivative(self, values: Array, axis: int) -> Array:
        """Return a Shenfun spectral coordinate derivative for one scalar field."""
        from shenfun import Array as ShenArray
        from shenfun import Dx, Function, project

        assert values.shape == self.shape
        # Shenfun transform order is (r, theta, x); repository order is (x, theta, r).
        physical = np.ascontiguousarray(np.transpose(values, (2, 1, 0)), dtype=np.float64)
        coefficients = Function(self.space)
        self.space.forward(physical, coefficients)
        derivative_hat = project(Dx(coefficients, axis, 1), self.space)
        derivative = ShenArray(self.space)
        self.space.backward(derivative_hat, derivative)
        return np.ascontiguousarray(np.transpose(np.asarray(derivative), (2, 1, 0)))

    def gradient_scalar(self, values: Array) -> Array:
        out = np.empty(values.shape + (3,), dtype=np.float64)
        out[..., 0] = self.derivative(values, 2)
        out[..., 1] = self.derivative(values, 1) / self._r3
        out[..., 2] = self.derivative(values, 0)
        return out

    def divergence(self, values: Array) -> Array:
        """Return divergence of a physical-frame flux with direction axis ``-1``."""
        assert values.shape[:3] == self.shape and values.shape[3] == 3
        trailing = values.shape[4:]
        out = np.empty(self.shape + trailing, dtype=np.float64)
        for index in np.ndindex(trailing):
            field = values[(slice(None), slice(None), slice(None), slice(None)) + index]
            out[(slice(None), slice(None), slice(None)) + index] = (
                self.derivative(field[..., 0], 2)
                + self.derivative(field[..., 1], 1) / self._r3
                + self.derivative(self._r3 * field[..., 2], 0) / self._r3
            )
        return out

    def laplacian_scalar(self, values: Array) -> Array:
        dr = self.derivative(values, 0)
        return (
            self.derivative(self.derivative(values, 2), 2)
            + self.derivative(dr, 0)
            + dr / self._r3
            + self.derivative(self.derivative(values, 1), 1) / (self._r3 * self._r3)
        )

    def gradient_vector(self, values: Array) -> Array:
        assert values.shape == self.shape + (3,)
        out = np.empty(self.shape + (3, 3), dtype=np.float64)
        for component in range(3):
            out[..., :, component] = self.gradient_scalar(values[..., component])
        return out

    def laplacian_vector(self, values: Array) -> Array:
        out = np.empty_like(values, dtype=np.float64)
        for component in range(3):
            out[..., component] = self.laplacian_scalar(values[..., component])
        return out

    def gradient_rank2(self, values: Array) -> Array:
        assert values.shape == self.shape + (3, 3)
        out = np.empty(self.shape + (3, 3, 3), dtype=np.float64)
        for row in range(3):
            for col in range(3):
                out[..., :, row, col] = self.gradient_scalar(values[..., row, col])
        return out

    def filter_two_thirds(self, values: Array) -> Array:
        """Apply the agreed 2/3 modal cutoff over every spatial axis."""
        coefficients = np.fft.fftn(dct(values, axis=2, norm="ortho"), axes=(0, 1))
        for axis in (0, 1):
            size = values.shape[axis]
            modes = np.abs(np.fft.fftfreq(size) * size)
            sl = [slice(None)] * coefficients.ndim
            sl[axis] = modes > size / 3.0
            coefficients[tuple(sl)] = 0.0
        radial_cutoff = (2 * values.shape[2]) // 3
        coefficients[:, :, radial_cutoff:, ...] = 0.0
        filtered = idct(np.fft.ifftn(coefficients, axes=(0, 1)).real, axis=2, norm="ortho")
        return np.asarray(filtered, dtype=np.float64)


def barycentric_matrix(source: Array, target: Array) -> Array:
    """Return the degree-N barycentric interpolation matrix from source to target."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    assert source.ndim == target.ndim == 1 and source.size >= 2
    weights = np.ones(source.size, dtype=np.float64)
    for index in range(source.size):
        weights[index] = 1.0 / np.prod(source[index] - np.delete(source, index))
    matrix = np.empty((target.size, source.size), dtype=np.float64)
    for row, point in enumerate(target):
        exact = np.flatnonzero(np.isclose(point, source, rtol=0.0, atol=1.0e-13))
        if exact.size:
            matrix[row] = 0.0
            matrix[row, exact[0]] = 1.0
        else:
            values = weights / (point - source)
            matrix[row] = values / np.sum(values)
    return matrix


def transfer_radial(values: Array, matrix: Array, axis: int) -> Array:
    """Apply a precomputed radial interpolation matrix along one array axis."""
    moved = np.moveaxis(values, axis, -1)
    assert moved.shape[-1] == matrix.shape[1]
    return np.moveaxis(np.einsum("...j,ij->...i", moved, matrix), -1, axis)
