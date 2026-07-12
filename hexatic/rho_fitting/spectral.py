"""Shared Rust cylindrical spectral operators for rho fitting."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import _rho_fitting_core, _rho_fitting_core_import_error


Array = np.ndarray


@dataclass
class CylindricalSpectralOperators:
    """Rust spectral operators on physical ``(x, theta, r)`` arrays."""

    lx: float
    theta_period: float
    r_min: float
    r_max: float
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        assert self.r_min > 0.0 and self.r_max > self.r_min
        nx, ntheta, nr = self.shape
        assert nx > 1 and ntheta > 1 and nr > 1
        if _rho_fitting_core is None:
            raise ImportError(
                f"rho-fitting Rust core is unavailable: {_rho_fitting_core_import_error}"
            )
        self._core = _rho_fitting_core.CylindricalSpectralOperators(
            float(self.lx),
            float(self.theta_period),
            float(self.r_min),
            float(self.r_max),
            int(nx),
            int(ntheta),
            int(nr),
        )
        self.r = np.asarray(self._core.radial_nodes())

    @property
    def spectral_shape(self) -> tuple[int, int, int]:
        return (self.shape[2], self.shape[1], self.shape[0])

    def radial_nodes(self) -> Array:
        return self.r.copy()

    def derivative(self, values: Array, axis: int) -> Array:
        """Return a coordinate derivative using the legacy ``(r,theta,x)`` axis id."""
        assert values.shape == self.shape
        assert axis in {0, 1, 2}
        direction = {0: 2, 1: 1, 2: 0}[axis]
        return np.asarray(
            self._core.derivative(
                np.ascontiguousarray(values, dtype=np.float64), direction
            )
        )

    def gradient_scalar(self, values: Array) -> Array:
        return np.asarray(
            self._core.gradient(np.ascontiguousarray(values, dtype=np.float64))
        )

    def gradient_scalar_frames(self, values: Array, *, label: str = "gradient") -> Array:
        """Differentiate a complete ``(T, Nx, Ntheta, Nr)`` batch with one plan."""
        assert values.ndim == 4 and values.shape[1:] == self.shape
        print(f"[rho_fitting.spectral] {label}: frames=0/{values.shape[0]}", flush=True)
        out = np.asarray(
            self._core.gradient(
                np.ascontiguousarray(values, dtype=np.float64), grid_offset=1
            )
        )
        _progress(label, values.shape[0], values.shape[0])
        return out

    def divergence(self, values: Array) -> Array:
        """Return divergence of a physical-frame flux with direction axis ``-1``."""
        return np.asarray(
            self._core.divergence(np.ascontiguousarray(values, dtype=np.float64))
        )

    def divergence_frames(self, values: Array, *, label: str = "divergence") -> Array:
        """Diverge a complete time batch while reusing the spatial transform plan."""
        assert values.ndim >= 5 and values.shape[1:4] == self.shape
        print(f"[rho_fitting.spectral] {label}: frames=0/{values.shape[0]}", flush=True)
        out = np.asarray(
            self._core.divergence(
                np.ascontiguousarray(values, dtype=np.float64), grid_offset=1
            )
        )
        _progress(label, values.shape[0], values.shape[0])
        return out

    def laplacian_scalar(self, values: Array) -> Array:
        return np.asarray(
            self._core.laplacian(np.ascontiguousarray(values, dtype=np.float64))
        )

    def gradient_vector(self, values: Array) -> Array:
        return np.asarray(
            self._core.gradient(np.ascontiguousarray(values, dtype=np.float64))
        )

    def laplacian_vector(self, values: Array) -> Array:
        return np.asarray(
            self._core.laplacian(np.ascontiguousarray(values, dtype=np.float64))
        )

    def gradient_rank2(self, values: Array) -> Array:
        return np.asarray(
            self._core.gradient(np.ascontiguousarray(values, dtype=np.float64))
        )

    def filter_two_thirds(self, values: Array) -> Array:
        """Apply the agreed 2/3 modal cutoff over every spatial axis."""
        return np.asarray(
            self._core.filter_two_thirds(
                np.ascontiguousarray(values, dtype=np.float64)
            )
        )


def barycentric_matrix(source: Array, target: Array) -> Array:
    """Return the Rust-built degree-N barycentric interpolation matrix."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    assert source.ndim == target.ndim == 1 and source.size >= 2
    from . import _rho_fitting_core

    assert _rho_fitting_core is not None, "rho-fitting Rust core is unavailable"
    return np.asarray(
        _rho_fitting_core.barycentric_matrix(
            np.ascontiguousarray(source), np.ascontiguousarray(target)
        )
    )


def transfer_radial(values: Array, matrix: Array, axis: int) -> Array:
    """Apply a precomputed radial interpolation matrix in Rust."""
    from . import _rho_fitting_core

    assert _rho_fitting_core is not None, "rho-fitting Rust core is unavailable"
    return np.asarray(
        _rho_fitting_core.transfer_radial(
            np.ascontiguousarray(values, dtype=np.float64),
            np.ascontiguousarray(matrix, dtype=np.float64),
            int(axis),
        )
    )


def _progress(label: str, completed: int, total: int) -> None:
    interval = max(1, min(10, total // 10))
    if completed == total or completed % interval == 0:
        print(f"[rho_fitting.spectral] {label}: frames={completed}/{total}", flush=True)


@lru_cache(maxsize=8)
def cached_cylindrical_operators(
    lx: float,
    theta_period: float,
    r_min: float,
    r_max: float,
    nx: int,
    ntheta: int,
    nr: int,
) -> CylindricalSpectralOperators:
    """Return one reusable Shenfun plan per cylindrical grid geometry."""
    return CylindricalSpectralOperators(lx, theta_period, r_min, r_max, (nx, ntheta, nr))
