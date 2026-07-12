"""Typed raw and temporally filtered mechanical field bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, Any

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[Any]


@dataclass(frozen=True)
class MechanicalRawFields:
    """Raw coarse-grained fields with canonical grid and component axes."""

    rho: Array
    P: Array
    Q: Array
    A: Array
    psi6_sq: Array
    J_rho: Array
    J_P: Array
    J_Q: Array
    r_edges: Array
    r_centers: Array

    @property
    def J_density(self) -> Array:
        """Return the density current under its legacy cache name."""
        return self.J_rho


@dataclass(frozen=True)
class MechanicalSpectralFields:
    """Temporally filtered fields and mechanical targets for one fitting run."""

    rho: Array
    P: Array
    Q: Array
    A: Array
    psi6_sq: Array
    J_rho: Array
    J_P: Array
    J_Q: Array
    Y_rho: Array
    Y_P: Array
    Y_Q: Array
    partial_t_rho: Array
    temporal_power: Array
    cheb_times: Array
    cheb_scaled_times: Array

    @property
    def J_density(self) -> Array:
        """Return the density current under its legacy cache name."""
        return self.J_rho


@dataclass(frozen=True)
class MechanicalTargets:
    """Mechanical closure targets in the canonical vector/tensor component order."""

    Y_rho: Array
    Y_P: Array
    Y_Q: Array
