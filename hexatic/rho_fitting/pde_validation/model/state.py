"""Grid construction, field packing, and validation projection bounds."""

from __future__ import annotations

from typing import Any

import numpy as np
from pde import CartesianGrid, FieldCollection, ScalarField

from ..cache import ValidationInputs
from .types import Array, COMPONENTS, VALIDATION_BOUND_SCALE


def make_grid(inputs: ValidationInputs) -> CartesianGrid:
    """Create a 3D storage grid for validation fields."""
    r_min = float(inputs.r_centers[0])
    r_max = float(inputs.r_centers[-1])
    if inputs.r_centers.size > 1:
        padding = 0.5 * float(np.mean(np.diff(inputs.r_centers)))
        r_min -= padding
        r_max += padding
    return CartesianGrid(
        [(0.0, inputs.lx), (0.0, inputs.theta_period), (r_min, r_max)],
        inputs.rho.shape[1:],
        periodic=[True, True, False],
    )


def pack_state(grid: Any, rho: Array, p: Array, q: Array) -> FieldCollection:
    """Pack rho, three P components, and nine Q components into a FieldCollection."""
    fields = [ScalarField(grid, rho, label="rho")]
    fields.extend(ScalarField(grid, p[..., component], label=f"P{component}") for component in range(3))
    fields.extend(ScalarField(grid, q[..., a, b], label=f"Q{a}{b}") for a in range(3) for b in range(3))
    return FieldCollection(fields)


def unpack_state(state: FieldCollection) -> tuple[Array, Array, Array]:
    """Unpack a FieldCollection into ``rho``, ``P``, and ``Q`` arrays."""
    assert len(state) == COMPONENTS, f"expected {COMPONENTS} fields"
    rho = np.asarray(state[0].data, dtype=np.float64)
    p = np.stack([np.asarray(state[1 + component].data, dtype=np.float64) for component in range(3)], axis=-1)
    offset = 4
    q = np.empty(rho.shape + (3, 3), dtype=np.float64)
    for a in range(3):
        for b in range(3):
            q[..., a, b] = np.asarray(state[offset + 3 * a + b].data, dtype=np.float64)
    return rho, p, q


def scalar_bounds(values: Array) -> tuple[float, float]:
    """Return padded scalar bounds derived from finite validation input values."""
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    assert finite.size > 0, "validation input has no finite scalar values"
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    span = max(maximum - minimum, abs(maximum), abs(minimum), 1.0)
    padding = VALIDATION_BOUND_SCALE * span
    return minimum - padding, maximum + padding


def symmetric_bound(values: Array) -> float:
    """Return a symmetric absolute bound derived from finite tensor input values."""
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    assert finite.size > 0, "validation input has no finite tensor values"
    return max(float(np.max(np.abs(finite))) * VALIDATION_BOUND_SCALE, 1.0)
