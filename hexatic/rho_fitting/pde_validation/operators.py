"""Periodic surface operators for rho-fitting PDE validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


Array = NDArray[Any]


@dataclass(frozen=True)
class ClosureFields:
    """Evaluated closure fluxes and auxiliary mechanical fields for PDE rollouts."""

    f_rho: Array
    f_p: Array
    f_q: Array
    ubar: Array
    a_surface: Array


def gradient_scalar(values: Array, dx: float, dy: float) -> Array:
    """Return centered periodic gradients of a scalar grid as ``(..., 2)`` components."""
    out = np.empty(values.shape + (2,), dtype=np.float64)
    out[..., 0] = (np.roll(values, -1, axis=0) - np.roll(values, 1, axis=0)) / (2.0 * dx)
    out[..., 1] = (np.roll(values, -1, axis=1) - np.roll(values, 1, axis=1)) / (2.0 * dy)
    return out


def laplacian_scalar(values: Array, dx: float, dy: float) -> Array:
    """Return the centered periodic scalar Laplacian on a uniform surface grid."""
    return (
        (np.roll(values, -1, axis=0) - 2.0 * values + np.roll(values, 1, axis=0)) / (dx * dx)
        + (np.roll(values, -1, axis=1) - 2.0 * values + np.roll(values, 1, axis=1)) / (dy * dy)
    )


def gradient_vector(values: Array, dx: float, dy: float) -> Array:
    """Differentiate each vector component and insert a two-component derivative axis."""
    out = np.empty(values.shape[:-1] + (2, values.shape[-1]), dtype=np.float64)
    for component in range(values.shape[-1]):
        out[..., :, component] = gradient_scalar(values[..., component], dx, dy)
    return out


def laplacian_vector(values: Array, dx: float, dy: float) -> Array:
    """Apply the scalar Laplacian independently to every vector component."""
    out = np.empty_like(values, dtype=np.float64)
    for component in range(values.shape[-1]):
        out[..., component] = laplacian_scalar(values[..., component], dx, dy)
    return out


def gradient_rank2(values: Array, dx: float, dy: float) -> Array:
    """Differentiate each rank-2 tensor component over the two surface directions."""
    out = np.empty(values.shape[:-2] + (2, values.shape[-2], values.shape[-1]), dtype=np.float64)
    for row in range(values.shape[-2]):
        for col in range(values.shape[-1]):
            out[..., :, row, col] = gradient_scalar(values[..., row, col], dx, dy)
    return out


def laplacian_rank2(values: Array, dx: float, dy: float) -> Array:
    """Apply the scalar Laplacian independently to every rank-2 tensor component."""
    out = np.empty_like(values, dtype=np.float64)
    for row in range(values.shape[-2]):
        for col in range(values.shape[-1]):
            out[..., row, col] = laplacian_scalar(values[..., row, col], dx, dy)
    return out


def divergence_vector(values: Array, dx: float, dy: float) -> Array:
    """Return the periodic divergence of a surface vector field with last axis length 2."""
    return (
        (np.roll(values[..., 0], -1, axis=0) - np.roll(values[..., 0], 1, axis=0)) / (2.0 * dx)
        + (np.roll(values[..., 1], -1, axis=1) - np.roll(values[..., 1], 1, axis=1)) / (2.0 * dy)
    )


def divergence_surface_flux(values: Array, dx: float, dy: float) -> Array:
    """Diverge a two-direction surface flux while preserving its moment-component axes.

    Parameters:
        values: Flux array with shape ``(Nx, Ny, 2, ...)``; the third axis stores the
            surface flux direction, not a moment component.
        dx: Axial grid spacing.
        dy: Unwrapped angular grid spacing.

    Returns:
        Array with shape ``(Nx, Ny, ...)``.

    Edge cases:
        The function assumes periodic boundaries on both surface axes and does not check
        that axis 2 has length 2.
    """
    return (
        (np.roll(values[..., 0, :, :], -1, axis=0) - np.roll(values[..., 0, :, :], 1, axis=0)) / (2.0 * dx)
        + (np.roll(values[..., 1, :, :], -1, axis=1) - np.roll(values[..., 1, :, :], 1, axis=1)) / (2.0 * dy)
    )


def q_dot_grad_rho(q: Array, grad_rho: Array) -> Array:
    """Contract the in-surface block of ``Q`` against ``grad(rho)``."""
    return np.einsum("...ka,...a->...k", q[..., :2, :2], grad_rho)


def closure_fields(
    rho: Array,
    p: Array,
    q: Array,
    psi6_sq_fixed: Array,
    y_rho_coefficients: Array,
    y_p_coefficients: Array,
    y_q_coefficients: Array,
    dx: float,
    dy: float,
    ubar_override: Array | None = None,
) -> ClosureFields:
    """Evaluate fitted mechanical closure fields from rho, P, Q, and coefficient vectors.

    Parameters:
        rho: Scalar density field with shape ``(Nx, Ny)``.
        p: Polarization field with shape ``(Nx, Ny, 3)``.
        q: Nematic moment field with shape ``(Nx, Ny, 3, 3)``.
        psi6_sq_fixed: Fixed hexatic-amplitude field sampled at the validation time.
        y_rho_coefficients: Coefficients for the three density-flux library terms.
        y_p_coefficients: Coefficients for the six polarization-flux library terms.
        y_q_coefficients: Coefficients for the five nematic-flux library terms.
        dx: Axial grid spacing.
        dy: Unwrapped angular grid spacing.
        ubar_override: Optional precomputed scalar speed field for validating cached
            transport separately from the fitted ``F_P`` projection.

    Returns:
        ``ClosureFields`` with density flux, P flux, Q flux, estimated speed, and the
        surface block of ``A = Q + rho I / 3``.

    Examples:
        ``fields = closure_fields(rho, p, q, psi6_sq, c_rho, c_p, c_q, dx, dy)``

    Edge cases:
        Coefficient order is positional and must match the Rust/Python library builders;
        this routine does not realign by term name.
    """
    identity = np.eye(3, dtype=np.float64)
    a = q + rho[..., None, None] * identity / 3.0
    a_surface = a[..., :2, :]

    grad_rho = gradient_scalar(rho, dx, dy)
    grad_lap_rho = gradient_scalar(laplacian_scalar(rho, dx, dy), dx, dy)
    f_rho = (
        y_rho_coefficients[0] * grad_rho
        + y_rho_coefficients[1] * grad_lap_rho
        + y_rho_coefficients[2] * q_dot_grad_rho(q, grad_rho)
    )

    grad_p = gradient_vector(p, dx, dy)
    grad_lap_p = gradient_vector(laplacian_vector(p, dx, dy), dx, dy)
    f_p = (
        y_p_coefficients[0] * a_surface
        + y_p_coefficients[1] * rho[..., None, None] * a_surface
        + y_p_coefficients[2] * psi6_sq_fixed[..., None, None] * a_surface
        + y_p_coefficients[3] * grad_p
        + y_p_coefficients[4] * rho[..., None, None] * grad_p
        + y_p_coefficients[5] * grad_lap_p
    )

    ubar = estimate_ubar(f_p, a_surface) if ubar_override is None else ubar_override
    grad_q = gradient_rank2(q, dx, dy)
    grad_lap_q = gradient_rank2(laplacian_rank2(q, dx, dy), dx, dy)
    f_q = (
        y_q_coefficients[0] * ubar[..., None, None, None] * p_dot_alpha_traceless(p)
        + y_q_coefficients[1] * grad_p_symmetric_traceless(p, dx, dy)
        + y_q_coefficients[2] * grad_q
        + y_q_coefficients[3] * rho[..., None, None, None] * grad_q
        + y_q_coefficients[4] * grad_lap_q
    )
    return ClosureFields(f_rho=f_rho, f_p=f_p, f_q=f_q, ubar=ubar, a_surface=a_surface)


def estimate_ubar(y_p: Array, a: Array) -> Array:
    """Project the fitted P flux onto the surface alignment tensor to estimate speed."""
    a_surface = a[..., :2, :]
    denominator = np.sum(a_surface * a_surface, axis=(-2, -1))
    numerator = np.sum(y_p * a_surface, axis=(-2, -1))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.0)


def p_dot_alpha_traceless(p: Array) -> Array:
    """Build the symmetric traceless ``P``-alignment tensor used in the Q closure."""
    out = np.zeros(p.shape[:-1] + (2, 3, 3), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for k in range(2):
        for a in range(3):
            for b in range(3):
                out[..., k, a, b] = p[..., a] * float(k == b) + p[..., b] * float(k == a) - (2.0 / 3.0) * p[..., k] * identity[a, b]
    return out


def grad_p_symmetric_traceless(p: Array, dx: float, dy: float) -> Array:
    """Build the symmetric traceless gradient-of-P tensor used in the Q closure."""
    grad_p = gradient_vector(p, dx, dy)
    out = np.zeros(p.shape[:-1] + (2, 3, 3), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for k in range(2):
        trace_part = (2.0 / 3.0) * grad_p[..., k, k]
        for a in range(3):
            for b in range(3):
                out[..., k, a, b] = (
                    grad_p[..., k, a] * float(k == b)
                    + grad_p[..., k, b] * float(k == a)
                    - trace_part * identity[a, b]
                )
    return out
