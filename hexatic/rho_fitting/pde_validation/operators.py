"""Periodic surface operators for rho-fitting PDE validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


Array = NDArray[Any]

from hexatic.rho_fitting.spectral import CylindricalSpectralOperators


@dataclass(frozen=True)
class ClosureFields:
    """Evaluated closure fluxes and auxiliary mechanical fields for PDE rollouts."""

    f_rho: Array
    f_p: Array
    f_q: Array
    s_q: Array
    ubar: Array
    a_surface: Array


def gradient_scalar(values: Array, operators: CylindricalSpectralOperators) -> Array:
    """Return cylindrical gradients of a scalar grid as ``(..., 3)`` components."""
    return operators.gradient_scalar(values)


def laplacian_scalar(values: Array, operators: CylindricalSpectralOperators) -> Array:
    """Return the cylindrical scalar Laplacian."""
    return operators.laplacian_scalar(values)


def gradient_vector(values: Array, operators: CylindricalSpectralOperators) -> Array:
    """Differentiate each vector component and insert a three-component derivative axis."""
    return operators.gradient_vector(values)


def laplacian_vector(values: Array, operators: CylindricalSpectralOperators) -> Array:
    """Apply the scalar Laplacian independently to every vector component."""
    return operators.laplacian_vector(values)


def gradient_rank2(values: Array, operators: CylindricalSpectralOperators) -> Array:
    """Differentiate each rank-2 tensor component over the three cylindrical directions."""
    return operators.gradient_rank2(values)


def laplacian_rank2(values: Array, operators: CylindricalSpectralOperators) -> Array:
    """Apply the scalar Laplacian independently to every rank-2 tensor component."""
    out = np.empty_like(values, dtype=np.float64)
    for row in range(values.shape[-2]):
        for col in range(values.shape[-1]):
            out[..., row, col] = laplacian_scalar(values[..., row, col], operators)
    return out

def divergence_vector(values: Array, operators: CylindricalSpectralOperators) -> Array:
    """Return cylindrical divergence of a flux field with direction axis 3."""
    assert values.shape[3] == 3, "flux direction axis must have length 3"
    return operators.divergence(values)


def divergence_surface_flux(values: Array, operators: CylindricalSpectralOperators) -> Array:
    """Compatibility alias for cylindrical flux divergence."""
    return divergence_vector(values, operators)


def a_dot_grad_rho(a: Array, grad_rho: Array) -> Array:
    """Contract the full second orientation moment against the cylindrical density gradient."""
    return np.einsum("...ka,...a->...k", a, grad_rho)


def closure_fields(
    rho: Array,
    p: Array,
    q: Array,
    psi6_sq_fixed: Array,
    y_rho_coefficients: Array,
    y_p_coefficients: Array,
    y_q_coefficients: Array,
    operators: CylindricalSpectralOperators,
    ubar_override: Array | None = None,
) -> ClosureFields:
    """Evaluate fitted mechanical closure fields from rho, P, Q, and coefficient vectors.

    Parameters:
        rho: Scalar density field with shape ``(Nx, Ny)``.
        p: Polarization field with shape ``(Nx, Ny, 3)``.
        q: Nematic moment field with shape ``(Nx, Ny, 3, 3)``.
        psi6_sq_fixed: Fixed hexatic-amplitude field sampled at the validation time.
        y_rho_coefficients: Coefficients for the density-flux library terms.
        y_p_coefficients: Coefficients for the six polarization-flux library terms.
        y_q_coefficients: Coefficients for the five nematic-flux library terms.
        dx: Axial grid spacing.
        dtheta: Angular grid spacing.
        r_centers: Radial bin centers.
        ubar_override: Optional precomputed scalar speed field for validating cached
            transport separately from the fitted ``F_P`` projection.

    Returns:
        ``ClosureFields`` with 3D density flux, P flux, Q flux, estimated speed,
        and the surface block of ``A = Q + rho I / 3``.

    Examples:
        ``fields = closure_fields(rho, p, q, psi6_sq, c_rho, c_p, c_q, dx, dy)``

    Edge cases:
        Coefficient order is positional and must match the Rust/Python library builders;
        this routine does not realign by term name.
    """
    identity = np.eye(3, dtype=np.float64)
    a = q + rho[..., None, None] * identity / 3.0
    a_surface = a

    grad_rho = gradient_scalar(rho, operators)
    f_rho = (
        y_rho_coefficients[0] * grad_rho
        + y_rho_coefficients[1] * a_dot_grad_rho(a_surface, grad_rho)
        + y_rho_coefficients[2] * p
    )

    grad_p = gradient_vector(p, operators)
    grad_lap_p = gradient_vector(laplacian_vector(p, operators), operators)
    f_p = (
        y_p_coefficients[0] * a_surface
        + y_p_coefficients[1] * rho[..., None, None] * a_surface
        + y_p_coefficients[2] * psi6_sq_fixed[..., None, None] * a_surface
        + y_p_coefficients[3] * grad_p
        + y_p_coefficients[4] * rho[..., None, None] * grad_p
        + y_p_coefficients[5] * grad_lap_p
    )

    ubar = estimate_ubar(f_p, a_surface) if ubar_override is None else ubar_override
    grad_q = gradient_rank2(q, operators)
    ubar_p_alpha = ubar[..., None, None, None] * p_dot_alpha_traceless(p)
    f_q = (
        y_q_coefficients[0] * project_flux_directions(ubar_p_alpha, "tangential")
        + y_q_coefficients[1] * project_flux_directions(ubar_p_alpha, "radial")
        + y_q_coefficients[2] * project_flux_directions(grad_q, "tangential")
        + y_q_coefficients[3] * project_flux_directions(grad_q, "radial")
    )
    s_q = y_q_coefficients[4] * q + y_q_coefficients[5] * psi6_sq_fixed[..., None, None] * q
    return ClosureFields(f_rho=f_rho, f_p=f_p, f_q=f_q, s_q=s_q, ubar=ubar, a_surface=a_surface)


def estimate_ubar(y_p: Array, a: Array) -> Array:
    """Project the fitted P flux onto the alignment tensor to estimate speed."""
    denominator = np.sum(a * a, axis=(-2, -1))
    numerator = np.sum(y_p * a, axis=(-2, -1))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.0)


def project_flux_directions(values: Array, mode: str) -> Array:
    """Project a cylindrical Q flux tensor onto tangential or radial flux directions."""
    out = np.zeros_like(values, dtype=np.float64)
    if mode == "tangential":
        out[..., 0, :, :] = values[..., 0, :, :]
        out[..., 1, :, :] = values[..., 1, :, :]
    elif mode == "radial":
        out[..., 2, :, :] = values[..., 2, :, :]
    else:
        raise AssertionError(f"unknown flux projection mode: {mode}")
    return out


def p_dot_alpha_traceless(p: Array) -> Array:
    """Build the symmetric traceless ``P``-alignment tensor used in the Q closure."""
    out = np.zeros(p.shape[:-1] + (3, 3, 3), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for k in range(3):
        for a in range(3):
            for b in range(3):
                out[..., k, a, b] = p[..., a] * float(k == b) + p[..., b] * float(k == a) - (2.0 / 3.0) * p[..., k] * identity[a, b]
    return out


def grad_p_symmetric_traceless(p: Array, operators: CylindricalSpectralOperators) -> Array:
    """Build the symmetric traceless gradient-of-P tensor used in the Q closure."""
    grad_p = gradient_vector(p, operators)
    out = np.zeros(p.shape[:-1] + (3, 3, 3), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for k in range(3):
        trace_part = (2.0 / 3.0) * grad_p[..., k, k]
        for a in range(3):
            for b in range(3):
                out[..., k, a, b] = (
                    grad_p[..., k, a] * float(k == b)
                    + grad_p[..., k, b] * float(k == a)
                    - trace_part * identity[a, b]
                )
    return out
