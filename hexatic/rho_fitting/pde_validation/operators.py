"""Periodic surface operators for rho-fitting PDE validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


Array = NDArray[Any]

from hexatic.rho_fitting import _rho_fitting_core, _rho_fitting_core_import_error
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


def _core() -> Any:
    if _rho_fitting_core is None:
        raise ImportError(f"rho-fitting Rust core is unavailable: {_rho_fitting_core_import_error}")
    return _rho_fitting_core


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


def alignment_tensor(rho: Array, q: Array) -> Array:
    """Build the shared alignment tensor ``A = Q + rho I / 3``."""
    return np.asarray(
        _core().build_alignment_tensor(
            np.ascontiguousarray(rho, dtype=np.float64),
            np.ascontiguousarray(q, dtype=np.float64),
        )
    )


def a_dot_grad_rho(a: Array, grad_rho: Array) -> Array:
    """Contract the full second orientation moment against the cylindrical density gradient."""
    return np.asarray(
        _core().contract_alignment_gradient(
            np.ascontiguousarray(a, dtype=np.float64),
            np.ascontiguousarray(grad_rho, dtype=np.float64),
        )
    )


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
        y_p_coefficients: Coefficients for the three polarization-flux library terms.
        y_q_coefficients: Coefficients for the three nematic-flux library terms.
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
    a = alignment_tensor(rho, q)
    a_surface = a

    grad_rho = gradient_scalar(rho, operators)
    f_rho = weighted_linear_combination(
        [
            grad_rho,
            a_dot_grad_rho(a_surface, grad_rho),
            p,
        ],
        y_rho_coefficients,
    )

    grad_p = gradient_vector(p, operators)
    f_p = weighted_linear_combination(
        [
            a_surface,
            scale_by_scalar(psi6_sq_fixed, a_surface),
            grad_p,
        ],
        y_p_coefficients,
    )

    ubar = estimate_ubar(f_p, a_surface) if ubar_override is None else ubar_override
    grad_q = gradient_rank2(q, operators)
    ubar_p_alpha = scale_by_scalar(ubar, p_dot_alpha_traceless(p))
    f_q = weighted_linear_combination(
        [
            project_flux_directions(ubar_p_alpha, "tangential"),
            project_flux_directions(ubar_p_alpha, "radial"),
            project_flux_directions(grad_q, "radial"),
        ],
        y_q_coefficients,
    )
    s_q = np.zeros_like(q)
    return ClosureFields(f_rho=f_rho, f_p=f_p, f_q=f_q, s_q=s_q, ubar=ubar, a_surface=a_surface)


def estimate_ubar(y_p: Array, a: Array) -> Array:
    """Project the fitted P flux onto the alignment tensor to estimate speed."""
    return np.asarray(
        _core().estimate_ubar(
            np.ascontiguousarray(y_p, dtype=np.float64),
            np.ascontiguousarray(a, dtype=np.float64),
        )
    )


def project_flux_directions(values: Array, mode: str) -> Array:
    """Project a cylindrical Q flux tensor onto tangential or radial flux directions."""
    if mode not in {"tangential", "radial"}:
        raise AssertionError(f"unknown flux projection mode: {mode}")
    return np.asarray(
        _core().project_flux_directions(
            np.ascontiguousarray(values, dtype=np.float64),
            0 if mode == "tangential" else 1,
        )
    )


def p_dot_alpha_traceless(p: Array) -> Array:
    """Build the symmetric traceless ``P``-alignment tensor used in the Q closure."""
    return np.asarray(
        _core().build_p_alignment(np.ascontiguousarray(p, dtype=np.float64))
    )


def scale_by_scalar(scalar: Array, values: Array) -> Array:
    """Multiply every trailing tensor component by a scalar field."""
    return np.asarray(
        _core().scale_by_scalar(
            np.ascontiguousarray(scalar, dtype=np.float64),
            np.ascontiguousarray(values, dtype=np.float64),
        )
    )


def weighted_linear_combination(fields: list[Array], coefficients: Array) -> Array:
    """Return the Rust-owned coefficient-weighted sum of same-shaped fields."""
    return np.asarray(
        _core().weighted_linear_combination(
            [np.ascontiguousarray(field, dtype=np.float64) for field in fields],
            np.ascontiguousarray(coefficients, dtype=np.float64),
        )
    )


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
