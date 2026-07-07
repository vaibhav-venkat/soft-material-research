"""Mechanical term labels."""

from __future__ import annotations

MECHANICAL_LABELS = {
    "grad_rho": "grad rho",
    "grad_lap_rho": "grad lap rho",
    "Q_dot_grad_rho": "Q dot grad rho",
    "A_dot_grad_rho": "A dot grad rho",
    "P": "P",
    "A": "A",
    "rho_A": "rho A",
    "psi6sq_A": "psi6^2 A",
    "rho_delta_psi6sq_A": "rho delta psi6^2 A",
    "grad_P": "grad P",
    "rho_grad_P": "rho grad P",
    "grad_lap_P": "grad lap P",
    "tangential_projected_Ubar_P_alpha": "tangential Ubar P alpha",
    "radial_projected_Ubar_P_alpha": "radial Ubar P alpha",
    "tangential_grad_Q": "tangential grad Q",
    "radial_grad_Q": "radial grad Q",
    "Q": "Q",
    "psi6sq_Q": "psi6^2 Q",
}


def mechanical_labels(names: tuple[str, ...]) -> tuple[str, ...]:
    """Map mechanical library term names to report labels while preserving unknown names."""
    return tuple(MECHANICAL_LABELS.get(name, name) for name in names)
