"""Mechanical term labels."""

from __future__ import annotations

MECHANICAL_LABELS = {
    "grad_rho": "grad rho",
    "grad_lap_rho": "grad lap rho",
    "Q_dot_grad_rho": "Q dot grad rho",
    "A": "A",
    "rho_A": "rho A",
    "psi6sq_A": "psi6^2 A",
    "rho_delta_psi6sq_A": "rho delta psi6^2 A",
    "grad_P": "grad P",
    "rho_grad_P": "rho grad P",
    "grad_lap_P": "grad lap P",
    "Ubar_P_dot_alpha_traceless": "Ubar P dot alpha",
    "grad_P_symmetric_traceless": "grad P sym traceless",
    "grad_Q": "grad Q",
    "rho_grad_Q": "rho grad Q",
    "grad_lap_Q": "grad lap Q",
}


def mechanical_labels(names: tuple[str, ...]) -> tuple[str, ...]:
    """Map mechanical library term names to report labels while preserving unknown names."""
    return tuple(MECHANICAL_LABELS.get(name, name) for name in names)
