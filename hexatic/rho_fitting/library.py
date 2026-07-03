"""Density terms derived from flux fields."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DensityTerm:
    name: str
    label: str
    flux: str


DENSITY_TERMS: tuple[DensityTerm, ...] = (
    DensityTerm("neg_div_grad_rho", "-div(grad rho)", "grad_rho"),
    DensityTerm("neg_div_grad_lap_rho", "-div(grad lap rho)", "grad_lap_rho"),
    DensityTerm("neg_div_lap_rho_grad_rho", "-div(lap rho grad rho)", "lap_rho_grad_rho"),
    DensityTerm("neg_div_grad_rho_cubed", "-div(|grad rho|^2 grad rho)", "grad_rho_cubed"),
)


def term_names() -> tuple[str, ...]:
    return tuple(term.name for term in DENSITY_TERMS)


def term_labels() -> tuple[str, ...]:
    return tuple(term.label for term in DENSITY_TERMS)


def flux_names() -> tuple[str, ...]:
    return tuple(term.flux for term in DENSITY_TERMS)


MECHANICAL_LABELS = {
    "grad_rho": "grad rho",
    "grad_lap_rho": "grad lap rho",
    "grad_rho_cubed": "|grad rho|^2 grad rho",
    "Q_dot_grad_rho": "Q dot grad rho",
    "rho2_Q_dot_grad_rho": "rho^2 Q dot grad rho",
    "trQ2_grad_rho": "tr(Q^2) grad rho",
    "P2_grad_rho": "|P|^2 grad rho",
    "grad_P2": "grad |P|^2",
    "grad_Qnn": "grad Q_nn",
    "A": "A",
    "delta_psi6sq_A": "delta (psi6)^2 A",
    "P_dot_alpha": "P dot alpha",
    "rho2_P_dot_alpha": "rho^2 P dot alpha",
    "P2_P_dot_alpha_traceless": "|P|^2 P dot alpha",
    "trQ2_P_dot_alpha_traceless": "tr(Q^2) P dot alpha",
    "Ubar_P_dot_alpha_traceless": "Ubar P dot alpha",
}


def mechanical_labels(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(MECHANICAL_LABELS.get(name, name) for name in names)
