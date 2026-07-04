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
    "phi1_rho_grad_rho": "phi1(rho) grad rho",
    "grad_lap_rho": "grad lap rho",
    "Q_dot_grad_rho": "Q dot grad rho",
    "grad_bilap_rho": "bilap grad",
    "Q2_dot_grad_rho": "Q^2 dot grad rho",
    "trQ2_grad_rho": "tr(Q^2) grad rho",
    "rho_Q_dot_grad_rho": "rho Q dot grad rho",
    "div_Q": "div Q",
    "rho_div_Q": "rho div Q",
    "div_rho_Q": "div(rho Q)",
    "Q_dot_grad_lap_rho": "Q dot grad lap rho",
    "P": "P",
    "rho_P": "rho P",
    "P_dot_grad_rho_P": "(P dot grad rho) P",
    "P2_grad_rho": "|P|^2 grad rho",
    "div_P_P": "(div P) P",
    "P_dot_grad_P": "(P dot grad) P",
    "grad_P2": "grad |P|^2",
    "div_A": "div A",
    "grad_trA": "grad tr(A)",
    "A_dot_grad_rho": "A dot grad rho",
    "trA_grad_rho": "tr(A) grad rho",
    "Q_dot_div_A": "Q dot div A",
    "A": "A",
    "rho_A": "rho A",
    "delta_A": "delta A",
    "psi6_A": "psi6 A",
    "psi6sq_A": "psi6^2 A",
    "delta_psi6_A": "delta psi6 A",
    "delta_psi6sq_A": "delta (psi6)^2 A",
    "rho_psi6sq_A": "rho psi6^2 A",
    "rho_delta_psi6sq_A": "rho delta psi6^2 A",
    "rho_grad_div_P": "rho grad div P",
    "rho_lap_P": "rho lap P",
    "trQ2_A": "tr(Q^2) A",
    "P2_A": "|P|^2 A",
    "P2_P": "|P|^2 P",
    "trQ2_P": "tr(Q^2) P",
    "psi6sq_P": "psi6^2 P",
    "lap_P": "lap P",
    "grad_div_P": "grad div P",
    "Q_dot_P": "Q dot P",
    "Q2_dot_P": "Q^2 dot P",
    "A_dot_Q": "A dot Q",
    "Q_dot_A_dot_Q": "Q dot A dot Q",
    "Q_colon_A_P": "(Q:A) P",
    "P_dot_alpha_traceless": "P dot alpha",
    "Ubar_P_dot_alpha_traceless": "Ubar P dot alpha",
    "rho_Ubar_P_dot_alpha_traceless": "rho Ubar P dot alpha",
    "delta_Ubar_P_dot_alpha_traceless": "delta Ubar P dot alpha",
    "psi6sq_Ubar_P_dot_alpha_traceless": "psi6^2 Ubar P dot alpha",
    "trQ2_Ubar_P_dot_alpha_traceless": "tr(Q^2) Ubar P dot alpha",
    "P2_Ubar_P_dot_alpha_traceless": "|P|^2 Ubar P dot alpha",
    "Q": "Q",
    "rho_Q": "rho Q",
    "delta_Q": "delta Q",
    "trQ2_Q": "tr(Q^2) Q",
    "psi6sq_Q": "psi6^2 Q",
    "P2_Q": "|P|^2 Q",
    "lap_Q": "lap Q",
    "bilap_Q": "lap^2 Q",
    "PP_traceless_2d": "PP - |P|^2 I/2",
    "rho_PP_traceless_2d": "rho (PP - |P|^2 I/2)",
    "AA_traceless_2d": "AA - tr(A^2) I/2",
    "QA_plus_AQ": "QA + AQ",
    "Q2_traceless_2d": "Q^2 - tr(Q^2) I/2",
    "sym_grad_P_traceless_2d": "grad P + grad P^T - div(P) I",
    "hess_rho_traceless_2d": "hess rho - lap(rho) I/2",
}


def mechanical_labels(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(MECHANICAL_LABELS.get(name, name) for name in names)
