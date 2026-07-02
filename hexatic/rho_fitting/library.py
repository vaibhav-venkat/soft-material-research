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
    "rho_grad_rho": "rho grad rho",
    "rho2_grad_rho": "rho^2 grad rho",
    "grad_lap_rho": "grad lap rho",
    "lap_rho_grad_rho": "lap rho grad rho",
    "grad_rho_cubed": "|grad rho|^2 grad rho",
    "grad_grad_rho_norm2": "grad |grad rho|^2",
    "A": "A",
    "rho_A": "rho A",
    "rho2_A": "rho^2 A",
    "rho3_A": "rho^3 A",
    "P_dot_alpha": "P dot alpha",
    "rho_P_dot_alpha": "rho P dot alpha",
    "rho2_P_dot_alpha": "rho^2 P dot alpha",
    "P_dot_II": "P dot II",
    "F_rho_I": "F_rho I",
}


def mechanical_labels(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(MECHANICAL_LABELS.get(name, name) for name in names)
