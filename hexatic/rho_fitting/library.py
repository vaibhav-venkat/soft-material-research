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
