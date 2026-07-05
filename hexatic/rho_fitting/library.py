"""Density terms derived from flux fields."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DensityTerm:
    """Name, report label, and flux-cache key for one density-library term."""

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
    """Return density-library term names in coefficient order."""
    return tuple(term.name for term in DENSITY_TERMS)


def term_labels() -> tuple[str, ...]:
    """Return display labels for density-library terms in coefficient order."""
    return tuple(term.label for term in DENSITY_TERMS)


def flux_names() -> tuple[str, ...]:
    """Return flux field keys corresponding to density-library terms."""
    return tuple(term.flux for term in DENSITY_TERMS)


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
