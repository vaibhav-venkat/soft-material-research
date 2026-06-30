"""Density candidate library registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Term:
    name: str
    label: str
    rust_name: str
    family: str
    order: int


def density_terms(n_rho_power: int, n_rho_lap_power: int) -> tuple[Term, ...]:
    if n_rho_power < 0 or n_rho_lap_power < 0:
        raise ValueError("term counts must be nonnegative")
    return DENSITY_TERMS


def term_names(terms: tuple[Term, ...]) -> tuple[str, ...]:
    return tuple(term.name for term in terms)


def term_labels(terms: tuple[Term, ...]) -> tuple[str, ...]:
    return tuple(term.label for term in terms)


def validate_terms(names: tuple[str, ...], registry: tuple[Term, ...]) -> None:
    known = {term.name for term in registry}
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"unknown terms: {', '.join(unknown)}")


def _term(name: str, label: str, family: str, order: int = 0) -> Term:
    return Term(name=name, label=label, rust_name=name, family=family, order=order)


DENSITY_TERMS: tuple[Term, ...] = (
    _term("source_cross", "S_cross", "source_cross"),
    _term("neg_div_j", "-div(J)", "neg_div_j"),
    _term("lap_rho", "lap(rho)", "lap_rho", 1),
)
