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
    _term("div_p", "div(P)", "div_p"),
    _term("lap_rho", "lap(rho)", "lap_rho", 1),
    _term("lap_rho2", "lap(rho^2)", "lap_rho_power", 2),
    _term("lap_rho3", "lap(rho^3)", "lap_rho_power", 3),
    _term("div_rho_p", "div(rho P)", "div_rho_p", 1),
    _term("div_rho2_p", "div(rho^2 P)", "div_rho_p", 2),
    _term("div_p_norm2_p", "div(|P|^2 P)", "div_p_norm2_p"),
    _term("div_p_perp", "div(P_perp)", "div_p_perp"),
    _term("div_rho_p_perp", "div(rho P_perp)", "div_rho_p_perp", 1),
    _term("div_rho2_p_perp", "div(rho^2 P_perp)", "div_rho_p_perp", 2),
    _term("div_p_norm2_p_perp", "div(|P|^2 P_perp)", "div_p_norm2_p_perp"),
)
