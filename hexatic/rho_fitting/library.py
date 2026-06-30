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
    terms = [_rho_power_term(power) for power in range(1, n_rho_power + 1)]
    terms.extend(_rho_lap_term(order) for order in range(1, n_rho_lap_power + 1))
    return tuple(terms)


def term_names(terms: tuple[Term, ...]) -> tuple[str, ...]:
    return tuple(term.name for term in terms)


def term_labels(terms: tuple[Term, ...]) -> tuple[str, ...]:
    return tuple(term.label for term in terms)


def validate_terms(names: tuple[str, ...], registry: tuple[Term, ...]) -> None:
    known = {term.name for term in registry}
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"unknown terms: {', '.join(unknown)}")


def _rho_power_term(power: int) -> Term:
    name = "rho" if power == 1 else f"rho{power}"
    label = "rho" if power == 1 else f"rho^{power}"
    return Term(name=name, label=label, rust_name=name, family="rho_power", order=power)


def _rho_lap_term(order: int) -> Term:
    name = "lap_rho" if order == 1 else f"lap{order}_rho"
    label = "rho"
    for _ in range(order):
        label = f"lap({label})"
    return Term(name=name, label=label, rust_name=name, family="rho_lap", order=order)
