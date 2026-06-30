"""Candidate library registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Term:
    name: str
    label: str
    kind: str
    rust_name: str
    requires: tuple[str, ...] = ()


DENSITY_TERMS: tuple[Term, ...] = (
    Term("div_p", "div(p)", "scalar", "div_p", ("p",)),
    Term("lap_rho", "lap(rho)", "scalar", "lap_rho", ("rho",)),
    Term("div_rho_p", "div(rho p)", "scalar", "div_rho_p", ("rho", "p")),
    Term("lap_rho2", "lap(rho^2)", "scalar", "lap_rho2", ("rho",)),
    Term("lap_p_norm2", "lap(|p|^2)", "scalar", "lap_p_norm2", ("p",)),
    Term("div_rho2_p", "div(rho^2 p)", "scalar", "div_rho2_p", ("rho", "p")),
    Term("lap_rho3", "lap(rho^3)", "scalar", "lap_rho3", ("rho",)),
    Term("div_p_norm2_p", "div(|p|^2 p)", "scalar", "div_p_norm2_p", ("p",)),
    Term("div_rho_grad_p_norm2", "div(rho grad(|p|^2))", "scalar", "div_rho_grad_p_norm2", ("rho", "p")),
    Term("div_p_norm2_grad_rho", "div(|p|^2 grad(rho))", "scalar", "div_p_norm2_grad_rho", ("rho", "p")),
    Term("div_p_perp", "div(p_perp)", "scalar", "div_p_perp", ("p",)),
    Term("div_rho_p_perp", "div(rho p_perp)", "scalar", "div_rho_p_perp", ("rho", "p")),
    Term("div_rho2_p_perp", "div(rho^2 p_perp)", "scalar", "div_rho2_p_perp", ("rho", "p")),
    Term("div_p_norm2_p_perp", "div(|p|^2 p_perp)", "scalar", "div_p_norm2_p_perp", ("p",)),
)

POLARIZATION_TERMS: tuple[Term, ...] = (
    Term("p", "p", "vector", "p", ("p",)),
    Term("rho_p", "rho p", "vector", "rho_p", ("rho", "p")),
    Term("p_perp", "p_perp", "vector", "p_perp", ("p",)),
    Term("rho_p_perp", "rho p_perp", "vector", "rho_p_perp", ("rho", "p")),
    Term("p_norm2_p", "|p|^2 p", "vector", "p_norm2_p", ("p",)),
    Term("p_norm2_p_perp", "|p|^2 p_perp", "vector", "p_norm2_p_perp", ("p",)),
    Term("grad_rho", "grad(rho)", "vector", "grad_rho", ("rho",)),
    Term("p_dot_grad_p", "(p . grad) p", "vector", "p_dot_grad_p", ("p",)),
    Term("p_dot_grad_p_perp", "(p . grad) p_perp", "vector", "p_dot_grad_p_perp", ("p",)),
    Term("p_perp_dot_grad_p", "(p_perp . grad) p", "vector", "p_perp_dot_grad_p", ("p",)),
    Term("grad_div_p", "grad(div(p))", "vector", "grad_div_p", ("p",)),
    Term("grad_div_p_perp", "grad(div(p_perp))", "vector", "grad_div_p_perp", ("p",)),
    Term("lap_p", "lap(p)", "vector", "lap_p", ("p",)),
    Term("lap_p_perp", "lap(p_perp)", "vector", "lap_p_perp", ("p",)),
    Term("grad_p_norm2", "grad(|p|^2)", "vector", "grad_p_norm2", ("p",)),
    Term("div_p_p", "div(p) p", "vector", "div_p_p", ("p",)),
    Term("div_p_p_perp", "div(p) p_perp", "vector", "div_p_p_perp", ("p",)),
    Term("bilap_p", "bilap(p)", "vector", "bilap_p", ("p",)),
    Term("bilap_p_perp", "bilap(p_perp)", "vector", "bilap_p_perp", ("p",)),
)


def validate_terms(names: tuple[str, ...], registry: tuple[Term, ...]) -> None:
    known = {term.name for term in registry}
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"unknown terms: {', '.join(unknown)}")
