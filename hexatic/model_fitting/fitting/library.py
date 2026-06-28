"""Hydrodynamic model libraries for density and polarization equations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fields import HydrodynamicFields


@dataclass(frozen=True)
class ScalarLibrary:
    names: tuple[str, ...]
    labels: tuple[str, ...]
    values: np.ndarray  # (transitions, nx, ntheta, terms)


@dataclass(frozen=True)
class VectorLibrary:
    names: tuple[str, ...]
    labels: tuple[str, ...]
    values: np.ndarray  # (transitions, nx, ntheta, terms, 2)


DENSITY_TERM_NAMES = (
    "minus_div_P",
    "minus_div_chiral_P_perp",
    "laplacian_rho",
    "laplacian_hexatic_order",
    "laplacian_D",
)
DENSITY_TERM_LABELS = (
    "-div P",
    "-div(chirality P_perp)",
    "laplacian rho",
    "laplacian hexatic_order",
    "laplacian D",
)

POLARIZATION_TERM_NAMES = (
    "P",
    "chiral_P_perp",
    "grad_rho",
    "grad_D",
    "grad_hexatic_order",
    "D_P",
    "D_chiral_P_perp",
    "P_norm2_P",
    "P_dot_grad_P",
    "P_perp_dot_grad_P",
    "laplacian_P",
    "laplacian_P_perp",
)
POLARIZATION_TERM_LABELS = (
    "P",
    "chirality P_perp",
    "grad rho",
    "grad D",
    "grad hexatic_order",
    "D P",
    "D chirality P_perp",
    "|P|^2 P",
    "(P · grad)P",
    "(P_perp · grad)P",
    "laplacian P",
    "laplacian P_perp",
)


def build_density_library(fields: HydrodynamicFields) -> ScalarLibrary:
    values = np.stack(
        (
            -fields.div_P,
            -fields.div_chiral_P_perp,
            fields.laplacian_rho,
            fields.laplacian_hexatic_order,
            fields.laplacian_D,
        ),
        axis=-1,
    )
    return ScalarLibrary(DENSITY_TERM_NAMES, DENSITY_TERM_LABELS, values)


def build_polarization_library(fields: HydrodynamicFields) -> VectorLibrary:
    P = fields.mid_P
    P_perp = np.stack((-P[..., 1], P[..., 0]), axis=-1)
    chiral_P_perp = fields.mid_chirality[..., None] * P_perp
    P_norm2 = np.sum(P * P, axis=-1)

    values = np.stack(
        (
            P,
            chiral_P_perp,
            fields.grad_rho,
            fields.grad_D,
            fields.grad_hexatic_order,
            fields.mid_D[..., None] * P,
            fields.mid_D[..., None] * chiral_P_perp,
            P_norm2[..., None] * P,
            fields.P_dot_grad_P,
            fields.P_perp_dot_grad_P,
            fields.laplacian_P,
            fields.laplacian_P_perp,
        ),
        axis=-2,
    )
    return VectorLibrary(POLARIZATION_TERM_NAMES, POLARIZATION_TERM_LABELS, values)


def density_target(fields: HydrodynamicFields) -> np.ndarray:
    return fields.partial_t_rho - fields.S_cross


def polarization_target(fields: HydrodynamicFields) -> np.ndarray:
    return fields.partial_t_P
