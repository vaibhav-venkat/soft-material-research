"""Hydrodynamic model libraries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft

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

CURRENT_TERM_NAMES = (
    "P",
    "chiral_P_perp",
    "force_density",
    "D_P",
    "D_chiral_P_perp",
    "D_force_density",
    "minus_grad_rho",
    "minus_grad_hexatic_order",
    "minus_grad_D",
)
CURRENT_TERM_LABELS = (
    "P",
    "chirality P_perp",
    "force_density",
    "D P",
    "D chirality P_perp",
    "D force_density",
    "-grad rho",
    "-grad hexatic_order",
    "-grad D",
)

S_CROSS_TERM_NAMES = (
    "constant",
    "rho",
    "laplacian_rho",
    "D",
    "hexatic_order",
    "P_r",
    "h",
    "h_rho",
    "h_P_r",
    "D_P_r",
    "D_rho",
)
S_CROSS_TERM_LABELS = (
    "1",
    "rho",
    "laplacian rho",
    "D",
    "|psi6|",
    "P_r",
    "h",
    "h rho",
    "h P_r",
    "D P_r",
    "D rho",
)

NO_FORCE_LOW_K_TERM_NAMES = (
    "low_k_P",
    "low_k_D_P",
    "low_k_P_r_P",
    "low_k_force_density",
    "low_k_grad_rho",
    "low_k_grad_hexatic_order",
    "low_k_grad_D",
)
NO_FORCE_LOW_K_TERM_LABELS = (
    "low-k(P)",
    "low-k(D P)",
    "low-k(P_r P)",
    "low-k(f)",
    "low-k(grad rho)",
    "low-k(grad |psi6|)",
    "low-k(grad D)",
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


def build_current_library(fields: HydrodynamicFields) -> VectorLibrary:
    P = fields.mid_P
    P_perp = np.stack((-P[..., 1], P[..., 0]), axis=-1)
    chiral_P_perp = fields.mid_chirality[..., None] * P_perp
    values = np.stack(
        (
            P,
            chiral_P_perp,
            fields.mid_force_density,
            fields.mid_D[..., None] * P,
            fields.mid_D[..., None] * chiral_P_perp,
            fields.mid_D[..., None] * fields.mid_force_density,
            -fields.grad_rho,
            -fields.grad_hexatic_order,
            -fields.grad_D,
        ),
        axis=-2,
    )
    return VectorLibrary(CURRENT_TERM_NAMES, CURRENT_TERM_LABELS, values)


def build_s_cross_library(fields: HydrodynamicFields) -> ScalarLibrary:
    """Build the scalar source library for S_cross."""
    ones = np.ones_like(fields.S_cross, dtype=float)
    values = np.stack(
        (
            ones,
            fields.mid_rho,
            fields.laplacian_rho,
            fields.mid_D,
            fields.mid_hexatic_order,
            fields.mid_P_r,
            fields.mid_h,
            fields.mid_h * fields.mid_rho,
            fields.mid_h * fields.mid_P_r,
            fields.mid_D * fields.mid_P_r,
            fields.mid_D * fields.mid_rho,
        ),
        axis=-1,
    )
    return ScalarLibrary(S_CROSS_TERM_NAMES, S_CROSS_TERM_LABELS, values)


def build_no_force_low_k_library(
    fields: HydrodynamicFields,
    *,
    drop_terms: tuple[str, ...] = (),
) -> VectorLibrary:
    """Build low-k fields used only by the no-force residual-split current fit."""
    drop_set = set(drop_terms)
    unknown = drop_set.difference(NO_FORCE_LOW_K_TERM_NAMES)
    if unknown:
        raise ValueError(
            "unknown no-force low-k terms to drop: "
            + ", ".join(sorted(unknown))
        )

    sources = (
        fields.mid_P,
        fields.mid_D[..., None] * fields.mid_P,
        fields.mid_P_r[..., None] * fields.mid_P,
        fields.mid_force_density,
        fields.grad_rho,
        fields.grad_hexatic_order,
        fields.grad_D,
    )
    values = []
    names = []
    labels = []
    for name, label, source in zip(
        NO_FORCE_LOW_K_TERM_NAMES,
        NO_FORCE_LOW_K_TERM_LABELS,
        sources,
        strict=True,
    ):
        if name in drop_set:
            continue
        names.append(name)
        labels.append(label)
        values.append(_low_k_fourier_modes(source, fields.mid_P))

    if not values:
        empty = np.empty((*fields.mid_P.shape[:-1], 0, fields.mid_P.shape[-1]))
        return VectorLibrary(tuple(names), tuple(labels), empty)
    return VectorLibrary(tuple(names), tuple(labels), np.stack(values, axis=-2))


def _low_k_fourier_modes(
    vec_field: np.ndarray | None,
    template: np.ndarray,
    *,
    max_mode: int = 1,
) -> np.ndarray:
    """Return the nonzero low-frequency Fourier reconstruction of a vector field."""
    if vec_field is None:
        return np.zeros_like(template)

    vec_field = np.asarray(vec_field, dtype=float)
    if vec_field.shape != template.shape:
        raise ValueError(
            "low-k Fourier mode field must match current-library vector shape "
            f"{template.shape}, got {vec_field.shape}."
        )

    nx = vec_field.shape[1]
    ntheta = vec_field.shape[2]
    x_modes = fft.fftfreq(nx) * nx
    theta_modes = fft.fftfreq(ntheta) * ntheta
    low_k = (
        (np.abs(x_modes[:, None]) <= max_mode)
        & (np.abs(theta_modes[None, :]) <= max_mode)
    )
    low_k &= ~((x_modes[:, None] == 0.0) & (theta_modes[None, :] == 0.0))

    filtered = np.empty_like(vec_field)
    for component in range(vec_field.shape[-1]):
        field_hat = fft.fft2(vec_field[..., component], axes=(1, 2))
        field_hat *= low_k[None, :, :]
        filtered[..., component] = fft.ifft2(field_hat, axes=(1, 2)).real
    return filtered


def density_target(fields: HydrodynamicFields) -> np.ndarray:
    return fields.partial_t_rho - fields.S_cross


def polarization_target(fields: HydrodynamicFields) -> np.ndarray:
    return fields.partial_t_P


def current_target(fields: HydrodynamicFields) -> np.ndarray:
    return fields.material_current
