"""Hydrodynamic model libraries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft

from .fields import HydrodynamicFields
from . import operators as ops


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
    "minus_div_P_density",
    "minus_div_chiral_P_density_perp",
    "laplacian_rho",
    "laplacian_hexatic_order",
    "laplacian_D",
)
DENSITY_TERM_LABELS = (
    "-div P_density",
    "-div(chirality P_density_perp)",
    "laplacian rho",
    "laplacian hexatic_order",
    "laplacian D",
)

POLARIZATION_TERM_NAMES = (
    "P_density",
    "chiral_P_density_perp",
    "grad_rho",
    "grad_D",
    "grad_hexatic_order",
    "D_P_density",
    "D_chiral_P_density_perp",
    "P_density_norm2_P_density",
    "P_density_dot_grad_P_density",
    "P_density_perp_dot_grad_P_density",
    "laplacian_P_density",
    "laplacian_P_density_perp",
)
POLARIZATION_TERM_LABELS = (
    "P_density",
    "chirality P_density_perp",
    "grad rho",
    "grad D",
    "grad hexatic_order",
    "D P_density",
    "D chirality P_density_perp",
    "|P_density|^2 P_density",
    "(P_density · grad)P_density",
    "(P_density_perp · grad)P_density",
    "laplacian P_density",
    "laplacian P_density_perp",
)

CURRENT_TERM_NAMES = (
    "P_density",
    "chiral_P_density_perp",
    "force_density",
    "D_P_density",
    "D_chiral_P_density_perp",
    "D_force_density",
)
CURRENT_TERM_LABELS = (
    "P_density",
    "chirality P_density_perp",
    "force_density",
    "D P_density",
    "D chirality P_density_perp",
    "D force_density",
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
    "low_k_P_density",
    "low_k_D_P_density",
    "low_k_P_r_P_density",
    "low_k_force_density",
    "low_k_grad_rho",
    "low_k_grad_hexatic_order",
    "low_k_grad_D",
)
NO_FORCE_LOW_K_TERM_LABELS = (
    "low-k(P_density)",
    "low-k(D P_density)",
    "low-k(P_r P_density)",
    "low-k(f)",
    "low-k(grad rho)",
    "low-k(grad |psi6|)",
    "low-k(grad D)",
)


def build_density_library(fields: HydrodynamicFields) -> ScalarLibrary:
    values = np.stack(
        (
            -fields.div_P_density,
            -fields.div_chiral_P_density_perp,
            fields.laplacian_rho,
            fields.laplacian_hexatic_order,
            fields.laplacian_D,
        ),
        axis=-1,
    )
    return ScalarLibrary(DENSITY_TERM_NAMES, DENSITY_TERM_LABELS, values)


def build_polarization_library(fields: HydrodynamicFields) -> VectorLibrary:
    P_density = fields.mid_P_density
    P_density_perp = np.stack((-P_density[..., 1], P_density[..., 0]), axis=-1)
    chiral_P_density_perp = fields.mid_chirality[..., None] * P_density_perp
    P_norm2 = np.sum(P_density * P_density, axis=-1)

    values = np.stack(
        (
            P_density,
            chiral_P_density_perp,
            fields.grad_rho,
            fields.grad_D,
            fields.grad_hexatic_order,
            fields.mid_D[..., None] * P_density,
            fields.mid_D[..., None] * chiral_P_density_perp,
            P_norm2[..., None] * P_density,
            fields.P_density_dot_grad_P_density,
            fields.P_density_perp_dot_grad_P_density,
            fields.laplacian_P_density,
            fields.laplacian_P_density_perp,
        ),
        axis=-2,
    )
    return VectorLibrary(POLARIZATION_TERM_NAMES, POLARIZATION_TERM_LABELS, values)


def build_current_library(
    fields: HydrodynamicFields,
    *,
    included_bonus_terms: tuple[str, ...] = (),
    rho_N_power: int = 0,
) -> VectorLibrary:
    P_density = fields.mid_P_density
    P_density_perp = np.stack((-P_density[..., 1], P_density[..., 0]), axis=-1)
    chiral_P_density_perp = fields.mid_chirality[..., None] * P_density_perp
    values = [
        P_density,
        chiral_P_density_perp,
        fields.mid_force_density,
        fields.mid_D[..., None] * P_density,
        fields.mid_D[..., None] * chiral_P_density_perp,
        fields.mid_D[..., None] * fields.mid_force_density,
    ]
    names = list(CURRENT_TERM_NAMES)
    labels = list(CURRENT_TERM_LABELS)

    bonus = _current_bonus_terms(
        fields,
        included_bonus_terms,
        rho_N_power=rho_N_power,
    )
    names.extend(bonus.names)
    labels.extend(bonus.labels)
    values.extend(np.moveaxis(bonus.values, -2, 0))

    return VectorLibrary(
        tuple(names),
        tuple(labels),
        np.stack(values, axis=-2),
    )


def current_library_term_names(
    included_bonus_terms: tuple[str, ...] = (),
    rho_N_power: int = 0,
) -> tuple[str, ...]:
    """Return current-library names for cache validation without field arrays."""
    _validate_current_bonus_terms(included_bonus_terms)
    return (
        *CURRENT_TERM_NAMES,
        *_expand_current_bonus_terms(included_bonus_terms, rho_N_power),
    )


def _current_bonus_terms(
    fields: HydrodynamicFields,
    included_bonus_terms: tuple[str, ...],
    *,
    rho_N_power: int = 0,
) -> VectorLibrary:
    _validate_current_bonus_terms(included_bonus_terms)
    all_bonus_terms = _expand_current_bonus_terms(included_bonus_terms, rho_N_power)
    if not all_bonus_terms:
        empty = np.empty((*fields.mid_P_density.shape[:-1], 0, fields.mid_P_density.shape[-1]))
        return VectorLibrary((), (), empty)

    kx, ky = _field_wavenumbers(fields)
    P_density = fields.mid_P_density
    P_density_perp = np.stack((-P_density[..., 1], P_density[..., 0]), axis=-1)
    Q = _nematic_tensor_from_polarization(P_density)
    grad_laplacian_rho = _gradient(fields.laplacian_rho, kx, ky)
    grad_laplacian_hexatic_order = _gradient(fields.laplacian_hexatic_order, kx, ky)
    grad_laplacian_D = _gradient(fields.laplacian_D, kx, ky)

    sources = {
        "Q_dot_P_density": _tensor_dot_vector(Q, P_density),
        "Q_dot_P_density_perp": _tensor_dot_vector(Q, P_density_perp),
        "Q_dot_grad_rho": _tensor_dot_vector(Q, fields.grad_rho),
        "div_Q": _tensor_divergence(Q, kx, ky),
        "minus_grad_hexatic_order": -fields.grad_hexatic_order,
        "minus_grad_hexatic_order2": -_gradient(fields.mid_hexatic_order ** 2, kx, ky),
        "minus_grad_D": -fields.grad_D,
        "minus_grad_D2": -_gradient(fields.mid_D ** 2, kx, ky),
        "minus_grad_laplacian_rho": -grad_laplacian_rho,
        "minus_grad_laplacian_hexatic_order": -grad_laplacian_hexatic_order,
        "minus_grad_laplacian_D": -grad_laplacian_D,
    }
    labels = {
        "Q_dot_P_density": "Q dot P_density",
        "Q_dot_P_density_perp": "Q dot P_density_perp",
        "Q_dot_grad_rho": "Q dot grad rho",
        "div_Q": "div Q",
        "minus_grad_hexatic_order": "-grad |psi6|",
        "minus_grad_hexatic_order2": "-grad(|psi6|^2)",
        "minus_grad_D": "-grad D",
        "minus_grad_D2": "-grad(D^2)",
        "minus_grad_laplacian_rho": "-grad(laplacian rho)",
        "minus_grad_laplacian_hexatic_order": "-grad(laplacian |psi6|)",
        "minus_grad_laplacian_D": "-grad(laplacian D)",
    }
    for power in range(1, _validate_rho_N_power(rho_N_power) + 1):
        name = _rho_power_term_name(power)
        sources[name] = -_gradient(fields.mid_rho ** power, kx, ky)
        labels[name] = "-grad rho" if power == 1 else f"-grad(rho^{power})"
    return VectorLibrary(
        all_bonus_terms,
        tuple(labels[name] for name in all_bonus_terms),
        np.stack([sources[name] for name in all_bonus_terms], axis=-2),
    )


def _validate_current_bonus_terms(included_bonus_terms: tuple[str, ...]) -> None:
    unknown = set(included_bonus_terms).difference(_CURRENT_BONUS_TERM_NAMES)
    if unknown:
        raise ValueError(
            "included_bonus_terms contains unknown current terms: "
            + ", ".join(sorted(unknown))
        )


_CURRENT_BONUS_TERM_NAMES = frozenset(
    {
        "Q_dot_P_density",
        "Q_dot_P_density_perp",
        "Q_dot_grad_rho",
        "div_Q",
        "minus_grad_hexatic_order",
        "minus_grad_hexatic_order2",
        "minus_grad_D",
        "minus_grad_D2",
        "minus_grad_laplacian_rho",
        "minus_grad_laplacian_hexatic_order",
        "minus_grad_laplacian_D",
    }
)


def _expand_current_bonus_terms(
    included_bonus_terms: tuple[str, ...],
    rho_N_power: int,
) -> tuple[str, ...]:
    _validate_current_bonus_terms(included_bonus_terms)
    rho_terms = tuple(
        _rho_power_term_name(power)
        for power in range(1, _validate_rho_N_power(rho_N_power) + 1)
    )
    expanded: list[str] = []
    inserted_rho_terms = False
    for term in included_bonus_terms:
        expanded.append(term)
        if term == "div_Q":
            expanded.extend(rho_terms)
            inserted_rho_terms = True
    if not inserted_rho_terms:
        expanded.extend(rho_terms)
    return tuple(expanded)


def _validate_rho_N_power(rho_N_power: int) -> int:
    power = int(rho_N_power)
    if power < 0:
        raise ValueError("rho_N_power must be non-negative.")
    return power


def _rho_power_term_name(power: int) -> str:
    return "minus_grad_rho" if power == 1 else f"minus_grad_rho{power}"


def _field_wavenumbers(fields: HydrodynamicFields) -> tuple[np.ndarray, np.ndarray]:
    ly = fields.cylinder_radius * (fields.theta_edges[-1] - fields.theta_edges[0])
    return ops.build_k_vectors(
        fields.x_centers.size,
        fields.theta_centers.size,
        fields.lx,
        ly,
    )


def _gradient(field: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    grad_x, grad_y = ops.fft_gradient(field, kx, ky)
    return np.stack((grad_x, grad_y), axis=-1)


def _nematic_tensor_from_polarization(P_density: np.ndarray) -> np.ndarray:
    P_norm2 = np.sum(P_density * P_density, axis=-1)
    Q = P_density[..., :, None] * P_density[..., None, :]
    Q[..., 0, 0] -= 0.5 * P_norm2
    Q[..., 1, 1] -= 0.5 * P_norm2
    return Q


def _tensor_dot_vector(tensor: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...j->...i", tensor, vector)


def _tensor_divergence(tensor: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    dQxx_dx, _ = ops.fft_gradient(tensor[..., 0, 0], kx, ky)
    _, dQxy_dy = ops.fft_gradient(tensor[..., 0, 1], kx, ky)
    dQyx_dx, _ = ops.fft_gradient(tensor[..., 1, 0], kx, ky)
    _, dQyy_dy = ops.fft_gradient(tensor[..., 1, 1], kx, ky)
    return np.stack((dQxx_dx + dQxy_dy, dQyx_dx + dQyy_dy), axis=-1)


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
        fields.mid_P_density,
        fields.mid_D[..., None] * fields.mid_P_density,
        fields.mid_P_r[..., None] * fields.mid_P_density,
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
        values.append(_low_k_fourier_modes(source, fields.mid_P_density))

    if not values:
        empty = np.empty((*fields.mid_P_density.shape[:-1], 0, fields.mid_P_density.shape[-1]))
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
    return fields.partial_t_P_density


def current_target(fields: HydrodynamicFields) -> np.ndarray:
    return fields.material_current
