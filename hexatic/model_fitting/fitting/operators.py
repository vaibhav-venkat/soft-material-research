"""Fourier spectral derivative operators for periodic (x, y=Rtheta) cylinder grids."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import fft


def build_k_vectors(
    nx: int, ntheta: int, lx: float, ly: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return kx of shape (nx,) and ky of shape (ntheta,) using scipy.fft.fftfreq.

    k = 2*pi*fftfreq(N, d=L/N) so that k[i]*x gives the correct phase on [0, L).
    """
    kx = 2.0 * np.pi * fft.fftfreq(nx, d=float(lx) / nx)
    ky = 2.0 * np.pi * fft.fftfreq(ntheta, d=float(ly) / ntheta)
    return kx, ky


def fft_gradient(
    field: np.ndarray, kx: np.ndarray, ky: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (d/dx, d/dy) of scalar field (T, nx, ntheta) via FFT.

    Returns (grad_x, grad_y), each same shape as *field*.
    Assumes periodic boundary conditions in both axes.
    """
    validate_grid_shape(field, expected_ndim=3, name="field")
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    field_hat = fft.fft2(field, axes=(1, 2))
    grad_x = fft.ifft2(1j * kx[None, :, None] * field_hat, axes=(1, 2)).real
    grad_y = fft.ifft2(1j * ky[None, None, :] * field_hat, axes=(1, 2)).real
    return grad_x, grad_y


def fft_divergence(
    vec_field: np.ndarray, kx: np.ndarray, ky: np.ndarray
) -> np.ndarray:
    """Compute div V = dV_x/dx + dV_y/dy for vec_field (T, nx, ntheta, 2).

    Returns scalar field (T, nx, ntheta).
    """
    if vec_field.ndim != 4 or vec_field.shape[3] != 2:
        raise ValueError(
            f"vec_field must have shape (T, nx, ntheta, 2), got {vec_field.shape}."
        )
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    vx_hat = fft.fft2(vec_field[..., 0], axes=(1, 2))
    vy_hat = fft.fft2(vec_field[..., 1], axes=(1, 2))
    div = fft.ifft2(
        1j * kx[None, :, None] * vx_hat + 1j * ky[None, None, :] * vy_hat,
        axes=(1, 2),
    ).real
    return div


def fft_laplacian(
    field: np.ndarray, kx: np.ndarray, ky: np.ndarray
) -> np.ndarray:
    """Compute Laplacian nabla^2 field via -|k|^2 * field_hat in Fourier space.

    Input field shape (T, nx, ntheta), output same shape.
    """
    validate_grid_shape(field, expected_ndim=3, name="field")
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    ksq = kx[np.newaxis, :, np.newaxis] ** 2 + ky[np.newaxis, np.newaxis, :] ** 2
    field_hat = fft.fft2(field, axes=(1, 2))
    return fft.ifft2(-ksq * field_hat, axes=(1, 2)).real


def fft_vector_laplacian(
    vec_field: np.ndarray, kx: np.ndarray, ky: np.ndarray
) -> np.ndarray:
    """Apply fft_laplacian to each component of vec_field (T, nx, ntheta, 2).

    Returns same-shape array.
    """
    if vec_field.ndim != 4 or vec_field.shape[3] != 2:
        raise ValueError(
            f"vec_field must have shape (T, nx, ntheta, 2), got {vec_field.shape}."
        )
    lap_x = fft_laplacian(vec_field[..., 0], kx, ky)
    lap_y = fft_laplacian(vec_field[..., 1], kx, ky)
    return np.stack((lap_x, lap_y), axis=-1)


def fft_directional_derivative(
    v: np.ndarray, kx: np.ndarray, ky: np.ndarray
) -> np.ndarray:
    """Compute (v . grad) v = (v_x d/dx + v_y d/dy) v for vector field v.

    Input v shape (T, nx, ntheta, 2), output same shape.
    Uses fft_gradient on each component to get partial derivatives,
    then combines: result_x = v_x * dv_x/dx + v_y * dv_x/dy,
                  result_y = v_x * dv_y/dx + v_y * dv_y/dy.
    """
    if v.ndim != 4 or v.shape[3] != 2:
        raise ValueError(
            f"v must have shape (T, nx, ntheta, 2), got {v.shape}."
        )
    dvx_dx, dvx_dy = fft_gradient(v[..., 0], kx, ky)
    dvy_dx, dvy_dy = fft_gradient(v[..., 1], kx, ky)
    result_x = v[..., 0] * dvx_dx + v[..., 1] * dvx_dy
    result_y = v[..., 0] * dvy_dx + v[..., 1] * dvy_dy
    return np.stack((result_x, result_y), axis=-1)


def fft_curl(
    vec_field: np.ndarray, kx: np.ndarray, ky: np.ndarray
) -> np.ndarray:
    """Curl of 2D vector field: dV_y/dx - dV_x/dy.

    Input shape (T, nx, ntheta, 2), output scalar (T, nx, ntheta).
    """
    dVy_dx, dVy_dy = fft_gradient(vec_field[..., 1], kx, ky)
    dVx_dx, dVx_dy = fft_gradient(vec_field[..., 0], kx, ky)
    return dVy_dx - dVx_dy



def validate_grid_shape(
    field: Any, *, expected_ndim: int = 3, name: str = "field"
) -> None:
    field = np.asarray(field)
    if field.ndim != expected_ndim:
        raise ValueError(
            f"{name} must have {expected_ndim} dimensions, got {field.ndim} "
            f"with shape {field.shape}."
        )
