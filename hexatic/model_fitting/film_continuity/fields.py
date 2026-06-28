from __future__ import annotations

import numpy as np


def rho_film(counts: np.ndarray, area_bin: np.ndarray) -> np.ndarray:
    return np.asarray(counts, dtype=float) / np.asarray(area_bin, dtype=float)[None, :, :]


def J_film(sum_vx: np.ndarray, sum_vy: np.ndarray, area_bin: np.ndarray) -> np.ndarray:
    area = np.asarray(area_bin, dtype=float)[None, :, :]
    jx = np.asarray(sum_vx, dtype=float) / area
    jy = np.asarray(sum_vy, dtype=float) / area
    return np.stack((jx, jy), axis=-1)


def J_film_from_face_crossings(
    x_face_crossings: np.ndarray,
    theta_face_crossings: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    cylinder_radius: float,
    dt: float,
) -> np.ndarray:
    """Return cell-centered flux from finite-volume face crossing counts."""
    x_faces = np.asarray(x_face_crossings, dtype=float)
    theta_faces = np.asarray(theta_face_crossings, dtype=float)
    if x_faces.shape != theta_faces.shape:
        raise ValueError("x and theta face crossing arrays must have matching shapes.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    dx = np.diff(np.asarray(x_edges, dtype=float))
    dtheta = np.diff(np.asarray(theta_edges, dtype=float))
    if x_faces.shape[1:] != (dx.size, dtheta.size):
        raise ValueError("face crossing arrays must have shape (transitions, nx, ntheta).")

    x_face_flux = x_faces / (dt * float(cylinder_radius) * dtheta[None, None, :])
    theta_face_flux = theta_faces / (dt * dx[None, :, None])
    jx = 0.5 * (np.roll(x_face_flux, 1, axis=1) + x_face_flux)
    jy = 0.5 * (np.roll(theta_face_flux, 1, axis=2) + theta_face_flux)
    return np.stack((jx, jy), axis=-1)


def partial_t_rho(rho_frame: np.ndarray, dt: float) -> np.ndarray:
    rho_frame = np.asarray(rho_frame, dtype=float)
    if rho_frame.shape[0] < 2:
        raise ValueError("At least two density frames are required.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    return (rho_frame[1:] - rho_frame[:-1]) / dt


def neg_div_J(
    j_film: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    cylinder_radius: float,
) -> np.ndarray:
    j_film = np.asarray(j_film, dtype=float)
    if j_film.ndim != 4 or j_film.shape[-1] != 2:
        raise ValueError("j_film must have shape (transitions, nx, ntheta, 2).")
    d_jx_dx = _cyclic_central_derivative(
        j_film[..., 0],
        _centers_from_edges(np.asarray(x_edges, dtype=float)),
        axis=1,
        period=float(x_edges[-1] - x_edges[0]),
    )
    d_jy_dtheta = _cyclic_central_derivative(
        j_film[..., 1],
        _centers_from_edges(np.asarray(theta_edges, dtype=float)),
        axis=2,
        period=float(theta_edges[-1] - theta_edges[0]),
    )
    return -d_jx_dx - (d_jy_dtheta / float(cylinder_radius))


def neg_div_J_from_face_crossings(
    x_face_crossings: np.ndarray,
    theta_face_crossings: np.ndarray,
    area_bin: np.ndarray,
    dt: float,
) -> np.ndarray:
    x_faces = np.asarray(x_face_crossings, dtype=float)
    theta_faces = np.asarray(theta_face_crossings, dtype=float)
    area = np.asarray(area_bin, dtype=float)
    if x_faces.shape != theta_faces.shape:
        raise ValueError("x and theta face crossing arrays must have matching shapes.")
    if x_faces.shape[1:] != area.shape:
        raise ValueError("face crossing arrays must match area_bin on spatial axes.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    net_count_change = (
        np.roll(x_faces, 1, axis=1)
        - x_faces
        + np.roll(theta_faces, 1, axis=2)
        - theta_faces
    )
    return net_count_change / (area[None, :, :] * dt)


def S_cross(
    n_in: np.ndarray,
    n_out: np.ndarray,
    area_bin: np.ndarray,
    dt: float,
) -> np.ndarray:
    return (np.asarray(n_in, dtype=float) - np.asarray(n_out, dtype=float)) / (
        np.asarray(area_bin, dtype=float)[None, :, :] * dt
    )


def _centers_from_edges(edges: np.ndarray) -> np.ndarray:
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("edges must be a one-dimensional edge array.")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("edges must be strictly increasing.")
    return 0.5 * (edges[:-1] + edges[1:])


def _cyclic_central_derivative(
    values: np.ndarray,
    centers: np.ndarray,
    *,
    axis: int,
    period: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centers = np.asarray(centers, dtype=float)
    n_bins = values.shape[axis]
    if centers.shape != (n_bins,):
        raise ValueError("centers length must match the selected values axis.")

    forward = np.roll(values, -1, axis=axis)
    backward = np.roll(values, 1, axis=axis)
    next_centers = np.roll(centers, -1)
    prev_centers = np.roll(centers, 1)
    spacing = np.mod(next_centers - prev_centers, period)
    spacing = np.where(spacing == 0.0, period, spacing)
    shape = [1] * values.ndim
    shape[axis] = n_bins
    return (forward - backward) / spacing.reshape(shape)
