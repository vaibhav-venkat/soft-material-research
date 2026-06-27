from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft

from hexatic.active_matter_cylinder.math_utils import _density_sum

from ..film_continuity.config import FilmContinuityConfig, scalars_from_active_fields
from ..film_continuity.continuity import FilmContinuityResult, compute_film_continuity
from ..film_continuity.io_cache import load_active_matter_fields
from .config import FittingConfig
from .io_cache import load_npz_arrays


RHO_GRID_KEYS = (
    "rho_gaussian",
    "rho_grid",
    "rho_density_grid",
    "rho_film_frame",
    "rho_film_frames",
)
J_GRID_KEYS = ("J_film", "J_film_b", "j_film", "j_film_b")
COUNT_GRID_KEYS = (
    "film_count_per_bin_b",
    "counts",
    "count",
    "counts_b",
)


@dataclass(frozen=True)
class FittingFields:
    transition_steps: np.ndarray
    dt: float
    cylinder_radius: float
    lx: float
    x_edges: np.ndarray
    x_centers: np.ndarray
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    rho: np.ndarray
    J: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray
    grad_x_mid: np.ndarray
    grad_y_mid: np.ndarray
    counts: np.ndarray | None
    pocket_radius: float | None = None


def load_or_compute_fields(config: FittingConfig) -> FittingFields:
    print(f"[fitting] Loading active fields from {config.active_matter_path}...")
    active = load_active_matter_fields(config.active_matter_path)
    print(
        "[fitting] Active fields loaded: "
        f"{active.steps.size} frames, "
        f"{active.coords.shape[1]} particles, "
        f"grid {active.x_edges.size - 1} x {active.theta_edges.size - 1}."
    )
    scalars = scalars_from_active_fields(
        FilmContinuityConfig(
            case_id=config.case_id,
            npz_path=config.npz_path,
            gsd_path=config.gsd_path,
            min_count=config.min_count,
        ),
        steps=active.steps,
        x_edges=active.x_edges,
        theta_edges=active.theta_edges,
        pocket_radius=active.pocket_radius,
    )
    print(
        "[fitting] Scalars resolved: "
        f"R={scalars.cylinder_radius:.6g}, "
        f"Lx={scalars.lx:.6g}, "
        f"dt={scalars.dt:.6g}."
    )
    active_arrays = load_npz_arrays(config.active_matter_path)

    rho = _grid_array_from_keys(
        active_arrays,
        RHO_GRID_KEYS,
        frame_count=active.steps.size,
        nx=scalars.x_centers.size,
        ntheta=scalars.theta_centers.size,
    )
    if rho is None:
        print(
            "[fitting] No cached grid rho found; recomputing Gaussian density "
            "on the film grid..."
        )
        rho = gaussian_rho_frames(
            active.coords,
            active.shell_mask,
            scalars.x_centers,
            scalars.theta_centers,
            scalars.lx,
            scalars.cylinder_radius,
            _pocket_radius(active.pocket_radius, scalars.cylinder_radius),
        )
        print(f"[fitting] Gaussian rho computed with shape {rho.shape}.")
    else:
        print(f"[fitting] Using cached grid rho with shape {rho.shape}.")

    J = _grid_array_from_keys(
        active_arrays,
        J_GRID_KEYS,
        transition_count=active.steps.size - 1,
        nx=scalars.x_centers.size,
        ntheta=scalars.theta_centers.size,
        vector_components=2,
    )
    if J is not None:
        print(f"[fitting] Using cached J grid with shape {J.shape}.")

    counts = _grid_array_from_keys(
        active_arrays,
        COUNT_GRID_KEYS,
        transition_count=active.steps.size - 1,
        nx=scalars.x_centers.size,
        ntheta=scalars.theta_centers.size,
    )
    if counts is not None:
        print(f"[fitting] Using cached bin counts with shape {counts.shape}.")
    if J is None or counts is None:
        print("[fitting] Loading film-continuity fallback for J/counts...")
        film_result = _load_or_compute_film_result(config)
        if J is None:
            J = film_result.J_film_b
            print(f"[fitting] Loaded fallback J grid with shape {J.shape}.")
        if counts is None:
            counts = film_result.film_count_per_bin_b
            print(f"[fitting] Loaded fallback bin counts with shape {counts.shape}.")

    _validate_rho_and_flux(rho, J)
    print("[fitting] Computing FFT gradients...")
    grad_x, grad_y = fft_gradients(
        rho,
        lx=scalars.lx,
        cylinder_radius=scalars.cylinder_radius,
        theta_period=float(scalars.theta_edges[-1] - scalars.theta_edges[0]),
    )
    grad_x_mid = 0.5 * (grad_x[1:] + grad_x[:-1])
    grad_y_mid = 0.5 * (grad_y[1:] + grad_y[:-1])
    print(
        "[fitting] Gradient fields ready: "
        f"grad_x_mid {grad_x_mid.shape}, grad_y_mid {grad_y_mid.shape}."
    )

    return FittingFields(
        transition_steps=np.stack((active.steps[:-1], active.steps[1:]), axis=1),
        dt=scalars.dt,
        cylinder_radius=scalars.cylinder_radius,
        lx=scalars.lx,
        x_edges=scalars.x_edges,
        x_centers=scalars.x_centers,
        theta_edges=scalars.theta_edges,
        theta_centers=scalars.theta_centers,
        rho=rho,
        J=J,
        grad_x=grad_x,
        grad_y=grad_y,
        grad_x_mid=grad_x_mid,
        grad_y_mid=grad_y_mid,
        counts=counts,
        pocket_radius=active.pocket_radius,
    )


def gaussian_rho_frames(
    coords: np.ndarray,
    shell_mask: np.ndarray,
    x_centers: np.ndarray,
    theta_centers: np.ndarray,
    lx: float,
    cylinder_radius: float,
    pocket_radius: float,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    shell_mask = np.asarray(shell_mask, dtype=bool)
    if coords.ndim != 3 or coords.shape[2] < 3:
        raise ValueError("coords must have shape (frames, particles, >=3).")
    if shell_mask.shape != coords.shape[:2]:
        raise ValueError("shell_mask must match coords frame/particle axes.")

    grid_points = _surface_grid_points(x_centers, theta_centers, cylinder_radius)
    rho = np.zeros((coords.shape[0], x_centers.size, theta_centers.size), dtype=float)
    for frame_idx in range(coords.shape[0]):
        print(
            "[fitting]   Gaussian rho frame "
            f"{frame_idx + 1}/{coords.shape[0]}..."
        )
        mask = shell_mask[frame_idx]
        positions = _cylindrical_coords_to_cartesian(coords[frame_idx, mask])
        values = np.ones(positions.shape[0], dtype=float)
        density = _density_sum(
            grid_points,
            positions,
            values,
            lx,
            pocket_radius,
        )
        rho[frame_idx] = density.reshape(x_centers.size, theta_centers.size)
    return rho


def fft_gradients(
    rho: np.ndarray,
    *,
    lx: float,
    cylinder_radius: float,
    theta_period: float = 2.0 * np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 3:
        raise ValueError("rho must have shape (frames, nx, ntheta).")
    if lx <= 0.0:
        raise ValueError("lx must be positive.")
    if cylinder_radius <= 0.0:
        raise ValueError("cylinder_radius must be positive.")
    if theta_period <= 0.0:
        raise ValueError("theta_period must be positive.")

    _, nx, ntheta = rho.shape
    ly = float(cylinder_radius) * float(theta_period)
    kx = 2.0 * np.pi * fft.fftfreq(nx, d=float(lx) / nx)
    ky = 2.0 * np.pi * fft.fftfreq(ntheta, d=ly / ntheta)
    rho_hat = fft.fft2(rho, axes=(1, 2))
    grad_x = fft.ifft2(1j * kx[None, :, None] * rho_hat, axes=(1, 2)).real
    grad_y = fft.ifft2(1j * ky[None, None, :] * rho_hat, axes=(1, 2)).real
    return grad_x, grad_y


def _load_or_compute_film_result(config: FittingConfig) -> FilmContinuityResult:
    if config.film_continuity_cache_path.exists():
        print(
            "[fitting] Using film-continuity cache "
            f"{config.film_continuity_cache_path}."
        )
        return FilmContinuityResult.from_cache_arrays(
            load_npz_arrays(config.film_continuity_cache_path)
        )
    print("[fitting] Film-continuity cache missing; computing fallback fields...")
    return compute_film_continuity(
        FilmContinuityConfig(
            case_id=config.case_id,
            npz_path=config.npz_path,
            gsd_path=config.gsd_path,
            min_count=config.min_count,
        )
    )


def _grid_array_from_keys(
    arrays: dict[str, np.ndarray],
    keys: tuple[str, ...],
    *,
    nx: int,
    ntheta: int,
    frame_count: int | None = None,
    transition_count: int | None = None,
    vector_components: int | None = None,
) -> np.ndarray | None:
    if frame_count is None and transition_count is None:
        raise ValueError("frame_count or transition_count is required.")
    expected_leading = frame_count if frame_count is not None else transition_count
    expected_shape = (expected_leading, nx, ntheta)
    if vector_components is not None:
        expected_shape = expected_shape + (vector_components,)
    for key in keys:
        if key in arrays and arrays[key].shape == expected_shape:
            return np.asarray(arrays[key], dtype=float)
    return None


def _validate_rho_and_flux(rho: np.ndarray, J: np.ndarray) -> None:
    if rho.ndim != 3:
        raise ValueError("rho must have shape (frames, nx, ntheta).")
    if rho.shape[0] < 2:
        raise ValueError("At least two density frames are required.")
    if J.shape != (rho.shape[0] - 1, rho.shape[1], rho.shape[2], 2):
        raise ValueError(
            "J must have shape (frames - 1, nx, ntheta, 2) matching rho."
        )


def _surface_grid_points(
    x_centers: np.ndarray,
    theta_centers: np.ndarray,
    cylinder_radius: float,
) -> np.ndarray:
    x_grid, theta_grid = np.meshgrid(x_centers, theta_centers, indexing="ij")
    return np.column_stack(
        (
            x_grid.ravel(),
            cylinder_radius * np.sin(theta_grid.ravel()),
            cylinder_radius * np.cos(theta_grid.ravel()),
        )
    )


def _cylindrical_coords_to_cartesian(coords: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            coords[:, 0],
            coords[:, 2] * np.sin(coords[:, 1]),
            coords[:, 2] * np.cos(coords[:, 1]),
        )
    )


def _pocket_radius(pocket_radius: float | None, cylinder_radius: float) -> float:
    if pocket_radius is not None and np.isfinite(pocket_radius) and pocket_radius > 0.0:
        return float(pocket_radius)
    return 0.5 * float(cylinder_radius)
