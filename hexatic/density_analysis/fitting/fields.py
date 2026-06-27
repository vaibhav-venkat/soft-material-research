from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft

from hexatic.active_matter_cylinder.math_utils import _density_sum
from hexatic.chirality.compute import compute_chirality_fields
from hexatic.chirality.config import ChiralityConfig

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
    J: np.ndarray
    frame_fields: dict[str, np.ndarray]
    mid_fields: dict[str, np.ndarray]
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
    P = polarization_grid_frames(
        active.coords,
        active.shell_mask,
        active_arrays,
        scalars.x_edges,
        scalars.theta_edges,
    )
    force_density = force_density_grid_frames(
        active.coords,
        active.shell_mask,
        active_arrays,
        scalars.x_edges,
        scalars.theta_edges,
    )
    chirality = load_or_compute_chirality_frames(
        config,
        scalars.x_centers.size,
        scalars.theta_centers.size,
    )
    _validate_frame_field_steps(
        "chirality",
        chirality.shape[0],
        active.steps.size,
    )

    frame_fields = {
        "rho": rho,
        "grad_rho": np.stack((grad_x, grad_y), axis=-1),
        "P": P,
        "force_density": force_density,
        "chirality": chirality,
    }
    chiral_P_perp = np.stack(
        (
            chirality * (-P[..., 1]),
            chirality * P[..., 0],
        ),
        axis=-1,
    )
    frame_fields["chiral_P_perp"] = chiral_P_perp
    mid_fields = {
        "grad_rho": _mid(frame_fields["grad_rho"]),
        "P": _mid(frame_fields["P"]),
        "chiral_P_perp": _mid(frame_fields["chiral_P_perp"]),
        "force_density": _mid(frame_fields["force_density"]),
    }
    print(
        "[fitting] Gradient fields ready: "
        f"grad_rho {mid_fields['grad_rho'].shape}."
    )
    print(
        "[fitting] Candidate fields ready: "
        + ", ".join(f"{name} {field.shape}" for name, field in mid_fields.items())
        + "."
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
        J=J,
        frame_fields=frame_fields,
        mid_fields=mid_fields,
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


def polarization_grid_frames(
    coords: np.ndarray,
    shell_mask: np.ndarray,
    arrays: dict[str, np.ndarray],
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> np.ndarray:
    if "polar_mean" not in arrays or "polar_cylindrical" not in arrays:
        raise KeyError("active matter fields must include polar_mean and polar_cylindrical.")
    return _particle_cylindrical_vector_grid_frames(
        coords,
        shell_mask,
        cartesian_values=arrays["polar_mean"],
        cylindrical_values=arrays["polar_cylindrical"],
        x_edges=x_edges,
        theta_edges=theta_edges,
        field_name="polarization",
    )


def force_density_grid_frames(
    coords: np.ndarray,
    shell_mask: np.ndarray,
    arrays: dict[str, np.ndarray],
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> np.ndarray:
    if "force_density" not in arrays or "force_density_cylindrical" not in arrays:
        raise KeyError(
            "active matter fields must include force_density and "
            "force_density_cylindrical."
        )
    return _particle_cylindrical_vector_grid_frames(
        coords,
        shell_mask,
        cartesian_values=arrays["force_density"],
        cylindrical_values=arrays["force_density_cylindrical"],
        x_edges=x_edges,
        theta_edges=theta_edges,
        field_name="force_density",
    )


def _particle_cylindrical_vector_grid_frames(
    coords: np.ndarray,
    shell_mask: np.ndarray,
    *,
    cartesian_values: np.ndarray,
    cylindrical_values: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    field_name: str,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    shell_mask = np.asarray(shell_mask, dtype=bool)
    cartesian_values = np.asarray(cartesian_values, dtype=float)
    cylindrical_values = np.asarray(cylindrical_values, dtype=float)
    if cartesian_values.shape != coords.shape or cylindrical_values.shape != coords.shape:
        raise ValueError(f"{field_name} arrays must match coords shape.")
    if shell_mask.shape != coords.shape[:2]:
        raise ValueError("shell_mask must match coords frame/particle axes.")

    frames, _, _ = coords.shape
    nx = len(x_edges) - 1
    ntheta = len(theta_edges) - 1
    field = np.full((frames, nx, ntheta, 2), np.nan, dtype=float)
    lx = float(x_edges[-1] - x_edges[0])
    for frame_idx in range(frames):
        values = np.column_stack(
            (
                cartesian_values[frame_idx, :, 0],
                cylindrical_values[frame_idx, :, 2],
            )
        )
        field[frame_idx] = _bin_vector_mean(
            coords[frame_idx],
            values,
            shell_mask[frame_idx],
            x_edges,
            theta_edges,
            lx,
        )
    return np.nan_to_num(field, nan=0.0)


def load_or_compute_chirality_frames(
    config: FittingConfig,
    nx: int,
    ntheta: int,
    metric_name: str = "instant_helix_relative",
) -> np.ndarray:
    if config.chirality_fields_path.exists():
        print(f"[fitting] Loading chirality fields from {config.chirality_fields_path}...")
        arrays = load_npz_arrays(config.chirality_fields_path)
        return _chirality_from_arrays(arrays, metric_name, nx, ntheta)

    print(
        "[fitting] Chirality fields cache missing; recomputing "
        f"{metric_name!r} on the fitting grid..."
    )
    fields = compute_chirality_fields(
        config.trajectory_path,
        config=ChiralityConfig(n_x_bins=nx, n_theta_bins=ntheta),
    )
    metric_index = {name: idx for idx, name in enumerate(fields.metric_names)}
    if metric_name not in metric_index:
        raise KeyError(f"Computed chirality fields do not include {metric_name!r}.")
    return np.nan_to_num(
        np.asarray(fields.xtheta_values[metric_index[metric_name]], dtype=float),
        nan=0.0,
    )


def _chirality_from_arrays(
    arrays: dict[str, np.ndarray],
    metric_name: str,
    nx: int,
    ntheta: int,
) -> np.ndarray:
    if "xtheta_values" not in arrays or "metric_names" not in arrays:
        raise KeyError("chirality npz must include xtheta_values and metric_names.")
    metric_names = tuple(str(name) for name in arrays["metric_names"])
    if metric_name not in metric_names:
        raise KeyError(f"chirality npz does not include metric {metric_name!r}.")
    values = np.asarray(arrays["xtheta_values"], dtype=float)
    field = values[metric_names.index(metric_name)]
    if field.ndim != 3 or field.shape[1:] != (nx, ntheta):
        raise ValueError(
            f"chirality field shape {field.shape} does not match grid (*, {nx}, {ntheta})."
        )
    return np.nan_to_num(field, nan=0.0)


def _bin_vector_mean(
    coords: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    lx: float,
) -> np.ndarray:
    nx = len(x_edges) - 1
    ntheta = len(theta_edges) - 1
    result = np.zeros((nx, ntheta, 2), dtype=float)
    counts = np.zeros((nx, ntheta), dtype=np.int64)
    finite = (
        mask
        & np.isfinite(coords[:, 0])
        & np.isfinite(coords[:, 1])
        & np.all(np.isfinite(values), axis=1)
    )
    if not np.any(finite):
        return result

    x0 = float(x_edges[0])
    x_values = ((coords[finite, 0] - x0) % lx) + x0
    x_idx = np.searchsorted(x_edges, x_values, side="right") - 1
    x_idx = np.clip(x_idx, 0, nx - 1)
    theta_period = float(theta_edges[-1] - theta_edges[0])
    theta0 = float(theta_edges[0])
    theta_values = ((coords[finite, 1] - theta0) % theta_period) + theta0
    theta_idx = np.searchsorted(theta_edges, theta_values, side="right") - 1
    theta_idx = np.clip(theta_idx, 0, ntheta - 1)

    np.add.at(counts, (x_idx, theta_idx), 1)
    for component in range(2):
        np.add.at(result[..., component], (x_idx, theta_idx), values[finite, component])
        result[..., component] = np.divide(
            result[..., component],
            counts,
            out=np.zeros((nx, ntheta), dtype=float),
            where=counts > 0,
        )
    return result


def _mid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape[0] < 2:
        raise ValueError("At least two frames are required for midpoint fields.")
    return 0.5 * (values[1:] + values[:-1])


def _validate_frame_field_steps(name: str, frame_count: int, expected: int) -> None:
    if frame_count != expected:
        raise ValueError(
            f"{name} has {frame_count} frames, expected {expected} to match active fields."
        )


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
