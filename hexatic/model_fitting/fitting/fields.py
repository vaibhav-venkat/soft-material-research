"""Build hydrodynamic fields from particle data.

Owns field construction: active-matter loading, chirality loading/computation,
hexatic-order loading, D construction, measured S_cross, smoothed frame fields,
transition fields, time derivatives, geometry metadata, and shared mask.
"""

from __future__ import annotations
from hexatic.model_fitting.film_continuity import FilmContinuityScalars

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hexatic.active_matter_cylinder.math_utils import _density_sum
from hexatic.chirality.compute import compute_chirality_fields
from hexatic.chirality.config import ChiralityConfig

from ..film_continuity.config import FilmContinuityConfig, scalars_from_active_fields
from ..film_continuity.continuity import FilmContinuityResult, compute_film_continuity
from ..film_continuity.io_cache import ActiveMatterFields, load_active_matter_fields
from . import operators as ops
from .config import FittingConfig
from .io_cache import load_npz_arrays


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HydrodynamicFields:
    """All hydrodynamic fields on the (x, y=R*theta) grid."""

    # geometry
    transition_steps: np.ndarray  # (T-1, 2)
    dt: float
    cylinder_radius: float
    lx: float
    x_edges: np.ndarray
    x_centers: np.ndarray
    theta_edges: np.ndarray
    theta_centers: np.ndarray

    rho: np.ndarray
    P: np.ndarray           # (T, nx, ntheta, 2) = (P_x, P_y)
    chirality: np.ndarray
    D: np.ndarray
    hexatic_order: np.ndarray

    S_cross: np.ndarray

    partial_t_rho: np.ndarray
    partial_t_P: np.ndarray   # (T-1, nx, ntheta, 2)

    # spatial derivatives evaluated at transition midpoints
    grad_rho: np.ndarray              # (T-1, nx, ntheta, 2)
    grad_D: np.ndarray                # (T-1, nx, ntheta, 2)
    grad_hexatic_order: np.ndarray    # (T-1, nx, ntheta, 2)
    div_P: np.ndarray                 # (T-1, nx, ntheta)
    div_chiral_P_perp: np.ndarray     # (T-1, nx, ntheta)
    laplacian_rho: np.ndarray         # (T-1, nx, ntheta)
    laplacian_D: np.ndarray           # (T-1, nx, ntheta)
    laplacian_hexatic_order: np.ndarray  # (T-1, nx, ntheta)
    P_dot_grad_P: np.ndarray          # (T-1, nx, ntheta, 2)
    P_perp_dot_grad_P: np.ndarray     # (T-1, nx, ntheta, 2)
    laplacian_P: np.ndarray           # (T-1, nx, ntheta, 2)
    laplacian_P_perp: np.ndarray      # (T-1, nx, ntheta, 2)

    mid_rho: np.ndarray
    mid_chirality: np.ndarray
    mid_D: np.ndarray
    mid_hexatic_order: np.ndarray
    mid_P: np.ndarray   # (T-1, nx, ntheta, 2)

    # shared validity mask (T-1, nx, ntheta)
    mask: np.ndarray

    pocket_radius: float | None = None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_or_compute_fields(config: FittingConfig) -> HydrodynamicFields:
    """Load or compute all hydrodynamic fields for the given case."""
    cache_path = config.hydrodynamic_fields_cache_path
    if cache_path.exists():
        print("[fitting] Loading cached hydrodynamic fields...")
        result = _load_hydrodynamic_cache(cache_path)
        print("[fitting] Cache loaded.")
        return result

    print("[fitting] Loading active fields...")
    active = load_active_matter_fields(config.active_matter_path)
    scalars = scalars_from_active_fields(
        FilmContinuityConfig(
            case_id=config.case_id,
            npz_path=config.npz_path,
            gsd_path=config.gsd_path,
        ),
        steps=active.steps,
        x_edges=active.x_edges,
        theta_edges=active.theta_edges,
        pocket_radius=active.pocket_radius,
    )
    active_arrays = load_npz_arrays(config.active_matter_path)
    pocket_radius = _pocket_radius(active.pocket_radius, scalars.cylinder_radius)

    # --- build k-vectors for spectral derivatives ---
    ly = float(scalars.cylinder_radius) * float(
        scalars.theta_edges[-1] - scalars.theta_edges[0]
    )
    kx, ky = ops.build_k_vectors(
        scalars.x_centers.size, scalars.theta_centers.size, scalars.lx, ly,
    )

    # --- load Gaussian-smoothed scalars on grid ---
    rho, hexatic_order, D = _load_smoothed_scalars(
        config, active, scalars, active_arrays, pocket_radius,
    )

    # --- polarization on grid ---
    print("[fitting] Gridding polarization...")
    P = _polarization_grid_frames(
        active.coords, active.shell_mask, active_arrays,
        scalars.x_edges, scalars.theta_edges,
    )

    # --- chirality on grid ---
    chirality = _load_or_compute_chirality_frames(
        config, scalars.x_centers.size, scalars.theta_centers.size,
    )
    assert chirality.shape[0] == active.steps.size, (
        f"chirality frames {chirality.shape[0]} != steps {active.steps.size}"
    )

    # --- measured crossing source (transition field) ---
    print("[fitting] Loading S_cross...")
    S_cross = _load_s_cross(config)

    # --- frame fields and mid-frame fields ---
    print("[fitting] Computing derivatives...")
    mid_rho = _mid(rho)
    mid_P = _mid(P)
    mid_chirality = _mid(chirality)
    mid_D = _mid(D)
    mid_hexatic_order = _mid(hexatic_order)

    # time derivatives from adjacent smoothed frames
    partial_t_rho = (rho[1:] - rho[:-1]) / scalars.dt
    partial_t_P = (P[1:] - P[:-1]) / scalars.dt

    # spatial derivatives on midpoint fields
    grad_rho_x, grad_rho_y = ops.fft_gradient(mid_rho, kx, ky)
    grad_rho = np.stack((grad_rho_x, grad_rho_y), axis=-1)

    grad_D_x, grad_D_y = ops.fft_gradient(mid_D, kx, ky)
    grad_D = np.stack((grad_D_x, grad_D_y), axis=-1)

    grad_hex_x, grad_hex_y = ops.fft_gradient(mid_hexatic_order, kx, ky)
    grad_hexatic_order = np.stack((grad_hex_x, grad_hex_y), axis=-1)

    div_P = ops.fft_divergence(mid_P, kx, ky)

    # chiral_P_perp = chirality * P_perp = (-chirality*P_y, chirality*P_x)
    chiral_P_perp = np.stack(
        (mid_chirality * (-mid_P[..., 1]), mid_chirality * mid_P[..., 0]),
        axis=-1,
    )
    div_chiral_P_perp = ops.fft_divergence(chiral_P_perp, kx, ky)

    laplacian_rho = ops.fft_laplacian(mid_rho, kx, ky)
    laplacian_D = ops.fft_laplacian(mid_D, kx, ky)
    laplacian_hexatic_order = ops.fft_laplacian(mid_hexatic_order, kx, ky)

    P_dot_grad_P = ops.fft_directional_derivative(mid_P, kx, ky)

    # (P_perp · grad)P computed manually
    P_perp_dot_grad_P = _p_perp_dot_grad_P(mid_P, kx, ky)

    laplacian_P = ops.fft_vector_laplacian(mid_P, kx, ky)

    mid_P_perp = np.stack((-mid_P[..., 1], mid_P[..., 0]), axis=-1)
    laplacian_P_perp = ops.fft_vector_laplacian(mid_P_perp, kx, ky)

    # --- shared mask: valid density and valid polarization ---
    mask = (
        np.isfinite(mid_rho)
        & (mid_rho > config.density_threshold)
        & np.isfinite(mid_P[..., 0])
        & np.isfinite(mid_P[..., 1])
    )
    masked_count = int(np.count_nonzero(mask))
    total_count = mask.size
    print(f"[fitting] Shared mask: {masked_count}/{total_count} valid samples.")

    transition_steps = np.stack((active.steps[:-1], active.steps[1:]), axis=1)

    result = HydrodynamicFields(
        transition_steps=transition_steps,
        dt=scalars.dt,
        cylinder_radius=scalars.cylinder_radius,
        lx=scalars.lx,
        x_edges=scalars.x_edges,
        x_centers=scalars.x_centers,
        theta_edges=scalars.theta_edges,
        theta_centers=scalars.theta_centers,
        rho=rho,
        P=P,
        chirality=chirality,
        D=D,
        hexatic_order=hexatic_order,
        S_cross=S_cross,
        partial_t_rho=partial_t_rho,
        partial_t_P=partial_t_P,
        grad_rho=grad_rho,
        grad_D=grad_D,
        grad_hexatic_order=grad_hexatic_order,
        div_P=div_P,
        div_chiral_P_perp=div_chiral_P_perp,
        laplacian_rho=laplacian_rho,
        laplacian_D=laplacian_D,
        laplacian_hexatic_order=laplacian_hexatic_order,
        P_dot_grad_P=P_dot_grad_P,
        P_perp_dot_grad_P=P_perp_dot_grad_P,
        laplacian_P=laplacian_P,
        laplacian_P_perp=laplacian_P_perp,
        mid_rho=mid_rho,
        mid_chirality=mid_chirality,
        mid_D=mid_D,
        mid_hexatic_order=mid_hexatic_order,
        mid_P=mid_P,
        mask=mask,
        pocket_radius=pocket_radius,
    )

    print("[fitting] Caching hydrodynamic fields...")
    _save_hydrodynamic_cache(cache_path, result)
    print("[fitting] Cache written.")
    return result


# ---------------------------------------------------------------------------
# S_cross loading
# ---------------------------------------------------------------------------

def _load_s_cross(config: FittingConfig) -> np.ndarray:
    """Load measured crossing source from film-continuity cache."""
    fc_cache = config.film_continuity_cache_path
    if fc_cache.exists():
        fc = FilmContinuityResult.from_cache_arrays(load_npz_arrays(fc_cache))
        return np.asarray(fc.S_cross_b, dtype=float)

    print("[fitting] Film-continuity cache missing; computing S_cross...")
    fc_config = FilmContinuityConfig(
        case_id=config.case_id,
        npz_path=config.npz_path,
        gsd_path=config.gsd_path,
    )
    fc = compute_film_continuity(fc_config)
    return np.asarray(fc.S_cross_b, dtype=float)


# ---------------------------------------------------------------------------
# Cache I/O for hydrodynamic fields
# ---------------------------------------------------------------------------

def _save_hydrodynamic_cache(path: Path, fields: HydrodynamicFields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "transition_steps": fields.transition_steps,
        "dt": np.asarray(fields.dt),
        "cylinder_radius": np.asarray(fields.cylinder_radius),
        "lx": np.asarray(fields.lx),
        "x_edges": fields.x_edges,
        "x_centers": fields.x_centers,
        "theta_edges": fields.theta_edges,
        "theta_centers": fields.theta_centers,
        "rho": fields.rho,
        "P": fields.P,
        "chirality": fields.chirality,
        "D": fields.D,
        "hexatic_order": fields.hexatic_order,
        "S_cross": fields.S_cross,
        "partial_t_rho": fields.partial_t_rho,
        "partial_t_P": fields.partial_t_P,
        "grad_rho": fields.grad_rho,
        "grad_D": fields.grad_D,
        "grad_hexatic_order": fields.grad_hexatic_order,
        "div_P": fields.div_P,
        "div_chiral_P_perp": fields.div_chiral_P_perp,
        "laplacian_rho": fields.laplacian_rho,
        "laplacian_D": fields.laplacian_D,
        "laplacian_hexatic_order": fields.laplacian_hexatic_order,
        "P_dot_grad_P": fields.P_dot_grad_P,
        "P_perp_dot_grad_P": fields.P_perp_dot_grad_P,
        "laplacian_P": fields.laplacian_P,
        "laplacian_P_perp": fields.laplacian_P_perp,
        "mid_rho": fields.mid_rho,
        "mid_chirality": fields.mid_chirality,
        "mid_D": fields.mid_D,
        "mid_hexatic_order": fields.mid_hexatic_order,
        "mid_P": fields.mid_P,
        "mask": fields.mask,
        "pocket_radius": np.asarray(
            fields.pocket_radius if fields.pocket_radius is not None else np.nan
        ),
    }
    np.savez_compressed(path, **arrays)


def _load_hydrodynamic_cache(path: Path) -> HydrodynamicFields:
    arrays = load_npz_arrays(path)
    pocket_radius_val = float(np.asarray(arrays["pocket_radius"]))
    pocket_radius = None if np.isnan(pocket_radius_val) else pocket_radius_val
    return HydrodynamicFields(
        transition_steps=arrays["transition_steps"],
        dt=float(np.asarray(arrays["dt"])),
        cylinder_radius=float(np.asarray(arrays["cylinder_radius"])),
        lx=float(np.asarray(arrays["lx"])),
        x_edges=arrays["x_edges"],
        x_centers=arrays["x_centers"],
        theta_edges=arrays["theta_edges"],
        theta_centers=arrays["theta_centers"],
        rho=arrays["rho"],
        P=arrays["P"],
        chirality=arrays["chirality"],
        D=arrays["D"],
        hexatic_order=arrays["hexatic_order"],
        S_cross=arrays["S_cross"],
        partial_t_rho=arrays["partial_t_rho"],
        partial_t_P=arrays["partial_t_P"],
        grad_rho=arrays["grad_rho"],
        grad_D=arrays["grad_D"],
        grad_hexatic_order=arrays["grad_hexatic_order"],
        div_P=arrays["div_P"],
        div_chiral_P_perp=arrays["div_chiral_P_perp"],
        laplacian_rho=arrays["laplacian_rho"],
        laplacian_D=arrays["laplacian_D"],
        laplacian_hexatic_order=arrays["laplacian_hexatic_order"],
        P_dot_grad_P=arrays["P_dot_grad_P"],
        P_perp_dot_grad_P=arrays["P_perp_dot_grad_P"],
        laplacian_P=arrays["laplacian_P"],
        laplacian_P_perp=arrays["laplacian_P_perp"],
        mid_rho=arrays["mid_rho"],
        mid_chirality=arrays["mid_chirality"],
        mid_D=arrays["mid_D"],
        mid_hexatic_order=arrays["mid_hexatic_order"],
        mid_P=arrays["mid_P"],
        mask=np.asarray(arrays["mask"], dtype=bool),
        pocket_radius=pocket_radius,
    )


# ---------------------------------------------------------------------------
# Smoothed scalar loading
# ---------------------------------------------------------------------------

def _load_smoothed_scalars(
    config: FittingConfig,
    active: ActiveMatterFields,
    scalars:  FilmContinuityScalars,
    active_arrays: dict[str, np.ndarray],
    pocket_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load or compute rho, hexatic_order, D via Gaussian smoothing."""
    gaussian_cache = _load_gaussian_field_cache(
        config, active.steps, scalars.x_edges, scalars.theta_edges,
    )

    gaussian_density = _cached_grid_field(
        gaussian_cache, "rho_gaussian",
        frame_count=active.steps.size,
        nx=scalars.x_centers.size,
        ntheta=scalars.theta_centers.size,
    )
    hexatic_numerator = _cached_grid_field(
        gaussian_cache, "hexatic_order_numerator",
        frame_count=active.steps.size,
        nx=scalars.x_centers.size,
        ntheta=scalars.theta_centers.size,
    )
    D_numerator = _cached_grid_field(
        gaussian_cache, "D_numerator",
        frame_count=active.steps.size,
        nx=scalars.x_centers.size,
        ntheta=scalars.theta_centers.size,
    )

    hexatic_values = _load_hexatic_order_frames(
        config.hexatic_order_table_path, active.steps, active.coords.shape[1],
    )
    neighbor_counts = _load_neighbor_count_frames(
        config.neighbor_count_table_path, active.steps, active.coords.shape[1],
    )
    D_values = (6.0 - neighbor_counts) ** 2

    missing: dict[str, np.ndarray] = {}
    if gaussian_density is None:
        missing["rho_gaussian"] = np.ones(active.coords.shape[:2], dtype=float)
    if hexatic_numerator is None:
        missing["hexatic_order_numerator"] = hexatic_values
    if D_numerator is None:
        missing["D_numerator"] = D_values

    if missing:
        print("[fitting] Computing Gaussian fields on grid...")
        computed = _gaussian_scalar_field_frames(
            active.coords, active.shell_mask, missing,
            scalars.x_centers, scalars.theta_centers,
            scalars.lx, scalars.cylinder_radius, pocket_radius,
        )
        gaussian_cache.update(computed)
        gaussian_density = _cached_grid_field(
            gaussian_cache, "rho_gaussian",
            frame_count=active.steps.size,
            nx=scalars.x_centers.size,
            ntheta=scalars.theta_centers.size,
        )
        hexatic_numerator = _cached_grid_field(
            gaussian_cache, "hexatic_order_numerator",
            frame_count=active.steps.size,
            nx=scalars.x_centers.size,
            ntheta=scalars.theta_centers.size,
        )
        D_numerator = _cached_grid_field(
            gaussian_cache, "D_numerator",
            frame_count=active.steps.size,
            nx=scalars.x_centers.size,
            ntheta=scalars.theta_centers.size,
        )
        _save_gaussian_field_cache(
            config, active.steps, scalars.x_edges, scalars.theta_edges,
            gaussian_cache,
        )

    assert gaussian_density is not None
    assert hexatic_numerator is not None
    assert D_numerator is not None

    rho = gaussian_density
    hexatic_order = _divide_by_density(hexatic_numerator, gaussian_density)
    D = _divide_by_density(D_numerator, gaussian_density)
    return rho, hexatic_order, D


# ---------------------------------------------------------------------------
# Gaussian field cache
# ---------------------------------------------------------------------------

def _load_gaussian_field_cache(
    config: FittingConfig,
    steps: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    path = config.gaussian_fields_cache_path
    if not path.exists():
        return {}
    arrays = load_npz_arrays(path)
    if not _gaussian_cache_metadata_matches(arrays, steps, x_edges, theta_edges):
        return {}
    return arrays


def _save_gaussian_field_cache(
    config: FittingConfig,
    steps: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> None:
    path = config.gaussian_fields_cache_path
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        steps=np.asarray(steps),
        x_edges=np.asarray(x_edges),
        theta_edges=np.asarray(theta_edges),
        **{
            k: v for k, v in arrays.items()
            if k not in {"steps", "x_edges", "theta_edges"}
        },
    )


def _gaussian_cache_metadata_matches(
    arrays: dict[str, np.ndarray],
    steps: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> bool:
    required = ("steps", "x_edges", "theta_edges")
    if any(k not in arrays for k in required):
        return False
    return (
        np.array_equal(np.asarray(arrays["steps"]), np.asarray(steps))
        and np.array_equal(np.asarray(arrays["x_edges"]), np.asarray(x_edges))
        and np.array_equal(np.asarray(arrays["theta_edges"]), np.asarray(theta_edges))
    )


def _cached_grid_field(
    arrays: dict[str, np.ndarray],
    name: str,
    *,
    frame_count: int,
    nx: int,
    ntheta: int,
) -> np.ndarray | None:
    values = arrays.get(name)
    if values is None:
        return None
    values = np.asarray(values, dtype=float)
    if values.shape != (frame_count, nx, ntheta):
        return None
    return values


# ---------------------------------------------------------------------------
# Particle table loading
# ---------------------------------------------------------------------------

def _load_hexatic_order_frames(
    hexatic_order_path: str | Path,
    steps: np.ndarray,
    n_particles: int,
) -> np.ndarray:
    return _load_particle_scalar_table(
        hexatic_order_path, steps, n_particles,
        value_column=5, table_name="hexatic order",
    )


def _load_neighbor_count_frames(
    neighbor_count_path: str | Path,
    steps: np.ndarray,
    n_particles: int,
) -> np.ndarray:
    return _load_particle_scalar_table(
        neighbor_count_path, steps, n_particles,
        value_column=3, table_name="neighbor count",
    )


def _load_particle_scalar_table(
    path: str | Path,
    steps: np.ndarray,
    n_particles: int,
    *,
    value_column: int,
    table_name: str,
) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{table_name} table not found: {path}")
    table = np.loadtxt(path, dtype=np.float64)
    if table.ndim == 1:
        table = table[np.newaxis, :]
    if table.ndim != 2 or table.shape[1] <= value_column:
        raise ValueError(
            f"{table_name} table must have at least {value_column + 1} columns."
        )

    steps = np.asarray(steps, dtype=np.int64)
    frame_indices = table[:, 0].astype(np.int64)
    table_steps = table[:, 1].astype(np.int64)
    particle_indices = table[:, 2].astype(np.int64)
    scalar_values = np.asarray(table[:, value_column], dtype=np.float64)
    assert not np.any(frame_indices < 0) and not np.any(frame_indices >= steps.size)
    assert not np.any(particle_indices < 0) and not np.any(particle_indices >= n_particles)
    assert np.array_equal(table_steps, steps[frame_indices])

    flat_indices = frame_indices * n_particles + particle_indices
    assert np.unique(flat_indices).size == flat_indices.size

    values = np.full((steps.size, n_particles), np.nan, dtype=np.float64)
    values[frame_indices, particle_indices] = scalar_values
    assert np.all(np.isfinite(values))
    return values


# ---------------------------------------------------------------------------
# Gaussian smoothing (particle -> grid)
# ---------------------------------------------------------------------------

def _gaussian_scalar_field_frames(
    coords: np.ndarray,
    shell_mask: np.ndarray,
    values_by_name: dict[str, np.ndarray],
    x_centers: np.ndarray,
    theta_centers: np.ndarray,
    lx: float,
    cylinder_radius: float,
    pocket_radius: float,
) -> dict[str, np.ndarray]:
    coords = np.asarray(coords, dtype=np.float64)
    shell_mask = np.asarray(shell_mask, dtype=bool)
    if not values_by_name:
        return {}

    names = tuple(values_by_name)
    value_arrays = [np.asarray(values_by_name[n], dtype=np.float64) for n in names]
    grid_points = _surface_grid_points(x_centers, theta_centers, cylinder_radius)
    fields = {
        name: np.zeros(
            (coords.shape[0], x_centers.size, theta_centers.size), dtype=float,
        )
        for name in names
    }
    for frame_idx in range(coords.shape[0]):
        stacked = np.column_stack([va[frame_idx] for va in value_arrays])
        finite = shell_mask[frame_idx] & np.all(np.isfinite(stacked), axis=1)
        positions = _cylindrical_coords_to_cartesian(coords[frame_idx, finite])
        density = _density_sum(
            grid_points, positions, stacked[finite], lx, pocket_radius,
        ).reshape(x_centers.size, theta_centers.size, len(names))
        for field_idx, name in enumerate(names):
            fields[name][frame_idx] = density[..., field_idx]
    return fields


def _divide_by_density(numerator: np.ndarray, density: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    assert density.shape == numerator.shape
    return np.divide(
        numerator, density,
        out=np.zeros_like(numerator),
        where=np.isfinite(density) & (density > 0.0),
    )


# ---------------------------------------------------------------------------
# Polarization gridding
# ---------------------------------------------------------------------------

def _polarization_grid_frames(
    coords: np.ndarray,
    shell_mask: np.ndarray,
    arrays: dict[str, np.ndarray],
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> np.ndarray:
    assert "polar_mean" in arrays and "polar_cylindrical" in arrays
    return _particle_cylindrical_vector_grid_frames(
        coords, shell_mask,
        cartesian_values=arrays["polar_mean"],
        cylindrical_values=arrays["polar_cylindrical"],
        x_edges=x_edges,
        theta_edges=theta_edges,
    )


def _particle_cylindrical_vector_grid_frames(
    coords: np.ndarray,
    shell_mask: np.ndarray,
    *,
    cartesian_values: np.ndarray,
    cylindrical_values: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    shell_mask = np.asarray(shell_mask, dtype=bool)
    cartesian_values = np.asarray(cartesian_values, dtype=float)
    cylindrical_values = np.asarray(cylindrical_values, dtype=float)
    frames, _, _ = coords.shape
    nx = len(x_edges) - 1
    ntheta = len(theta_edges) - 1
    field = np.full((frames, nx, ntheta, 2), np.nan, dtype=float)
    lx = float(x_edges[-1] - x_edges[0])
    for frame_idx in range(frames):
        values = np.column_stack(
            (cartesian_values[frame_idx, :, 0], cylindrical_values[frame_idx, :, 2])
        )
        field[frame_idx] = _bin_vector_mean(
            coords[frame_idx], values, shell_mask[frame_idx],
            x_edges, theta_edges, lx,
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
    x_idx = np.clip(np.searchsorted(x_edges, x_values, side="right") - 1, 0, nx - 1)
    theta_period = float(theta_edges[-1] - theta_edges[0])
    theta0 = float(theta_edges[0])
    theta_values = ((coords[finite, 1] - theta0) % theta_period) + theta0
    theta_idx = np.clip(
        np.searchsorted(theta_edges, theta_values, side="right") - 1, 0, ntheta - 1,
    )

    np.add.at(counts, (x_idx, theta_idx), 1)
    for component in range(2):
        np.add.at(result[..., component], (x_idx, theta_idx), values[finite, component])
        result[..., component] = np.divide(
            result[..., component], counts,
            out=np.zeros((nx, ntheta), dtype=float),
            where=counts > 0,
        )
    return result


# ---------------------------------------------------------------------------
# Chirality
# ---------------------------------------------------------------------------

def _load_or_compute_chirality_frames(
    config: FittingConfig,
    nx: int,
    ntheta: int,
    metric_name: str = "instant_helix_relative",
) -> np.ndarray:
    if config.chirality_fields_path.exists():
        print("[fitting] Loading chirality fields...")
        arrays = load_npz_arrays(config.chirality_fields_path)
        return _chirality_from_arrays(arrays, metric_name, nx, ntheta)

    print("[fitting] Computing chirality fields...")
    fields = compute_chirality_fields(
        config.trajectory_path,
        config=ChiralityConfig(n_x_bins=nx, n_theta_bins=ntheta),
    )
    metric_index = {name: idx for idx, name in enumerate(fields.metric_names)}
    assert metric_name in metric_index
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
    assert "xtheta_values" in arrays and "metric_names" in arrays
    metric_names = tuple(str(n) for n in arrays["metric_names"])
    assert metric_name in metric_names
    values = np.asarray(arrays["xtheta_values"], dtype=float)
    field = values[metric_names.index(metric_name)]
    assert field.ndim == 3 and field.shape[1:] == (nx, ntheta)
    return np.nan_to_num(field, nan=0.0)


# ---------------------------------------------------------------------------
# (P_perp · grad)P  -- directional derivative with perpendicular velocity
# ---------------------------------------------------------------------------

def _p_perp_dot_grad_P(
    P: np.ndarray, kx: np.ndarray, ky: np.ndarray,
) -> np.ndarray:
    """Compute (P_perp · grad)P where P_perp = (-P_y, P_x).

    Input P shape (T, nx, ntheta, 2), output same shape.
    """
    assert P.ndim == 4 and P.shape[3] == 2
    dvx_dx, dvx_dy = ops.fft_gradient(P[..., 0], kx, ky)
    dvy_dx, dvy_dy = ops.fft_gradient(P[..., 1], kx, ky)
    # P_perp = (-P_y, P_x)
    result_x = (-P[..., 1]) * dvx_dx + P[..., 0] * dvx_dy
    result_y = (-P[..., 1]) * dvy_dx + P[..., 0] * dvy_dy
    return np.stack((result_x, result_y), axis=-1)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _mid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    assert values.shape[0] >= 2
    return 0.5 * (values[1:] + values[:-1])


def _pocket_radius(pocket_radius: float | None, cylinder_radius: float) -> float:
    if pocket_radius is not None and np.isfinite(pocket_radius) and pocket_radius > 0.0:
        return float(pocket_radius)
    return 0.5 * float(cylinder_radius)


def _surface_grid_points(
    x_centers: np.ndarray,
    theta_centers: np.ndarray,
    cylinder_radius: float,
) -> np.ndarray:
    x_grid, theta_grid = np.meshgrid(x_centers, theta_centers, indexing="ij")
    return np.column_stack((
        x_grid.ravel(),
        cylinder_radius * np.sin(theta_grid.ravel()),
        cylinder_radius * np.cos(theta_grid.ravel()),
    ))


def _cylindrical_coords_to_cartesian(coords: np.ndarray) -> np.ndarray:
    return np.column_stack((
        coords[:, 0],
        coords[:, 2] * np.sin(coords[:, 1]),
        coords[:, 2] * np.cos(coords[:, 1]),
    ))
