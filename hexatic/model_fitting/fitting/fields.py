"""Build hydrodynamic fields from particle data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hexatic.active_matter_cylinder.math_utils import _density_sum, _minimum_image_delta
from hexatic.chirality.compute import compute_chirality_fields
from hexatic.chirality.config import ChiralityConfig

from ..film_continuity.config import (
    FilmContinuityConfig,
    FilmContinuityScalars,
    scalars_from_active_fields,
)
from ..film_continuity.io_cache import ActiveMatterFields, load_active_matter_fields
from . import operators as ops
from .config import FittingConfig
from .fields_cache import (
    _cached_grid_field,
    _load_gaussian_field_cache,
    _load_hydrodynamic_cache,
    _save_gaussian_field_cache,
    _save_hydrodynamic_cache,
)
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
    material_current: np.ndarray      # (T-1, nx, ntheta, 2)

    mid_rho: np.ndarray
    mid_chirality: np.ndarray
    mid_D: np.ndarray
    mid_hexatic_order: np.ndarray
    mid_P: np.ndarray   # (T-1, nx, ntheta, 2)
    mid_force_density: np.ndarray     # (T-1, nx, ntheta, 2)

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
        try:
            result = _load_hydrodynamic_cache(cache_path)
            print("[fitting] Cache loaded.")
            return result
        except ValueError:
            print("[fitting] Hydrodynamic cache stale; recomputing...")

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

    # --- load Gaussian-smoothed scalars and polarization on grid ---
    rho, hexatic_order, D, P, chirality, force_density = _load_smoothed_scalars(
        config, active, scalars, active_arrays, pocket_radius,
    )

    # --- crossing source in the same Gaussian representation as rho ---
    print("[fitting] Computing Gaussian S_cross...")
    S_cross = _gaussian_crossing_source(
        active.coords,
        active.shell_mask,
        scalars.x_centers,
        scalars.theta_centers,
        scalars.lx,
        scalars.cylinder_radius,
        pocket_radius,
        scalars.dt,
    )
    material_current = _gaussian_material_current(
        active.coords,
        active.shell_mask,
        scalars.x_centers,
        scalars.theta_centers,
        scalars.lx,
        scalars.cylinder_radius,
        pocket_radius,
        scalars.dt,
    )

    # --- frame fields and mid-frame fields ---
    print("[fitting] Computing derivatives...")
    mid_rho = _mid(rho)
    mid_P = _mid(P)
    mid_chirality = _mid(chirality)
    mid_D = _mid(D)
    mid_hexatic_order = _mid(hexatic_order)
    mid_force_density = _mid(force_density)

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

    # --- shared mask: valid density and polarization rows use identical samples ---
    mask = _shared_valid_mask(
        density_threshold=config.density_threshold,
        mid_rho=mid_rho,
        scalar_fields=(
            partial_t_rho,
            S_cross,
            mid_chirality,
            mid_D,
            mid_hexatic_order,
            div_P,
            div_chiral_P_perp,
            laplacian_rho,
            laplacian_D,
            laplacian_hexatic_order,
        ),
        vector_fields=(
            mid_P,
            partial_t_P,
            grad_rho,
            grad_D,
            grad_hexatic_order,
            P_dot_grad_P,
            P_perp_dot_grad_P,
            laplacian_P,
            laplacian_P_perp,
            material_current,
            mid_force_density,
        ),
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
        material_current=material_current,
        mid_rho=mid_rho,
        mid_chirality=mid_chirality,
        mid_D=mid_D,
        mid_hexatic_order=mid_hexatic_order,
        mid_P=mid_P,
        mid_force_density=mid_force_density,
        mask=mask,
        pocket_radius=pocket_radius,
    )

    print("[fitting] Caching hydrodynamic fields...")
    _save_hydrodynamic_cache(cache_path, result)
    print("[fitting] Cache written.")
    return result


# ---------------------------------------------------------------------------
# Smoothed scalar / polarization loading
# ---------------------------------------------------------------------------

def _load_smoothed_scalars(
    config: FittingConfig,
    active: ActiveMatterFields,
    scalars: FilmContinuityScalars,
    active_arrays: dict[str, np.ndarray],
    pocket_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load or compute rho, hexatic_order, D, P, chirality, force density."""
    gaussian_cache = _load_gaussian_field_cache(
        config.gaussian_fields_cache_path,
        active.steps, scalars.x_edges, scalars.theta_edges,
    )

    def _cached(name: str) -> np.ndarray | None:
        return _cached_grid_field(
            gaussian_cache, name,
            frame_count=active.steps.size,
            nx=scalars.x_centers.size,
            ntheta=scalars.theta_centers.size,
        )

    gaussian_density = _cached("rho_gaussian")
    hexatic_numerator = _cached("hexatic_order_numerator")
    D_numerator = _cached("D_numerator")
    Px_numerator = _cached("P_x_numerator")
    Py_numerator = _cached("P_y_numerator")
    chirality_numerator = _cached("chirality_numerator")
    Fx_numerator = _cached("force_density_x_numerator")
    Fy_numerator = _cached("force_density_y_numerator")

    # particle-level source values
    hexatic_values = _load_hexatic_order_frames(
        config.hexatic_order_table_path, active.steps, active.coords.shape[1],
    )
    neighbor_counts = _load_neighbor_count_frames(
        config.neighbor_count_table_path, active.steps, active.coords.shape[1],
    )
    D_values = (6.0 - neighbor_counts) ** 2
    Px_values = np.asarray(active_arrays["polar_mean"][..., 0], dtype=float)
    Py_values = np.asarray(active_arrays["polar_cylindrical"][..., 2], dtype=float)
    Fx_values = np.asarray(
        active_arrays["force_density_cylindrical"][..., 0], dtype=float,
    )
    Fy_values = np.asarray(
        active_arrays["force_density_cylindrical"][..., 2], dtype=float,
    )
    if chirality_numerator is None:
        chirality_grid = _load_or_compute_chirality_frames(
            config, scalars.x_centers.size, scalars.theta_centers.size,
        )
        assert chirality_grid.shape[0] == active.steps.size, (
            f"chirality frames {chirality_grid.shape[0]} != steps {active.steps.size}"
        )
        chirality_values = _particle_values_from_grid_field(
            chirality_grid, active.coords, scalars.x_edges, scalars.theta_edges,
        )

    missing: dict[str, np.ndarray] = {}
    if gaussian_density is None:
        missing["rho_gaussian"] = np.ones(active.coords.shape[:2], dtype=float)
    if hexatic_numerator is None:
        missing["hexatic_order_numerator"] = hexatic_values
    if D_numerator is None:
        missing["D_numerator"] = D_values
    if Px_numerator is None:
        missing["P_x_numerator"] = Px_values
    if Py_numerator is None:
        missing["P_y_numerator"] = Py_values
    if chirality_numerator is None:
        missing["chirality_numerator"] = chirality_values
    if Fx_numerator is None:
        missing["force_density_x_numerator"] = Fx_values
    if Fy_numerator is None:
        missing["force_density_y_numerator"] = Fy_values

    if missing:
        print("[fitting] Computing Gaussian fields on grid...")
        computed = _gaussian_scalar_field_frames(
            active.coords, active.shell_mask, missing,
            scalars.x_centers, scalars.theta_centers,
            scalars.lx, scalars.cylinder_radius, pocket_radius,
        )
        gaussian_cache.update(computed)
        _save_gaussian_field_cache(
            config.gaussian_fields_cache_path,
            active.steps, scalars.x_edges, scalars.theta_edges,
            gaussian_cache,
        )
        gaussian_density = _cached("rho_gaussian")
        hexatic_numerator = _cached("hexatic_order_numerator")
        D_numerator = _cached("D_numerator")
        Px_numerator = _cached("P_x_numerator")
        Py_numerator = _cached("P_y_numerator")
        chirality_numerator = _cached("chirality_numerator")
        Fx_numerator = _cached("force_density_x_numerator")
        Fy_numerator = _cached("force_density_y_numerator")

    assert gaussian_density is not None
    assert hexatic_numerator is not None
    assert D_numerator is not None
    assert Px_numerator is not None
    assert Py_numerator is not None
    assert chirality_numerator is not None
    assert Fx_numerator is not None
    assert Fy_numerator is not None

    rho = gaussian_density
    hexatic_order = _divide_by_density(hexatic_numerator, gaussian_density)
    D = _divide_by_density(D_numerator, gaussian_density)
    P_x = _divide_by_density(Px_numerator, gaussian_density)
    P_y = _divide_by_density(Py_numerator, gaussian_density)
    chirality = _divide_by_density(chirality_numerator, gaussian_density)
    force_x = _divide_by_density(Fx_numerator, gaussian_density)
    force_y = _divide_by_density(Fy_numerator, gaussian_density)
    P = np.stack((P_x, P_y), axis=-1)
    force_density = np.stack((force_x, force_y), axis=-1)
    return rho, hexatic_order, D, P, chirality, force_density


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


def _gaussian_crossing_source(
    coords: np.ndarray,
    shell_mask: np.ndarray,
    x_centers: np.ndarray,
    theta_centers: np.ndarray,
    lx: float,
    cylinder_radius: float,
    pocket_radius: float,
    dt: float,
) -> np.ndarray:
    """Return S_cross from shell entry/exit in the Gaussian density basis."""
    coords = np.asarray(coords, dtype=np.float64)
    shell_mask = np.asarray(shell_mask, dtype=bool)
    assert coords.ndim == 3 and coords.shape[2] >= 3
    assert shell_mask.shape == coords.shape[:2]
    assert dt > 0.0

    grid_points = _surface_grid_points(x_centers, theta_centers, cylinder_radius)
    source = np.zeros(
        (coords.shape[0] - 1, x_centers.size, theta_centers.size),
        dtype=np.float64,
    )
    for frame_idx in range(coords.shape[0] - 1):
        entering = (~shell_mask[frame_idx]) & shell_mask[frame_idx + 1]
        exiting = shell_mask[frame_idx] & (~shell_mask[frame_idx + 1])
        density_in = _gaussian_density_for_particles(
            grid_points, coords[frame_idx + 1, entering], lx, pocket_radius,
        )
        density_out = _gaussian_density_for_particles(
            grid_points, coords[frame_idx, exiting], lx, pocket_radius,
        )
        source[frame_idx] = ((density_in - density_out) / dt).reshape(
            x_centers.size, theta_centers.size,
        )
    return source


def _gaussian_material_current(
    coords: np.ndarray,
    shell_mask: np.ndarray,
    x_centers: np.ndarray,
    theta_centers: np.ndarray,
    lx: float,
    cylinder_radius: float,
    pocket_radius: float,
    dt: float,
) -> np.ndarray:
    """Return measured material current J_m from same-particle displacement."""
    coords = np.asarray(coords, dtype=np.float64)
    shell_mask = np.asarray(shell_mask, dtype=bool)
    assert coords.ndim == 3 and coords.shape[2] >= 3
    assert shell_mask.shape == coords.shape[:2]
    assert dt > 0.0

    grid_points = _surface_grid_points(x_centers, theta_centers, cylinder_radius)
    current = np.zeros(
        (coords.shape[0] - 1, x_centers.size, theta_centers.size, 2),
        dtype=np.float64,
    )
    theta_period = 2.0 * np.pi
    for frame_idx in range(coords.shape[0] - 1):
        valid = shell_mask[frame_idx] & shell_mask[frame_idx + 1]
        if not np.any(valid):
            continue
        start = coords[frame_idx, valid]
        stop = coords[frame_idx + 1, valid]
        dx = _minimum_image_delta(stop[:, 0] - start[:, 0], lx)
        dtheta = _minimum_image_delta(stop[:, 1] - start[:, 1], theta_period)
        mid = start.copy()
        mid[:, 0] = start[:, 0] + 0.5 * dx
        mid[:, 1] = np.mod(start[:, 1] + 0.5 * dtheta, theta_period)
        values = np.column_stack((dx / dt, cylinder_radius * dtheta / dt))
        current[frame_idx] = _density_sum(
            grid_points,
            _cylindrical_coords_to_cartesian(mid),
            values,
            lx,
            pocket_radius,
        ).reshape(x_centers.size, theta_centers.size, 2)
    return current


def _gaussian_density_for_particles(
    grid_points: np.ndarray,
    coords: np.ndarray,
    lx: float,
    pocket_radius: float,
) -> np.ndarray:
    if coords.size == 0:
        return np.zeros(grid_points.shape[0], dtype=np.float64)
    positions = _cylindrical_coords_to_cartesian(coords)
    values = np.ones(positions.shape[0], dtype=np.float64)
    return _density_sum(grid_points, positions, values, lx, pocket_radius)


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
def _particle_values_from_grid_field(
    grid_field: np.ndarray,
    coords: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> np.ndarray:
    """Sample a gridded scalar at particle (x, theta) bins for later smoothing."""
    grid_field = np.asarray(grid_field, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    x_edges = np.asarray(x_edges, dtype=np.float64)
    theta_edges = np.asarray(theta_edges, dtype=np.float64)
    assert grid_field.ndim == 3
    assert coords.ndim == 3 and coords.shape[2] >= 2
    assert grid_field.shape[0] == coords.shape[0]
    assert grid_field.shape[1:] == (x_edges.size - 1, theta_edges.size - 1)

    values = np.empty(coords.shape[:2], dtype=np.float64)
    for frame_idx in range(coords.shape[0]):
        ix = _periodic_bin_indices(coords[frame_idx, :, 0], x_edges)
        itheta = _periodic_bin_indices(coords[frame_idx, :, 1], theta_edges)
        values[frame_idx] = grid_field[frame_idx, ix, itheta]
    return np.nan_to_num(values, nan=0.0)


def _periodic_bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    period = float(edges[-1] - edges[0])
    assert period > 0.0
    wrapped = edges[0] + np.mod(np.asarray(values, dtype=np.float64) - edges[0], period)
    indices = np.searchsorted(edges, wrapped, side="right") - 1
    return np.mod(indices, edges.size - 1)


def _shared_valid_mask(
    *,
    density_threshold: float,
    mid_rho: np.ndarray,
    scalar_fields: tuple[np.ndarray, ...] = (),
    vector_fields: tuple[np.ndarray, ...] = (),
) -> np.ndarray:
    mid_rho = np.asarray(mid_rho, dtype=float)
    mask = np.isfinite(mid_rho) & (mid_rho > density_threshold)
    for field in scalar_fields:
        field = np.asarray(field, dtype=float)
        assert field.shape == mid_rho.shape
        mask &= np.isfinite(field)
    for field in vector_fields:
        field = np.asarray(field, dtype=float)
        assert field.shape[:-1] == mid_rho.shape and field.shape[-1] == 2
        mask &= np.all(np.isfinite(field), axis=-1)
    return mask


# Polarization gridding
# ---------------------------------------------------------------------------

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
