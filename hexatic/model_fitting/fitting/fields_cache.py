"""Cache I/O for hydrodynamic and Gaussian-smoothed fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io_cache import load_npz_arrays

HYDRO_CACHE_VERSION = 1


def _save_hydrodynamic_cache(path: Path, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "hydro_cache_version": np.asarray(HYDRO_CACHE_VERSION),
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


def _load_hydrodynamic_cache(path: Path):
    from .fields import HydrodynamicFields

    arrays = load_npz_arrays(path)
    cache_ver = int(np.asarray(arrays.get("hydro_cache_version", 0)))
    if cache_ver < HYDRO_CACHE_VERSION:
        raise ValueError(
            f"Hydrodynamic cache version {cache_ver} < {HYDRO_CACHE_VERSION}; recompute."
        )
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


def _load_gaussian_field_cache(
    gaussian_cache_path: Path,
    steps: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    if not gaussian_cache_path.exists():
        return {}
    arrays = load_npz_arrays(gaussian_cache_path)
    if not _gaussian_cache_metadata_matches(arrays, steps, x_edges, theta_edges):
        return {}
    return arrays


def _save_gaussian_field_cache(
    gaussian_cache_path: Path,
    steps: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> None:
    gaussian_cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        gaussian_cache_path,
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
