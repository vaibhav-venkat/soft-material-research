"""Load rho-fitting validation inputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hexatic.constants import cylinder


Array = NDArray[Any]


Y_RHO_NAMES = (
    "grad_rho",
    "grad_lap_rho",
    "Q_dot_grad_rho",
    "P",
)
Y_P_NAMES = ("A", "rho_A", "psi6sq_A", "grad_P", "rho_grad_P", "grad_lap_P")
Y_Q_NAMES = (
    "tangential_projected_Ubar_P_alpha",
    "radial_projected_Ubar_P_alpha",
    "tangential_grad_Q",
    "radial_grad_Q",
    "Q",
    "psi6sq_Q",
)


@dataclass(frozen=True)
class ValidationInputs:
    """Validated fields, coefficients, geometry, and constants for PDE rollout tests."""

    cache_path: Path
    metadata: dict[str, Any]
    rho: Array
    p: Array
    q: Array
    a: Array
    psi6_sq: Array
    y_p: Array
    times: Array
    y_rho_coefficients: Array
    y_p_coefficients: Array
    y_q_coefficients: Array
    lx: float
    theta_period: float
    r_centers: Array
    radius: float
    u0: float
    gamma: float
    tau_r: float


def load_validation_inputs(cache_path: Path) -> ValidationInputs:
    """Load and validate PDE validation inputs from a rho-fitting NPZ cache.

    Parameters:
        cache_path: Mechanical fit cache containing filtered fields, fitted coefficients,
            library names, Chebyshev times, and metadata.

    Returns:
        ``ValidationInputs`` with shape-checked arrays and positive domain constants.

    Edge cases:
        Library names must exactly match the current validation operators because
        coefficients are interpreted positionally.
    """
    with np.load(cache_path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata_json"])) if "metadata_json" in cache.files else {}
        _validate_library_names(cache)
        rho, p, q, a, psi6_sq, y_p = _load_field_arrays(cache)
        times = _load_array(cache, "cheb_times")
        y_rho_coefficients = _load_array(cache, "Y_rho_coefficients")
        y_p_coefficients = _load_array(cache, "Y_P_coefficients")
        y_q_coefficients = _load_array(cache, "Y_Q_coefficients")

    _validate_shapes(
        rho=rho,
        p=p,
        q=q,
        a=a,
        psi6_sq=psi6_sq,
        y_p=y_p,
        times=times,
        y_rho_coefficients=y_rho_coefficients,
        y_p_coefficients=y_p_coefficients,
        y_q_coefficients=y_q_coefficients,
    )
    lx, theta_period, r_centers, radius, u0, gamma, tau_r = _metadata_values(metadata)

    return ValidationInputs(
        cache_path=cache_path,
        metadata=metadata,
        rho=rho,
        p=p,
        q=q,
        a=a,
        psi6_sq=psi6_sq,
        y_p=y_p,
        times=times,
        y_rho_coefficients=y_rho_coefficients,
        y_p_coefficients=y_p_coefficients,
        y_q_coefficients=y_q_coefficients,
        lx=lx,
        theta_period=theta_period,
        r_centers=r_centers,
        radius=radius,
        u0=u0,
        gamma=gamma,
        tau_r=tau_r,
    )


def _validate_shapes(
    *,
    rho: Array,
    p: Array,
    q: Array,
    a: Array,
    psi6_sq: Array,
    y_p: Array,
    times: Array,
    y_rho_coefficients: Array,
    y_p_coefficients: Array,
    y_q_coefficients: Array,
) -> None:
    """Validate cached field and coefficient shapes expected by PDE validation."""
    assert rho.ndim == 4, "rho must be (T,Nx,Ntheta,Nr)"
    assert p.shape == rho.shape + (3,), "P must be (T,Nx,Ntheta,Nr,3)"
    assert q.shape == rho.shape + (3, 3), "Q must be (T,Nx,Ntheta,Nr,3,3)"
    assert a.shape == rho.shape + (3, 3), "A must be (T,Nx,Ntheta,Nr,3,3)"
    assert psi6_sq.shape == rho.shape, "psi6_sq must be (T,Nx,Ntheta,Nr)"
    assert y_p.shape == rho.shape + (3, 3), "Y_P must be (T,Nx,Ntheta,Nr,3,3)"
    assert times.shape == (rho.shape[0],), "cheb_times must match rho time axis"
    assert y_rho_coefficients.shape == (4,), "Y_rho coefficients must match current library"
    assert y_p_coefficients.shape == (6,), "Y_P coefficients must match current library"
    assert y_q_coefficients.shape == (6,), "Y_Q coefficients must match current library"


def _metadata_values(metadata: dict[str, Any]) -> tuple[float, float, Array, float, float, float, float]:
    """Extract positive geometry and simulation constants from cache metadata."""
    assert metadata.get("coordinate_system") == "cylindrical_3d", "validation cache must use cylindrical_3d"
    lx = float(metadata["lx"])
    theta_period = float(metadata["theta_period"])
    r_centers = np.asarray(metadata["r_centers"], dtype=np.float64)
    radius = float(metadata["radius"])
    u0 = float(metadata.get("u0", cylinder.SIMULATION.u0))
    gamma = float(metadata.get("gamma", cylinder.SIMULATION.gamma))
    tau_r = float(metadata.get("tau_r", cylinder.SIMULATION.tau_r))
    assert lx > 0.0 and theta_period > 0.0 and radius > 0.0, "domain metadata must be positive"
    assert r_centers.ndim == 1 and r_centers.size >= 2 and np.all(r_centers > 0.0), "invalid radial centers"
    assert gamma != 0.0 and tau_r > 0.0, "gamma must be nonzero and tau_r positive"
    return lx, theta_period, r_centers, radius, u0, gamma, tau_r


def _validate_library_names(cache: np.lib.npyio.NpzFile) -> None:
    """Assert cached mechanical library names match the validation coefficient order."""
    y_rho_names = tuple(str(name) for name in cache["Y_rho_names"])
    y_p_names = tuple(str(name) for name in cache["Y_P_names"])
    y_q_names = tuple(str(name) for name in cache["Y_Q_names"])
    assert y_rho_names == Y_RHO_NAMES, f"unexpected Y_rho names: {y_rho_names}"
    assert y_p_names == Y_P_NAMES, f"unexpected Y_P names: {y_p_names}"
    assert y_q_names == Y_Q_NAMES, f"unexpected Y_Q names: {y_q_names}"


def _load_field_arrays(
    cache: np.lib.npyio.NpzFile,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Load the filtered field arrays required by validation rollouts."""
    return (
        _load_array(cache, "rho"),
        _load_array(cache, "P"),
        _load_array(cache, "Q"),
        _load_array(cache, "A"),
        _load_array(cache, "psi6_sq"),
        _load_array(cache, "Y_P"),
    )


def _load_array(cache: np.lib.npyio.NpzFile, name: str) -> Array:
    """Load one cache array as float64 without allowing object pickles."""
    return np.asarray(cache[name], dtype=np.float64)
