"""Load rho-fitting validation inputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


Array = NDArray[Any]


Y_RHO_NAMES = ("grad_rho", "grad_lap_rho", "Q_dot_grad_rho")
Y_P_NAMES = ("A", "rho_delta_psi6sq_A")
Y_Q_NAMES = ("Ubar_P_dot_alpha_traceless",)


@dataclass(frozen=True)
class ValidationInputs:
    cache_path: Path
    metadata: dict[str, Any]
    rho: Array
    p: Array
    q: Array
    psi6_sq: Array
    times: Array
    y_rho_coefficients: Array
    y_p_coefficients: Array
    y_q_coefficients: Array
    lx: float
    ly: float
    radius: float
    u0: float
    gamma: float
    tau_r: float


def load_validation_inputs(cache_path: Path) -> ValidationInputs:
    from hexatic.constants import cylinder

    with np.load(cache_path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata_json"])) if "metadata_json" in cache.files else {}
        y_rho_names = tuple(str(name) for name in cache["Y_rho_names"])
        y_p_names = tuple(str(name) for name in cache["Y_P_names"])
        y_q_names = tuple(str(name) for name in cache["Y_Q_names"])
        assert y_rho_names == Y_RHO_NAMES, f"unexpected Y_rho names: {y_rho_names}"
        assert y_p_names == Y_P_NAMES, f"unexpected Y_P names: {y_p_names}"
        assert y_q_names == Y_Q_NAMES, f"unexpected Y_Q names: {y_q_names}"

        rho = np.asarray(cache["rho"], dtype=np.float64)
        p = np.asarray(cache["P"], dtype=np.float64)
        q = np.asarray(cache["Q"], dtype=np.float64)
        psi6_sq = np.asarray(cache["psi6_sq"], dtype=np.float64)
        times = np.asarray(cache["cheb_times"], dtype=np.float64)
        y_rho_coefficients = np.asarray(cache["Y_rho_coefficients"], dtype=np.float64)
        y_p_coefficients = np.asarray(cache["Y_P_coefficients"], dtype=np.float64)
        y_q_coefficients = np.asarray(cache["Y_Q_coefficients"], dtype=np.float64)

    assert rho.ndim == 3, "rho must be (T,Nx,Ny)"
    assert p.shape == rho.shape + (3,), "P must be (T,Nx,Ny,3)"
    assert q.shape == rho.shape + (3, 3), "Q must be (T,Nx,Ny,3,3)"
    assert psi6_sq.shape == rho.shape, "psi6_sq must be (T,Nx,Ny)"
    assert times.shape == (rho.shape[0],), "cheb_times must match rho time axis"
    assert y_rho_coefficients.shape == (3,), "Y_rho coefficients must match current library"
    assert y_p_coefficients.shape == (2,), "Y_P coefficients must match current library"
    assert y_q_coefficients.shape == (1,), "Y_Q coefficients must match current library"

    lx = float(metadata["lx"])
    ly = float(metadata["ly"])
    radius = float(metadata["radius"])
    u0 = float(metadata.get("u0", cylinder.SIMULATION.u0))
    gamma = float(metadata.get("gamma", cylinder.SIMULATION.gamma))
    tau_r = float(metadata.get("tau_r", cylinder.SIMULATION.tau_r))
    assert lx > 0.0 and ly > 0.0 and radius > 0.0, "domain metadata must be positive"
    assert gamma != 0.0 and tau_r > 0.0, "gamma must be nonzero and tau_r positive"

    return ValidationInputs(
        cache_path=cache_path,
        metadata=metadata,
        rho=rho,
        p=p,
        q=q,
        psi6_sq=psi6_sq,
        times=times,
        y_rho_coefficients=y_rho_coefficients,
        y_p_coefficients=y_p_coefficients,
        y_q_coefficients=y_q_coefficients,
        lx=lx,
        ly=ly,
        radius=radius,
        u0=u0,
        gamma=gamma,
        tau_r=tau_r,
    )
