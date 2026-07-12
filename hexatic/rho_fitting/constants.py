"""Numerical constants exported by the compiled Rust core."""

from __future__ import annotations

from . import _rho_fitting_core, _rho_fitting_core_import_error


if _rho_fitting_core is None:
    raise ImportError(f"rho-fitting Rust core is unavailable: {_rho_fitting_core_import_error}")

_VALUES = _rho_fitting_core.numerical_constants()
P_RELAXATION_COEFFICIENT = float(_VALUES["p_relaxation_coefficient"])
Q_RELAXATION_COEFFICIENT = float(_VALUES["q_relaxation_coefficient"])
DEFAULT_PDE_DT_MAX = float(_VALUES["default_pde_dt_max"])
REGRESSION_TOLERANCE = float(_VALUES["regression_tolerance"])
