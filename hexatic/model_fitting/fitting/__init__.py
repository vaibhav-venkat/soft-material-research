"""Fit film fluxes to density-gradient fields."""

from .config import DEFAULT_CASE_ID, FittingConfig
from .fields import HydrodynamicFields
from .fit import FittingResult, compute_fitting, stlsq
from .library import build_density_library, build_polarization_library
from .regression import RegressionResult

__all__ = [
    "DEFAULT_CASE_ID",
    "FittingConfig",
    "HydrodynamicFields",
    "FittingResult",
    "RegressionResult",
    "build_density_library",
    "build_polarization_library",
    "compute_fitting",
    "stlsq",
]
