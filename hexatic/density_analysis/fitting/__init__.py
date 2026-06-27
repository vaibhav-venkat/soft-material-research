"""Fit film fluxes to density-gradient fields."""

from .config import DEFAULT_CASE_ID, FittingConfig
from .fit import FittingResult, compute_fitting, stlsq
from .types import DEFAULT_CANDIDATES, FIELD_REGISTRY, FieldRegistry, FieldSpec

__all__ = [
    "DEFAULT_CANDIDATES",
    "DEFAULT_CASE_ID",
    "FIELD_REGISTRY",
    "FieldRegistry",
    "FieldSpec",
    "FittingConfig",
    "FittingResult",
    "compute_fitting",
    "stlsq",
]
