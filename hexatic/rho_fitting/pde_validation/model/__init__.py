"""Native spectral RK4 model for validating rho-fitting closures."""

from .core import run_validation, run_validation_from_cache, validation_metric_arrays
from .interpolation import (
    interpolated_cached_fields,
    interpolated_fields,
    interpolated_p_q,
    interpolate_time_series,
    interpolation_index_weight,
)
from .types import Array, ValidationOptions, ValidationResult

__all__ = [
    "Array",
    "ValidationOptions",
    "ValidationResult",
    "interpolate_time_series",
    "interpolated_cached_fields",
    "interpolated_fields",
    "interpolated_p_q",
    "interpolation_index_weight",
    "run_validation",
    "run_validation_from_cache",
    "validation_metric_arrays",
]
