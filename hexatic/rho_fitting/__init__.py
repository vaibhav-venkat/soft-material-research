"""Rho fitting workflow package."""

from __future__ import annotations

try:
    from . import _rho_fitting_core
    _rho_fitting_core_import_error: ImportError | None = None
except ImportError as error:
    _rho_fitting_core = None
    _rho_fitting_core_import_error = error

__all__ = ["_rho_fitting_core", "_rho_fitting_core_import_error"]
