"""Configuration for rho fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from hexatic.constants import cylinder
from hexatic.constants.cylinder import PARTICLE_DIAMETER


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output"


@dataclass(frozen=True)
class CasePaths:
    """Canonical input paths for one rho-fitting case id."""

    case_id: str
    gsd_path: Path
    active_fields_path: Path
    hexatic_order_path: Path
    hexatic_velocity_path: Path
    neighbor_counts_path: Path

    @classmethod
    def from_case_id(cls, case_id: str) -> "CasePaths":
        """Build the standard local file paths for a case such as ``radius_15D``."""
        return cls(
            case_id=case_id,
            gsd_path=PACKAGE_DIR / "gsd" / f"trajectory_{case_id}.gsd",
            active_fields_path=PACKAGE_DIR / "npz" / f"{case_id}_active_matter_fields.npz",
            hexatic_order_path=PACKAGE_DIR / "hexatic_output" / f"{case_id}_hexatic_order.txt",
            hexatic_velocity_path=PACKAGE_DIR / "hexatic_output" / f"{case_id}_hexatic_velocity.gsd",
            neighbor_counts_path=PACKAGE_DIR / "hexatic_output" / f"{case_id}_neighbor_counts.txt",
        )


def radius_from_case_id(case_id: str) -> float | None:
    """Parse a radius in particle diameters from case ids like ``radius_15D``."""
    match = re.search(r"radius_([0-9]+(?:\.[0-9]+)?)D", case_id)
    return float(match.group(1)) if match else None


@dataclass(frozen=True)
class NumericalSettings:
    """Numerical controls for coarse-graining, filtering, sampling, and sparse regression."""

    sigma: float = 5.0 * PARTICLE_DIAMETER
    cheb_cutoff: int = 10
    timestep: float = cylinder.SIMULATION.timestep
    nd: int = 500_000
    seed: int = 0
    replace: bool = False
    tau_count: int = 40
    tau_eps: float = 1e-3
    subsamples: int = 200
    importance_threshold: float = 0.6
    alpha: float = 1e-6
    stlsq_max_iter: int = 20
    mechanical_flux_weight: float = 1.0
    gamma: float = float(cylinder.SIMULATION.gamma)
    u0: float = float(cylinder.SIMULATION.u0)
    radial_bins: int = 16
    radial_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class RhoFittingConfig:
    """Top-level rho-fitting run configuration and derived case paths."""

    case_id: str = "radius_15D"
    overwrite: bool = False
    make_plots: bool = True
    correlations_only: bool = False
    fit_only: bool = False
    output_dir: Path = DEFAULT_OUTPUT_DIR
    settings: NumericalSettings | None = None

    def __post_init__(self) -> None:
        """Fill default numerical settings and reject invalid scalar controls."""
        settings = self.settings or NumericalSettings()
        assert settings.nd > 0, "nd must be positive"
        assert settings.sigma > 0.0, "sigma must be positive"
        assert settings.cheb_cutoff > 0, "cheb_cutoff must be positive"
        assert settings.tau_count > 0, "tau_count must be positive"
        assert settings.tau_eps > 0.0, "tau_eps must be positive"
        assert settings.mechanical_flux_weight >= 0.0, "mechanical_flux_weight must be non-negative"
        assert settings.timestep > 0.0, "timestep must be positive"
        assert settings.u0 != 0.0, "u0 must be nonzero"
        assert settings.radial_bins > 0, "radial_bins must be positive"
        if settings.radial_range is not None:
            r_min, r_max = settings.radial_range
            assert r_min < r_max, "radial_range must be ordered"
        object.__setattr__(self, "settings", settings)

    @property
    def paths(self) -> CasePaths:
        """Return canonical input paths for this configuration's case id."""
        return CasePaths.from_case_id(self.case_id)
