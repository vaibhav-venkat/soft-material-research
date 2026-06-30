"""Configuration for rho fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output"


@dataclass(frozen=True)
class CasePaths:
    case_id: str
    gsd_path: Path
    active_fields_path: Path
    hexatic_order_path: Path
    hexatic_velocity_path: Path
    neighbor_counts_path: Path

    @classmethod
    def from_case_id(cls, case_id: str) -> "CasePaths":
        return cls(
            case_id=case_id,
            gsd_path=PACKAGE_DIR / "gsd" / f"trajectory_{case_id}.gsd",
            active_fields_path=PACKAGE_DIR / "npz" / f"{case_id}_active_matter_fields.npz",
            hexatic_order_path=PACKAGE_DIR / "hexatic_output" / f"{case_id}_hexatic_order.txt",
            hexatic_velocity_path=PACKAGE_DIR / "hexatic_output" / f"{case_id}_hexatic_velocity.gsd",
            neighbor_counts_path=PACKAGE_DIR / "hexatic_output" / f"{case_id}_neighbor_counts.txt",
        )


@dataclass(frozen=True)
class NumericalSettings:
    sigma: float = 1.0
    cheb_cutoff: int = 20
    nd: int = 500_000
    seed: int = 0
    replace: bool = False
    tau_count: int = 40
    tau_eps: float = 1e-2
    subsamples: int = 200
    importance_threshold: float = 0.6
    alpha: float = 1e-6
    stlsq_max_iter: int = 20


@dataclass(frozen=True)
class RhoFittingConfig:
    case_id: str = "radius_15D"
    nd: int = 500_000
    seed: int = 0
    overwrite: bool = False
    make_plots: bool = True
    output_dir: Path = DEFAULT_OUTPUT_DIR
    settings: NumericalSettings | None = None

    def __post_init__(self) -> None:
        if self.nd <= 0:
            raise ValueError("nd must be positive")
        settings = self.settings or NumericalSettings(nd=self.nd, seed=self.seed)
        object.__setattr__(self, "settings", settings)

    @property
    def paths(self) -> CasePaths:
        return CasePaths.from_case_id(self.case_id)
