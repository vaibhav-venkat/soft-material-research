from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hexatic.radii_analysis.cases import RadiusCase, get_case

from ..film_continuity.config import GSD_DIR, NPZ_DIR


DENSITY_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
HEXATIC_OUTPUT_DIR = DENSITY_ANALYSIS_DIR / "hexatic_output"
OUTPUT_DIR = DENSITY_ANALYSIS_DIR / "output" / "fitting"
DEFAULT_CASE_ID = "radius_15D"
DEFAULT_RIDGE_ALPHA = 1.0e-6
DEFAULT_STLSQ_THRESHOLD = 1.0e-8
DEFAULT_STLSQ_MAX_ITER = 20
# Block average this many adjacent transition fields before regression. This is
# separate from the raw hydrodynamic cache, which stays at native transition time.
DEFAULT_COARSE_GRAIN_TRANSITIONS = 50


@dataclass(frozen=True)
class FittingConfig:
    case_id: str = DEFAULT_CASE_ID
    npz_path: Path | None = None
    gsd_path: Path | None = None
    chirality_path: Path | None = None
    hexatic_order_path: Path | None = None
    neighbor_count_path: Path | None = None
    output_dir: Path = OUTPUT_DIR
    density_threshold: float = 0.0
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA
    stlsq_threshold: float = DEFAULT_STLSQ_THRESHOLD
    stlsq_max_iter: int = DEFAULT_STLSQ_MAX_ITER
    coarse_grain_transitions: int = DEFAULT_COARSE_GRAIN_TRANSITIONS

    def __post_init__(self) -> None:
        if self.npz_path is not None:
            object.__setattr__(self, "npz_path", Path(self.npz_path))
        if self.gsd_path is not None:
            object.__setattr__(self, "gsd_path", Path(self.gsd_path))
        if self.chirality_path is not None:
            object.__setattr__(self, "chirality_path", Path(self.chirality_path))
        if self.hexatic_order_path is not None:
            object.__setattr__(
                self,
                "hexatic_order_path",
                Path(self.hexatic_order_path),
            )
        if self.neighbor_count_path is not None:
            object.__setattr__(
                self,
                "neighbor_count_path",
                Path(self.neighbor_count_path),
            )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.density_threshold < 0.0:
            raise ValueError("density_threshold must be non-negative.")
        if self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be non-negative.")
        if self.stlsq_threshold < 0.0:
            raise ValueError("stlsq_threshold must be non-negative.")
        if self.stlsq_max_iter < 1:
            raise ValueError("stlsq_max_iter must be at least 1.")
        if self.coarse_grain_transitions < 1:
            raise ValueError("coarse_grain_transitions must be at least 1.")

    @property
    def case(self) -> RadiusCase:
        return get_case(self.case_id)

    @property
    def active_matter_path(self) -> Path:
        if self.npz_path is not None:
            return self.npz_path
        return NPZ_DIR / f"{self.case_id}_active_matter_fields.npz"

    @property
    def trajectory_path(self) -> Path:
        if self.gsd_path is not None:
            return self.gsd_path
        return GSD_DIR / f"trajectory_{self.case_id}.gsd"

    @property
    def chirality_fields_path(self) -> Path:
        if self.chirality_path is not None:
            return self.chirality_path
        return NPZ_DIR / f"{self.case_id}_chirality_fields.npz"

    @property
    def hexatic_order_table_path(self) -> Path:
        if self.hexatic_order_path is not None:
            return self.hexatic_order_path
        return HEXATIC_OUTPUT_DIR / f"{self.case_id}_hexatic_order.txt"

    @property
    def neighbor_count_table_path(self) -> Path:
        if self.neighbor_count_path is not None:
            return self.neighbor_count_path
        return HEXATIC_OUTPUT_DIR / f"{self.case_id}_neighbor_counts.txt"

    @property
    def cache_path(self) -> Path:
        return self.output_dir / f"{self.case_id}_fitting.npz"

    @property
    def gaussian_fields_cache_path(self) -> Path:
        return self.output_dir / f"{self.case_id}_gaussian_fields.npz"

    @property
    def hydrodynamic_fields_cache_path(self) -> Path:
        return self.output_dir / f"{self.case_id}_hydrodynamic_fields.npz"

    @property
    def film_continuity_cache_path(self) -> Path:
        return (
            DENSITY_ANALYSIS_DIR
            / "output"
            / "film_continuity"
            / f"{self.case_id}_film_continuity.npz"
        )
