from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hexatic.radii_analysis.cases import RadiusCase, get_case

from ..film_continuity.config import GSD_DIR, NPZ_DIR


DENSITY_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
HEXATIC_OUTPUT_DIR = DENSITY_ANALYSIS_DIR / "hexatic_output"
OUTPUT_DIR = DENSITY_ANALYSIS_DIR / "output" / "fitting"
DEFAULT_CASE_ID = "radius_15D"


@dataclass(frozen=True)
class FittingConfig:
    case_id: str = DEFAULT_CASE_ID
    npz_path: str | Path | None = None
    gsd_path: str | Path | None = None
    chirality_path: str | Path | None = None
    hexatic_order_path: str | Path | None = None
    neighbor_count_path: str | Path | None = None
    output_dir: str | Path = OUTPUT_DIR
    density_threshold: float = 0.0
    smoothing_bins: float = 0.0

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
        if self.smoothing_bins < 0.0:
            raise ValueError("smoothing_bins must be non-negative.")

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
        if self.smoothing_bins > 0.0:
            bins_text = f"{self.smoothing_bins:g}".replace(".", "p")
            return self.output_dir / f"{self.case_id}_fitting_bins_{bins_text}.npz"
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
