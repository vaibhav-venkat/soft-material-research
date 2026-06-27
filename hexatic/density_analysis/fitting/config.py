from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hexatic.radii_analysis.cases import RadiusCase, get_case

from ..film_continuity.config import GSD_DIR, NPZ_DIR


DENSITY_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = DENSITY_ANALYSIS_DIR / "output" / "fitting"
DEFAULT_CASE_ID = "radius_15D"
DEFAULT_MIN_COUNT = 2


@dataclass(frozen=True)
class FittingConfig:
    case_id: str = DEFAULT_CASE_ID
    npz_path: str | Path | None = None
    gsd_path: str | Path | None = None
    output_dir: str | Path = OUTPUT_DIR
    min_count: int = DEFAULT_MIN_COUNT

    def __post_init__(self) -> None:
        if self.npz_path is not None:
            object.__setattr__(self, "npz_path", Path(self.npz_path))
        if self.gsd_path is not None:
            object.__setattr__(self, "gsd_path", Path(self.gsd_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

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
    def cache_path(self) -> Path:
        return self.output_dir / f"{self.case_id}_fitting.npz"

    @property
    def film_continuity_cache_path(self) -> Path:
        return (
            DENSITY_ANALYSIS_DIR
            / "output"
            / "film_continuity"
            / f"{self.case_id}_film_continuity.npz"
        )

    def plot_path(self, quantity: str) -> Path:
        return self.output_dir / f"{self.case_id}_fit_{quantity}.png"
