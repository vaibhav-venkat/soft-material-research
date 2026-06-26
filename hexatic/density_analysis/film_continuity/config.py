from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np

from hexatic.constants import cylinder
from hexatic.radii_analysis.cases import RadiusCase, get_case


DENSITY_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
NPZ_DIR = DENSITY_ANALYSIS_DIR / "npz"
GSD_DIR = DENSITY_ANALYSIS_DIR / "gsd"
OUTPUT_DIR = DENSITY_ANALYSIS_DIR / "output" / "film_continuity"
DEFAULT_CASE_ID = "radius_15D"
DEFAULT_MIN_COUNT = 2
FIXED_LX_REFERENCE = (
    4000.0 / (cylinder.RHO * np.pi * cylinder.BASELINE_CYLINDER_RADIUS**2)
)


@dataclass(frozen=True)
class FilmContinuityConfig:
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
        return self.output_dir / f"{self.case_id}_film_continuity.npz"

    def plot_path(self, quantity: str) -> Path:
        return self.output_dir / f"{self.case_id}_film_continuity_map_{quantity}.html"


@dataclass(frozen=True)
class FilmContinuityScalars:
    cylinder_radius: float
    lx: float
    dt: float
    x_edges: np.ndarray
    theta_edges: np.ndarray
    x_centers: np.ndarray
    theta_centers: np.ndarray
    area_bin: np.ndarray
    particle_diameter: float = cylinder.PARTICLE_DIAMETER
    timestep: float = cylinder.TIMESTEP
    pocket_radius: float | None = None

    @classmethod
    def from_edges(
        cls,
        *,
        cylinder_radius: float,
        lx: float,
        dt: float,
        x_edges: np.ndarray,
        theta_edges: np.ndarray,
        pocket_radius: float | None = None,
    ) -> "FilmContinuityScalars":
        x_edges = np.asarray(x_edges, dtype=float)
        theta_edges = np.asarray(theta_edges, dtype=float)
        dx = np.diff(x_edges)
        dtheta = np.diff(theta_edges)
        area_bin = dx[:, None] * cylinder_radius * dtheta[None, :]
        return cls(
            cylinder_radius=float(cylinder_radius),
            lx=float(lx),
            dt=float(dt),
            x_edges=x_edges,
            theta_edges=theta_edges,
            x_centers=0.5 * (x_edges[:-1] + x_edges[1:]),
            theta_centers=0.5 * (theta_edges[:-1] + theta_edges[1:]),
            area_bin=area_bin,
            pocket_radius=pocket_radius,
        )


def scalars_from_active_fields(
    config: FilmContinuityConfig,
    *,
    steps: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    pocket_radius: float | None = None,
) -> FilmContinuityScalars:
    case = config.case
    lx = lx_from_edges(x_edges)
    case_lx = float(case.lx)
    if not np.isclose(lx, case_lx, rtol=1.0e-6, atol=1.0e-9):
        warnings.warn(
            f"Using Lx={lx} from cached x_edges instead of case Lx={case_lx}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return FilmContinuityScalars.from_edges(
        cylinder_radius=float(case.radius),
        lx=lx,
        dt=dt_from_steps(steps),
        x_edges=x_edges,
        theta_edges=theta_edges,
        pocket_radius=pocket_radius,
    )


def dt_from_steps(steps: np.ndarray, timestep: float = cylinder.TIMESTEP) -> float:
    steps = np.asarray(steps)
    if steps.size < 2:
        raise ValueError("At least two steps are required to derive dt.")
    step_deltas = np.diff(steps).astype(float)
    if np.any(step_deltas <= 0.0):
        raise ValueError("Frame steps must be strictly increasing.")
    if not np.allclose(step_deltas, step_deltas[0]):
        raise ValueError("Film continuity currently expects uniformly spaced frames.")
    return float(step_deltas[0] * timestep)


def lx_from_edges(x_edges: np.ndarray) -> float:
    edges = np.asarray(x_edges, dtype=float)
    if edges.size < 2:
        raise ValueError("At least two x edges are required to derive lx.")
    return float(edges[-1] - edges[0])
