from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hexatic.constants import cylinder

ANALYSIS_DIR = Path(__file__).resolve().parent
GSD_DIR = ANALYSIS_DIR / "gsd"
INITIAL_GSD_DIR = ANALYSIS_DIR / "initial"
HEXATIC_OUTPUT_DIR = ANALYSIS_DIR / "hexatic_output"
NPZ_FIELDS_DIR = ANALYSIS_DIR / "npz_fields"
LOG_DIR = ANALYSIS_DIR / "logs"
METADATA_DIR = ANALYSIS_DIR / "metadata"

RUN_STEPS = int(1e7)
TRAJECTORY_WRITE_PERIOD = int(1e5)
LONG_AXIS_LX = 500.0 * cylinder.PARTICLE_DIAMETER


@dataclass(frozen=True)
class RadiusCase:
    case_id: str
    radius: float
    lx: float | None = None
    run_steps: int = RUN_STEPS
    trajectory_write_period: int = TRAJECTORY_WRITE_PERIOD
    seed: int = cylinder.SEED
    label: str = ""

    def __post_init__(self) -> None:
        if self.lx is None:
            object.__setattr__(self, "lx", cylinder.lx_for_radius(self.radius))

    @property
    def n_particles(self) -> int:
        return cylinder.n_particles_for_radius(
            self.radius,
            lx=self.lx,
            rho=cylinder.RHO,
        )

    @property
    def initial_gsd(self) -> Path:
        return INITIAL_GSD_DIR / f"initial_{self.case_id}.gsd"

    @property
    def trajectory_gsd(self) -> Path:
        return GSD_DIR / f"trajectory_{self.case_id}.gsd"

    @property
    def metadata_json(self) -> Path:
        return METADATA_DIR / f"{self.case_id}.json"

    @property
    def simulation_log(self) -> Path:
        return LOG_DIR / f"{self.case_id}_simulation.log"

    @property
    def analysis_log(self) -> Path:
        return LOG_DIR / f"{self.case_id}_analysis.log"

    def as_metadata(self) -> dict[str, float | int | str | None]:
        return {
            "case_id": self.case_id,
            "label": self.label or self.case_id,
            "radius": self.radius,
            "lx": self.lx,
            "n_particles": self.n_particles,
            "rho": cylinder.RHO,
            "particle_diameter": cylinder.PARTICLE_DIAMETER,
            "run_steps": self.run_steps,
            "trajectory_write_period": self.trajectory_write_period,
            "seed": self.seed,
            "initial_gsd": str(self.initial_gsd),
            "trajectory_gsd": str(self.trajectory_gsd),
        }


def radius_from_diameters(multiplier: float) -> float:
    return multiplier * cylinder.PARTICLE_DIAMETER


SCALED_RADIUS_CASES: tuple[RadiusCase, ...] = tuple(
    RadiusCase(
        case_id=f"radius_{int(value)}D",
        radius=radius_from_diameters(value),
        label=f"R = {int(value)}D",
    )
    # for value in (15.0, 20.0, 25.0, 30.0)
    for value in (15.0, 20.0, 25.0, 30.0)
)

SWEEP_CASES: tuple[RadiusCase, ...] = SCALED_RADIUS_CASES

LONG_AXIS_CASE = RadiusCase(
    case_id="long_axis_R10D_Lx500D",
    radius=radius_from_diameters(10.0),
    lx=LONG_AXIS_LX,
    label="R = 10D, Lx = 500D",
)


def all_cases(include_long_axis: bool = False) -> tuple[RadiusCase, ...]:
    if include_long_axis:
        return SWEEP_CASES + (LONG_AXIS_CASE,)
    return SWEEP_CASES


def get_case(case_id: str, include_long_axis: bool = False) -> RadiusCase:
    for case in all_cases(include_long_axis=include_long_axis):
        if case.case_id == case_id:
            return case
    known = ", ".join(case.case_id for case in all_cases(include_long_axis=True))
    raise KeyError(f"Unknown radius case {case_id!r}. Known cases: {known}")


def ensure_output_dirs() -> None:
    for path in (
        GSD_DIR,
        INITIAL_GSD_DIR,
        HEXATIC_OUTPUT_DIR,
        NPZ_FIELDS_DIR,
        LOG_DIR,
        METADATA_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
