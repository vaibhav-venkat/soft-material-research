from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from hexatic.constants import cylinder

ANALYSIS_DIR = Path(__file__).resolve().parent
GSD_DIR = ANALYSIS_DIR / "gsd"
INITIAL_GSD_DIR = ANALYSIS_DIR / "initial"
HEXATIC_OUTPUT_DIR = ANALYSIS_DIR / "hexatic_output"
NPZ_FIELDS_DIR = ANALYSIS_DIR / "npz_fields"
LOG_DIR = ANALYSIS_DIR / "logs"
METADATA_DIR = ANALYSIS_DIR / "metadata"

RUN_STEPS = int(1e8)
TRAJECTORY_WRITE_PERIOD = int(1e5)


def _round_to_nearest_even(value: float) -> int:
    rounded = int(math.floor(value / 2.0 + 0.5) * 2)
    return max(2, rounded)


def circumference_from_diameters(multiplier: float) -> float:
    return multiplier * cylinder.PARTICLE_DIAMETER


def radius_from_diameters(multiplier: float) -> float:
    return multiplier * cylinder.PARTICLE_DIAMETER


@dataclass(frozen=True)
class UnwrappedCase:
    case_id: str
    radius: float
    circumference: float
    run_steps: int = RUN_STEPS
    trajectory_write_period: int = TRAJECTORY_WRITE_PERIOD
    seed: int = cylinder.SEED
    label: str = ""

    @property
    def n_theta(self) -> int:
        return int(math.floor(self.circumference / cylinder.PARTICLE_DIAMETER))

    @property
    def a(self) -> float:
        return self.circumference / self.n_theta

    @property
    def h(self) -> float:
        return self.a * math.sqrt(3.0) / 2.0

    @property
    def lx_target(self) -> float:
        return cylinder.lx_for_radius(self.radius, cylinder.X_RATIO)

    @property
    def n_x(self) -> int:
        return _round_to_nearest_even(self.lx_target / self.h)

    @property
    def lx(self) -> float:
        return self.n_x * self.h

    @property
    def wall_radius(self) -> float:
        analysis = cylinder.ANALYSIS
        simulation = cylinder.SIMULATION
        clearance = simulation.wall_clearance_epsilon * analysis.particle_diameter
        return self.radius + analysis.wall_cutoff + clearance

    @property
    def n_particles(self) -> int:
        return self.n_theta * self.n_x

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
            "surface_radius": self.radius,
            "wall_radius": self.wall_radius,
            "circumference": self.circumference,
            "n_theta": self.n_theta,
            "n_x": self.n_x,
            "a": self.a,
            "h": self.h,
            "lx_target": self.lx_target,
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


def case_from_circumference(multiplier: float) -> UnwrappedCase:
    circumference = circumference_from_diameters(multiplier)
    suffix = str(multiplier).replace(".", "_")
    return UnwrappedCase(
        case_id=f"circ_{suffix}D",
        radius=circumference / (2.0 * math.pi),
        circumference=circumference,
        label=f"C = {multiplier:g}D",
    )


def case_from_radius(multiplier: float) -> UnwrappedCase:
    radius = radius_from_diameters(multiplier)
    return UnwrappedCase(
        case_id=f"radius_{int(multiplier)}D",
        radius=radius,
        circumference=2.0 * math.pi * radius,
        label=f"R = {multiplier:g}D",
    )


SWEEP_CASES: tuple[UnwrappedCase, ...] = (
    case_from_circumference(60.0),
    case_from_circumference(60.25),
    case_from_circumference(60.5),
    case_from_circumference(60.75),
    case_from_circumference(61.0),
    case_from_radius(15.0),
)


def all_cases() -> tuple[UnwrappedCase, ...]:
    return SWEEP_CASES


def get_case(case_id: str) -> UnwrappedCase:
    for case in all_cases():
        if case.case_id == case_id:
            return case
    known = ", ".join(case.case_id for case in all_cases())
    raise KeyError(f"Unknown unwrapped case {case_id!r}. Known cases: {known}")


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
