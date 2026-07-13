from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from hexatic.constants import cylinder

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent
RUN_STEPS = int(1e8)
TRAJECTORY_WRITE_PERIOD = int(1e5)
LX_MULTIPLIERS = (1, 2, 4, 8, 16)
CIRCUMFERENCE_MULTIPLIERS = (60.0, 60.5)


def _number_token(value: float) -> str:
    return str(value).replace(".", "_")


@dataclass(frozen=True)
class BigLxCase:
    case_id: str
    circumference_diameters: float
    lx_multiplier: int
    run_steps: int = RUN_STEPS
    trajectory_write_period: int = TRAJECTORY_WRITE_PERIOD
    seed: int = cylinder.SEED

    def __post_init__(self) -> None:
        if self.lx_multiplier not in LX_MULTIPLIERS:
            raise ValueError(f"lx_multiplier must be one of {LX_MULTIPLIERS}")

    @property
    def circumference(self) -> float:
        return self.circumference_diameters * cylinder.PARTICLE_DIAMETER

    @property
    def radius(self) -> float:
        return self.circumference / (2.0 * math.pi)

    @property
    def n_theta(self) -> int:
        return int(math.floor(self.circumference / cylinder.PARTICLE_DIAMETER))

    @property
    def circumference_lattice_vector(self) -> tuple[int, int]:
        return 1, self.n_theta - 1

    @property
    def primitive_axial_lattice_vector(self) -> tuple[int, int]:
        m, n = self.circumference_lattice_vector
        divisor = math.gcd(2 * n + m, 2 * m + n)
        return (2 * n + m) // divisor, -(2 * m + n) // divisor

    @property
    def axial_lattice_vector(self) -> tuple[int, int]:
        p, q = self.primitive_axial_lattice_vector
        return self.lx_multiplier * p, self.lx_multiplier * q

    @property
    def lattice_spacing(self) -> float:
        m, n = self.circumference_lattice_vector
        return self.circumference / math.sqrt(m * m + m * n + n * n)

    @property
    def base_lx(self) -> float:
        p, q = self.primitive_axial_lattice_vector
        return self.lattice_spacing * math.sqrt(p * p + p * q + q * q)

    @property
    def lx(self) -> float:
        return self.lx_multiplier * self.base_lx

    @property
    def base_n_particles(self) -> int:
        m, n = self.circumference_lattice_vector
        p, q = self.primitive_axial_lattice_vector
        return abs(m * q - n * p)

    @property
    def n_particles(self) -> int:
        return self.lx_multiplier * self.base_n_particles

    @property
    def volume(self) -> float:
        return math.pi * self.radius * self.radius * self.lx

    @property
    def volume_density(self) -> float:
        return self.n_particles / self.volume

    @property
    def surface_density(self) -> float:
        return self.n_particles / (self.circumference * self.lx)

    @property
    def wall_radius(self) -> float:
        clearance = (
            cylinder.SIMULATION.wall_clearance_epsilon
            * cylinder.ANALYSIS.particle_diameter
        )
        return self.radius + cylinder.ANALYSIS.wall_cutoff + clearance

    @property
    def label(self) -> str:
        return f"C = {self.circumference_diameters:g}D, Lx = {self.lx_multiplier}x"

    def as_metadata(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "label": self.label,
            "circumference_diameters": self.circumference_diameters,
            "circumference": self.circumference,
            "radius": self.radius,
            "wall_radius": self.wall_radius,
            "lx_multiplier": self.lx_multiplier,
            "base_lx": self.base_lx,
            "lx": self.lx,
            "base_n_particles": self.base_n_particles,
            "n_particles": self.n_particles,
            "volume": self.volume,
            "volume_density": self.volume_density,
            "surface_density": self.surface_density,
            "particle_diameter": cylinder.PARTICLE_DIAMETER,
            "lattice_spacing": self.lattice_spacing,
            "n_theta": self.n_theta,
            "circumference_lattice_vector": self.circumference_lattice_vector,
            "primitive_axial_lattice_vector": self.primitive_axial_lattice_vector,
            "axial_lattice_vector": self.axial_lattice_vector,
            "run_steps": self.run_steps,
            "trajectory_write_period": self.trajectory_write_period,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class CasePaths:
    case: BigLxCase
    output_root: Path = DEFAULT_OUTPUT_ROOT

    @property
    def initial_gsd(self) -> Path:
        return self.output_root / "initial" / f"initial_{self.case.case_id}.gsd"

    @property
    def trajectory_gsd(self) -> Path:
        return self.output_root / "gsd" / f"trajectory_{self.case.case_id}.gsd"

    @property
    def metadata_json(self) -> Path:
        return self.output_root / "metadata" / f"{self.case.case_id}.json"

    @property
    def simulation_complete_json(self) -> Path:
        return self.output_root / "metadata" / f"{self.case.case_id}_simulation_complete.json"

    @property
    def simulation_log(self) -> Path:
        return self.output_root / "logs" / f"{self.case.case_id}_simulation.log"

    @property
    def analysis_log(self) -> Path:
        return self.output_root / "logs" / f"{self.case.case_id}_analysis.log"

    @property
    def analysis_dir(self) -> Path:
        return self.output_root / "safetensors_output" / self.case.case_id

    def ensure_parent_dirs(self) -> None:
        for path in (
            self.initial_gsd.parent,
            self.trajectory_gsd.parent,
            self.metadata_json.parent,
            self.simulation_log.parent,
            self.analysis_dir.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _make_case(circumference_diameters: float, lx_multiplier: int) -> BigLxCase:
    circumference_token = _number_token(circumference_diameters)
    return BigLxCase(
        case_id=f"circ_{circumference_token}D_lx_{lx_multiplier}x",
        circumference_diameters=circumference_diameters,
        lx_multiplier=lx_multiplier,
    )


SWEEP_CASES = tuple(
    _make_case(circumference, multiplier)
    for circumference in CIRCUMFERENCE_MULTIPLIERS
    for multiplier in LX_MULTIPLIERS
)


def all_cases() -> tuple[BigLxCase, ...]:
    return SWEEP_CASES


def ordered_cases(cases: tuple[BigLxCase, ...]) -> tuple[BigLxCase, ...]:
    return tuple(
        sorted(
            cases,
            key=lambda case: (case.n_particles, case.circumference_diameters),
            reverse=True,
        )
    )


def get_case(case_id: str) -> BigLxCase:
    for case in SWEEP_CASES:
        if case.case_id == case_id:
            return case
    known = ", ".join(case.case_id for case in SWEEP_CASES)
    raise KeyError(f"Unknown big-Lx case {case_id!r}. Known cases: {known}")


def select_cases(all_selected: bool, case_ids: list[str]) -> tuple[BigLxCase, ...]:
    if all_selected:
        return ordered_cases(all_cases())
    if case_ids:
        return ordered_cases(tuple(get_case(case_id) for case_id in case_ids))
    raise SystemExit("Select --all or one or more --case values.")
