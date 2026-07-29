from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from pathlib import Path

from hexatic.big_lx.cases import BigLxCase, get_case as get_big_lx_case
from hexatic.constants import cylinder

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent
BASE_CASE_ID = "circ_60_5D_lx_1x"
RUN_STEPS = int(1e8)
TRAJECTORY_WRITE_PERIOD = int(1e5)
FRAMES_PER_SHARD = 100
PASSIVE_KT = 1.0
PASSIVE_STIFFNESS_MULTIPLIER = 50.0


class CaseKind(StrEnum):
    IDEAL_ABP_CYLINDER = "ideal_abp_cylinder"
    X_WALLED_PRISM = "x_walled_prism"
    PERIODIC_2D_PLANE = "periodic_2d_plane"
    IDEAL_2D_X_WALLS = "ideal_2d_x_walls"
    IDEAL_2D_PERIODIC = "ideal_2d_periodic"
    PERIODIC_3D_BULK = "periodic_3d_bulk"
    IDEAL_3D_BULK = "ideal_3d_bulk"
    IDEAL_3D_BULK_PERIOD_10 = "ideal_3d_bulk_period_10"
    IDEAL_3D_BULK_PERIOD_1 = "ideal_3d_bulk_period_1"
    SINGLE_ACTIVE_TRACER_CYLINDER = "single_active_tracer_cylinder"
    INVERSION_ACTIVE_PAIR_CYLINDER = "inversion_active_pair_cylinder"


@dataclass(frozen=True)
class NewSimCase:
    case_id: str
    kind: CaseKind
    base: BigLxCase
    run_steps: int = RUN_STEPS
    trajectory_write_period: int = TRAJECTORY_WRITE_PERIOD
    seed: int = cylinder.SEED
    diffusion_period: int | None = None
    tau_r: float = cylinder.SIMULATION.tau_r

    @property
    def lx(self) -> float:
        return self.base.lx

    @property
    def circumference(self) -> float:
        return self.base.circumference

    @property
    def radius(self) -> float:
        return self.base.radius

    @property
    def wall_radius(self) -> float:
        return self.base.wall_radius

    @property
    def prism_side(self) -> float:
        return self.radius * math.sqrt(math.pi)

    @property
    def n_particles(self) -> int:
        return self.base.n_particles

    @property
    def dimensions(self) -> int:
        return 2 if self.is_2d else 3

    @property
    def stored_box(self) -> tuple[float, float, float]:
        if self.is_cylinder:
            width = 2.0 * self.wall_radius
            return self.lx, width, width
        if self.is_2d:
            return self.lx, self.circumference, 0.0
        return self.lx, self.prism_side, self.prism_side

    @property
    def is_cylinder(self) -> bool:
        return self.kind in {
            CaseKind.IDEAL_ABP_CYLINDER,
            CaseKind.SINGLE_ACTIVE_TRACER_CYLINDER,
            CaseKind.INVERSION_ACTIVE_PAIR_CYLINDER,
        }

    @property
    def is_tracer(self) -> bool:
        return self.kind in {
            CaseKind.SINGLE_ACTIVE_TRACER_CYLINDER,
            CaseKind.INVERSION_ACTIVE_PAIR_CYLINDER,
        }

    @property
    def is_2d(self) -> bool:
        return self.kind in {
            CaseKind.PERIODIC_2D_PLANE,
            CaseKind.IDEAL_2D_X_WALLS,
            CaseKind.IDEAL_2D_PERIODIC,
        }

    @property
    def has_x_walls(self) -> bool:
        return self.kind in {
            CaseKind.X_WALLED_PRISM,
            CaseKind.IDEAL_2D_X_WALLS,
        }

    @property
    def has_pair_interaction(self) -> bool:
        return self.kind not in {
            CaseKind.IDEAL_ABP_CYLINDER,
            CaseKind.IDEAL_2D_X_WALLS,
            CaseKind.IDEAL_2D_PERIODIC,
            CaseKind.IDEAL_3D_BULK,
            CaseKind.IDEAL_3D_BULK_PERIOD_10,
            CaseKind.IDEAL_3D_BULK_PERIOD_1,
        }

    @property
    def is_3d_bulk(self) -> bool:
        return self.kind in {
            CaseKind.PERIODIC_3D_BULK,
            CaseKind.IDEAL_3D_BULK,
            CaseKind.IDEAL_3D_BULK_PERIOD_10,
            CaseKind.IDEAL_3D_BULK_PERIOD_1,
        }

    @property
    def rotational_diffusion_period(self) -> int:
        if self.diffusion_period is not None:
            return self.diffusion_period
        if self.kind == CaseKind.IDEAL_3D_BULK_PERIOD_1:
            return 1
        return cylinder.SIMULATION.rotational_diffusion_period

    @property
    def active_count(self) -> int:
        if self.kind == CaseKind.SINGLE_ACTIVE_TRACER_CYLINDER:
            return 1
        if self.kind == CaseKind.INVERSION_ACTIVE_PAIR_CYLINDER:
            return 2
        return self.n_particles

    @property
    def periodic_axes(self) -> tuple[str, ...]:
        if self.kind == CaseKind.X_WALLED_PRISM:
            return "y", "z"
        if self.kind == CaseKind.IDEAL_2D_X_WALLS:
            return ("y",)
        if self.is_2d:
            return "x", "y"
        if self.is_3d_bulk:
            return "x", "y", "z"
        return ("x",)

    @property
    def wall_faces(self) -> tuple[str, ...]:
        if self.has_x_walls:
            return "+x", "-x"
        if self.is_cylinder:
            return ("radial",)
        return ()

    @property
    def has_hexatic(self) -> bool:
        return not self.is_3d_bulk

    @property
    def label(self) -> str:
        labels = {
            CaseKind.IDEAL_ABP_CYLINDER: "ideal non-interacting ABPs in a cylinder",
            CaseKind.X_WALLED_PRISM: "active prism with x walls and periodic y/z",
            CaseKind.PERIODIC_2D_PLANE: "fully periodic unwrapped 2D ABPs",
            CaseKind.IDEAL_2D_X_WALLS: (
                "non-interacting 2D ABPs with x walls and periodic y"
            ),
            CaseKind.IDEAL_2D_PERIODIC: "non-interacting fully periodic 2D ABPs",
            CaseKind.PERIODIC_3D_BULK: "fully periodic 3D ABPs",
            CaseKind.IDEAL_3D_BULK: "non-interacting fully periodic 3D ABPs",
            CaseKind.IDEAL_3D_BULK_PERIOD_10: (
                "non-interacting fully periodic 3D ABPs with diffusion period 10"
            ),
            CaseKind.IDEAL_3D_BULK_PERIOD_1: (
                "non-interacting fully periodic 3D ABPs with diffusion period 1"
            ),
            CaseKind.SINGLE_ACTIVE_TRACER_CYLINDER: (
                "one active tracer in a passive twisted cylinder film"
            ),
            CaseKind.INVERSION_ACTIVE_PAIR_CYLINDER: (
                "two inversion-symmetric active tracers in a passive twisted cylinder film"
            ),
        }
        return labels[self.kind]

    def as_metadata(self) -> dict[str, object]:
        standard_epsilon = (
            cylinder.SIMULATION.interaction_epsilon_multiplier
            * cylinder.SIMULATION.gamma
            * cylinder.SIMULATION.u0
            * cylinder.ANALYSIS.sigma
        )
        payload: dict[str, object] = {
            "schema": "hexatic.new_sims.case.v1",
            "case_id": self.case_id,
            "geometry_kind": self.kind.value,
            "label": self.label,
            "base_case_id": self.base.case_id,
            "seed": self.seed,
            "run_steps": self.run_steps,
            "trajectory_write_period": self.trajectory_write_period,
            "expected_frame_count": self.run_steps // self.trajectory_write_period,
            "n_particles": self.n_particles,
            "active_count": self.active_count,
            "passive_count": self.n_particles - self.active_count,
            "particle_diameter": cylinder.ANALYSIS.particle_diameter,
            "polydisperse": False,
            "lx": self.lx,
            "circumference": self.circumference,
            "radius": self.radius,
            "wall_radius": self.wall_radius,
            "prism_side": self.prism_side,
            "stored_box": self.stored_box,
            "dimensions": self.dimensions,
            "periodic_axes": self.periodic_axes,
            "wall_faces": self.wall_faces,
            "wall_interaction": bool(self.wall_faces),
            "wall_interaction_epsilon": (
                None
                if not self.wall_faces
                else PASSIVE_STIFFNESS_MULTIPLIER * PASSIVE_KT
                if self.is_tracer
                else standard_epsilon
            ),
            "has_hexatic": self.has_hexatic,
            "initialization": (
                "exact_wrapped_twisted_triangular_supercell"
                if self.is_tracer
                else "x_compressed_flattened_triangular_supercell_random_planar_polarization"
                if self.kind == CaseKind.IDEAL_2D_X_WALLS
                else "flattened_triangular_supercell_random_planar_polarization"
                if self.is_2d
                else "bulk_lattice_random_uniform_3d_polarization"
            ),
            "pair_interaction": self.has_pair_interaction,
            "interaction_epsilon": (
                None
                if not self.has_pair_interaction
                else PASSIVE_STIFFNESS_MULTIPLIER * PASSIVE_KT
                if self.is_tracer
                else standard_epsilon
            ),
            "active_force_magnitude": (
                cylinder.SIMULATION.gamma * cylinder.SIMULATION.u0
            ),
            "tau_r": self.tau_r,
            "rotational_diffusion": 1.0 / self.tau_r,
            "rotational_diffusion_period": self.rotational_diffusion_period,
            "timestep": cylinder.SIMULATION.timestep,
        }
        if self.is_tracer:
            payload.update(
                passive_dynamics="brownian",
                passive_kT=PASSIVE_KT,
                active_dynamics="overdamped_viscous",
                pair_type_rule="same_50kT_LJ_for_all_type_pairs",
            )
        else:
            payload["dynamics"] = "active_overdamped_viscous"
        return payload


@dataclass(frozen=True)
class CasePaths:
    case: NewSimCase
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
            self.initial_gsd,
            self.trajectory_gsd,
            self.metadata_json,
            self.simulation_log,
            self.analysis_log,
            self.analysis_dir / "manifest.json",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)


_BASE = get_big_lx_case(BASE_CASE_ID)
SWEEP_CASES = tuple(
    NewSimCase(case_id=kind.value, kind=kind, base=_BASE) for kind in CaseKind
)


def all_cases() -> tuple[NewSimCase, ...]:
    return SWEEP_CASES


def get_case(case_id: str) -> NewSimCase:
    for case in SWEEP_CASES:
        if case.case_id == case_id:
            return case
    known = ", ".join(case.case_id for case in SWEEP_CASES)
    raise KeyError(f"Unknown new simulation {case_id!r}; known cases: {known}")
