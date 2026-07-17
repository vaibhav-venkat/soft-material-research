from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import StrEnum
import math
from pathlib import Path

from hexatic.big_lx.cases import BigLxCase, get_case as get_big_lx_case
from hexatic.constants import cylinder

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent
RUN_STEPS = int(1e8)
TRAJECTORY_WRITE_PERIOD = int(1e5)
VACANCY_RUN_FRAMES = 300
VACANCY_RUN_STEPS = VACANCY_RUN_FRAMES * TRAJECTORY_WRITE_PERIOD
BASE_CYLINDER_CASE_ID = "circ_60_5D_lx_1x"
DENSE_2D_NX = 88
DENSE_2D_NY = 60
PASSIVE_KT = 1.0
PASSIVE_STIFFNESS_MULTIPLIER = 50.0


class CaseKind(StrEnum):
    PASSIVE_CYLINDER = "passive_cylinder_60_5D"
    DENSE_2D = "dense_2d_60D"
    DENSE_2D_CENTER_VACANCY = "dense_2d_60D_center_vacancy"
    DENSE_2D_WALL_VACANCY = "dense_2d_60D_wall_vacancy"
    DENSE_2D_OPPOSITE_WALL_VACANCIES = (
        "dense_2d_60D_opposite_wall_vacancies"
    )


@dataclass(frozen=True)
class PassiveDenseCase:
    case_id: str
    kind: CaseKind
    base: BigLxCase
    run_steps: int = RUN_STEPS
    trajectory_write_period: int = TRAJECTORY_WRITE_PERIOD
    seed: int = cylinder.SEED

    @property
    def is_passive_cylinder(self) -> bool:
        return self.kind == CaseKind.PASSIVE_CYLINDER

    @property
    def is_dense_2d(self) -> bool:
        return not self.is_passive_cylinder

    @property
    def vacancy_count(self) -> int:
        if self.kind in {
            CaseKind.DENSE_2D_CENTER_VACANCY,
            CaseKind.DENSE_2D_WALL_VACANCY,
        }:
            return 1
        if self.kind == CaseKind.DENSE_2D_OPPOSITE_WALL_VACANCIES:
            return 2
        return 0

    @property
    def lattice_spacing(self) -> float:
        return cylinder.ANALYSIS.particle_diameter

    @property
    def lattice_height(self) -> float:
        return math.sqrt(3.0) * self.lattice_spacing / 2.0

    @property
    def nx(self) -> int:
        return DENSE_2D_NX

    @property
    def ny(self) -> int:
        return DENSE_2D_NY

    @property
    def n_particles(self) -> int:
        if self.is_passive_cylinder:
            return self.base.n_particles
        return DENSE_2D_NX * DENSE_2D_NY - self.vacancy_count

    @property
    def lx(self) -> float:
        if self.is_passive_cylinder:
            return self.base.lx
        return DENSE_2D_NX * self.lattice_height

    @property
    def ly(self) -> float:
        if self.is_passive_cylinder:
            return 2.0 * self.base.wall_radius
        # Alternating columns span (ny - 0.5) * D. Leave D/2 between
        # each outermost site and its wall without changing a = D.
        return (DENSE_2D_NY + 0.5) * self.lattice_spacing

    @property
    def stored_box(self) -> tuple[float, float, float]:
        if self.is_passive_cylinder:
            return self.lx, self.ly, self.ly
        return self.lx, self.ly, 0.0

    @property
    def dimensions(self) -> int:
        return 3 if self.is_passive_cylinder else 2

    @property
    def label(self) -> str:
        if self.is_passive_cylinder:
            return "Passive Brownian hard-sphere approximation, twisted C = 60.5D"
        labels = {
            CaseKind.DENSE_2D: "Active dense 2D untwisted crystal, a = D",
            CaseKind.DENSE_2D_CENTER_VACANCY: (
                "Active dense 2D crystal with one center vacancy"
            ),
            CaseKind.DENSE_2D_WALL_VACANCY: (
                "Active dense 2D crystal with one wall vacancy"
            ),
            CaseKind.DENSE_2D_OPPOSITE_WALL_VACANCIES: (
                "Active dense 2D crystal with inversion-symmetric wall vacancies"
            ),
        }
        return labels[self.kind]

    def as_metadata(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": "hexatic.confinement_comparison.passive_dense.case.v1",
            "case_id": self.case_id,
            "geometry_kind": self.kind.value,
            "label": self.label,
            "seed": self.seed,
            "run_steps": self.run_steps,
            "trajectory_write_period": self.trajectory_write_period,
            "n_particles": self.n_particles,
            "stored_box": self.stored_box,
            "dimensions": self.dimensions,
            "particle_diameter": cylinder.ANALYSIS.particle_diameter,
            "lattice_spacing": self.lattice_spacing,
            "periodic_axes": ("x",),
        }
        if self.is_passive_cylinder:
            payload.update(
                base_case_id=self.base.case_id,
                circumference=self.base.circumference,
                radius=self.base.radius,
                wall_radius=self.base.wall_radius,
                circumference_lattice_vector=self.base.circumference_lattice_vector,
                axial_lattice_vector=self.base.axial_lattice_vector,
                dynamics="brownian",
                kT=PASSIVE_KT,
                gamma=cylinder.SIMULATION.gamma,
                interaction_epsilon=PASSIVE_STIFFNESS_MULTIPLIER * PASSIVE_KT,
                active_force=False,
                manifold_constrained=False,
                initialization="exact_twisted_integer_supercell",
            )
        else:
            payload.update(
                nx=DENSE_2D_NX,
                ny=DENSE_2D_NY,
                lattice_height=self.lattice_height,
                wall_faces=("+y", "-y"),
                initial_wall_distance=0.5 * self.lattice_spacing,
                dynamics="active_overdamped_viscous",
                active_force=True,
                initialization="exact_untwisted_triangular_patch",
                vacancy_count=self.vacancy_count,
                vacancy_rule=(
                    "closest_to_origin"
                    if self.kind == CaseKind.DENSE_2D_CENTER_VACANCY
                    else "closest_to_positive_y_wall"
                    if self.kind == CaseKind.DENSE_2D_WALL_VACANCY
                    else "inversion_pair_near_opposite_walls"
                    if self.kind == CaseKind.DENSE_2D_OPPOSITE_WALL_VACANCIES
                    else "none"
                ),
                wall_r_extrap=0.98 * cylinder.ANALYSIS.wall_cutoff,
            )
        return payload


@dataclass(frozen=True)
class CasePaths:
    case: PassiveDenseCase
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

    def ensure_parent_dirs(self) -> None:
        for path in (
            self.initial_gsd,
            self.trajectory_gsd,
            self.metadata_json,
            self.simulation_log,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)


_BASE_CASE = get_big_lx_case(BASE_CYLINDER_CASE_ID)
SWEEP_CASES = tuple(
    PassiveDenseCase(
        case_id=kind.value,
        kind=kind,
        base=_BASE_CASE,
        run_steps=(
            VACANCY_RUN_STEPS
            if kind
            in {
                CaseKind.DENSE_2D_CENTER_VACANCY,
                CaseKind.DENSE_2D_WALL_VACANCY,
                CaseKind.DENSE_2D_OPPOSITE_WALL_VACANCIES,
            }
            else RUN_STEPS
        ),
    )
    for kind in (
        CaseKind.PASSIVE_CYLINDER,
        CaseKind.DENSE_2D,
        CaseKind.DENSE_2D_CENTER_VACANCY,
        CaseKind.DENSE_2D_WALL_VACANCY,
        CaseKind.DENSE_2D_OPPOSITE_WALL_VACANCIES,
    )
)


def all_cases() -> tuple[PassiveDenseCase, ...]:
    return SWEEP_CASES


def get_case(case_id: str) -> PassiveDenseCase:
    for case in SWEEP_CASES:
        if case.case_id == case_id:
            return case
    known = ", ".join(case.case_id for case in SWEEP_CASES)
    raise KeyError(f"Unknown passive/dense case {case_id!r}. Known cases: {known}")


def select_cases(all_selected: bool, case_ids: list[str]) -> tuple[PassiveDenseCase, ...]:
    if all_selected:
        return all_cases()
    if case_ids:
        return tuple(get_case(case_id) for case_id in case_ids)
    raise SystemExit("Select --all or one or more --sim values.")


def add_case_selection_arguments(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument(
        "--sim",
        dest="case",
        action="extend",
        nargs="+",
        choices=tuple(case.case_id for case in SWEEP_CASES),
        default=[],
        help="Run only the listed simulations.",
    )
