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
EXPECTED_FRAME_COUNT = RUN_STEPS // TRAJECTORY_WRITE_PERIOD
BASE_CASE_ID = "circ_60_5D_lx_1x"


class GeometryKind(StrEnum):
    PRISM_VOLUME = "prism_volume"
    PRISM_SURFACE_AREA = "prism_surface_area"
    SANDWICH_VOLUME = "sandwich_volume"
    SANDWICH_SURFACE_AREA = "sandwich_surface_area"
    TWO_DIMENSION = "two_dimension"
    CYLINDER_RATTLE = "cylinder_rattle"
    CYLINDER_RATTLE_TANGENT = "cylinder_rattle_tangent"


@dataclass(frozen=True)
class ComparisonCase:
    case_id: str
    kind: GeometryKind
    base: BigLxCase
    run_steps: int = RUN_STEPS
    trajectory_write_period: int = TRAJECTORY_WRITE_PERIOD
    seed: int = cylinder.SEED

    @property
    def label(self) -> str:
        labels = {
            GeometryKind.PRISM_VOLUME: "equal-volume square prism",
            GeometryKind.PRISM_SURFACE_AREA: "equal-surface-area square prism",
            GeometryKind.SANDWICH_VOLUME: "equal-volume two-wall sandwich",
            GeometryKind.SANDWICH_SURFACE_AREA: (
                "equal-surface-area two-wall sandwich"
            ),
            GeometryKind.TWO_DIMENSION: "two-dimensional two-wall channel",
            GeometryKind.CYLINDER_RATTLE: "cylinder surface (RATTLE)",
            GeometryKind.CYLINDER_RATTLE_TANGENT: (
                "cylinder surface (RATTLE, tangent active force)"
            ),
        }
        return f"C = 60.5D, Lx = 1x, {labels[self.kind]}"

    @property
    def lx(self) -> float:
        return self.base.lx

    @property
    def radius(self) -> float:
        return self.base.radius

    @property
    def circumference(self) -> float:
        return self.base.circumference

    @property
    def n_particles(self) -> int:
        if self.kind in {
            GeometryKind.PRISM_VOLUME,
            GeometryKind.SANDWICH_VOLUME,
            GeometryKind.CYLINDER_RATTLE,
            GeometryKind.CYLINDER_RATTLE_TANGENT,
        }:
            return self.base.n_particles
        if self.kind == GeometryKind.TWO_DIMENSION:
            return int(
                round(self.base.surface_density * self.lx * self.transverse_span)
            )
        return int(
            round(self.base.volume_density * self.lx * self.transverse_span**2)
        )

    @property
    def prism_side(self) -> float:
        if self.kind == GeometryKind.PRISM_SURFACE_AREA:
            return 0.5 * math.pi * self.radius
        return self.radius * math.sqrt(math.pi)

    @property
    def transverse_span(self) -> float:
        if self.kind == GeometryKind.PRISM_SURFACE_AREA:
            return 0.5 * math.pi * self.radius
        if self.kind in {
            GeometryKind.SANDWICH_SURFACE_AREA,
            GeometryKind.TWO_DIMENSION,
        }:
            return math.pi * self.radius
        return self.radius * math.sqrt(math.pi)

    @property
    def initial_span_y(self) -> float:
        if self.kind == GeometryKind.PRISM_VOLUME:
            return self.prism_side
        if self.is_prism:
            return self.transverse_span - 2.0 * self.wall_clearance
        return self.transverse_span

    @property
    def initial_span_z(self) -> float:
        if self.kind == GeometryKind.PRISM_VOLUME:
            return self.prism_side
        if self.is_prism or self.is_sandwich:
            return self.transverse_span - 2.0 * self.wall_clearance
        return 0.0

    @property
    def wall_clearance(self) -> float:
        return (
            cylinder.ANALYSIS.wall_cutoff
            + cylinder.SIMULATION.wall_clearance_epsilon
            * cylinder.ANALYSIS.particle_diameter
        )

    @property
    def prism_wall_half_width(self) -> float:
        if self.kind == GeometryKind.PRISM_SURFACE_AREA:
            return 0.5 * self.transverse_span
        return 0.5 * self.prism_side + self.wall_clearance

    @property
    def prism_box_width(self) -> float:
        return 2.0 * self.prism_wall_half_width

    @property
    def cylinder_box_width(self) -> float:
        return 2.0 * self.base.wall_radius

    @property
    def stored_box(self) -> tuple[float, float, float]:
        if self.kind == GeometryKind.PRISM_VOLUME:
            return self.lx, self.prism_box_width, self.prism_box_width
        if self.kind == GeometryKind.PRISM_SURFACE_AREA:
            return self.lx, self.transverse_span, self.transverse_span
        if self.kind in {
            GeometryKind.SANDWICH_VOLUME,
            GeometryKind.SANDWICH_SURFACE_AREA,
        }:
            return self.lx, self.transverse_span, self.transverse_span
        if self.kind == GeometryKind.TWO_DIMENSION:
            return self.lx, self.transverse_span, 0.0
        # HOOMD's cylinder manifold is fixed along the stored z axis.
        return self.cylinder_box_width, self.cylinder_box_width, self.lx

    @property
    def logical_to_stored_axes(self) -> tuple[int, int, int]:
        if not self.is_cylinder:
            return 0, 1, 2
        return 1, 2, 0

    @property
    def is_constrained(self) -> bool:
        return self.is_cylinder

    @property
    def is_cylinder(self) -> bool:
        return self.kind in {
            GeometryKind.CYLINDER_RATTLE,
            GeometryKind.CYLINDER_RATTLE_TANGENT,
        }

    @property
    def is_prism(self) -> bool:
        return self.kind in {
            GeometryKind.PRISM_VOLUME,
            GeometryKind.PRISM_SURFACE_AREA,
        }

    @property
    def is_sandwich(self) -> bool:
        return self.kind in {
            GeometryKind.SANDWICH_VOLUME,
            GeometryKind.SANDWICH_SURFACE_AREA,
        }

    @property
    def is_2d(self) -> bool:
        return self.kind == GeometryKind.TWO_DIMENSION

    @property
    def dimensions(self) -> int:
        return 2 if self.is_2d else 3

    @property
    def periodic_axes(self) -> tuple[str, ...]:
        if self.is_cylinder:
            return ("x", "theta")
        if self.is_sandwich:
            return ("x", "y")
        return ("x",)

    @property
    def wall_faces(self) -> tuple[str, ...]:
        if self.is_prism:
            return ("+y", "-y", "+z", "-z")
        if self.is_sandwich:
            return ("+z", "-z")
        if self.is_2d:
            return ("+y", "-y")
        return ("radial",)

    def as_metadata(self) -> dict[str, object]:
        metadata = self.base.as_metadata()
        metadata.update(
            {
                "schema": "hexatic.confinement_comparison.case.v1",
                "case_id": self.case_id,
                "base_case_id": self.base.case_id,
                "label": self.label,
                "geometry_kind": self.kind.value,
                "run_steps": self.run_steps,
                "trajectory_write_period": self.trajectory_write_period,
                "expected_frame_count": self.run_steps
                // self.trajectory_write_period,
                "seed": self.seed,
                "n_particles": self.n_particles,
                "prism_side": self.prism_side,
                "prism_wall_half_width": self.prism_wall_half_width,
                "prism_box_width": self.prism_box_width,
                "transverse_span": self.transverse_span,
                "initial_span_y": self.initial_span_y,
                "initial_span_z": self.initial_span_z,
                "stored_box": self.stored_box,
                "wall_to_wall_dimensions": self.stored_box,
                "confinement_volume": (
                    self.lx * self.transverse_span**2
                    if not self.is_cylinder and not self.is_2d
                    else None
                ),
                "confinement_area": (
                    self.lx * self.transverse_span if self.is_2d else None
                ),
                "dimensions": self.dimensions,
                "periodic_axes": self.periodic_axes,
                "wall_faces": self.wall_faces,
                "wall_r_extrap": (
                    0.98 * cylinder.ANALYSIS.wall_cutoff
                    if self.kind
                    in {
                        GeometryKind.PRISM_SURFACE_AREA,
                        GeometryKind.SANDWICH_VOLUME,
                        GeometryKind.SANDWICH_SURFACE_AREA,
                        GeometryKind.TWO_DIMENSION,
                    }
                    else 0.0
                ),
                "particle_count_rule": (
                    "cylinder_surface_density"
                    if self.is_2d
                    else "cylinder_volume_density"
                    if self.kind in {
                        GeometryKind.PRISM_SURFACE_AREA,
                        GeometryKind.SANDWICH_VOLUME,
                        GeometryKind.SANDWICH_SURFACE_AREA,
                    }
                    else "base_case"
                ),
                "matching_rule": (
                    "equal_surface_area"
                    if self.kind
                    in {
                        GeometryKind.PRISM_SURFACE_AREA,
                        GeometryKind.SANDWICH_SURFACE_AREA,
                    }
                    else "equal_volume"
                    if self.kind
                    in {GeometryKind.PRISM_VOLUME, GeometryKind.SANDWICH_VOLUME}
                    else "cylinder_surface_density"
                    if self.is_2d
                    else "cylinder_manifold"
                ),
                "logical_to_stored_axes": self.logical_to_stored_axes,
                "logical_axial_axis": "x",
                "stored_axial_axis": "z" if self.is_cylinder else "x",
            }
        )
        return metadata


@dataclass(frozen=True)
class CasePaths:
    case: ComparisonCase
    output_root: Path = DEFAULT_OUTPUT_ROOT

    @property
    def initial_gsd(self) -> Path:
        return self.output_root / "initial" / f"initial_{self.case.case_id}.gsd"

    @property
    def trajectory_gsd(self) -> Path:
        return self.output_root / "gsd" / f"trajectory_{self.case.case_id}.gsd"

    @property
    def diagnostic_gsd(self) -> Path:
        return self.output_root / "diagnostics" / f"diagnostic_{self.case.case_id}.gsd"

    @property
    def diagnostic_json(self) -> Path:
        return self.output_root / "diagnostics" / f"diagnostic_{self.case.case_id}.json"

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
            self.diagnostic_gsd,
            self.metadata_json,
            self.simulation_log,
            self.analysis_dir / "manifest.json",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)


_BASE_CASE = get_big_lx_case(BASE_CASE_ID)
SWEEP_CASES = tuple(
    ComparisonCase(case_id=kind.value, kind=kind, base=_BASE_CASE)
    for kind in (
        GeometryKind.CYLINDER_RATTLE_TANGENT,
        GeometryKind.CYLINDER_RATTLE,
        GeometryKind.PRISM_VOLUME,
        GeometryKind.PRISM_SURFACE_AREA,
        GeometryKind.SANDWICH_SURFACE_AREA,
        GeometryKind.SANDWICH_VOLUME,
        GeometryKind.TWO_DIMENSION,
    )
)


def all_cases() -> tuple[ComparisonCase, ...]:
    return SWEEP_CASES


def get_case(case_id: str) -> ComparisonCase:
    for case in SWEEP_CASES:
        if case.case_id == case_id:
            return case
    known = ", ".join(case.case_id for case in SWEEP_CASES)
    raise KeyError(f"Unknown confinement-comparison case {case_id!r}: {known}")


def select_cases(all_selected: bool, case_ids: list[str]) -> tuple[ComparisonCase, ...]:
    if all_selected and case_ids:
        raise SystemExit("--all cannot be combined with --sim/--case.")
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
        "--case",
        dest="case",
        action="extend",
        nargs="+",
        choices=tuple(case.case_id for case in SWEEP_CASES),
        default=[],
        help="Run only the listed simulations; --case is a compatibility alias.",
    )
