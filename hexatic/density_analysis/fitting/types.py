from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


FIELD_COMPONENTS = ("x", "y")
ROLE_CANDIDATE = "candidate"
ROLE_TARGET = "target"
ROLE_AUXILIARY = "auxiliary"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    role: str
    label: str
    components: tuple[str, ...] = FIELD_COMPONENTS
    at_frames: bool = False


class FieldRegistry:
    def __init__(self, specs: Iterable[FieldSpec]) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def get(self, name: str) -> FieldSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._specs))
            raise KeyError(f"Unknown fitting field {name!r}. Known fields: {known}") from exc

    def names_for_role(self, role: str) -> tuple[str, ...]:
        return tuple(
            spec.name for spec in self._specs.values() if spec.role == role
        )

    def candidate_names(
        self,
        candidate_names: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        names = candidate_names or self.names_for_role(ROLE_CANDIDATE)
        if not candidate_names:
            raise ValueError("At least one fitting candidate is required.")
        for name in candidate_names:
            spec = self.get(name)
            if spec.role != ROLE_CANDIDATE:
                raise ValueError(f"{name!r} is registered as {spec.role!r}, not a candidate.")
        return candidate_names


DEFAULT_CANDIDATES = (
    "grad_rho",
    "P",
    "chiral_P_perp",
    "force_density",
    "grad_hexatic_order",
)

FIELD_REGISTRY = FieldRegistry(
    (
        FieldSpec("J", ROLE_TARGET, "film flux", at_frames=False),
        FieldSpec("rho", ROLE_AUXILIARY, "density", components=(), at_frames=True),
        FieldSpec("chirality", ROLE_AUXILIARY, "chirality", components=(), at_frames=True),
        FieldSpec("grad_rho", ROLE_CANDIDATE, "density gradient", at_frames=False),
        FieldSpec(
            "hexatic_order",
            ROLE_AUXILIARY,
            "hexatic order",
            components=(),
            at_frames=True,
        ),
        FieldSpec("P", ROLE_CANDIDATE, "polarization", at_frames=False),
        FieldSpec(
            "chiral_P_perp",
            ROLE_CANDIDATE,
            "chiral perpendicular polarization",
            at_frames=False,
        ),
        FieldSpec("force_density", ROLE_CANDIDATE, "force density", at_frames=False),
        FieldSpec(
            "grad_hexatic_order",
            ROLE_CANDIDATE,
            "hexatic-order gradient",
            at_frames=False,
        ),
    )
)
