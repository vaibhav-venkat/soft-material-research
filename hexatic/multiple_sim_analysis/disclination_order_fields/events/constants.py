from __future__ import annotations

from dataclasses import dataclass

from hexatic.constants import cylinder


@dataclass(frozen=True)
class EventAnalysisConstants:
    particle_diameter: float = cylinder.PARTICLE_DIAMETER
    neighbor_count_radius: float = cylinder.ANALYSIS.neighbor_count_radius
    match_tolerance: float = 1.5 * cylinder.ANALYSIS.neighbor_count_radius
    cluster_bond_length: float = 4.5 * cylinder.ANALYSIS.neighbor_count_radius
    annulus_core_radius: float = cylinder.ANALYSIS.neighbor_count_radius
    annulus_outer_radius: float = 3.0 * cylinder.ANALYSIS.neighbor_count_radius
    persistence_frames: int = 2


DEFAULT_EVENT_CONSTANTS = EventAnalysisConstants()
