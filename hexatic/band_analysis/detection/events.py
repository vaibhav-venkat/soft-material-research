"""Bipartite event structure to the padded maps (H, g, C, o, R, m') of one transition.

Classification reads the overlap degrees, never ``EventCode``: row degree > 1 is a
split, column degree > 1 a merge, an unmatched column a birth, an unmatched row a
death. Simultaneous events compose into the same maps and must touch disjoint
pre-event slots, post-event slots, and observation rows; a violation raises
``EventCompositionError`` so the chain builder can break the chain.

``R`` is ``sigma_E**2`` on selected diagonal rows and ``sigma_E`` is fitted, so the
matrix cannot be precomputed. ``EventStep.sigma_e_rows`` marks those rows and the
likelihood multiplies by the parameter.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from .tracking import EventEdges


class EventCompositionError(RuntimeError):
    """Simultaneous events overlap; the caller must break the chain here."""


@dataclass(frozen=True)
class EventStep:
    """Precomputed NumPy maps for one k -> k+1 transition, all at size ``N_max``."""

    H: np.ndarray  # (N, N) propagation
    g: np.ndarray  # (N,) propagation offset
    C: np.ndarray  # (N, N) observation
    o: np.ndarray  # (N,) bool informative rows
    sigma_e_rows: np.ndarray  # (N,) bool rows carrying sigma_E**2 in R
    mask_next: np.ndarray  # (N,) bool post-event mask m'
    y: np.ndarray  # (N,) observations, zero off informative rows


def initial_slots(track_ids: Iterable[int]) -> dict[int, int]:
    """Assign the lowest slots to the initial tracks, in track-id order."""
    return {track_id: slot for slot, track_id in enumerate(sorted(track_ids))}


def slot_mask(slots: Mapping[int, int], n_max: int) -> np.ndarray:
    mask = np.zeros(n_max, dtype=bool)
    mask[list(slots.values())] = True
    return mask


def pack_areas(
    slots: Mapping[int, int], areas: Mapping[int, float], n_max: int
) -> np.ndarray:
    """Place per-track areas into their slots, zero elsewhere."""
    packed = np.zeros(n_max)
    for track_id, slot in slots.items():
        packed[slot] = areas[track_id]
    return packed


def no_event_step(mask: np.ndarray, y: np.ndarray) -> EventStep:
    """Ordinary transition (also `EventCode.RESTORED`): H = C = M, g = 0, o = m, R = 0."""
    n_max = mask.size
    projector = np.diag(mask.astype(float))
    return EventStep(
        H=projector,
        g=np.zeros(n_max),
        C=projector.copy(),
        o=mask.copy(),
        sigma_e_rows=np.zeros(n_max, dtype=bool),
        mask_next=mask.copy(),
        y=np.where(mask, y, 0.0),
    )


def _claim(taken: set[int], indices: Iterable[int], kind: str) -> None:
    for index in indices:
        if index in taken:
            raise EventCompositionError(f"overlapping {kind} slot {index}")
        taken.add(index)


def _classify(edges: np.ndarray) -> tuple[dict[int, int], list[int], list[int], list[int]]:
    """Return identity pairs (row -> column) and the split rows, merge and birth columns."""
    column_degree = edges.sum(axis=0)
    pairs = {}
    splits = []
    for row in range(edges.shape[0]):
        columns = np.flatnonzero(edges[row])
        if columns.size > 1:
            splits.append(row)
        elif columns.size == 1 and column_degree[columns[0]] == 1:
            pairs[row] = int(columns[0])
    merges = [
        column for column in range(edges.shape[1]) if column_degree[column] > 1
    ]
    births = [
        column for column in range(edges.shape[1]) if column_degree[column] == 0
    ]
    return pairs, splits, merges, births


def build_event_step(
    edges: EventEdges,
    slots: Mapping[int, int],
    observed: Mapping[int, float],
    n_max: int,
) -> tuple[EventStep, dict[int, int]]:
    """Build the maps for one event frame and the post-event slot assignment.

    ``slots`` maps pre-event track ids to slots, ``observed`` maps post-event track
    ids to their observed areas. Raises ``EventCompositionError`` if simultaneous
    events are not disjoint.
    """
    matrix = edges.edges
    old_ids = edges.old_track_ids
    new_ids = edges.new_track_ids
    pairs, splits, merges, births = _classify(matrix)

    # Slot allocation: survivors and split parents keep their slot, a merge lands in
    # the lowest slot of its parents, everything created takes the lowest free slot.
    next_slots: dict[int, int] = {}
    pending: list[int] = []
    for row, column in pairs.items():
        next_slots[new_ids[column]] = slots[old_ids[row]]
    for row in splits:
        columns = np.flatnonzero(matrix[row])
        next_slots[new_ids[int(columns[0])]] = slots[old_ids[row]]
        pending.extend(int(column) for column in columns[1:])
    for column in merges:
        rows = np.flatnonzero(matrix[:, column])
        next_slots[new_ids[column]] = min(slots[old_ids[int(r)]] for r in rows)
    pending.extend(births)

    free = [slot for slot in range(n_max) if slot not in set(next_slots.values())]
    for offset, column in enumerate(sorted(pending)):
        next_slots[new_ids[column]] = free[offset]

    H = np.zeros((n_max, n_max))
    g = np.zeros(n_max)
    C = np.zeros((n_max, n_max))
    o = np.zeros(n_max, dtype=bool)
    sigma_e_rows = np.zeros(n_max, dtype=bool)
    mask_next = np.zeros(n_max, dtype=bool)
    y = np.zeros(n_max)

    taken_pre: set[int] = set()
    taken_post: set[int] = set()
    taken_obs: set[int] = set()

    for row, column in pairs.items():
        slot = slots[old_ids[row]]
        _claim(taken_pre, (slot,), "pre-event")
        _claim(taken_post, (slot,), "post-event")
        _claim(taken_obs, (slot,), "observation")
        H[slot, slot] = 1.0
        C[slot, slot] = 1.0
        o[slot] = True
        mask_next[slot] = True
        y[slot] = observed[new_ids[column]]

    for row in splits:
        parent = slots[old_ids[row]]
        columns = [int(column) for column in np.flatnonzero(matrix[row])]
        daughter_slots = [next_slots[new_ids[column]] for column in columns]
        daughter_areas = np.array([observed[new_ids[column]] for column in columns])
        weights = daughter_areas / daughter_areas.sum()
        _claim(taken_pre, (parent,), "pre-event")
        _claim(taken_post, daughter_slots, "post-event")
        _claim(taken_obs, (parent,), "observation")
        for slot, weight in zip(daughter_slots, weights):
            H[slot, parent] = weight
            mask_next[slot] = True
        # One observation row on the daughter sum; the other daughters are silent.
        C[parent, parent] = 1.0
        o[parent] = True
        sigma_e_rows[parent] = True
        y[parent] = daughter_areas.sum()

    for column in merges:
        parents = [slots[old_ids[int(row)]] for row in np.flatnonzero(matrix[:, column])]
        target = next_slots[new_ids[column]]
        _claim(taken_pre, parents, "pre-event")
        _claim(taken_post, (target,), "post-event")
        _claim(taken_obs, (target,), "observation")
        H[target, parents] = 1.0
        C[target, parents] = 1.0
        o[target] = True
        sigma_e_rows[target] = True
        mask_next[target] = True
        y[target] = observed[new_ids[column]]

    for column in births:
        slot = next_slots[new_ids[column]]
        _claim(taken_post, (slot,), "post-event")
        # Inserted exactly, carrying no density: o = 0 and no observation row.
        g[slot] = observed[new_ids[column]]
        mask_next[slot] = True

    deaths = [
        slots[old_ids[row]]
        for row in range(matrix.shape[0])
        if not matrix[row].any()
    ]
    # H row, C row, o and m' are already zero on a dead slot; only the claim matters.
    _claim(taken_pre, deaths, "pre-event")

    step = EventStep(
        H=H,
        g=g,
        C=C,
        o=o,
        sigma_e_rows=sigma_e_rows,
        mask_next=mask_next,
        y=y,
    )
    return step, next_slots
