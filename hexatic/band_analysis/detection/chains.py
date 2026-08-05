"""Event-capable chains: padded masked state plus one ``EventStep`` per transition.

A chain is broken only by ``EventCode.UNCERTAIN``, ``EventCode.TRANSIENT_GAP`` and a
non-disjoint simultaneous event (``EventCompositionError``). ``EventCode.RESTORED``
is *not* a break: the band-less gap frames are dropped as rows and the pre-gap frame
connects straight to the restored frame, so the transition carries the full elapsed
``delta tau``.

``N_max`` is global across every seed, so building is two-pass: :func:`plan_chains`
segments the tracked frames without allocating anything, :func:`global_n_max` reduces
over all plans, and :func:`build_event_chains` then builds at that single padded shape
(one JIT trace, ``vmap``-able).

``EventChain.events`` has uniform length ``T - 1`` -- one step per transition, with
``no_event_step`` filling the eventless ones -- rather than only the event-carrying
transitions, because the likelihood needs a step for every transition.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .events import (
    EventCompositionError,
    EventStep,
    build_event_step,
    initial_slots,
    no_event_step,
    pack_areas,
    slot_mask,
)
from .tracking import EventCode, TrackedFrame


@dataclass(frozen=True)
class EventChain:
    seed_id: str
    slot_ids: np.ndarray  # (T, N_max) int64, track_id per slot, -1 if empty
    frame_indices: np.ndarray  # (T,)
    tau: np.ndarray  # (T,)
    areas: np.ndarray  # (T, N_max), zero where inactive
    mask: np.ndarray  # (T, N_max) bool
    events: tuple[EventStep, ...]  # (T - 1,) one per transition

    @property
    def n_steps(self) -> int:
        return self.tau.size

    @property
    def n_max(self) -> int:
        return self.mask.shape[1]

    @property
    def n_active(self) -> np.ndarray:
        return self.mask.sum(axis=1)


@dataclass(frozen=True)
class ChainPlan:
    """Rows of one unbroken chain, before padding to the global ``N_max``."""

    seed_id: str
    rows: tuple[TrackedFrame, ...]

    @property
    def max_bands(self) -> int:
        return max(len(row.bands) for row in self.rows)


def plan_chains(frames: Sequence[TrackedFrame], *, seed_id: str) -> list[ChainPlan]:
    """Segment tracked frames into chains, bridging restored gaps."""
    plans: list[ChainPlan] = []
    current: list[TrackedFrame] = []
    in_gap = False

    def finish() -> None:
        if len(current) >= 2:
            plans.append(ChainPlan(seed_id, tuple(current)))

    for frame in frames:
        if not frame.bands:
            in_gap = True
            continue
        codes = {record.code for record in frame.events}
        restored = EventCode.RESTORED in codes
        if EventCode.UNCERTAIN in codes or (in_gap and not restored):
            finish()
            current = []
        in_gap = False
        current.append(frame)
    finish()
    return plans


def global_n_max(plans: Iterable[ChainPlan]) -> int:
    """The single padded width shared by every chain of every seed."""
    return max(plan.max_bands for plan in plans)


def _slot_ids(slots: Mapping[int, int], n_max: int) -> np.ndarray:
    ids = np.full(n_max, -1, dtype=np.int64)
    for track_id, slot in slots.items():
        ids[slot] = track_id
    return ids


def _build_one(
    seed_id: str, rows: Sequence[TrackedFrame], n_max: int
) -> tuple[EventChain | None, int]:
    """Build one chain from ``rows[0]`` onward; return it and where the next starts."""
    slots = initial_slots(band.track_id for band in rows[0].bands)
    observed = {band.track_id: band.area for band in rows[0].bands}

    slot_ids = [_slot_ids(slots, n_max)]
    masks = [slot_mask(slots, n_max)]
    areas = [pack_areas(slots, observed, n_max)]
    events: list[EventStep] = []
    end = len(rows)

    for k in range(len(rows) - 1):
        nxt = rows[k + 1]
        observed = {band.track_id: band.area for band in nxt.bands}
        if nxt.edges is None:
            step = no_event_step(masks[-1], pack_areas(slots, observed, n_max))
            next_slots = dict(slots)
        else:
            try:
                step, next_slots = build_event_step(nxt.edges, slots, observed, n_max)
            except EventCompositionError:
                end = k + 1
                break
        events.append(step)
        slots = next_slots
        slot_ids.append(_slot_ids(slots, n_max))
        masks.append(slot_mask(slots, n_max))
        areas.append(pack_areas(slots, observed, n_max))

    if end < 2:
        return None, end
    chain = EventChain(
        seed_id=seed_id,
        slot_ids=np.asarray(slot_ids),
        frame_indices=np.asarray([row.frame_index for row in rows[:end]], dtype=np.int64),
        tau=np.asarray([row.tau for row in rows[:end]], dtype=np.float64),
        areas=np.asarray(areas),
        mask=np.asarray(masks),
        events=tuple(events),
    )
    return chain, end


def build_event_chains(plan: ChainPlan, n_max: int) -> list[EventChain]:
    """Build the chains of one plan, splitting again on a composition failure."""
    chains: list[EventChain] = []
    start = 0
    while start < len(plan.rows) - 1:
        chain, consumed = _build_one(plan.seed_id, plan.rows[start:], n_max)
        if chain is not None:
            chains.append(chain)
        start += consumed
    return chains


def build_chains(tracked: Mapping[str, Sequence[TrackedFrame]]) -> list[EventChain]:
    """Two-pass build over every seed against one global ``N_max``."""
    plans = [
        plan for seed_id, frames in tracked.items()
        for plan in plan_chains(frames, seed_id=seed_id)
    ]
    if not plans:
        return []
    n_max = global_n_max(plans)
    return [chain for plan in plans for chain in build_event_chains(plan, n_max)]


def _is_event(step: EventStep, mask: np.ndarray) -> bool:
    """Compare against the ordinary transition; a birth only moves ``g``, not ``H``."""
    plain = no_event_step(mask, step.y)
    return not all(
        np.array_equal(getattr(step, field), getattr(plain, field))
        for field in ("H", "g", "C", "o", "sigma_e_rows", "mask_next")
    )


def event_transitions(chain: EventChain) -> np.ndarray:
    """(T - 1,) bool: transitions carrying an event."""
    return np.asarray(
        [_is_event(step, chain.mask[k]) for k, step in enumerate(chain.events)],
        dtype=bool,
    )


@dataclass(frozen=True)
class KnotGrid:
    """Sampled indices of one (lag, phase), with mandatory knots at every event."""

    lag: int
    phase: int
    knots: np.ndarray  # (K,) int64, sorted and unique
    event_index: np.ndarray  # (K - 1,) index into EventChain.events, -1 if continuous

    @property
    def weight(self) -> float:
        return 1.0 / self.lag

    @property
    def start(self) -> np.ndarray:
        return self.knots[:-1]

    @property
    def end(self) -> np.ndarray:
        return self.knots[1:]


def build_knot_grid(
    chain: EventChain, lag: int, phase: int, flags: np.ndarray | None = None
) -> KnotGrid | None:
    """Ordinary indices ``phase, phase + lag, ...`` plus ``k, k + 1`` at every event.

    Because both ``k`` and ``k + 1`` are knots they are adjacent after sorting, so no
    transition can span an event boundary.
    """
    if flags is None:
        flags = event_transitions(chain)
    n_steps = chain.n_steps
    knots = set(range(phase, n_steps, lag))
    for k in np.flatnonzero(flags):
        knots.update((int(k), int(k) + 1))
    ordered = np.asarray(sorted(knots), dtype=np.int64)
    if ordered.size < 2:
        return None
    start, end = ordered[:-1], ordered[1:]
    event_index = np.where((end == start + 1) & flags[start], start, -1)
    return KnotGrid(lag=lag, phase=phase, knots=ordered, event_index=event_index)


def build_knot_grids(chain: EventChain, lags: Iterable[int]) -> tuple[KnotGrid, ...]:
    """All (lag, phase) grids of one chain."""
    flags = event_transitions(chain)
    grids = (
        build_knot_grid(chain, lag, phase, flags)
        for lag in lags
        for phase in range(lag)
    )
    return tuple(grid for grid in grids if grid is not None)
