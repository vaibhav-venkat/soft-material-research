"""Synthetic topology schedules shared by the event-model tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hexatic.band_analysis.detection.chains import EventChain
from hexatic.band_analysis.detection.events import EventStep, no_event_step
from hexatic.band_analysis.detection.segments import StableSegment


@dataclass(frozen=True)
class ScriptedSchedule:
    chain: EventChain
    event_kinds: tuple[str, ...]


def _topology_step(
    mask: np.ndarray,
    groups: tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...]], ...],
) -> EventStep:
    """Compose disjoint split/merge groups; identities fill the unaffected rows."""
    n_max = mask.size
    H = np.zeros((n_max, n_max))
    C = np.zeros_like(H)
    o = np.zeros(n_max, dtype=bool)
    event_rows = np.zeros(n_max, dtype=bool)
    mask_next = np.zeros(n_max, dtype=bool)
    claimed = set()
    targets = set()
    for sources, daughters, weights in groups:
        claimed.update(sources)
        targets.update(daughters)
        row = daughters[0]
        if len(sources) == 1:
            H[np.asarray(daughters), sources[0]] = weights
            C[row, sources[0]] = 1.0
        else:
            H[row, np.asarray(sources)] = 1.0
            C[row, np.asarray(sources)] = 1.0
        o[row] = True
        event_rows[row] = True
        mask_next[np.asarray(daughters)] = True
    for slot in np.flatnonzero(mask):
        if int(slot) not in claimed:
            H[slot, slot] = 1.0
            C[slot, slot] = 1.0
            o[slot] = True
            mask_next[slot] = True
            targets.add(int(slot))
    return EventStep(
        H=H,
        g=np.zeros(n_max),
        C=C,
        o=o,
        sigma_e_rows=event_rows,
        mask_next=mask_next,
        y=np.zeros(n_max),
    )


def _replacement_step(
    mask: np.ndarray, *, retired: int, created: int, birth_area: float
) -> EventStep:
    """A simultaneous death and birth that preserves active-band count."""
    n_max = mask.size
    survivors = mask.copy()
    survivors[retired] = False
    projector = np.diag(survivors.astype(float))
    mask_next = survivors.copy()
    mask_next[created] = True
    g = np.zeros(n_max)
    g[created] = birth_area
    return EventStep(
        H=projector,
        g=g,
        C=projector.copy(),
        o=survivors,
        sigma_e_rows=np.zeros(n_max, dtype=bool),
        mask_next=mask_next,
        y=np.zeros(n_max),
    )


def scripted_schedule(
    *,
    cycles: int,
    continuous_steps: int = 5,
    replacement_steps: int | None = None,
    dt: float = 0.2,
    seed_id: str = "synthetic",
) -> ScriptedSchedule:
    """Repeat k-way topology changes and count-preserving birth/death pairs."""
    n_max = 6
    mask = np.asarray([True, True, True, False, False, False])
    masks = [mask.copy()]
    steps: list[EventStep] = []
    kinds: list[str] = []

    def continuous(count: int = continuous_steps) -> None:
        nonlocal mask
        for _ in range(count):
            steps.append(no_event_step(mask, np.zeros(n_max)))
            masks.append(mask.copy())
            kinds.append("continuous")

    for _ in range(cycles):
        continuous()
        step = _topology_step(
            mask,
            (
                ((0,), (0, 3, 4), (0.5, 0.3, 0.2)),
                ((1, 2), (1,), (1.0,)),
            ),
        )
        steps.append(step)
        mask = step.mask_next
        masks.append(mask.copy())
        kinds.append("split3_merge2")

        continuous()
        step = _topology_step(
            mask,
            (
                ((0, 3, 4), (0,), (1.0,)),
                ((1,), (1, 2), (0.6, 0.4)),
            ),
        )
        steps.append(step)
        mask = step.mask_next
        masks.append(mask.copy())
        kinds.append("merge3_split2")

        continuous()
        step = _replacement_step(mask, retired=2, created=3, birth_area=5.0)
        steps.append(step)
        mask = step.mask_next
        masks.append(mask.copy())
        kinds.append("birth5_death2")

        continuous(replacement_steps or continuous_steps)
        step = _replacement_step(mask, retired=3, created=2, birth_area=2.0)
        steps.append(step)
        mask = step.mask_next
        masks.append(mask.copy())
        kinds.append("birth2_death3")

    length = len(masks)
    masks_array = np.asarray(masks)
    initial = np.zeros(n_max)
    initial[:3] = 4.0
    areas = np.zeros((length, n_max))
    areas[0] = initial
    slots = np.where(masks_array, np.arange(n_max)[None, :], -1)
    chain = EventChain(
        seed_id=seed_id,
        slot_ids=slots.astype(np.int64),
        frame_indices=np.arange(length, dtype=np.int64),
        tau=np.arange(length, dtype=np.float64) * dt,
        areas=areas,
        mask=masks_array,
        events=tuple(steps),
    )
    return ScriptedSchedule(chain, tuple(kinds))


def ignored_event_segments(chain: EventChain) -> list[StableSegment]:
    """Compact active slots, breaking only when fixed dimension requires it."""
    segments = []
    start = 0
    counts = chain.n_active
    for stop in range(1, chain.n_steps + 1):
        boundary = stop == chain.n_steps or counts[stop] != counts[start]
        if not boundary:
            continue
        active_rows = [chain.areas[index, chain.mask[index]] for index in range(start, stop)]
        if len(active_rows) >= 2:
            n_bands = int(counts[start])
            segments.append(
                StableSegment(
                    seed_id=chain.seed_id,
                    track_ids=tuple(range(n_bands)),
                    frame_indices=chain.frame_indices[start:stop].copy(),
                    tau=chain.tau[start:stop].copy(),
                    areas=np.asarray(active_rows),
                )
            )
        start = stop
    return segments
