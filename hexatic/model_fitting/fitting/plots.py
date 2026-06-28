from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .fit import FittingResult


ROBUST_PERCENTILE = 98.0


def write_all_plots(
    result: FittingResult,
    output_dir: str | Path,
    *,
    case_id: str = "radius_15D",
) -> list[Path]:
    """Write all diagnostic plots. Currently a no-op until Phase 7."""
    _ = result  # plots rebuilt in later phases
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    print("[fitting] No plots implemented yet (Phase 1 cleanup).")
    return []


def _robust_color_limits(
    values: np.ndarray,
    *,
    symmetric: bool,
    zero_floor: bool = False,
    percentile: float = ROBUST_PERCENTILE,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0

    if symmetric:
        limit = float(np.nanpercentile(np.abs(finite), percentile))
        if not np.isfinite(limit) or limit == 0.0:
            limit = float(np.nanmax(np.abs(finite)))
        if not np.isfinite(limit) or limit == 0.0:
            limit = 1.0
        return -limit, limit

    if zero_floor:
        high = float(np.nanpercentile(finite, percentile))
        if not np.isfinite(high) or high == 0.0:
            high = float(np.nanmax(finite))
        if not np.isfinite(high) or high == 0.0:
            high = 1.0
        return 0.0, high

    low = float(np.nanpercentile(finite, 100.0 - percentile))
    high = float(np.nanpercentile(finite, percentile))
    if not np.isfinite(low) or not np.isfinite(high) or np.isclose(low, high):
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
    if not np.isfinite(low) or not np.isfinite(high) or np.isclose(low, high):
        return 0.0, 1.0
    return low, high


def _select_transition_values(values: np.ndarray, frame_idx: int | None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if frame_idx is None:
        return np.nanmean(values, axis=0)
    if frame_idx < 0 or frame_idx >= values.shape[0]:
        raise IndexError(
            f"frame_idx={frame_idx} is outside transition range 0..{values.shape[0] - 1}."
        )
    return values[frame_idx]


def _frame_title_suffix(result: FittingResult, frame_idx: int | None) -> str:
    if frame_idx is None:
        return "(transition mean)"
    steps = np.asarray(result.transition_steps)
    if steps.ndim == 2 and frame_idx < steps.shape[0]:
        return f"(steps {steps[frame_idx, 0]} -> {steps[frame_idx, 1]})"
    return f"(transition {frame_idx})"
