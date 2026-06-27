from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .fit import FittingResult


PLOT_QUANTITIES = ("normalized_residual_x", "normalized_residual_y", "c_x", "c_y")
ROBUST_PERCENTILE = 98.0


def write_all_plots(
    result: FittingResult,
    output_dir: str | Path,
    *,
    frame_idx: int | None = None,
    case_id: str = "radius_15D",
) -> list[Path]:
    destination_dir = Path(output_dir)
    written = []
    for quantity in PLOT_QUANTITIES:
        print(f"[fitting] Writing {quantity} plot...")
        written.append(
            write_quantity_plot(
                result,
                quantity,
                destination_dir / f"{case_id}_fit_{quantity}.png",
                frame_idx=frame_idx,
            )
        )
    return written


def write_quantity_plot(
    result: FittingResult,
    quantity: str,
    output_path: str | Path,
    *,
    frame_idx: int | None = None,
) -> Path:
    values = _quantity_values(result, quantity)
    selected = (
        values
        if values.ndim == 2
        else _select_transition_values(values, frame_idx)
    )
    title_suffix = _frame_title_suffix(result, frame_idx)
    if values.ndim == 2:
        title_suffix = "(local coefficient map)"
    x_centers = np.asarray(result.x_centers, dtype=float)
    theta_centers = np.asarray(result.theta_centers, dtype=float)
    color_limits = _robust_color_limits(
        selected,
        symmetric="residual" in quantity,
    )

    fig, axis = plt.subplots(figsize=(10, 5))
    mesh = axis.pcolormesh(
        x_centers,
        theta_centers,
        selected.T,
        shading="auto",
        cmap="RdBu_r" if "residual" in quantity else "viridis",
        vmin=color_limits[0],
        vmax=color_limits[1],
    )
    colorbar = fig.colorbar(mesh, ax=axis, label=quantity)
    colorbar.ax.set_title("clipped", fontsize=8)
    axis.set_xlabel("x")
    axis.set_ylabel("theta")
    axis.set_title(f"{quantity} {title_suffix}")
    fig.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    print(f"[fitting] Wrote {destination}.")
    return destination


def _robust_color_limits(
    values: np.ndarray,
    *,
    symmetric: bool,
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

    low = float(np.nanpercentile(finite, 100.0 - percentile))
    high = float(np.nanpercentile(finite, percentile))
    if not np.isfinite(low) or not np.isfinite(high) or np.isclose(low, high):
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
    if not np.isfinite(low) or not np.isfinite(high) or np.isclose(low, high):
        return 0.0, 1.0
    return low, high


def _quantity_values(result: FittingResult, quantity: str) -> np.ndarray:
    if quantity == "normalized_residual_x":
        return _normalized_residual(result.residual_x, result.J[..., 0], result.mask)
    if quantity == "normalized_residual_y":
        return _normalized_residual(result.residual_y, result.J[..., 1], result.mask)
    if quantity == "c_x":
        return result.c_x
    if quantity == "c_y":
        return result.c_y
    raise ValueError(f"Unsupported fitting quantity: {quantity}")


def _normalized_residual(
    residual: np.ndarray,
    measured: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=float)
    measured = np.asarray(measured, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    valid = mask & np.isfinite(residual) & np.isfinite(measured)
    scale = float(np.nanmean(np.abs(measured[valid]))) if np.any(valid) else float("nan")
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    print(f"[fitting] Normalizing residual by mean(|J|)={scale:.6g}.")
    normalized = residual / scale
    return np.where(valid, normalized, np.nan)


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
