"""Visual outputs from a FittingResult: true-predicted views, residual maps,
contribution maps, and curl residual structure."""

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
    """Write all diagnostic plots to output_dir. Returns list of saved paths."""
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. True vs predicted scatter
    for label, target, prediction, fname in [
        ("density Y_rho", result.density_target, result.density.prediction,
         f"{case_id}_density_true_vs_predicted.png"),
        ("partial_t P_x", result.polarization_target[..., 0],
         result.polarization.prediction[..., 0],
         f"{case_id}_polarization_Px_true_vs_predicted.png"),
        ("partial_t P_y", result.polarization_target[..., 1],
         result.polarization.prediction[..., 1],
         f"{case_id}_polarization_Py_true_vs_predicted.png"),
    ]:
        p = _scatter_plot(result, target, prediction, label, dest / fname)
        saved.append(p)

    # 2. Density residual map (transition mean)
    p = _residual_map(
        result.density.residual,
        "Density residual",
        dest / f"{case_id}_density_residual_map.png",
        mask=result.mask,
    )
    saved.append(p)

    # 3. Polarization residual map (vector magnitude, transition mean)
    res_mag = np.sqrt(
        result.polarization.residual[..., 0] ** 2
        + result.polarization.residual[..., 1] ** 2
    )
    p = _residual_map(
        res_mag,
        "Polarization residual |r|",
        dest / f"{case_id}_polarization_residual_map.png",
        mask=result.mask,
    )
    saved.append(p)

    # 4. Density term contribution maps
    p = _density_contribution_plots(
        result, dest / f"{case_id}_density_contributions.png",
    )
    saved.append(p)

    # 5. Curl residual structure
    p = _curl_residual_plot(
        result, dest / f"{case_id}_polarization_curl_residual.png",
    )
    saved.append(p)

    print(f"[fitting] Plots saved to {dest}")
    return saved


# ---------------------------------------------------------------------------
# True vs predicted scatter
# ---------------------------------------------------------------------------

def _scatter_plot(
    result: FittingResult,
    target: np.ndarray,
    prediction: np.ndarray,
    label: str,
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    valid = result.mask & np.isfinite(target) & np.isfinite(prediction)
    t = target[valid].ravel()
    p = prediction[valid].ravel()
    ax.scatter(t, p, s=1, alpha=0.3, c="k")
    lim = [
        min(np.nanmin(t), np.nanmin(p)),
        max(np.nanmax(t), np.nanmax(p)),
    ]
    ax.plot(lim, lim, "r--", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")
    ax.set_title(label)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Residual map (2D field, transition mean)
# ---------------------------------------------------------------------------

def _residual_map(
    residual: np.ndarray,
    title: str,
    path: Path,
    *,
    mask: np.ndarray | None = None,
) -> Path:
    values = np.nanmean(np.asarray(residual, dtype=float), axis=0)
    if mask is not None:
        m = np.all(mask, axis=0)
        values[~m] = np.nan

    vmin, vmax = _robust_color_limits(values, symmetric=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(values.T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax,
                   cmap="RdBu_r")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("x bin")
    ax.set_ylabel("theta bin")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Density term contribution maps (one panel per term)
# ---------------------------------------------------------------------------

def _density_contribution_plots(result: FittingResult, path: Path) -> Path:
    n_terms = result.density_contributions.shape[-1]
    ncols = min(n_terms, 3)
    nrows = (n_terms + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)
    for i in range(n_terms):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        contrib = np.nanmean(result.density_contributions[..., i], axis=0)
        if result.mask is not None:
            m = np.all(result.mask, axis=0)
            contrib[~m] = np.nan
        vmin, vmax = _robust_color_limits(contrib, symmetric=True)
        im = ax.imshow(contrib.T, origin="lower", aspect="auto",
                       vmin=vmin, vmax=vmax, cmap="RdBu_r")
        plt.colorbar(im, ax=ax)
        ax.set_title(result.density.labels[i])
        ax.set_xlabel("x bin")
        ax.set_ylabel("theta bin")
    for i in range(n_terms, nrows * ncols):
        axes.flat[i].set_visible(False)
    fig.suptitle("Density term contributions", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Curl residual structure
# ---------------------------------------------------------------------------

def _curl_residual_plot(result: FittingResult, path: Path) -> Path:
    curl = np.asarray(result.curl_residual, dtype=float)
    curl_mean = np.nanmean(curl, axis=0)
    if result.mask is not None:
        m = np.all(result.mask, axis=0)
        curl_mean[~m] = np.nan

    vmin, vmax = _robust_color_limits(curl_mean, symmetric=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(curl_mean.T, origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax, cmap="RdBu_r")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("x bin")
    ax.set_ylabel("theta bin")
    ax.set_title("Curl of polarization residual (transition mean)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Color scaling helper
# ---------------------------------------------------------------------------

def _robust_color_limits(
    values: np.ndarray,
    *,
    symmetric: bool = True,
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
