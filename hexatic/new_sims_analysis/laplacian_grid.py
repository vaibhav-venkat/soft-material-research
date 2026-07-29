"""Shared-scale grids of cross-case Laplace transforms."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import seaborn as sns

from .plotting import _save_figure


# Panels are ~4 inches wide, so a case id has to wrap to stay inside one.
TITLE_WRAP_COLUMNS = 26

# Floor the shared color scale at this percentile of the pooled field. The raw
# minimum is a single cell out of r_points x omega_points x cases, so one
# near-zero cell would otherwise flatten every panel.
VMIN_PERCENTILE = 1.0


@dataclass(frozen=True)
class LaplaceEntry:
    key: str
    case_id: str
    seed_count: int
    r: NDArray[np.float64]
    omega: NDArray[np.float64]
    values: NDArray[np.complex128]


def _grid_description(values: NDArray[np.float64]) -> str:
    if values.size == 0:
        return f"shape={values.shape}, empty"
    return f"shape={values.shape}, endpoints=({values[0]!r}, {values[-1]!r})"


def validate_shared_grid(entries: list[LaplaceEntry]) -> None:
    if not entries:
        raise ValueError("no Laplace entries to validate")
    reference = entries[0]
    for entry in entries[1:]:
        if not np.array_equal(entry.r, reference.r):
            raise ValueError(
                f"Laplace r grid mismatch for {entry.key}: "
                f"expected {_grid_description(reference.r)}, "
                f"got {_grid_description(entry.r)}"
            )
        if not np.array_equal(entry.omega, reference.omega):
            raise ValueError(
                f"Laplace omega grid mismatch for {entry.key}: "
                f"expected {_grid_description(reference.omega)}, "
                f"got {_grid_description(entry.omega)}"
            )


def plot_grid(entries: list[LaplaceEntry], variant_label: str, output: Path) -> None:
    """Plot every case on one shared log10 color scale.

    Callers must have run ``validate_shared_grid`` first; the shared scale is
    only meaningful once every case is known to sit on the same ``(r, omega)``.
    """
    if not entries:
        raise ValueError("no Laplace entries to plot")
    sns.set_theme(context="paper", style="ticks")
    fields = [
        np.log10(np.maximum(np.abs(entry.values), np.finfo(float).tiny))
        for entry in entries
    ]
    finite = np.concatenate([field[np.isfinite(field)] for field in fields])
    if finite.size == 0:
        raise ValueError("Laplace grid has no finite values to plot")
    vmin = float(np.percentile(finite, VMIN_PERCENTILE))
    vmax = float(np.max(finite))
    if not vmin < vmax:
        vmax = float(np.nextafter(vmin, np.inf))

    ncols = math.ceil(math.sqrt(len(entries)))
    nrows = math.ceil(len(entries) / ncols)
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.2 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    used_axes = []
    for index, (entry, field) in enumerate(zip(entries, fields, strict=True)):
        row, column = divmod(index, ncols)
        axis = axes[row, column]
        used_axes.append(axis)
        image = axis.pcolormesh(
            entry.r,
            entry.omega,
            field,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            shading="auto",
        )
        axis.set_title(
            f"{textwrap.fill(entry.case_id, TITLE_WRAP_COLUMNS)}\n"
            f"({entry.seed_count} seed(s))"
        )
        if column == 0:
            axis.set_ylabel(r"angular frequency $\omega$")
        # sharex hides tick labels by subplot geometry, not by visibility, so an
        # axis sitting above a switched-off slot would lose its r axis entirely.
        # The bottom-most *visible* axis in a column is the one with no entry
        # one full row further on.
        if index + ncols >= len(entries):
            axis.set_xlabel(r"decay coordinate $r$")
            axis.tick_params(labelbottom=True)
        sns.despine(ax=axis)
    for index in range(len(entries), nrows * ncols):
        axes.flat[index].set_axis_off()
    assert image is not None
    figure.colorbar(
        image,
        ax=used_axes,
        label=r"$\log_{10}|L(r,\omega)|$",
        pad=0.02,
        # The used axes span every row even when the last one is mostly empty,
        # so scale the bar down to the fraction of the grid actually filled.
        shrink=min(1.0, max(0.5, len(entries) / (nrows * ncols))),
        extend="min" if np.min(finite) < vmin else "neither",
    )
    total_seeds = sum(entry.seed_count for entry in entries)
    figure.suptitle(
        f"Laplace transform — {variant_label} ({total_seeds} seeds, "
        f"{len(entries)} cases)"
    )
    _save_figure(figure, output)
