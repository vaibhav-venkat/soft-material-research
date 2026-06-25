from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gsd.hoomd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

from hexatic.active_matter_cylinder.config import (
    ACTIVE_RADIAL_BIN_WIDTH,
    LOCAL_POCKET_RADIUS,
)
from hexatic.active_matter_cylinder.grid_utils import _time_edges
from hexatic.active_matter_cylinder.math_utils import _gaussian_delta_weights
from hexatic.constants import cylinder

from .common import CYLINDER, CYLINDER_PATHS


@dataclass(frozen=True)
class RadialExchangeCurrentSeries:
    steps: np.ndarray
    radius_edges: np.ndarray
    radius_centers: np.ndarray
    rho: np.ndarray
    v_r: np.ndarray
    j_r: np.ndarray


def _radius_edges_and_centers(
    cylinder_radius: float,
    radial_bin_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    assert cylinder_radius > 0.0
    assert radial_bin_width > 0.0
    n_bins = int(np.ceil(cylinder_radius / radial_bin_width))
    edges = radial_bin_width * np.arange(n_bins + 1, dtype=np.float64)
    edges[-1] = cylinder_radius
    return edges, 0.5 * (edges[:-1] + edges[1:])


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=np.isfinite(denominator) & (denominator != 0.0),
    )


def _radii_from_positions(positions: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(positions, dtype=np.float64)[:, 1:3], axis=1)


def radial_exchange_current_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    radial_bin_width: float = ACTIVE_RADIAL_BIN_WIDTH,
    kernel_radius: float = LOCAL_POCKET_RADIUS,
    cylinder_radius: float = CYLINDER.cylinder_radius,
) -> RadialExchangeCurrentSeries:
    radius_edges, radius_centers = _radius_edges_and_centers(
        cylinder_radius,
        radial_bin_width,
    )
    steps: list[int] = []
    rho_frames: list[np.ndarray] = []
    vr_frames: list[np.ndarray] = []
    jr_frames: list[np.ndarray] = []
    previous_radii: np.ndarray | None = None
    previous_step: int | None = None

    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        for frame in source:
            particles = frame.particles
            assert particles.position is not None
            current_radii = _radii_from_positions(particles.position)
            current_step = int(frame.configuration.step)

            if previous_radii is not None and previous_step is not None:
                delta_t = (current_step - previous_step) * float(cylinder.TIMESTEP)
                if delta_t > 0.0:
                    radial_velocity = (current_radii - previous_radii) / delta_t
                    valid = (
                        np.isfinite(current_radii)
                        & np.isfinite(radial_velocity)
                        & (current_radii >= 0.0)
                        & (current_radii <= cylinder_radius)
                    )
                    if np.any(valid):
                        delta_sq = (
                            radius_centers[:, np.newaxis]
                            - current_radii[valid][np.newaxis, :]
                        ) ** 2
                        weights = _gaussian_delta_weights(delta_sq, kernel_radius)
                        rho = np.sum(weights, axis=1)
                        j_r = weights @ radial_velocity[valid]
                        v_r = _safe_divide(j_r, rho)
                    else:
                        rho = np.full_like(radius_centers, np.nan, dtype=np.float64)
                        v_r = np.full_like(radius_centers, np.nan, dtype=np.float64)
                        j_r = np.full_like(radius_centers, np.nan, dtype=np.float64)
                    steps.append(current_step)
                    rho_frames.append(rho)
                    vr_frames.append(v_r)
                    jr_frames.append(j_r)

            previous_radii = current_radii
            previous_step = current_step

    if not steps:
        empty = np.empty((0, len(radius_centers)), dtype=np.float64)
        return RadialExchangeCurrentSeries(
            steps=np.asarray([], dtype=np.int64),
            radius_edges=radius_edges,
            radius_centers=radius_centers,
            rho=empty,
            v_r=empty,
            j_r=empty,
        )

    return RadialExchangeCurrentSeries(
        steps=np.asarray(steps, dtype=np.int64),
        radius_edges=radius_edges,
        radius_centers=radius_centers,
        rho=np.vstack(rho_frames),
        v_r=np.vstack(vr_frames),
        j_r=np.vstack(jr_frames),
    )


def save_radial_exchange_current(
    series: RadialExchangeCurrentSeries,
    filename: str | Path,
    kernel_radius: float,
    radial_bin_width: float,
) -> None:
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        steps=series.steps,
        radius_edges=series.radius_edges,
        radius_centers=series.radius_centers,
        rho=series.rho,
        v_r=series.v_r,
        j_r=series.j_r,
        kernel_radius=np.asarray(kernel_radius, dtype=np.float64),
        radial_bin_width=np.asarray(radial_bin_width, dtype=np.float64),
    )


def _finite_limits(values: np.ndarray, symmetric: bool = False):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    low, high = np.percentile(finite, [1.0, 99.0])
    if symmetric:
        limit = max(abs(float(low)), abs(float(high)))
        if np.isclose(limit, 0.0):
            limit = 1.0
        return -limit, limit
    if np.isclose(low, high):
        low = float(np.min(finite))
        high = float(np.max(finite))
    if np.isclose(low, high):
        high = low + 1.0
    return float(low), float(high)


def _plot_heatmap(axis, steps, radius_edges, values, title, label, symmetric=False):
    vmin, vmax = _finite_limits(values, symmetric=symmetric)
    cmap = plt.get_cmap("coolwarm" if symmetric else "viridis").copy()
    cmap.set_bad("0.85")
    norm = (
        TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        if symmetric
        else Normalize(vmin=vmin, vmax=vmax)
    )
    mesh = axis.pcolormesh(
        _time_edges(steps),
        radius_edges,
        np.ma.masked_invalid(values.T),
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    axis.set_title(title)
    axis.set_ylabel("r")
    axis.grid(False)
    return mesh, label


def plot_radial_exchange_current(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    filename: str | Path | None = CYLINDER_PATHS.x_com_velocity_plot.with_name(
        "cylinder_radial_exchange_current.png"
    ),
    npz_filename: str | Path | None = CYLINDER_PATHS.in_gsd.parent
    / "radial_exchange_current.npz",
    radial_bin_width: float = ACTIVE_RADIAL_BIN_WIDTH,
    kernel_radius: float = LOCAL_POCKET_RADIUS,
    cylinder_radius: float = CYLINDER.cylinder_radius,
) -> RadialExchangeCurrentSeries:
    series = radial_exchange_current_series(
        input_gsd=input_gsd,
        radial_bin_width=radial_bin_width,
        kernel_radius=kernel_radius,
        cylinder_radius=cylinder_radius,
    )
    if npz_filename is not None:
        save_radial_exchange_current(
            series,
            npz_filename,
            kernel_radius=kernel_radius,
            radial_bin_width=radial_bin_width,
        )

    if filename is None:
        return series

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    plots = (
        (series.rho, r"$\rho_K(r,t)$", "Gaussian kernel density", False),
        (series.v_r, r"$v_r(r,t)$", "Kernel-averaged radial velocity", True),
        (series.j_r, r"$J_r(r,t)$", "Radial exchange current", True),
    )
    for axis, (values, label, title, symmetric) in zip(axes, plots):
        mesh, colorbar_label = _plot_heatmap(
            axis,
            series.steps,
            series.radius_edges,
            values,
            title,
            label,
            symmetric=symmetric,
        )
        fig.colorbar(mesh, ax=axis, label=colorbar_label)
    axes[-1].set_xlabel("Simulation step")
    fig.suptitle("Radial exchange current from Gaussian kernel density")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return series
