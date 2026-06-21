from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from ..common import _color_limits, _format_theta_axis, _frame_index, _radial_integral_mean
from ..config import ACTIVE_IMAGE_DIR, CYLINDER, ActiveMatterFields


def plot_rho_shell(
    fields: ActiveMatterFields,
    filename: str | Path,
    frame_index: int = -1,
) -> None:
    frame_idx = _frame_index(frame_index, len(fields.steps))
    mask = fields.shell_mask[frame_idx]
    coords = fields.coords[frame_idx, mask]
    values = fields.rho[frame_idx, mask]
    vmin, vmax = _color_limits(values)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 5))
    scatter = axis.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        s=9,
        cmap="magma",
        norm=Normalize(vmin=vmin, vmax=vmax),
        linewidths=0,
    )
    fig.colorbar(scatter, ax=axis, label="local rho")
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(f"Outer-shell local rho, step {fields.steps[frame_idx]}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_rho_radial_integral(
    fields: ActiveMatterFields,
    filename: str | Path,
    frame_index: int = -1,
) -> None:
    frame_idx = _frame_index(frame_index, len(fields.steps))
    values = _radial_integral_mean(
        fields.coords[frame_idx],
        fields.rho[frame_idx],
        fields.x_edges,
        fields.theta_edges,
        fields.x_edges[-1] - fields.x_edges[0],
    )
    x_grid, theta_grid = np.meshgrid(
        fields.x_centers,
        fields.theta_centers,
        indexing="ij",
    )
    vmin, vmax = _color_limits(values)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 5))
    scatter = axis.scatter(
        x_grid.ravel(),
        theta_grid.ravel(),
        c=values.ravel(),
        s=9,
        marker="o",
        cmap="magma",
        norm=Normalize(vmin=vmin, vmax=vmax),
        linewidths=0,
    )
    fig.colorbar(scatter, ax=axis, label="r-averaged local rho")
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(f"Integrated local rho, step {fields.steps[frame_idx]}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_polar_shell(
    fields: ActiveMatterFields,
    filename: str | Path,
    frame_index: int = -1,
) -> None:
    frame_idx = _frame_index(frame_index, len(fields.steps))
    mask = fields.shell_mask[frame_idx]
    coords = fields.coords[frame_idx, mask]
    polar = fields.polar_cylindrical[frame_idx, mask]
    projected_theta = polar[:, 2] / CYLINDER.cylinder_radius
    magnitude = np.hypot(polar[:, 0], polar[:, 2])
    vmin, vmax = _color_limits(magnitude)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 5))
    quiver = axis.quiver(
        coords[:, 0],
        coords[:, 1],
        polar[:, 0],
        projected_theta,
        magnitude,
        cmap="viridis",
        norm=Normalize(vmin=vmin, vmax=vmax),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
    )
    fig.colorbar(quiver, ax=axis, label="|polar mean in x-theta|")
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(f"Outer-shell polar mean, step {fields.steps[frame_idx]}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_polar_radial_integral(
    fields: ActiveMatterFields,
    filename: str | Path,
    frame_index: int = -1,
) -> None:
    frame_idx = _frame_index(frame_index, len(fields.steps))
    polar = _radial_integral_mean(
        fields.coords[frame_idx],
        fields.polar_cylindrical[frame_idx],
        fields.x_edges,
        fields.theta_edges,
        fields.x_edges[-1] - fields.x_edges[0],
    )
    x_grid, theta_grid = np.meshgrid(
        fields.x_centers,
        fields.theta_centers,
        indexing="ij",
    )
    projected_theta = polar[..., 2] / CYLINDER.cylinder_radius
    magnitude = np.hypot(polar[..., 0], polar[..., 2])
    vmin, vmax = _color_limits(magnitude)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 5))
    quiver = axis.quiver(
        x_grid,
        theta_grid,
        polar[..., 0],
        projected_theta,
        magnitude,
        cmap="viridis",
        norm=Normalize(vmin=vmin, vmax=vmax),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
    )
    fig.colorbar(quiver, ax=axis, label="r-averaged |polar mean in x-theta|")
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(f"Radially averaged polar mean, step {fields.steps[frame_idx]}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def _draw_rho_shell(fields: ActiveMatterFields, fig, axis, frame_idx: int) -> None:
    mask = fields.shell_mask[frame_idx]
    coords = fields.coords[frame_idx, mask]
    values = fields.rho[frame_idx, mask]
    vmin, vmax = _color_limits(values)
    scatter = axis.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        s=9,
        cmap="magma",
        norm=Normalize(vmin=vmin, vmax=vmax),
        linewidths=0,
    )
    fig.colorbar(scatter, ax=axis, label="local rho")
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(f"Outer-shell local rho, step {fields.steps[frame_idx]}")


def _draw_rho_radial_integral(
    fields: ActiveMatterFields,
    fig,
    axis,
    frame_idx: int,
) -> None:
    values = _radial_integral_mean(
        fields.coords[frame_idx],
        fields.rho[frame_idx],
        fields.x_edges,
        fields.theta_edges,
        fields.x_edges[-1] - fields.x_edges[0],
    )
    x_grid, theta_grid = np.meshgrid(
        fields.x_centers,
        fields.theta_centers,
        indexing="ij",
    )
    vmin, vmax = _color_limits(values)
    scatter = axis.scatter(
        x_grid.ravel(),
        theta_grid.ravel(),
        c=values.ravel(),
        s=9,
        marker="o",
        cmap="magma",
        norm=Normalize(vmin=vmin, vmax=vmax),
        linewidths=0,
    )
    fig.colorbar(scatter, ax=axis, label="r-averaged local rho")
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(f"Integrated local rho, step {fields.steps[frame_idx]}")


def _draw_polar_shell(fields: ActiveMatterFields, fig, axis, frame_idx: int) -> None:
    mask = fields.shell_mask[frame_idx]
    coords = fields.coords[frame_idx, mask]
    polar = fields.polar_cylindrical[frame_idx, mask]
    projected_theta = polar[:, 2] / CYLINDER.cylinder_radius
    magnitude = np.hypot(polar[:, 0], polar[:, 2])
    vmin, vmax = _color_limits(magnitude)
    quiver = axis.quiver(
        coords[:, 0],
        coords[:, 1],
        polar[:, 0],
        projected_theta,
        magnitude,
        cmap="viridis",
        norm=Normalize(vmin=vmin, vmax=vmax),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
    )
    fig.colorbar(quiver, ax=axis, label="|polar mean in x-theta|")
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(f"Outer-shell polar mean, step {fields.steps[frame_idx]}")


def _draw_polar_radial_integral(
    fields: ActiveMatterFields,
    fig,
    axis,
    frame_idx: int,
) -> None:
    polar = _radial_integral_mean(
        fields.coords[frame_idx],
        fields.polar_cylindrical[frame_idx],
        fields.x_edges,
        fields.theta_edges,
        fields.x_edges[-1] - fields.x_edges[0],
    )
    x_grid, theta_grid = np.meshgrid(
        fields.x_centers,
        fields.theta_centers,
        indexing="ij",
    )
    projected_theta = polar[..., 2] / CYLINDER.cylinder_radius
    magnitude = np.hypot(polar[..., 0], polar[..., 2])
    vmin, vmax = _color_limits(magnitude)
    quiver = axis.quiver(
        x_grid,
        theta_grid,
        polar[..., 0],
        projected_theta,
        magnitude,
        cmap="viridis",
        norm=Normalize(vmin=vmin, vmax=vmax),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
    )
    fig.colorbar(quiver, ax=axis, label="r-averaged |polar mean in x-theta|")
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(f"Radially averaged polar mean, step {fields.steps[frame_idx]}")
