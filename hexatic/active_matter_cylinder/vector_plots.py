from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .common import (
    _coarse_vector_density_grid,
    _color_limits,
    _fixed_length_quiver_components,
    _format_theta_axis,
    _frame_index,
    _radial_integral_mean,
    _xytheta_occupied,
)
from .config import (
    ACTIVE_FLUX_PLOT_THETA_BINS,
    ACTIVE_FLUX_PLOT_X_BINS,
    ACTIVE_IMAGE_DIR,
    CYLINDER,
    ActiveMatterFields,
)


def _plot_vector_density(
    fields: ActiveMatterFields,
    vectors: np.ndarray,
    filename: str | Path,
    title_prefix: str,
    colorbar_label: str,
    shell_only: bool,
    frame_index: int = -1,
    cmap: str = "plasma",
) -> None:
    frame_idx = _frame_index(frame_index, len(fields.steps))
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 5))
    _draw_vector_density(
        fields,
        vectors,
        fig,
        axis,
        frame_idx,
        title_prefix,
        colorbar_label,
        shell_only=shell_only,
        cmap=cmap,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _draw_vector_density(
    fields: ActiveMatterFields,
    vectors: np.ndarray,
    fig,
    axis,
    frame_idx: int,
    title_prefix: str,
    colorbar_label: str,
    shell_only: bool,
    cmap: str = "plasma",
) -> None:
    x_grid, theta_grid, vector_grid, x_span = _coarse_vector_density_grid(
        fields,
        vectors,
        frame_idx,
        shell_only=shell_only,
    )
    projected_theta = vector_grid[..., 2] / CYLINDER.cylinder_radius
    magnitude = np.linalg.norm(vector_grid, axis=2)
    vmin, vmax = _color_limits(magnitude)
    arrow_x, arrow_theta = _fixed_length_quiver_components(
        vector_grid[..., 0],
        projected_theta,
        x_span,
        ACTIVE_FLUX_PLOT_X_BINS,
        ACTIVE_FLUX_PLOT_THETA_BINS,
    )
    quiver = axis.quiver(
        x_grid,
        theta_grid,
        arrow_x,
        arrow_theta,
        magnitude,
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
    )
    fig.colorbar(quiver, ax=axis, label=colorbar_label)
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(f"{title_prefix}, step {fields.steps[frame_idx]}")


def plot_flux_shell(
    fields: ActiveMatterFields,
    filename: str | Path,
    frame_index: int = -1,
) -> None:
    _plot_vector_density(
        fields,
        fields.flux_cylindrical,
        filename,
        "Outer-shell flux mean",
        r"$|\langle\dot{\mathbf{r}}\rangle|$",
        shell_only=True,
        frame_index=frame_index,
    )


def plot_flux_radial_integral(
    fields: ActiveMatterFields,
    filename: str | Path,
    frame_index: int = -1,
) -> None:
    _plot_vector_density(
        fields,
        fields.flux_cylindrical,
        filename,
        "Radially averaged flux mean",
        r"$|R^{-1}\int dr\,\langle\dot{\mathbf{r}}\rangle_r|$",
        shell_only=False,
        frame_index=frame_index,
    )


def plot_force_density_shell(
    fields: ActiveMatterFields,
    filename: str | Path,
    frame_index: int = -1,
) -> None:
    _plot_vector_density(
        fields,
        fields.force_density_cylindrical,
        filename,
        "Outer-shell force mean",
        r"$|\langle\gamma^{-1}\mathbf{F}\rangle|$",
        shell_only=True,
        frame_index=frame_index,
        cmap="cividis",
    )


def plot_force_density_radial_integral(
    fields: ActiveMatterFields,
    filename: str | Path,
    frame_index: int = -1,
) -> None:
    _plot_vector_density(
        fields,
        fields.force_density_cylindrical,
        filename,
        "Radially averaged force mean",
        r"$|R^{-1}\int dr\,\langle\gamma^{-1}\mathbf{F}\rangle_r|$",
        shell_only=False,
        frame_index=frame_index,
        cmap="cividis",
    )


def _component_pair_mean(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.asarray([np.nan, np.nan], dtype=np.float64)
    return np.asarray(
        [
            np.mean(values[:, 0]),
            np.mean(values[:, 2]),
        ],
        dtype=np.float64,
    )


def _radially_averaged_component_pair(
    fields: ActiveMatterFields,
    values: np.ndarray,
    frame_idx: int,
) -> np.ndarray:
    x_span = fields.x_edges[-1] - fields.x_edges[0]
    grid = _radial_integral_mean(
        fields.coords[frame_idx],
        values[frame_idx],
        fields.x_edges,
        fields.theta_edges,
        x_span,
    )
    occupied = _xytheta_occupied(
        fields.coords[frame_idx],
        fields.x_edges,
        fields.theta_edges,
        x_span,
    )
    if not np.any(occupied):
        return np.asarray([np.nan, np.nan], dtype=np.float64)
    return np.asarray(
        [
            np.mean(grid[..., 0][occupied]),
            np.mean(grid[..., 2][occupied]),
        ],
        dtype=np.float64,
    )


def _active_component_series(
    fields: ActiveMatterFields,
    radial_average: bool,
) -> np.ndarray:
    series = np.full((len(fields.steps), 6), np.nan, dtype=np.float64)
    for frame_idx in range(len(fields.steps)):
        if radial_average:
            polar_pair = _radially_averaged_component_pair(
                fields,
                fields.direction_cylindrical,
                frame_idx,
            )
            flux_pair = _radially_averaged_component_pair(
                fields,
                fields.flux_cylindrical,
                frame_idx,
            )
            force_pair = _radially_averaged_component_pair(
                fields,
                fields.force_density_cylindrical,
                frame_idx,
            )
        else:
            mask = fields.shell_mask[frame_idx]
            polar_pair = _component_pair_mean(
                fields.direction_cylindrical[frame_idx, mask]
            )
            flux_pair = _component_pair_mean(fields.flux_cylindrical[frame_idx, mask])
            force_pair = _component_pair_mean(
                fields.force_density_cylindrical[frame_idx, mask]
            )

        series[frame_idx] = np.concatenate((polar_pair, flux_pair, force_pair))
    return series


def _plot_active_component_series(
    fields: ActiveMatterFields,
    filename: str | Path,
    radial_average: bool,
) -> None:
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series = _active_component_series(fields, radial_average=radial_average)
    title_prefix = "Radially averaged" if radial_average else "Outer-shell"
    groups = (
        ("P", 0, 1),
        ("J", 2, 3),
        ("F", 4, 5),
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis, (name, x_col, theta_col) in zip(axes, groups):
        axis.plot(
            fields.steps,
            series[:, x_col],
            color="tab:blue",
            label=f"{name}_x",
        )
        axis.plot(
            fields.steps,
            series[:, theta_col],
            color="tab:orange",
            linestyle="--",
            label=rf"{name}_\theta",
        )
        axis.set_ylabel(name)
        axis.grid(True, ls="--", alpha=0.35)
        axis.legend(loc="best")

    axes[-1].set_xlabel("Simulation step")
    fig.suptitle(f"{title_prefix} active component means")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_active_component_series(
    fields: ActiveMatterFields,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
) -> None:
    image_path = Path(image_dir)
    _plot_active_component_series(
        fields,
        image_path / "active_components_shell.png",
        radial_average=False,
    )
    _plot_active_component_series(
        fields,
        image_path / "active_components_radial_integral.png",
        radial_average=True,
    )
