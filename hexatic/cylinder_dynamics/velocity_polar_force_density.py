from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hexatic.constants import cylinder

from .common import CYLINDER_PATHS, _minimum_image_delta
from .plotting import ACTIVE_MATTER_FIELDS_FILE, _load_active_matter_fields, _masked_mean


@dataclass(frozen=True)
class VelocityPolarForceDensitySeries:
    steps: np.ndarray
    v_x: np.ndarray
    u0_p_x: np.ndarray
    force_density_x: np.ndarray
    v_theta: np.ndarray
    u0_p_theta: np.ndarray
    force_density_theta: np.ndarray


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=np.isfinite(denominator) & (denominator != 0.0),
    )


def _velocity_terms_from_active_fields(
    active_matter_fields_file: str | Path,
    shell_only: bool,
) -> VelocityPolarForceDensitySeries:
    fields = _load_active_matter_fields(active_matter_fields_file)
    if fields is None:
        raise FileNotFoundError(active_matter_fields_file)

    steps = np.asarray(fields.steps, dtype=np.int64)
    coords = np.asarray(fields.coords, dtype=np.float64)
    rho = np.asarray(fields.rho, dtype=np.float64)
    polar_density = np.asarray(fields.polar_cylindrical, dtype=np.float64)
    force_density = np.asarray(fields.force_density_cylindrical, dtype=np.float64)
    radii = coords[..., 2]
    shell_mask = np.asarray(fields.shell_mask, dtype=bool)
    n_intervals = max(0, len(fields.steps) - 1)
    interval_steps = steps[1:]
    v_x = np.full(n_intervals, np.nan, dtype=np.float64)
    v_theta = np.full(n_intervals, np.nan, dtype=np.float64)
    u0_p_x = np.full(n_intervals, np.nan, dtype=np.float64)
    u0_p_theta = np.full(n_intervals, np.nan, dtype=np.float64)
    force_density_x = np.full(n_intervals, np.nan, dtype=np.float64)
    force_density_theta = np.full(n_intervals, np.nan, dtype=np.float64)

    box_length_x = float(fields.x_edges[-1] - fields.x_edges[0])
    local_polar = _safe_divide(
        polar_density,
        rho[..., np.newaxis],
    )
    force_velocity = _safe_divide(
        force_density,
        rho[..., np.newaxis],
    )

    u0_p_x_frame = float(cylinder.U0) * local_polar[..., 0]
    force_x_frame = force_velocity[..., 0]
    u0_p_theta_frame = _safe_divide(
        float(cylinder.U0) * local_polar[..., 2],
        radii,
    )
    force_theta_frame = _safe_divide(
        force_velocity[..., 2],
        radii,
    )

    for frame_idx in range(1, len(fields.steps)):
        mask = np.ones(shell_mask.shape[1], dtype=bool)
        if shell_only:
            mask = shell_mask[frame_idx]
        if shell_only:
            mask = mask & shell_mask[frame_idx - 1]

        delta_t = (steps[frame_idx] - steps[frame_idx - 1]) * float(cylinder.TIMESTEP)
        interval_idx = frame_idx - 1
        if delta_t > 0.0:
            dx = _minimum_image_delta(
                coords[frame_idx, :, 0] - coords[frame_idx - 1, :, 0],
                box_length_x,
            )
            dtheta = _minimum_image_delta(
                coords[frame_idx, :, 1] - coords[frame_idx - 1, :, 1],
                2.0 * np.pi,
            )
            v_x[interval_idx] = _masked_mean(dx / delta_t, mask)
            v_theta[interval_idx] = _masked_mean(dtheta / delta_t, mask)

        u0_p_x[interval_idx] = _masked_mean(
            0.5 * (u0_p_x_frame[frame_idx - 1] + u0_p_x_frame[frame_idx]),
            mask,
        )
        u0_p_theta[interval_idx] = _masked_mean(
            0.5
            * (
                u0_p_theta_frame[frame_idx - 1]
                + u0_p_theta_frame[frame_idx]
            ),
            mask,
        )
        force_density_x[interval_idx] = _masked_mean(
            0.5 * (force_x_frame[frame_idx - 1] + force_x_frame[frame_idx]),
            mask,
        )
        force_density_theta[interval_idx] = _masked_mean(
            0.5
            * (
                force_theta_frame[frame_idx - 1]
                + force_theta_frame[frame_idx]
            ),
            mask,
        )

    return VelocityPolarForceDensitySeries(
        steps=interval_steps,
        v_x=v_x,
        u0_p_x=u0_p_x,
        force_density_x=force_density_x,
        v_theta=v_theta,
        u0_p_theta=u0_p_theta,
        force_density_theta=force_density_theta,
    )


def velocity_polar_force_density_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    active_matter_fields_file: str | Path = ACTIVE_MATTER_FIELDS_FILE,
    shell_only: bool = True,
) -> VelocityPolarForceDensitySeries:
    del input_gsd
    return _velocity_terms_from_active_fields(
        active_matter_fields_file,
        shell_only=shell_only,
    )


def _plot_component(
    axis,
    steps: np.ndarray,
    velocity: np.ndarray,
    active_polar: np.ndarray,
    force_density: np.ndarray,
    component: str,
    angular: bool = False,
) -> None:
    velocity_label = rf"$V_{component}$"
    active_label = rf"$U_0 P_{component}$"
    force_label = rf"$\rho^{{-1}}F_{{\mathrm{{density}},{component}}}$"
    sum_label = (
        rf"$U_0 P_{component} + "
        rf"\rho^{{-1}}F_{{\mathrm{{density}},{component}}}$"
    )
    if angular:
        velocity_label = r"$\dot{\theta}$"
        active_label = rf"$U_0 P_{component}/r$"
        force_label = (
            rf"$\rho^{{-1}}F_{{\mathrm{{density}},{component}}}/r$"
        )
        sum_label = (
            rf"$(U_0 P_{component} + "
            rf"\rho^{{-1}}F_{{\mathrm{{density}},{component}}})/r$"
        )

    axis.plot(steps, velocity, color="black", label=velocity_label)
    axis.plot(steps, active_polar, color="green", label=active_label)
    axis.plot(
        steps,
        active_polar + force_density,
        color="blue",
        linestyle="--",
        label=sum_label,
    )
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.3)
    axis.set_ylabel(f"{velocity_label}, {active_label}, {sum_label}")
    axis.grid(True, ls="--", alpha=0.35)

    force_axis = axis.twinx()
    force_axis.plot(
        steps,
        force_density,
        color="red",
        label=force_label,
    )
    force_axis.set_ylabel(force_label, color="red")
    force_axis.tick_params(axis="y", labelcolor="red")

    lines, labels = axis.get_legend_handles_labels()
    force_lines, force_labels = force_axis.get_legend_handles_labels()
    axis.legend(lines + force_lines, labels + force_labels, loc="best")


def plot_velocity_polar_force_density_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    filename: str | Path | None = CYLINDER_PATHS.x_com_velocity_plot.with_name(
        "cylinder_velocity_polar_force_density_series.png"
    ),
    active_matter_fields_file: str | Path = ACTIVE_MATTER_FIELDS_FILE,
    shell_only: bool = True,
) -> None:
    series = velocity_polar_force_density_series(
        input_gsd=input_gsd,
        active_matter_fields_file=active_matter_fields_file,
        shell_only=shell_only,
    )

    fig, (x_axis, theta_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
    )
    _plot_component(
        x_axis,
        series.steps,
        series.v_x,
        series.u0_p_x,
        series.force_density_x,
        "x",
    )
    _plot_component(
        theta_axis,
        series.steps,
        series.v_theta,
        series.u0_p_theta,
        series.force_density_theta,
        r"\theta",
        angular=True,
    )
    x_axis.set_title("Outer-shell interval means from active_matter_fields.npz")
    theta_axis.set_xlabel("Simulation step")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
