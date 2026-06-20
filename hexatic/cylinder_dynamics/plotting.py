from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.optimize import curve_fit

from hexatic.constants import cylinder

from .common import CYLINDER_PATHS, _format_theta_axis
from .series import (
    _outer_shell_xtheta_velocity_frames,
    center_of_mass_series,
    disclination_center_of_mass_series,
    dislocation_summary_series,
    theta_com_velocity_series,
    x_center_of_mass_velocity_series,
)


def _single_relaxation(time: np.ndarray, v_inf: float, amplitude: float, tau: float):
    return v_inf + amplitude * np.exp(-time / tau)


def _double_relaxation(
    time: np.ndarray,
    v_inf: float,
    amplitude_1: float,
    tau_1: float,
    amplitude_2: float,
    tau_2: float,
):
    return (
        v_inf
        + amplitude_1 * np.exp(-time / tau_1)
        + amplitude_2 * np.exp(-time / tau_2)
    )


def _aicc(residuals: np.ndarray, parameter_count: int) -> float:
    n_points = residuals.size
    rss = float(np.sum(residuals**2))
    if n_points <= parameter_count + 1:
        return np.inf
    rss = max(rss, np.finfo(float).tiny)
    return (
        n_points * np.log(rss / n_points)
        + 2 * parameter_count
        + (2 * parameter_count * (parameter_count + 1))
        / (n_points - parameter_count - 1)
    )


def _fit_x_velocity_relaxation(
    steps: np.ndarray,
    velocities: np.ndarray,
    fit_mode: str = "auto",
):
    if fit_mode not in {"auto", "single", "double"}:
        raise ValueError("fit_mode must be 'auto', 'single', or 'double'.")

    finite = np.isfinite(steps) & np.isfinite(velocities)
    fit_steps = np.asarray(steps[finite], dtype=np.float64)
    fit_velocities = np.asarray(velocities[finite], dtype=np.float64)
    if fit_steps.size < 4:
        return None

    time = (fit_steps - fit_steps[0]) * float(cylinder.TIMESTEP)
    time_span = float(np.ptp(time))
    if not np.isfinite(time_span) or time_span <= 0.0:
        return None

    tail_start = max(0, int(0.75 * fit_velocities.size))
    v_inf_guess = float(np.nanmedian(fit_velocities[tail_start:]))
    amplitude_guess = float(fit_velocities[0] - v_inf_guess)
    if amplitude_guess == 0.0:
        amplitude_guess = float(np.nanmax(fit_velocities) - np.nanmin(fit_velocities))
    if amplitude_guess == 0.0:
        amplitude_guess = 1.0

    tau_min = max(np.finfo(float).eps, time_span / 1.0e6)
    fits = []

    if fit_mode in {"auto", "single"}:
        try:
            single_params, _ = curve_fit(
                _single_relaxation,
                time,
                fit_velocities,
                p0=(v_inf_guess, amplitude_guess, max(time_span / 3.0, tau_min)),
                bounds=([-np.inf, -np.inf, tau_min], [np.inf, np.inf, np.inf]),
                maxfev=20000,
            )
            single_model = _single_relaxation(time, *single_params)
            fits.append(
                {
                    "stage_count": 1,
                    "params": tuple(float(value) for value in single_params),
                    "aicc": _aicc(fit_velocities - single_model, 3),
                }
            )
        except (RuntimeError, ValueError, FloatingPointError):
            pass

    if fit_mode in {"auto", "double"} and fit_steps.size >= 7:
        try:
            double_params, _ = curve_fit(
                _double_relaxation,
                time,
                fit_velocities,
                p0=(
                    v_inf_guess,
                    0.6 * amplitude_guess,
                    max(time_span / 10.0, tau_min),
                    0.4 * amplitude_guess,
                    max(time_span / 2.0, tau_min),
                ),
                bounds=(
                    [-np.inf, -np.inf, tau_min, -np.inf, tau_min],
                    [np.inf, np.inf, np.inf, np.inf, np.inf],
                ),
                maxfev=40000,
            )
            v_inf, amplitude_1, tau_1, amplitude_2, tau_2 = [
                float(value) for value in double_params
            ]
            if tau_1 > tau_2:
                amplitude_1, tau_1, amplitude_2, tau_2 = (
                    amplitude_2,
                    tau_2,
                    amplitude_1,
                    tau_1,
                )
            double_params = (v_inf, amplitude_1, tau_1, amplitude_2, tau_2)
            double_model = _double_relaxation(time, *double_params)
            fits.append(
                {
                    "stage_count": 2,
                    "params": double_params,
                    "aicc": _aicc(fit_velocities - double_model, 5),
                }
            )
        except (RuntimeError, ValueError, FloatingPointError):
            pass

    if not fits:
        return None

    best_fit = min(fits, key=lambda fit: fit["aicc"])
    if best_fit["stage_count"] == 1:
        fitted = _single_relaxation(time, *best_fit["params"])
    else:
        fitted = _double_relaxation(time, *best_fit["params"])

    residuals = fit_velocities - fitted
    late_start = max(0, int(0.67 * residuals.size))
    best_fit["steps"] = fit_steps
    best_fit["modeled_velocities"] = fitted
    best_fit["residual_noise"] = float(np.nanstd(residuals[late_start:]))
    best_fit["rms_residual"] = float(np.sqrt(np.nanmean(residuals**2)))
    best_fit["fit_mode"] = fit_mode
    return best_fit


def _print_x_velocity_relaxation_fit(fit: dict | None, shell_only: bool) -> None:
    population = "outer-shell" if shell_only else "all-particle"
    if fit is None:
        print(f"x velocity relaxation fit ({population}): no stable fit found.")
        return

    print(f"x velocity relaxation fit ({population}):")
    print(f"  requested mode = {fit['fit_mode']}")
    print(f"  stages = {fit['stage_count']}")
    print(f"  V_inf = {fit['params'][0]:.8g}")
    if fit["stage_count"] == 1:
        _, amplitude, tau = fit["params"]
        print(f"  A = {amplitude:.8g}")
        print(f"  tau = {tau:.8g}")
    else:
        _, amplitude_1, tau_1, amplitude_2, tau_2 = fit["params"]
        print(f"  A_1 = {amplitude_1:.8g}")
        print(f"  tau_1 = {tau_1:.8g}")
        print(f"  A_2 = {amplitude_2:.8g}")
        print(f"  tau_2 = {tau_2:.8g}")
    print(f"  residual noise = {fit['residual_noise']:.8g}")
    print(f"  RMS residual = {fit['rms_residual']:.8g}")


def _animate_outer_shell_xtheta_velocity(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    filename: str | Path | None = CYLINDER_PATHS.shell_xtheta_x_velocity_movie,
    start_frame: int = 100,
    n_x_bins: int = 80,
    n_theta_bins: int = 48,
    fps: int = 12,
    velocity_component: str = "x",
) -> None:
    if velocity_component not in {"x", "theta"}:
        raise ValueError("velocity_component must be 'x' or 'theta'.")

    frames = _outer_shell_xtheta_velocity_frames(
        input_gsd,
        start_frame=start_frame,
        n_x_bins=n_x_bins,
        n_theta_bins=n_theta_bins,
        velocity_component=velocity_component,
    )
    if not frames:
        raise ValueError(
            f"No consecutive frames available at or after frame {start_frame}."
        )

    all_velocities = np.concatenate(
        [
            np.asarray(frame["velocity"], dtype=np.float64)
            for frame in frames
            if np.asarray(frame["velocity"]).size
        ]
    )
    if all_velocities.size:
        vmax = float(np.nanpercentile(np.abs(all_velocities), 98.0))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = float(np.nanmax(np.abs(all_velocities)))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
    else:
        vmax = 1.0

    first_box_length_x = float(frames[0]["box_length_x"])
    fig, axis = plt.subplots(figsize=(11, 5.8))
    scatter = axis.scatter(
        [],
        [],
        c=[],
        s=[],
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        edgecolors="none",
    )
    colorbar = fig.colorbar(scatter, ax=axis, pad=0.02)
    if velocity_component == "x":
        colorbar.set_label(r"binned discrete $v_x = \Delta x / \Delta t$")
        title_prefix = "Outer-shell binned x velocity"
    else:
        colorbar.set_label(
            r"binned discrete $\omega_\theta = \Delta \theta / \Delta t$"
        )
        title_prefix = "Outer-shell binned theta velocity"
    axis.set_xlim(-0.5 * first_box_length_x, 0.5 * first_box_length_x)
    axis.set_ylim(0.0, 2.0 * np.pi)
    axis.set_xlabel("x")
    axis.set_ylabel(r"$\theta$")
    axis.set_yticks([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi])
    axis.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: {
                0.0: "0",
                0.5: r"$\pi/2$",
                1.0: r"$\pi$",
                1.5: r"$3\pi/2$",
                2.0: r"$2\pi$",
            }.get(round(value / np.pi, 1), "")
        )
    )
    axis.grid(True, ls="--", alpha=0.25)
    title = axis.set_title("")

    def update(animation_idx: int):
        frame = frames[animation_idx]
        x = np.asarray(frame["x"], dtype=np.float64)
        theta = np.asarray(frame["theta"], dtype=np.float64)
        velocity = np.asarray(frame["velocity"], dtype=np.float64)
        counts = np.asarray(frame["counts"], dtype=np.float64)
        scatter.set_offsets(np.column_stack((x, theta)) if x.size else np.empty((0, 2)))
        scatter.set_array(velocity)
        scatter.set_sizes(18.0 + 5.0 * np.sqrt(counts) if counts.size else [])
        axis.set_xlim(
            -0.5 * float(frame["box_length_x"]),
            0.5 * float(frame["box_length_x"]),
        )
        title.set_text(
            f"{title_prefix} on x-theta grid "
            f"(frame {int(frame['frame_idx'])}, step {int(frame['step'])})"
        )
        return scatter, title

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
    )
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".gif":
            animation.save(output_path, writer="pillow", fps=fps)
        else:
            animation.save(output_path, writer="ffmpeg", fps=fps, dpi=160)
        plt.close(fig)


def animate_outer_shell_xtheta_x_velocity(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    filename: str | Path | None = CYLINDER_PATHS.shell_xtheta_x_velocity_movie,
    start_frame: int = 100,
    n_x_bins: int = 80,
    n_theta_bins: int = 48,
    fps: int = 12,
) -> None:
    _animate_outer_shell_xtheta_velocity(
        input_gsd=input_gsd,
        filename=filename,
        start_frame=start_frame,
        n_x_bins=n_x_bins,
        n_theta_bins=n_theta_bins,
        fps=fps,
        velocity_component="x",
    )


def animate_outer_shell_xtheta_theta_velocity(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    filename: str | Path | None = CYLINDER_PATHS.shell_xtheta_theta_velocity_movie,
    start_frame: int = 100,
    n_x_bins: int = 80,
    n_theta_bins: int = 48,
    fps: int = 12,
) -> None:
    _animate_outer_shell_xtheta_velocity(
        input_gsd=input_gsd,
        filename=filename,
        start_frame=start_frame,
        n_x_bins=n_x_bins,
        n_theta_bins=n_theta_bins,
        fps=fps,
        velocity_component="theta",
    )


def plot_x_center_of_mass_velocity_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    filename: str | Path | None = CYLINDER_PATHS.x_com_velocity_plot,
    shell_only: bool = False,
    relaxation_fit_mode: str = "auto",
) -> None:
    series = x_center_of_mass_velocity_series(input_gsd, shell_only=shell_only)
    relaxation_fit = _fit_x_velocity_relaxation(
        series.steps,
        series.x_velocities,
        fit_mode=relaxation_fit_mode,
    )
    _print_x_velocity_relaxation_fit(relaxation_fit, shell_only)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        series.steps,
        series.x_velocities,
        color="tab:blue",
        label="velocity_x",
    )
    if relaxation_fit is not None:
        fit_label = (
            "best-fit relaxation"
            if relaxation_fit["stage_count"] == 1
            else "best-fit two-stage relaxation"
        )
        axis.plot(
            relaxation_fit["steps"],
            relaxation_fit["modeled_velocities"],
            color="purple",
            linewidth=2.0,
            label=fit_label,
        )
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.45)
    axis.set_xlabel("Simulation step")
    axis.set_ylabel("x center-of-mass velocity")
    population = "outer-shell" if shell_only else "all-particle"
    axis.set_title(f"Cylinder {population} x center-of-mass velocity")
    axis.grid(True, ls="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_theta_center_of_mass_velocity_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    filename: str | Path | None = CYLINDER_PATHS.theta_com_velocity_plot,
    shell_only: bool = False,
) -> None:
    series = theta_com_velocity_series(input_gsd, shell_only=shell_only)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        series.steps,
        series.theta_velocities,
        color="tab:orange",
        label=r"$\langle \Delta \theta_i\rangle / \Delta t$",
    )
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.45)
    axis.set_xlabel("Simulation step")
    axis.set_ylabel("theta center-of-mass angular velocity")
    population = "outer-shell" if shell_only else "all-particle"
    axis.set_title(f"Cylinder {population} theta center-of-mass velocity")
    axis.grid(True, ls="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_center_of_mass_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    filename: str | Path | None = CYLINDER_PATHS.com_plot,
) -> None:
    series = center_of_mass_series(input_gsd)

    fig, x_axis = plt.subplots(figsize=(9, 5))
    x_line = x_axis.plot(
        series.steps,
        series.x_centers,
        color="tab:blue",
        label="x center of mass",
    )
    x_axis.set_xlabel("Simulation step")
    x_axis.set_ylabel("Center of mass x", color="tab:blue")
    x_axis.tick_params(axis="y", labelcolor="tab:blue")

    theta_axis = x_axis.twinx()
    theta_line = theta_axis.plot(
        series.steps,
        series.theta_centers,
        color="tab:orange",
        label="theta center of mass",
    )
    _format_theta_axis(theta_axis)

    lines = x_line + theta_line
    labels = [line.get_label() for line in lines]
    x_axis.legend(lines, labels, loc="best")
    x_axis.set_title("Cylinder center of mass")
    x_axis.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_disclination_center_of_mass_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
    filename: str | Path | None = CYLINDER_PATHS.disclination_com_plot,
) -> None:
    series = disclination_center_of_mass_series(input_gsd, neighbor_count_txt)

    fig, x_axis = plt.subplots(figsize=(12, 6))
    plus_x_line = x_axis.plot(
        series.steps,
        series.plus_x_centers,
        color="#1f77b4",
        linewidth=1.8,
        alpha=0.9,
        label="+1 x COM",
    )
    minus_x_line = x_axis.plot(
        series.steps,
        series.minus_x_centers,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.8,
        alpha=0.65,
        label="-1 x COM",
    )
    x_axis.set_xlabel("Simulation step")
    x_axis.set_ylabel("Center of mass x", color="tab:blue")
    x_axis.tick_params(axis="y", labelcolor="tab:blue")
    x_axis.margins(x=0.02, y=0.15)

    theta_axis = x_axis.twinx()
    plus_theta_line = theta_axis.plot(
        series.steps,
        series.plus_theta_centers,
        color="#ff7f0e",
        linewidth=1.8,
        alpha=0.9,
        label="+1 theta COM",
    )
    minus_theta_line = theta_axis.plot(
        series.steps,
        series.minus_theta_centers,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.8,
        alpha=0.65,
        label="-1 theta COM",
    )
    _format_theta_axis(theta_axis)

    lines = plus_x_line + plus_theta_line + minus_x_line + minus_theta_line
    labels = [line.get_label() for line in lines]
    x_axis.legend(
        lines,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
    )
    x_axis.set_title("Cylinder disclination center of mass")
    x_axis.grid(True, ls="--", alpha=0.28)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

def plot_dislocation_center_of_mass_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
    filename: str | Path | None = CYLINDER_PATHS.dislocation_com_plot,
) -> None:
    series = dislocation_summary_series(
        input_gsd,
        neighbor_count_txt,
    )

    fig, x_axis = plt.subplots(figsize=(10, 5))
    x_line = x_axis.plot(
        series.steps,
        series.x_centers,
        color="tab:blue",
        label="dislocation x COM",
    )
    x_axis.set_xlabel("Simulation step")
    x_axis.set_ylabel("Center of mass x", color="tab:blue")
    x_axis.tick_params(axis="y", labelcolor="tab:blue")
    x_axis.margins(x=0.02, y=0.15)

    theta_axis = x_axis.twinx()
    theta_line = theta_axis.plot(
        series.steps,
        series.theta_centers,
        color="tab:orange",
        label="dislocation theta COM",
    )
    _format_theta_axis(theta_axis)

    lines = x_line + theta_line
    labels = [line.get_label() for line in lines]
    x_axis.legend(lines, labels, loc="best")
    x_axis.set_title("Cylinder dislocation center of mass")
    x_axis.grid(True, ls="--", alpha=0.35)
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_dislocation_count_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
    filename: str | Path | None = CYLINDER_PATHS.dislocation_count_plot,
) -> None:
    series = dislocation_summary_series(
        input_gsd,
        neighbor_count_txt,
    )

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        series.steps,
        series.dislocation_counts,
        color="tab:purple",
        label="dislocation particles",
    )
    axis.set_xlabel("Simulation step")
    axis.set_ylabel("Number of dislocation particles")
    axis.set_title("Cylinder dislocation count")
    axis.grid(True, ls="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_disclination_count_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
    filename: str | Path | None = CYLINDER_PATHS.disclination_count_plot,
) -> None:
    series = dislocation_summary_series(
        input_gsd,
        neighbor_count_txt,
    )

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        series.steps,
        series.plus_disclination_counts,
        color="tab:orange",
        label="+1 disclinations",
    )
    axis.plot(
        series.steps,
        series.minus_disclination_counts,
        color="tab:red",
        linestyle="--",
        label="-1 disclinations",
    )
    axis.set_xlabel("Simulation step")
    axis.set_ylabel("Number of disclinations")
    axis.set_title("Cylinder disclination counts")
    axis.grid(True, ls="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_net_disclination_charge_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
    filename: str | Path | None = CYLINDER_PATHS.net_charge_plot,
) -> None:
    series = dislocation_summary_series(
        input_gsd,
        neighbor_count_txt,
    )

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        series.steps,
        series.net_disclination_charges,
        color="tab:green",
        label=r"net charge $\sum_i(6 - n_i)$",
    )
    axis.axhline(0, color="black", linewidth=1.0, alpha=0.45)
    axis.set_xlabel("Simulation step")
    axis.set_ylabel("Net disclination charge")
    axis.set_title("Cylinder net disclination charge")
    axis.grid(True, ls="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
