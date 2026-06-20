from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.optimize import curve_fit

from hexatic.active_matter_cylinder.common import _theta_bin_indices, _x_bin_indices
from hexatic.active_matter_cylinder.config import ACTIVE_DATA_DIR, ActiveMatterFields
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


ACTIVE_MATTER_FIELDS_FILE = ACTIVE_DATA_DIR / "active_matter_fields.npz"


@dataclass(frozen=True)
class RelaxationParams:
    v_inf: float
    amplitude: float
    tau: float
    amplitude_2: float | None = None
    tau_2: float | None = None

    @property
    def stage_count(self) -> int:
        return 1 if self.amplitude_2 is None or self.tau_2 is None else 2

    def as_tuple(self) -> tuple[float, ...]:
        if self.stage_count == 1:
            return (self.v_inf, self.amplitude, self.tau)
        assert self.amplitude_2 is not None
        assert self.tau_2 is not None
        return (self.v_inf, self.amplitude, self.tau, self.amplitude_2, self.tau_2)


@dataclass(frozen=True)
class RelaxationFit:
    stage_count: int
    params: RelaxationParams
    aicc: float
    steps: np.ndarray
    modeled_values: np.ndarray
    residual_noise: float
    rms_residual: float
    fit_mode: str


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


def _fit_relaxation_series(
    steps: np.ndarray,
    values: np.ndarray,
    fit_mode: str = "auto",
) -> RelaxationFit | None:
    if fit_mode not in {"auto", "single", "double"}:
        raise ValueError("fit_mode must be 'auto', 'single', or 'double'.")

    finite = np.isfinite(steps) & np.isfinite(values)
    fit_steps = np.asarray(steps[finite], dtype=np.float64)
    fit_values = np.asarray(values[finite], dtype=np.float64)
    if fit_steps.size < 4:
        return None

    time = (fit_steps - fit_steps[0]) * float(cylinder.TIMESTEP)
    time_span = float(np.ptp(time))
    if not np.isfinite(time_span) or time_span <= 0.0:
        return None

    tail_start = max(0, int(0.75 * fit_values.size))
    v_inf_guess = float(np.nanmedian(fit_values[tail_start:]))
    amplitude_guess = float(fit_values[0] - v_inf_guess)
    if amplitude_guess == 0.0:
        amplitude_guess = float(np.nanmax(fit_values) - np.nanmin(fit_values))
    if amplitude_guess == 0.0:
        amplitude_guess = 1.0

    tau_min = max(np.finfo(float).eps, time_span / 1.0e6)
    fits = []

    if fit_mode in {"auto", "single"}:
        try:
            single_params, _ = curve_fit(
                _single_relaxation,
                time,
                fit_values,
                p0=(v_inf_guess, amplitude_guess, max(time_span / 3.0, tau_min)),
                bounds=([-np.inf, -np.inf, tau_min], [np.inf, np.inf, np.inf]),
                maxfev=20000,
            )
            single_model = _single_relaxation(time, *single_params)
            fits.append(
                {
                    "params": RelaxationParams(
                        *tuple(float(value) for value in single_params)
                    ),
                    "aicc": _aicc(fit_values - single_model, 3),
                }
            )
        except (RuntimeError, ValueError, FloatingPointError):
            pass

    if fit_mode in {"auto", "double"} and fit_steps.size >= 7:
        try:
            double_params, _ = curve_fit(
                _double_relaxation,
                time,
                fit_values,
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
            params = RelaxationParams(
                v_inf=v_inf,
                amplitude=amplitude_1,
                tau=tau_1,
                amplitude_2=amplitude_2,
                tau_2=tau_2,
            )
            double_model = _double_relaxation(time, *params.as_tuple())
            fits.append(
                {
                    "params": params,
                    "aicc": _aicc(fit_values - double_model, 5),
                }
            )
        except (RuntimeError, ValueError, FloatingPointError):
            pass

    if not fits:
        return None

    best_fit = min(fits, key=lambda fit: fit["aicc"])
    params = best_fit["params"]
    if params.stage_count == 1:
        fitted = _single_relaxation(time, *params.as_tuple())
    else:
        fitted = _double_relaxation(time, *params.as_tuple())

    residuals = fit_values - fitted
    late_start = max(0, int(0.67 * residuals.size))
    return RelaxationFit(
        stage_count=params.stage_count,
        params=params,
        aicc=float(best_fit["aicc"]),
        steps=fit_steps,
        modeled_values=fitted,
        residual_noise=float(np.nanstd(residuals[late_start:])),
        rms_residual=float(np.sqrt(np.nanmean(residuals**2))),
        fit_mode=fit_mode,
    )


def _print_x_velocity_relaxation_fit(
    fit: RelaxationFit | None,
    shell_only: bool,
) -> None:
    population = "outer-shell" if shell_only else "all-particle"
    if fit is None:
        print(f"x velocity relaxation fit ({population}): no stable fit found.")
        return

    print(f"x velocity relaxation fit ({population}):")
    print(f"  requested mode = {fit.fit_mode}")
    print(f"  stages = {fit.stage_count}")
    print(f"  V_inf = {fit.params.v_inf:.8g}")
    if fit.stage_count == 1:
        print(f"  A = {fit.params.amplitude:.8g}")
        print(f"  tau = {fit.params.tau:.8g}")
    else:
        print(f"  A_1 = {fit.params.amplitude:.8g}")
        print(f"  tau_1 = {fit.params.tau:.8g}")
        print(f"  A_2 = {fit.params.amplitude_2:.8g}")
        print(f"  tau_2 = {fit.params.tau_2:.8g}")
    print(f"  residual noise = {fit.residual_noise:.8g}")
    print(f"  RMS residual = {fit.rms_residual:.8g}")


def _load_active_matter_fields(filename: str | Path) -> ActiveMatterFields | None:
    input_path = Path(filename)
    if not input_path.exists():
        print(f"shell P_x series: active matter fields file not found at {input_path}.")
        return None

    with np.load(input_path) as data:
        return ActiveMatterFields(
            steps=np.asarray(data["steps"]),
            x_edges=np.asarray(data["x_edges"]),
            x_centers=np.asarray(data["x_centers"]),
            theta_edges=np.asarray(data["theta_edges"]),
            theta_centers=np.asarray(data["theta_centers"]),
            coords=np.asarray(data["coords"]),
            shell_mask=np.asarray(data["shell_mask"]),
            rho=np.asarray(data["rho"]),
            active_direction=np.asarray(data["active_direction"]),
            direction_cylindrical=np.asarray(data["direction_cylindrical"]),
            polar_mean=np.asarray(data["polar_mean"]),
            polar_cylindrical=np.asarray(data["polar_cylindrical"]),
            flux_cylindrical=np.asarray(data["flux_cylindrical"]),
            force_density=np.asarray(data["force_density"]),
            force_density_cylindrical=np.asarray(data["force_density_cylindrical"]),
        )


def _shell_binned_px_series(fields: ActiveMatterFields) -> tuple[np.ndarray, np.ndarray]:
    px_values = np.full(len(fields.steps), np.nan, dtype=np.float64)
    n_x_bins = len(fields.x_centers)
    n_theta_bins = len(fields.theta_centers)
    n_bins = n_x_bins * n_theta_bins

    for frame_idx in range(len(fields.steps)):
        mask = fields.shell_mask[frame_idx]
        if not np.any(mask):
            continue

        coords = fields.coords[frame_idx, mask]
        px = fields.polar_cylindrical[frame_idx, mask, 0]
        finite = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1]) & np.isfinite(px)
        if not np.any(finite):
            continue

        box_length_x = float(fields.x_edges[-1] - fields.x_edges[0])
        x_indices = _x_bin_indices(coords[finite, 0], box_length_x, n_x_bins)
        theta_indices = _theta_bin_indices(coords[finite, 1], n_theta_bins)
        groups = x_indices * n_theta_bins + theta_indices
        counts = np.bincount(groups, minlength=n_bins)
        sums = np.bincount(groups, weights=px[finite], minlength=n_bins)
        occupied = counts > 0
        if np.any(occupied):
            bin_means = sums[occupied] / counts[occupied]
            px_values[frame_idx] = float(np.mean(bin_means))

    return fields.steps, px_values


def _print_px_relaxation_fit(fit: RelaxationFit | None) -> None:
    if fit is None:
        print("shell P_x relaxation fit: no stable fit found.")
        return

    print("shell P_x relaxation fit:")
    print(f"  requested mode = {fit.fit_mode}")
    print(f"  stages = {fit.stage_count}")
    print(f"  P_inf = {fit.params.v_inf:.8g}")
    if fit.stage_count == 1:
        print(f"  A = {fit.params.amplitude:.8g}")
        print(f"  tau = {fit.params.tau:.8g}")
    else:
        print(f"  A_1 = {fit.params.amplitude:.8g}")
        print(f"  tau_1 = {fit.params.tau:.8g}")
        print(f"  A_2 = {fit.params.amplitude_2:.8g}")
        print(f"  tau_2 = {fit.params.tau_2:.8g}")
    print(f"  residual noise = {fit.residual_noise:.8g}")
    print(f"  RMS residual = {fit.rms_residual:.8g}")


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


def plot_velocity_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    filename: str | Path | None = CYLINDER_PATHS.x_com_velocity_plot,
    shell_only: bool = False,
    relaxation_fit_mode: str = "auto",
    active_matter_fields_file: str | Path = ACTIVE_MATTER_FIELDS_FILE,
) -> None:
    series = x_center_of_mass_velocity_series(input_gsd, shell_only=shell_only)
    vx_fit = _fit_relaxation_series(
        series.steps,
        series.x_velocities,
        fit_mode=relaxation_fit_mode,
    )
    assert vx_fit is not None
    _print_x_velocity_relaxation_fit(vx_fit, shell_only)
    active_fields = _load_active_matter_fields(active_matter_fields_file)
    px_steps = px_values = px_fit = None
    if active_fields is not None:
        px_steps, px_values = _shell_binned_px_series(active_fields)
        px_fit = _fit_relaxation_series(
            px_steps,
            px_values,
            fit_mode=relaxation_fit_mode,
        )
    _print_px_relaxation_fit(px_fit)
    assert px_fit is not None
    u0_eff = vx_fit.params.amplitude / px_fit.params.amplitude

    print(f"U0_eff = {u0_eff}")
    print(f"U0 actual = {cylinder.U0}")
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        series.steps,
        series.x_velocities,
        color="tab:blue",
        label="velocity_x",
    )
    if vx_fit is not None:
        fit_label = (
            "velocity_x best-fit relaxation"
            if vx_fit.stage_count == 1
            else "velocity_x best-fit two-stage relaxation"
        )
        axis.plot(
            vx_fit.steps,
            vx_fit.modeled_values,
            color="tab:blue",
            linestyle="--",
            linewidth=2.0,
            label=fit_label,
        )
    px_axis = axis.twinx()
    if px_steps is not None and px_values is not None:
        px_axis.plot(
            px_steps,
            px_values,
            color="purple",
            label=r"$P_x$",
        )
    if px_fit is not None:
        px_fit_label = (
            r"$P_x$ best-fit relaxation"
            if px_fit.stage_count == 1
            else r"$P_x$ best-fit two-stage relaxation"
        )
        px_axis.plot(
            px_fit.steps,
            px_fit.modeled_values,
            color="purple",
            linestyle="--",
            linewidth=2.0,
            label=px_fit_label,
        )
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.45)
    px_axis.axhline(0.0, color="purple", linewidth=1.0, alpha=0.2)
    axis.set_xlabel("Simulation step")
    axis.set_ylabel("x center-of-mass velocity", color="tab:blue")
    axis.tick_params(axis="y", labelcolor="tab:blue")
    px_axis.set_ylabel(r"shell-binned mean $P_x$", color="purple")
    px_axis.tick_params(axis="y", labelcolor="purple")
    population = "outer-shell" if shell_only else "all-particle"
    axis.set_title(f"Cylinder {population} x velocity and shell mean $P_x$")
    axis.grid(True, ls="--", alpha=0.35)
    lines, labels = axis.get_legend_handles_labels()
    px_lines, px_labels = px_axis.get_legend_handles_labels()
    axis.legend(lines + px_lines, labels + px_labels, loc="best")
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
