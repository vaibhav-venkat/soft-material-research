from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gsd.hoomd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

from hexatic import analysis as hx
from hexatic.active_matter_cylinder.config import (
    ACTIVE_RADIAL_BIN_WIDTH,
    LOCAL_POCKET_RADIUS,
)
from hexatic.active_matter_cylinder.grid_utils import _time_edges
from hexatic.active_matter_cylinder.math_utils import _gaussian_delta_weights
from hexatic.constants import cylinder
from hexatic.radii_analysis.cases import RadiusCase

from .best_fit import fit_payload
from .common import (
    FRAME_START,
    FRAME_STOP,
    NPZ_OUTPUT_DIR,
    PLOT_OUTPUT_DIR,
    active_fields_path,
    frame_indices,
    load_active_fields,
    load_cached_metric_values,
    load_metric_fit_curves,
    radii_for_cases,
    save_metric_npz,
)
from .plotting import plot_for_cases, plots_missing


@dataclass(frozen=True)
class RadialExchangeCurrentSeries:
    case_id: str
    label: str
    radius: float
    steps: np.ndarray
    radius_edges: np.ndarray
    radius_centers: np.ndarray
    rho: np.ndarray
    v_r: np.ndarray
    j_r: np.ndarray
    source: str


def _radius_edges_and_centers(
    cylinder_radius: float,
    radial_bin_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    if cylinder_radius <= 0.0:
        raise ValueError("cylinder_radius must be positive.")
    if radial_bin_width <= 0.0:
        raise ValueError("radial_bin_width must be positive.")
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


def _empty_series(
    case: RadiusCase,
    radius_edges: np.ndarray,
    radius_centers: np.ndarray,
    source: str,
) -> RadialExchangeCurrentSeries:
    empty = np.empty((0, len(radius_centers)), dtype=np.float64)
    return RadialExchangeCurrentSeries(
        case_id=case.case_id,
        label=case.label or case.case_id,
        radius=float(case.radius),
        steps=np.asarray([], dtype=np.int64),
        radius_edges=radius_edges,
        radius_centers=radius_centers,
        rho=empty,
        v_r=empty,
        j_r=empty,
        source=source,
    )


def _frame_values(
    current_radii: np.ndarray,
    radial_velocity: np.ndarray,
    radius_centers: np.ndarray,
    cylinder_radius: float,
    kernel_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = (
        np.isfinite(current_radii)
        & np.isfinite(radial_velocity)
        & (current_radii >= 0.0)
        & (current_radii <= cylinder_radius)
    )
    if not np.any(valid):
        empty = np.full_like(radius_centers, np.nan, dtype=np.float64)
        return empty, empty.copy(), empty.copy()

    delta_sq = (radius_centers[:, np.newaxis] - current_radii[valid][np.newaxis, :]) ** 2
    weights = _gaussian_delta_weights(delta_sq, kernel_radius)
    rho = np.sum(weights, axis=1)
    j_r = weights @ radial_velocity[valid]
    return rho, _safe_divide(j_r, rho), j_r


def _series_from_radii(
    case: RadiusCase,
    radii_by_frame: np.ndarray,
    steps: np.ndarray,
    source: str,
    frame_start: int,
    frame_stop: int,
    radial_bin_width: float,
    kernel_radius: float,
) -> RadialExchangeCurrentSeries:
    radius_edges, radius_centers = _radius_edges_and_centers(
        case.radius,
        radial_bin_width,
    )
    selected = frame_indices(len(steps), frame_start, frame_stop)
    selected = selected[selected > 0]
    if selected.size == 0:
        return _empty_series(case, radius_edges, radius_centers, source)

    out_steps: list[int] = []
    rho_frames: list[np.ndarray] = []
    vr_frames: list[np.ndarray] = []
    jr_frames: list[np.ndarray] = []
    for frame_idx in selected:
        previous_idx = int(frame_idx) - 1
        delta_t = (int(steps[frame_idx]) - int(steps[previous_idx])) * float(
            cylinder.TIMESTEP
        )
        if delta_t <= 0.0:
            continue
        current_radii = np.asarray(radii_by_frame[frame_idx], dtype=np.float64)
        previous_radii = np.asarray(radii_by_frame[previous_idx], dtype=np.float64)
        radial_velocity = (current_radii - previous_radii) / delta_t
        rho, v_r, j_r = _frame_values(
            current_radii,
            radial_velocity,
            radius_centers,
            case.radius,
            kernel_radius,
        )
        out_steps.append(int(steps[frame_idx]))
        rho_frames.append(rho)
        vr_frames.append(v_r)
        jr_frames.append(j_r)

    if not out_steps:
        return _empty_series(case, radius_edges, radius_centers, source)

    return RadialExchangeCurrentSeries(
        case_id=case.case_id,
        label=case.label or case.case_id,
        radius=float(case.radius),
        steps=np.asarray(out_steps, dtype=np.int64),
        radius_edges=radius_edges,
        radius_centers=radius_centers,
        rho=np.vstack(rho_frames),
        v_r=np.vstack(vr_frames),
        j_r=np.vstack(jr_frames),
        source=source,
    )


def _series_from_active_fields(
    case: RadiusCase,
    frame_start: int,
    frame_stop: int,
    radial_bin_width: float,
    kernel_radius: float,
) -> RadialExchangeCurrentSeries:
    fields = load_active_fields(active_fields_path(case))
    radii = np.asarray(fields.coords[..., 2], dtype=np.float64)
    steps = np.asarray(fields.steps, dtype=np.int64)
    return _series_from_radii(
        case,
        radii,
        steps,
        "active_matter_fields",
        frame_start,
        frame_stop,
        radial_bin_width,
        kernel_radius,
    )


def _series_from_gsd(
    case: RadiusCase,
    frame_start: int,
    frame_stop: int,
    radial_bin_width: float,
    kernel_radius: float,
) -> RadialExchangeCurrentSeries:
    radius_edges, radius_centers = _radius_edges_and_centers(
        case.radius,
        radial_bin_width,
    )
    selected: set[int]
    with gsd.hoomd.open(name=str(case.trajectory_gsd), mode="r") as source:
        selected = set(frame_indices(len(source), frame_start, frame_stop).tolist())
        steps: list[int] = []
        rho_frames: list[np.ndarray] = []
        vr_frames: list[np.ndarray] = []
        jr_frames: list[np.ndarray] = []
        previous_radii: np.ndarray | None = None
        previous_step: int | None = None
        for frame_idx, frame in enumerate(source):
            particles = frame.particles
            if particles.position is None:
                continue
            positions = np.asarray(particles.position, dtype=np.float64)
            current_radii = hx.get_new_coords(positions)[:, 2]
            current_step = int(frame.configuration.step)
            if frame_idx in selected and previous_radii is not None and previous_step is not None:
                delta_t = (current_step - previous_step) * float(cylinder.TIMESTEP)
                if delta_t > 0.0:
                    radial_velocity = (current_radii - previous_radii) / delta_t
                    rho, v_r, j_r = _frame_values(
                        current_radii,
                        radial_velocity,
                        radius_centers,
                        case.radius,
                        kernel_radius,
                    )
                    steps.append(current_step)
                    rho_frames.append(rho)
                    vr_frames.append(v_r)
                    jr_frames.append(j_r)
            previous_radii = current_radii
            previous_step = current_step

    if not steps:
        return _empty_series(case, radius_edges, radius_centers, "gsd")

    return RadialExchangeCurrentSeries(
        case_id=case.case_id,
        label=case.label or case.case_id,
        radius=float(case.radius),
        steps=np.asarray(steps, dtype=np.int64),
        radius_edges=radius_edges,
        radius_centers=radius_centers,
        rho=np.vstack(rho_frames),
        v_r=np.vstack(vr_frames),
        j_r=np.vstack(jr_frames),
        source="gsd",
    )


def radial_exchange_current_series(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    radial_bin_width: float = ACTIVE_RADIAL_BIN_WIDTH,
    kernel_radius: float = LOCAL_POCKET_RADIUS,
) -> RadialExchangeCurrentSeries:
    if kernel_radius <= 0.0:
        raise ValueError("kernel_radius must be positive.")
    if active_fields_path(case).exists():
        return _series_from_active_fields(
            case,
            frame_start,
            frame_stop,
            radial_bin_width,
            kernel_radius,
        )
    return _series_from_gsd(
        case,
        frame_start,
        frame_stop,
        radial_bin_width,
        kernel_radius,
    )


def _case_npz_path(case: RadiusCase) -> Path:
    return NPZ_OUTPUT_DIR / f"{case.case_id}_radial_exchange_current.npz"


def _case_plot_path(case: RadiusCase) -> Path:
    return PLOT_OUTPUT_DIR / f"{case.case_id}_radial_exchange_current.png"


def _aggregate_npz_path() -> Path:
    return NPZ_OUTPUT_DIR / "radial_exchange_current.npz"


def _aggregate_plot_path() -> Path:
    return PLOT_OUTPUT_DIR / "radial_exchange_current.png"


def save_radial_exchange_current(
    series: RadialExchangeCurrentSeries,
    filename: str | Path,
    frame_start: int,
    frame_stop: int,
    kernel_radius: float,
    radial_bin_width: float,
) -> None:
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        case_id=np.asarray(series.case_id),
        label=np.asarray(series.label),
        radius=np.asarray(series.radius, dtype=np.float64),
        frame_start=np.asarray(frame_start, dtype=np.int64),
        frame_stop=np.asarray(frame_stop, dtype=np.int64),
        steps=series.steps,
        radius_edges=series.radius_edges,
        radius_centers=series.radius_centers,
        rho=series.rho,
        v_r=series.v_r,
        j_r=series.j_r,
        kernel_radius=np.asarray(kernel_radius, dtype=np.float64),
        radial_bin_width=np.asarray(radial_bin_width, dtype=np.float64),
        source=np.asarray(series.source),
    )


def load_radial_exchange_current(filename: str | Path) -> RadialExchangeCurrentSeries:
    with np.load(filename) as data:
        return RadialExchangeCurrentSeries(
            case_id=str(np.asarray(data["case_id"]).item()),
            label=str(np.asarray(data["label"]).item()),
            radius=float(np.asarray(data["radius"]).item()),
            steps=np.asarray(data["steps"], dtype=np.int64),
            radius_edges=np.asarray(data["radius_edges"], dtype=np.float64),
            radius_centers=np.asarray(data["radius_centers"], dtype=np.float64),
            rho=np.asarray(data["rho"], dtype=np.float64),
            v_r=np.asarray(data["v_r"], dtype=np.float64),
            j_r=np.asarray(data["j_r"], dtype=np.float64),
            source=str(np.asarray(data["source"]).item()),
        )


def _validate_cached_case_window(
    case: RadiusCase,
    filename: str | Path,
    frame_start: int,
    frame_stop: int,
) -> None:
    input_path = Path(filename)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing cached radial exchange current NPZ for {case.case_id}: "
            f"{input_path}. Generate the per-case current cache first or skip "
            "radial_exchange_current."
        )

    with np.load(input_path) as data:
        cached_case_id = str(np.asarray(data["case_id"]).item())
        cached_start = int(np.asarray(data["frame_start"]).item())
        cached_stop = int(np.asarray(data["frame_stop"]).item())

    if (
        cached_case_id != case.case_id
        or cached_start != int(frame_start)
        or cached_stop != int(frame_stop)
    ):
        raise FileExistsError(
            f"Cached radial exchange current {input_path} does not match the "
            "requested case/frame window. Use the same --frame-start/--frame-stop "
            "as the cached current files, or regenerate those per-case caches."
        )


def shell_radial_exchange_current_value_for_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
) -> float:
    input_path = _case_npz_path(case)
    _validate_cached_case_window(case, input_path, frame_start, frame_stop)
    series = load_radial_exchange_current(input_path)
    shell = (
        (series.radius_centers >= case.radius - cylinder.ANALYSIS.wall_cutoff)
        & (series.radius_centers <= case.radius)
    )
    if not np.any(shell):
        return np.nan
    values = np.asarray(series.j_r[:, shell], dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan
    return float(np.mean(values[finite]))


def _cached_series(
    case: RadiusCase,
    filename: str | Path,
    frame_start: int,
    frame_stop: int,
    radial_bin_width: float,
    kernel_radius: float,
    overwrite: bool,
) -> RadialExchangeCurrentSeries | None:
    input_path = Path(filename)
    if overwrite or not input_path.exists():
        return None

    with np.load(input_path) as data:
        cached_case_id = str(np.asarray(data["case_id"]).item())
        cached_start = int(np.asarray(data["frame_start"]).item())
        cached_stop = int(np.asarray(data["frame_stop"]).item())
        cached_bin_width = float(np.asarray(data["radial_bin_width"]).item())
        cached_kernel = float(np.asarray(data["kernel_radius"]).item())
    if (
        cached_case_id != case.case_id
        or cached_start != int(frame_start)
        or cached_stop != int(frame_stop)
        or not np.isclose(cached_bin_width, radial_bin_width)
        or not np.isclose(cached_kernel, kernel_radius)
    ):
        raise FileExistsError(
            f"Cached radial exchange current {input_path} does not match the "
            "requested case/window/kernel settings. Use --overwrite to regenerate it."
        )
    return load_radial_exchange_current(input_path)


def _finite_limits(values: np.ndarray, symmetric: bool = False) -> tuple[float, float]:
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


def _plot_heatmap(axis, series: RadialExchangeCurrentSeries, values, title, label, symmetric=False):
    if series.steps.size == 0 or values.size == 0:
        axis.text(0.5, 0.5, "no frame overlap", ha="center", va="center")
        axis.set_title(title)
        axis.set_ylabel("r")
        axis.set_axis_off()
        return None

    vmin, vmax = _finite_limits(values, symmetric=symmetric)
    cmap = plt.get_cmap("coolwarm" if symmetric else "viridis").copy()
    cmap.set_bad("0.85")
    norm = (
        TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        if symmetric
        else Normalize(vmin=vmin, vmax=vmax)
    )
    mesh = axis.pcolormesh(
        _time_edges(series.steps),
        series.radius_edges,
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
    series: RadialExchangeCurrentSeries,
    filename: str | Path,
) -> None:
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    plots = (
        (series.rho, r"$\rho_K(r,t)$", "Gaussian kernel density", False),
        (series.v_r, r"$v_r(r,t)$", "Kernel-averaged radial velocity", True),
        (series.j_r, r"$J_r(r,t)$", "Radial exchange current", True),
    )
    for axis, (values, label, title, symmetric) in zip(axes, plots):
        mesh_info = _plot_heatmap(
            axis,
            series,
            values,
            title,
            label,
            symmetric=symmetric,
        )
        if mesh_info is not None:
            mesh, colorbar_label = mesh_info
            fig.colorbar(mesh, ax=axis, label=colorbar_label)
    axes[-1].set_xlabel("Simulation step")
    fig.suptitle(f"{series.label}: radial exchange current")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
    radial_bin_width: float = ACTIVE_RADIAL_BIN_WIDTH,
    kernel_radius: float = LOCAL_POCKET_RADIUS,
) -> dict[str, np.ndarray]:
    output_npz = _aggregate_npz_path()
    output_png = _aggregate_plot_path()
    value_names = ("shell",)
    arrays = load_cached_metric_values(
        output_npz,
        value_names,
        cases,
        frame_start,
        frame_stop,
        overwrite=overwrite,
    )
    if arrays is not None:
        if plots_missing(cases, output_png):
            fits = load_metric_fit_curves(output_npz, value_names)
            plot_for_cases(
                cases,
                arrays,
                output_png,
                title="Mean shell radial exchange current vs radius",
                ylabel="mean shell J_r",
                fits=fits,
            )
        print(f"using cached radial_exchange_current values from {output_npz}")
        return arrays

    values = {"shell": []}
    for case in cases:
        values["shell"].append(
            shell_radial_exchange_current_value_for_case(
                case,
                frame_start=frame_start,
                frame_stop=frame_stop,
            )
        )
    arrays = {name: np.asarray(series, dtype=np.float64) for name, series in values.items()}
    fits, payload = fit_payload(radii_for_cases(cases), arrays)
    payload.update(
        {
            "shell_cutoff": np.asarray(cylinder.ANALYSIS.wall_cutoff, dtype=np.float64),
            "radial_bin_width": np.asarray(radial_bin_width, dtype=np.float64),
            "kernel_radius": np.asarray(kernel_radius, dtype=np.float64),
        }
    )
    save_metric_npz(
        output_npz,
        cases,
        "radial_exchange_current",
        arrays,
        payload,
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    plot_for_cases(
        cases,
        arrays,
        output_png,
        title="Mean shell radial exchange current vs radius",
        ylabel="mean shell J_r",
        fits=fits,
    )
    return arrays


def run_per_case_heatmaps(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
    radial_bin_width: float = ACTIVE_RADIAL_BIN_WIDTH,
    kernel_radius: float = LOCAL_POCKET_RADIUS,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for case in cases:
        output_npz = _case_npz_path(case)
        output_png = _case_plot_path(case)
        series = _cached_series(
            case,
            output_npz,
            frame_start,
            frame_stop,
            radial_bin_width,
            kernel_radius,
            overwrite,
        )
        if series is None:
            series = radial_exchange_current_series(
                case,
                frame_start=frame_start,
                frame_stop=frame_stop,
                radial_bin_width=radial_bin_width,
                kernel_radius=kernel_radius,
            )
            save_radial_exchange_current(
                series,
                output_npz,
                frame_start=frame_start,
                frame_stop=frame_stop,
                kernel_radius=kernel_radius,
                radial_bin_width=radial_bin_width,
            )
            plot_radial_exchange_current(series, output_png)
        elif not output_png.exists():
            plot_radial_exchange_current(series, output_png)
        else:
            print(f"using cached radial_exchange_current values from {output_npz}")
        outputs[case.case_id] = output_npz
    return outputs
