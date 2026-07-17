from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.integrate import simpson
from scipy.optimize import least_squares

from .correlations import lagged_pearson
from .plot_velocity_laplace import AxialCenterSeries, LaplaceCase
from .plot_velocity_laplace import _axial_center_series


BIG_LX_SCHEMA = "hexatic.big_lx.analysis.v1"
CONFINEMENT_SCHEMA = "hexatic.confinement_comparison.analysis.v1"
REGULAR_CIRCUMFERENCE_DIAMETERS = 60.5
LX_MULTIPLIERS = (1, 2, 4, 8, 16)
CONFINEMENT_CASE_IDS = (
    "prism_volume",
    "prism_surface_area",
    "sandwich_volume",
    "sandwich_surface_area",
    "two_dimension",
    "cylinder_rattle",
    "cylinder_rattle_tangent",
)
CONFINEMENT_MARKERS = {
    "prism_volume": "s",
    "prism_surface_area": "X",
    "sandwich_volume": "D",
    "sandwich_surface_area": "P",
    "two_dimension": "^",
    "cylinder_rattle": "v",
    "cylinder_rattle_tangent": "*",
}


@dataclass(frozen=True)
class DiscoveredCase:
    case: LaplaceCase
    mode: str
    lx_multiplier: int
    circumference_diameters: float | None
    geometry_kind: str | None
    manifest_path: Path


@dataclass(frozen=True)
class CorrelationInput:
    discovered: DiscoveredCase
    lag_times: np.ndarray
    correlation: np.ndarray
    origin_counts: np.ndarray


@dataclass(frozen=True)
class CaseEstimate:
    discovered: DiscoveredCase
    coordinate: float
    score: float
    at_lower_boundary: bool
    at_upper_boundary: bool
    fit_omega: float | None = None
    fit_amplitude: float | None = None
    fit_phase: float | None = None
    fit_offset: float | None = None


@dataclass(frozen=True)
class RegularCoordinateSummary:
    circumference_diameters: float
    lx_multiplier: int
    mean_coordinate: float
    std_coordinate: float
    replicate_count: int


def _numeric(payload: dict[str, object], name: str, manifest_path: Path) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric {name!r} in {manifest_path}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Non-finite {name!r} in {manifest_path}")
    return result


def _integer(payload: dict[str, object], name: str, manifest_path: Path) -> int:
    value = _numeric(payload, name, manifest_path)
    result = int(value)
    if float(result) != value:
        raise ValueError(f"{name!r} must be an integer in {manifest_path}")
    return result


def _validate_manifest_index(
    manifest: dict[str, object],
    manifest_path: Path,
) -> None:
    if manifest.get("complete") is not True:
        raise ValueError(f"Analysis is not marked complete: {manifest_path}")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"Analysis manifest has no shards: {manifest_path}")

    analysis_dir = manifest_path.parent.resolve()
    expected_start = 0
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError(f"Invalid shard entry in {manifest_path}")
        start = shard.get("frame_start")
        stop = shard.get("frame_stop")
        filename = shard.get("file")
        if (
            not isinstance(start, int)
            or start != expected_start
            or not isinstance(stop, int)
            or stop <= start
            or not isinstance(filename, str)
        ):
            raise ValueError(
                f"Analysis shards are not contiguous from frame zero: {manifest_path}"
            )
        shard_path = (analysis_dir / filename).resolve()
        try:
            shard_path.relative_to(analysis_dir)
        except ValueError as error:
            raise ValueError(
                f"Shard path escapes its analysis directory: {shard_path}"
            ) from error
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing frame shard: {shard_path}")
        expected_start = stop

    frame_count = manifest.get("frame_count")
    if not isinstance(frame_count, int) or frame_count != expected_start:
        raise ValueError(f"Manifest frame count does not match shards: {manifest_path}")


def _case_from_manifest(manifest_path: Path) -> DiscoveredCase | None:
    manifest = json.loads(manifest_path.read_text())
    schema = manifest.get("schema")
    if schema not in {BIG_LX_SCHEMA, CONFINEMENT_SCHEMA}:
        return None
    _validate_manifest_index(manifest, manifest_path)

    payload = manifest.get("case")
    if not isinstance(payload, dict):
        raise ValueError(f"Analysis manifest has no case metadata: {manifest_path}")
    case_id = payload.get("case_id")
    label = payload.get("label")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"Missing case ID in {manifest_path}")
    if not isinstance(label, str) or not label:
        label = case_id
    lx = _numeric(payload, "lx", manifest_path)
    n_particles = _integer(payload, "n_particles", manifest_path)
    lx_multiplier = _integer(payload, "lx_multiplier", manifest_path)
    if lx <= 0.0 or n_particles < 1:
        raise ValueError(f"Invalid case geometry in {manifest_path}")
    if lx_multiplier not in LX_MULTIPLIERS:
        raise ValueError(
            f"Unsupported Lx multiplier {lx_multiplier} in {manifest_path}"
        )

    if schema == BIG_LX_SCHEMA:
        mode = "big-lx"
        circumference_diameters = _numeric(
            payload,
            "circumference_diameters",
            manifest_path,
        )
        if not np.isclose(
            circumference_diameters,
            REGULAR_CIRCUMFERENCE_DIAMETERS,
        ):
            return None
        geometry_kind = None
    else:
        mode = "confinement"
        circumference_diameters = None
        raw_geometry_kind = payload.get("geometry_kind")
        if not isinstance(raw_geometry_kind, str):
            raise ValueError(f"Missing confinement geometry in {manifest_path}")
        if raw_geometry_kind not in CONFINEMENT_CASE_IDS:
            return None
        if case_id != raw_geometry_kind:
            raise ValueError(
                f"Confinement case ID and geometry disagree in {manifest_path}"
            )
        if lx_multiplier != 1:
            raise ValueError(
                f"Confinement case must use Lx multiplier 1: {manifest_path}"
            )
        geometry_kind = raw_geometry_kind

    return DiscoveredCase(
        case=LaplaceCase(
            case_id=case_id,
            label=label,
            lx=lx,
            n_particles=n_particles,
            analysis_dir=manifest_path.parent,
        ),
        mode=mode,
        lx_multiplier=lx_multiplier,
        circumference_diameters=circumference_diameters,
        geometry_kind=geometry_kind,
        manifest_path=manifest_path,
    )


def scan_input_directories(input_dirs: list[Path]) -> list[DiscoveredCase]:
    discovered: list[DiscoveredCase] = []
    confinement_by_key: dict[tuple[str, str], DiscoveredCase] = {}
    seen_roots: set[Path] = set()
    for input_dir in input_dirs:
        root = input_dir.resolve()
        if root in seen_roots:
            continue
        seen_roots.add(root)
        safetensors_root = root / "safetensors_output"
        if not safetensors_root.is_dir():
            raise FileNotFoundError(
                f"Input directory has no safetensors_output directory: {root}"
            )
        for manifest_path in sorted(safetensors_root.glob("*/manifest.json")):
            item = _case_from_manifest(manifest_path)
            if item is None:
                continue
            key = (item.mode, item.case.case_id)
            if item.mode == "confinement":
                previous = confinement_by_key.get(key)
                if previous is not None:
                    raise ValueError(
                        f"Duplicate confinement case {item.case.case_id!r}: "
                        f"{previous.manifest_path} and {item.manifest_path}"
                    )
                confinement_by_key[key] = item
            discovered.append(item)
    if not discovered:
        raise ValueError("No eligible complete analysis cases were discovered")
    return discovered


def _correlation_input(
    discovered: DiscoveredCase,
    series: AxialCenterSeries,
    *,
    min_origins: int,
    max_lag: int | None,
) -> CorrelationInput:
    frame_count = series.elapsed_time.size
    if min_origins > frame_count:
        raise ValueError(
            f"min_origins={min_origins} exceeds {frame_count} frames for "
            f"{discovered.case.case_id}"
        )
    selected_max_lag = frame_count - min_origins
    if max_lag is not None:
        selected_max_lag = min(selected_max_lag, max_lag)
    if selected_max_lag < 1:
        raise ValueError(
            f"No positive lag remains for {discovered.case.case_id}"
        )

    spacing = np.diff(series.elapsed_time)
    dt = float(spacing[0])
    if not np.allclose(spacing, dt, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(
            f"Pearson correlation requires uniform samples for "
            f"{discovered.case.case_id}"
        )
    lag_times = np.arange(selected_max_lag + 1, dtype=np.float64) * dt
    correlation = lagged_pearson(
        series.x_velocity,
        selected_max_lag,
        f"Axial COM velocity for {discovered.case.case_id}",
    )
    return CorrelationInput(
        discovered=discovered,
        lag_times=lag_times,
        correlation=correlation,
        origin_counts=(
            frame_count - np.arange(selected_max_lag + 1, dtype=np.int64)
        ),
    )


def _shared_positive_omega(
    inputs: list[CorrelationInput],
    *,
    omega_max: float | None,
    omega_points: int,
) -> np.ndarray:
    reference_duration = min(float(item.lag_times[-1]) for item in inputs)
    if reference_duration <= 0.0:
        raise ValueError("Correlation inputs must span positive lag time")
    nyquist_limit = min(
        np.pi / float(np.diff(item.lag_times)[0]) for item in inputs
    )
    default_maximum = min(nyquist_limit, 20.0 * np.pi / reference_duration)
    maximum = default_maximum if omega_max is None else omega_max
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("--omega-max must be positive and finite")
    if maximum > nyquist_limit * (1.0 + 1.0e-12):
        raise ValueError(
            f"--omega-max={maximum:g} exceeds the shared Nyquist limit "
            f"{nyquist_limit:g}"
        )

    symmetric = np.linspace(-maximum, maximum, omega_points, dtype=np.float64)
    positive = symmetric[symmetric > 0.0]
    if positive.size < 2:
        raise ValueError("--omega-points leaves fewer than two positive frequencies")
    return positive


def preferred_omega(
    correlation_input: CorrelationInput,
    omega: np.ndarray,
) -> CaseEstimate:
    time = correlation_input.lag_times
    correlation = correlation_input.correlation
    values = simpson(
        np.exp(1j * omega[:, None] * time[None, :])
        * correlation[None, :],
        x=time,
        axis=1,
    )
    magnitudes = np.abs(values)
    if not np.all(np.isfinite(magnitudes)) or not np.any(magnitudes > 0.0):
        raise ValueError(
            f"Invalid r=0 transform for "
            f"{correlation_input.discovered.case.case_id}"
        )
    log_magnitudes = np.log10(
        np.maximum(magnitudes, np.finfo(np.float64).tiny)
    )
    peak_index = int(np.argmax(log_magnitudes))
    at_upper_boundary = peak_index == omega.size - 1
    if at_upper_boundary:
        warnings.warn(
            f"Preferred omega for "
            f"{correlation_input.discovered.case.case_id} is at the upper "
            "frequency boundary; consider increasing --omega-max if permitted "
            "by the Nyquist limit",
            stacklevel=2,
        )
    return CaseEstimate(
        discovered=correlation_input.discovered,
        coordinate=float(omega[peak_index]),
        score=float(log_magnitudes[peak_index]),
        at_lower_boundary=False,
        at_upper_boundary=at_upper_boundary,
    )


def _shared_r(
    inputs: list[CorrelationInput],
    *,
    r_min: float | None,
    r_max: float,
    r_points: int,
) -> np.ndarray:
    reference_duration = min(float(item.lag_times[-1]) for item in inputs)
    if reference_duration <= 0.0:
        raise ValueError("Correlation inputs must span positive lag time")
    minimum = -10.0 / reference_duration if r_min is None else r_min
    if not np.isfinite(minimum) or not np.isfinite(r_max) or minimum >= r_max:
        raise ValueError("The r range requires finite --r-min < --r-max")
    maximum_exponent = r_max * max(
        float(item.lag_times[-1]) for item in inputs
    )
    if maximum_exponent > 700.0:
        raise ValueError(
            "The requested positive --r-max overflows exp(r*t); reduce --r-max"
        )
    grid = np.linspace(minimum, r_max, r_points, dtype=np.float64)
    negative = grid[grid < 0.0]
    if negative.size < 2:
        raise ValueError(
            "The requested r grid must contain at least two strictly negative "
            "values for --preferred-r"
        )
    return negative


def preferred_r(
    correlation_input: CorrelationInput,
    r: np.ndarray,
) -> CaseEstimate:
    time = correlation_input.lag_times
    correlation = correlation_input.correlation
    values = simpson(
        np.exp(r[:, None] * time[None, :]) * correlation[None, :],
        x=time,
        axis=1,
    )
    magnitudes = np.abs(values)
    if not np.all(np.isfinite(magnitudes)) or not np.any(magnitudes > 0.0):
        raise ValueError(
            f"Invalid omega=0 transform for "
            f"{correlation_input.discovered.case.case_id}"
        )
    log_magnitudes = np.log10(
        np.maximum(magnitudes, np.finfo(np.float64).tiny)
    )
    peak_index = int(np.argmax(log_magnitudes))
    at_lower_boundary = peak_index == 0
    at_upper_boundary = peak_index == r.size - 1
    if at_lower_boundary or at_upper_boundary:
        boundary = "lower" if at_lower_boundary else "upper"
        detail = (
            "The upper boundary is the available negative r closest to zero, "
            "not r=0."
            if at_upper_boundary
            else "Use a more-negative --r-min to test whether the maximum is interior."
        )
        warnings.warn(
            f"Preferred r for {correlation_input.discovered.case.case_id} "
            f"is at the {boundary} search boundary; expand the r range before "
            f"interpreting it as an interior optimum. {detail}",
            stacklevel=2,
        )
    return CaseEstimate(
        discovered=correlation_input.discovered,
        coordinate=float(r[peak_index]),
        score=float(log_magnitudes[peak_index]),
        at_lower_boundary=at_lower_boundary,
        at_upper_boundary=at_upper_boundary,
    )


def _damped_cosine(
    time: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    amplitude, tau_r, omega, phase, offset = parameters
    return (
        amplitude
        * np.exp(-time / tau_r)
        * np.cos(omega * time + phase)
        + offset
    )


def fit_tau_r(
    correlation_input: CorrelationInput,
    omega_grid: np.ndarray,
    *,
    tau_max: float | None,
) -> CaseEstimate:
    time = correlation_input.lag_times
    correlation = correlation_input.correlation
    dt = float(time[1] - time[0])
    duration = float(time[-1])
    lower_tau = dt
    upper_tau = 10.0 * duration if tau_max is None else tau_max
    if not np.isfinite(upper_tau) or upper_tau <= lower_tau:
        raise ValueError(
            f"tau-max must exceed the lag spacing {lower_tau:g} for "
            f"{correlation_input.discovered.case.case_id}"
        )

    initial_peak = preferred_omega(correlation_input, omega_grid).coordinate
    nyquist = np.pi / dt
    tail_count = max(3, correlation.size // 10)
    initial_offset = float(np.clip(np.median(correlation[-tail_count:]), -0.5, 0.5))
    initial_amplitude = float(
        np.clip(np.max(np.abs(correlation - initial_offset)), 0.05, 2.0)
    )
    lower_bounds = np.asarray(
        (0.0, lower_tau, 0.0, -np.pi, -1.0),
        dtype=np.float64,
    )
    upper_bounds = np.asarray(
        (2.5, upper_tau, nyquist, np.pi, 1.0),
        dtype=np.float64,
    )
    weights = np.sqrt(
        correlation_input.origin_counts.astype(np.float64)
        / float(correlation_input.origin_counts[0])
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        return weights * (_damped_cosine(time, parameters) - correlation)

    omega_starts = np.unique(
        np.clip(
            np.asarray(
                (0.5 * initial_peak, initial_peak, 1.5 * initial_peak),
                dtype=np.float64,
            ),
            np.finfo(np.float64).eps,
            nyquist * (1.0 - 1.0e-8),
        )
    )
    tau_starts = np.clip(
        np.asarray((duration / 10.0, duration / 3.0, duration), dtype=np.float64),
        lower_tau * (1.0 + 1.0e-8),
        upper_tau * (1.0 - 1.0e-8),
    )
    best = None
    for initial_tau in np.unique(tau_starts):
        for initial_omega in omega_starts:
            fit = least_squares(
                residual,
                x0=np.asarray(
                    (
                        initial_amplitude,
                        initial_tau,
                        initial_omega,
                        0.0,
                        initial_offset,
                    ),
                    dtype=np.float64,
                ),
                bounds=(lower_bounds, upper_bounds),
                loss="soft_l1",
                f_scale=0.05,
                x_scale="jac",
                max_nfev=20_000,
            )
            if fit.success and np.all(np.isfinite(fit.x)):
                if best is None or fit.cost < best.cost:
                    best = fit
    if best is None:
        raise ValueError(
            f"Damped-cosine fit failed for "
            f"{correlation_input.discovered.case.case_id}"
        )

    amplitude, fitted_tau, fitted_omega, phase, offset = (
        float(value) for value in best.x
    )
    prediction = _damped_cosine(time, best.x)
    residual_sum = float(np.sum((correlation - prediction) ** 2))
    total_sum = float(np.sum((correlation - np.mean(correlation)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else float("nan")
    at_lower_boundary = fitted_tau <= lower_tau * (1.0 + 1.0e-4)
    at_upper_boundary = fitted_tau >= upper_tau * (1.0 - 1.0e-4)
    if at_lower_boundary or at_upper_boundary:
        boundary = "lower" if at_lower_boundary else "upper"
        warnings.warn(
            f"Fitted tau_r for {correlation_input.discovered.case.case_id} "
            f"is at the {boundary} fit boundary; treat the relaxation time as "
            "unresolved and adjust --tau-max if it is at the upper boundary",
            stacklevel=2,
        )
    return CaseEstimate(
        discovered=correlation_input.discovered,
        coordinate=fitted_tau,
        score=r_squared,
        at_lower_boundary=at_lower_boundary,
        at_upper_boundary=at_upper_boundary,
        fit_omega=fitted_omega,
        fit_amplitude=amplitude,
        fit_phase=phase,
        fit_offset=offset,
    )


def summarize_regular_coordinates(
    results: list[CaseEstimate],
) -> list[RegularCoordinateSummary]:
    grouped: dict[tuple[float, int], list[float]] = {}
    for result in results:
        if result.discovered.mode != "big-lx":
            continue
        circumference = result.discovered.circumference_diameters
        if circumference is None:
            raise RuntimeError("Regular Big-Lx result has no circumference")
        key = (circumference, result.discovered.lx_multiplier)
        grouped.setdefault(key, []).append(result.coordinate)

    summaries: list[RegularCoordinateSummary] = []
    for (circumference, lx_multiplier), values in sorted(grouped.items()):
        coordinates = np.asarray(values, dtype=np.float64)
        summaries.append(
            RegularCoordinateSummary(
                circumference_diameters=circumference,
                lx_multiplier=lx_multiplier,
                mean_coordinate=float(np.mean(coordinates)),
                std_coordinate=(
                    float(np.std(coordinates, ddof=1))
                    if coordinates.size > 1
                    else 0.0
                ),
                replicate_count=int(coordinates.size),
            )
        )
    return summaries


def plot_preferred_coordinate(
    results: list[CaseEstimate],
    output: Path,
    *,
    coordinate_name: str,
    dpi: int,
) -> Path:
    if not results:
        raise ValueError("At least one preferred-coordinate result is required")
    if output.suffix.lower() != ".png":
        raise ValueError("Preferred-coordinate output must use a .png suffix")
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 7.5))
    regular = summarize_regular_coordinates(results)
    circumferences = sorted(
        {
            result.circumference_diameters
            for result in regular
        }
    )
    regular_colors = (
        plt.colormaps["tab10"](
            np.linspace(0.0, 0.3, len(circumferences))
        )
        if circumferences
        else np.empty((0, 4), dtype=np.float64)
    )
    for color, circumference in zip(
        regular_colors,
        circumferences,
        strict=True,
    ):
        family = sorted(
            (
                result
                for result in regular
                if result.circumference_diameters == circumference
            ),
            key=lambda result: result.lx_multiplier,
        )
        axis.errorbar(
            [result.lx_multiplier for result in family],
            [result.mean_coordinate for result in family],
            yerr=[result.std_coordinate for result in family],
            color=color,
            marker="o",
            linewidth=1.7,
            markersize=6,
            elinewidth=1.25,
            capsize=5,
            capthick=1.25,
            label=f"regular cylinder, C = {circumference:g}D",
        )

    confinement = [
        result for result in results if result.discovered.mode == "confinement"
    ]
    confinement_colors = (
        plt.colormaps["Dark2"](
            np.linspace(0.0, 1.0, len(confinement))
        )
        if confinement
        else np.empty((0, 4), dtype=np.float64)
    )
    for color, result in zip(
        confinement_colors,
        sorted(confinement, key=lambda item: item.discovered.case.case_id),
        strict=True,
    ):
        geometry = result.discovered.geometry_kind
        if geometry is None:
            raise RuntimeError("Confinement result has no geometry kind")
        marker = CONFINEMENT_MARKERS[geometry]
        axis.scatter(
            [1],
            [result.coordinate],
            color=[color],
            edgecolors="black",
            linewidths=0.7,
            marker=marker,
            s=95 if marker != "*" else 140,
            zorder=4,
            label=result.discovered.case.label,
        )

    axis.set_xscale("log", base=2)
    axis.set_xlim(0.85, 18.0)
    axis.set_xticks(LX_MULTIPLIERS)
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.set_xlabel(r"axial length multiplier $L_x/L_{x,1}$")
    if coordinate_name == "tau_r":
        axis.set_ylabel(r"fitted relaxation time $\tau_r$")
        axis.set_title(
            r"Damped-cosine relaxation time from "
            r"$C_v(\tau)=A e^{-\tau/\tau_r}"
            r"\cos(\omega_0\tau+\phi)+B$"
        )
    elif coordinate_name == "r":
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_ylabel(r"preferred real coordinate $r_*$")
        axis.set_title(
            r"Preferred real coordinate at $\omega=0$: "
            r"$\arg\max_{r<0}\log_{10}|\widehat C_v(r)|$"
        )
    else:
        axis.set_ylabel(r"preferred positive angular frequency $\omega_*$")
        axis.set_title(
            r"Preferred frequency at $r=0$: "
            r"$\arg\max_{\omega>0}\log_{10}|\widehat C_v(i\omega)|$"
        )
    axis.grid(alpha=0.22, which="both")
    axis.legend(loc="best", fontsize=8.5)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan Big-Lx and confinement analysis safetensors and plot the "
            "preferred coordinate of the velocity-correlation transform "
            "versus Lx: positive omega at r=0 by default, or r at omega=0 "
            "with --preferred-r. Use --fit-tau-r to fit and plot the physical "
            "damped-cosine relaxation time instead. "
            "Repeated Big-Lx cases are combined as seed replicates with "
            "mean and sample-standard-deviation error bars."
        )
    )
    parser.add_argument(
        "--input-dir",
        "--input_dir",
        dest="input_dir",
        action="append",
        type=Path,
        required=True,
        help=(
            "Production root containing safetensors_output/; repeat for "
            "additional roots."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--min-origins", type=int, default=10)
    parser.add_argument("--max-lag", type=int)
    analysis_mode = parser.add_mutually_exclusive_group()
    analysis_mode.add_argument(
        "--preferred-r",
        action="store_true",
        help=(
            "Plot the maximizing strictly negative real coordinate r at "
            "omega=0 instead of the maximizing positive omega at r=0."
        ),
    )
    analysis_mode.add_argument(
        "--fit-tau-r",
        action="store_true",
        help=(
            "Fit a robust damped cosine to C_v(tau) and plot its relaxation "
            "time tau_r."
        ),
    )
    parser.add_argument("--omega-max", type=float)
    parser.add_argument("--omega-points", type=int, default=241)
    parser.add_argument("--r-min", type=float)
    parser.add_argument("--r-max", type=float, default=0.0)
    parser.add_argument("--r-points", type=int, default=241)
    parser.add_argument(
        "--tau-max",
        type=float,
        help=(
            "Upper bound for fitted tau_r; defaults to ten times each case's "
            "fitted lag duration."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.dpi < 1:
        raise SystemExit("--dpi must be positive")
    if args.min_origins < 2:
        raise SystemExit("--min-origins must be at least two")
    if args.max_lag is not None and args.max_lag < 1:
        raise SystemExit("--max-lag must be positive")
    if args.omega_points < 5:
        raise SystemExit("--omega-points must be at least five")
    if args.r_points < 2:
        raise SystemExit("--r-points must be at least two")
    if args.tau_max is not None and args.tau_max <= 0.0:
        raise SystemExit("--tau-max must be positive")

    discovered = scan_input_directories(args.input_dir)
    correlation_inputs = [
        _correlation_input(
            item,
            _axial_center_series(item.case, item.mode),
            min_origins=args.min_origins,
            max_lag=args.max_lag,
        )
        for item in discovered
    ]
    if args.fit_tau_r:
        coordinate_name = "tau_r"
        omega_grid = _shared_positive_omega(
            correlation_inputs,
            omega_max=args.omega_max,
            omega_points=args.omega_points,
        )
        results = [
            fit_tau_r(item, omega_grid, tau_max=args.tau_max)
            for item in correlation_inputs
        ]
        coordinate = np.asarray(
            [result.coordinate for result in results],
            dtype=np.float64,
        )
        default_filename = "fitted_tau_r.png"
    elif args.preferred_r:
        coordinate_name = "r"
        coordinate = _shared_r(
            correlation_inputs,
            r_min=args.r_min,
            r_max=args.r_max,
            r_points=args.r_points,
        )
        results = [preferred_r(item, coordinate) for item in correlation_inputs]
        default_filename = "preferred_r.png"
    else:
        coordinate_name = "omega"
        coordinate = _shared_positive_omega(
            correlation_inputs,
            omega_max=args.omega_max,
            omega_points=args.omega_points,
        )
        results = [preferred_omega(item, coordinate) for item in correlation_inputs]
        default_filename = "preferred_omega.png"
    output = args.output or args.input_dir[0] / "plots" / default_filename
    result_path = plot_preferred_coordinate(
        results,
        output,
        coordinate_name=coordinate_name,
        dpi=args.dpi,
    )

    for result in sorted(
        results,
        key=lambda item: (
            item.discovered.lx_multiplier,
            item.discovered.case.case_id,
        ),
    ):
        print(
            f"[preferred_{coordinate_name}.case] mode={result.discovered.mode} "
            f"case={result.discovered.case.case_id} "
            f"lx_multiplier={result.discovered.lx_multiplier} "
            f"{coordinate_name}={result.coordinate:.8g} "
            f"{'r_squared' if coordinate_name == 'tau_r' else 'log10_magnitude'}="
            f"{result.score:.8g} "
            f"lower_boundary={result.at_lower_boundary} "
            f"upper_boundary={result.at_upper_boundary} "
            f"fit_omega={result.fit_omega} "
            f"fit_amplitude={result.fit_amplitude} "
            f"fit_phase={result.fit_phase} "
            f"fit_offset={result.fit_offset} "
            f"manifest={result.discovered.manifest_path}",
            flush=True,
        )
    for summary in summarize_regular_coordinates(results):
        print(
            f"[preferred_{coordinate_name}.regular_summary] "
            f"circumference_diameters={summary.circumference_diameters:.8g} "
            f"lx_multiplier={summary.lx_multiplier} "
            f"replicates={summary.replicate_count} "
            f"mean_{coordinate_name}={summary.mean_coordinate:.8g} "
            f"std_{coordinate_name}={summary.std_coordinate:.8g}",
            flush=True,
        )
    print(
        f"[preferred_{coordinate_name}] cases={len(results)} "
        f"{coordinate_name}=[{np.min(coordinate):.8g}, "
        f"{np.max(coordinate):.8g}] "
        f"output={result_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
