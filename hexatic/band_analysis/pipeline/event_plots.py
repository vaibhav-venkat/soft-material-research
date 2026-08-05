"""Diagnostics specific to masked event-chain inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from ..detection.chains import EventChain
from .workflow import LagOutcome


def _save(path: Path, figure: plt.Figure) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _placeholder(path: Path, title: str, message: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 3.5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    _save(path, figure)


def _arrays(outcome: LagOutcome) -> dict[str, np.ndarray]:
    prefix = "training_"
    return {
        key[len(prefix) :]: value
        for key, value in outcome.arrays.items()
        if key.startswith(prefix)
    }


def _one_step(outcome: LagOutcome, arrays: dict[str, np.ndarray], path: Path) -> None:
    observed = arrays["one_step_observed"]
    mean = arrays["one_step_mean"]
    active = arrays["one_step_mask"].astype(bool)
    event = arrays["one_step_event"].astype(bool)
    low = arrays["one_step_low"]
    high = arrays["one_step_high"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for selected, label, color in (
        (active & ~event, "continuous", "tab:blue"),
        (active & event, "event step", "tab:orange"),
    ):
        axes[0].scatter(observed[selected], mean[selected], s=5, alpha=0.25, label=label)
    extent = max(
        np.max(np.abs(observed[active]), initial=0.0),
        np.max(np.abs(mean[active]), initial=0.0),
        1e-12,
    )
    axes[0].plot((-extent, extent), (-extent, extent), color="black", linewidth=0.8)
    axes[0].set(xlabel="observed area", ylabel="posterior predictive mean")
    axes[0].legend()
    residual = observed[active] - mean[active]
    interval = high[active] - low[active]
    axes[1].scatter(interval, residual, s=5, alpha=0.25)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(xlabel="95% predictive interval width", ylabel="residual")
    figure.suptitle(f"Event fit lag {outcome.lag}: masked one-step prediction")
    _save(path, figure)


def _trajectories(
    outcome: LagOutcome, arrays: dict[str, np.ndarray], path: Path
) -> None:
    areas = arrays["path_area"]
    masks = arrays["path_mask"].astype(bool)
    tau = arrays["path_tau"]
    offsets = arrays["path_offset"]
    n_max = int(outcome.metadata["data"]["n_max"])
    figure, axis = plt.subplots(figsize=(9, 4))
    for start, stop in zip(offsets[:-1][:20], offsets[1:][:20], strict=True):
        start, stop = int(start), int(stop)
        path_area = areas[start:stop].reshape(-1, n_max)
        path_mask = masks[start:stop].reshape(-1, n_max)
        time_start, time_stop = start // n_max, stop // n_max
        path_tau = tau[time_start:time_stop]
        total = np.sum(path_area * path_mask, axis=1)
        axis.plot(path_tau - path_tau[0], total, alpha=0.25)
    axis.set(xlabel="physical time from chain start", ylabel=r"$A_T$")
    axis.set_title(f"Event fit lag {outcome.lag}: posterior scheduled paths")
    _save(path, figure)


def _event_rows(outcome: LagOutcome, arrays: dict[str, np.ndarray], path: Path) -> None:
    event = arrays["one_step_event"].astype(bool)
    informative = arrays["one_step_observation_mask"].astype(bool)
    active = arrays["one_step_mask"].astype(bool)
    counts = (
        np.count_nonzero(active & ~event),
        np.count_nonzero(informative & event),
        np.count_nonzero(active & event & ~informative),
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(("continuous active", "event informative", "event marginalized"), counts)
    axis.set_ylabel("component rows")
    axis.tick_params(axis="x", rotation=15)
    axis.set_title(f"Event fit lag {outcome.lag}: observation routing")
    _save(path, figure)


def _negative(outcome: LagOutcome, metrics: dict[str, Any], path: Path) -> None:
    negative = float(metrics.get("negative_area_count", 0))
    generated = float(metrics.get("generated_area_count", 0))
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(("negative", "nonnegative"), (negative, max(generated - negative, 0)))
    axis.set_yscale("symlog", linthresh=1.0)
    axis.set_title(
        f"Event fit lag {outcome.lag}: negative-area fraction "
        f"{metrics.get('negative_area_fraction', float('nan')):.3g}"
    )
    _save(path, figure)


def _whitened(outcome: LagOutcome, arrays: dict[str, np.ndarray], path: Path) -> None:
    active = arrays["one_step_mask"].astype(bool)
    observed = arrays["one_step_observed"][active]
    mean = arrays["one_step_mean"][active]
    width = arrays["one_step_high"][active] - arrays["one_step_low"][active]
    residual = (observed - mean) / np.maximum(width / (2.0 * 1.96), 1e-12)
    event = arrays["one_step_event"].astype(bool)[active]
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes[0, 0].hist(residual, bins=40, density=True, alpha=0.7)
    x = np.linspace(-4.0, 4.0, 200)
    axes[0, 0].plot(x, stats.norm.pdf(x), color="black")
    axes[0, 0].set_title("standardized predictive residual")
    if residual.size:
        stats.probplot(residual, dist="norm", plot=axes[0, 1])
    for selected, label, color in (
        (~event, "continuous", "tab:blue"),
        (event, "event", "tab:orange"),
    ):
        axes[0, 2].hist(
            residual[selected], bins=35, density=True, alpha=0.5,
            label=label, color=color,
        )
    axes[0, 2].legend()
    axes[0, 2].set_title("residual by transition type")
    centered = residual - residual.mean() if residual.size else residual
    denominator = float(np.dot(centered, centered))
    acf = np.asarray(
        [
            float(np.dot(centered[:-lag], centered[lag:])) / denominator
            if lag and denominator and len(centered) > lag
            else 1.0 if lag == 0 else np.nan
            for lag in range(min(50, len(centered)))
        ]
    )
    axes[1, 0].stem(np.arange(len(acf)), acf)
    axes[1, 0].set(xlabel="transition separation", ylabel="residual ACF")
    axes[1, 1].scatter(observed, residual, s=4, alpha=0.2)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set(xlabel="observed area", ylabel="standardized residual")
    axes[1, 2].scatter(width, residual, s=4, alpha=0.2)
    axes[1, 2].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 2].set(xlabel="95% interval width", ylabel="standardized residual")
    figure.suptitle(f"Event fit lag {outcome.lag}: residual diagnostics")
    _save(path, figure)


def _predictive(outcome: LagOutcome, arrays: dict[str, np.ndarray], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    comparisons = (
        ("delta_total", r"topology-relative $\Delta A_T$"),
        ("delta_area", r"topology-relative $\Delta A_i$"),
        ("area_deviation", r"$A_i-\bar A$"),
        ("conservative_increment", "conservative increment"),
    )
    for axis, (suffix, label) in zip(axes.ravel(), comparisons, strict=True):
        axis.hist(
            arrays[f"observed_{suffix}"], bins=40, density=True,
            alpha=0.55, label="observed",
        )
        axis.hist(
            arrays[f"predicted_{suffix}"], bins=40, density=True,
            alpha=0.55, label="posterior predictive",
        )
        axis.set_xlabel(label)
        axis.legend()
    figure.suptitle(f"Event fit lag {outcome.lag}: one-step predictive check")
    _save(path, figure)


def _path_records(
    arrays: dict[str, np.ndarray], prefix: str, n_max: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    areas = arrays[f"{prefix}path_area"]
    masks = arrays[f"{prefix}path_mask"].astype(bool)
    slots = arrays[f"{prefix}path_slot_id"].astype(np.int64)
    tau = arrays[f"{prefix}path_tau"]
    segment_break = arrays[f"{prefix}path_segment_break"].astype(bool)
    offsets = arrays[f"{prefix}path_offset"]
    records = []
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        start, stop = int(start), int(stop)
        time_start, time_stop = start // n_max, stop // n_max
        records.append(
            (
                areas[start:stop].reshape(-1, n_max),
                masks[start:stop].reshape(-1, n_max),
                slots[start:stop].reshape(-1, n_max),
                tau[time_start:time_stop],
                segment_break[time_start:time_stop],
            )
        )
    return records


def _pooled_acf(series: list[np.ndarray]) -> np.ndarray:
    usable = [values for values in series if len(values)]
    if not usable:
        return np.empty(0)
    mean = float(np.mean(np.concatenate(usable)))
    centered = [values - mean for values in usable]
    denominator = sum(float(np.dot(values, values)) for values in centered)
    maximum = min(100, max(len(values) for values in centered))
    return np.asarray(
        [
            sum(
                float(np.dot(values[:-lag], values[lag:]))
                for values in centered if lag and len(values) > lag
            ) / denominator
            if lag and denominator else 1.0 if lag == 0 else np.nan
            for lag in range(maximum)
        ]
    )


def _track_series(
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *, conservative: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    values = []
    times = []
    for area, mask, slots, tau, segment_break in records:
        row_mean = np.divide(
            (area * mask).sum(axis=1), mask.sum(axis=1),
            out=np.zeros(len(area)), where=mask.sum(axis=1) > 0,
        )
        source = area - row_mean[:, None] if conservative else area
        for slot in range(area.shape[1]):
            valid = mask[:, slot] & (slots[:, slot] >= 0)
            boundary = segment_break[1:] | (slots[1:, slot] != slots[:-1, slot])
            starts = np.flatnonzero(valid & np.r_[True, ~valid[:-1] | boundary])
            stops = (
                np.flatnonzero(valid & np.r_[~valid[1:] | boundary, True]) + 1
            )
            for start, stop in zip(starts, stops, strict=True):
                values.append(source[start:stop, slot])
                times.append(tau[start:stop])
    return values, times


def _pooled_msd(series: list[np.ndarray], maximum: int = 100) -> np.ndarray:
    if not series:
        return np.empty(0)
    stop = min(maximum, max(len(values) for values in series))
    return np.asarray(
        [
            0.0 if lag == 0 else np.mean(
                np.concatenate(
                    [values[lag:] - values[:-lag] for values in series if len(values) > lag]
                ) ** 2
            )
            for lag in range(stop)
        ]
    )


def _shared_msd_limit(
    observed: list[np.ndarray], simulated: list[np.ndarray], maximum: int = 100
) -> int:
    """Use one segment-length censor for both observed and simulated MSDs."""
    if not observed or not simulated:
        return 0
    return min(maximum, max(map(len, observed)), max(map(len, simulated)))


def _lag_times(times: list[np.ndarray], maximum: int) -> np.ndarray:
    return np.asarray(
        [
            0.0 if lag == 0 else np.median(
                np.concatenate(
                    [values[lag:] - values[:-lag] for values in times if len(values) > lag]
                )
            )
            for lag in range(maximum)
        ]
    )


def _long_time(outcome: LagOutcome, arrays: dict[str, np.ndarray], path: Path) -> None:
    n_max = int(outcome.metadata["data"]["n_max"])
    observed = _path_records(arrays, "observed_", n_max)
    simulated = _path_records(arrays, "", n_max)
    observed_area_series, observed_area_times = _track_series(
        observed, conservative=False
    )
    simulated_area_series, simulated_area_times = _track_series(
        simulated, conservative=False
    )
    msd_limit = _shared_msd_limit(observed_area_series, simulated_area_series)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for records, area_series, area_times, label, color in (
        (observed, observed_area_series, observed_area_times, "observed", "black"),
        (
            simulated,
            simulated_area_series,
            simulated_area_times,
            "simulated",
            "tab:blue",
        ),
    ):
        totals = [np.sum(area * mask, axis=1) for area, mask, _, _, _ in records]
        total_values = np.concatenate(totals)
        axes[0, 0].hist(total_values, bins=40, density=True, alpha=0.5, label=label)
        total_acf = _pooled_acf(totals)
        total_times = [tau for _, _, _, tau, _ in records]
        lag_time = _lag_times(total_times, len(total_acf))
        axes[0, 1].plot(lag_time, total_acf, color=color, label=label)
        conservative, _ = _track_series(records, conservative=True)
        conservative_acf = _pooled_acf(conservative)
        conservative_time = _lag_times(area_times, len(conservative_acf))
        axes[0, 2].plot(
            conservative_time, conservative_acf, color=color, label=label
        )
        active_area = np.concatenate(
            [area[mask] for area, mask, _, _, _ in records]
        )
        active_deviation = (
            np.concatenate(conservative) if conservative else np.empty(0)
        )
        axes[1, 0].hist(active_area, bins=40, density=True, alpha=0.5, label=label)
        axes[1, 1].hist(
            active_deviation, bins=40, density=True, alpha=0.5, label=label
        )
        msd = _pooled_msd(area_series, maximum=msd_limit)
        axes[1, 2].plot(
            _lag_times(area_times, len(msd)), msd, color=color, label=label
        )
    axes[0, 0].set_xlabel(r"$A_T$")
    axes[0, 1].set(xlabel="physical lag time", ylabel=r"$A_T$ ACF")
    axes[0, 2].set(xlabel="physical lag time", ylabel="track conservative ACF")
    axes[1, 0].set_xlabel(r"active $A_i$")
    axes[1, 1].set_xlabel(r"$A_i-\bar A$")
    axes[1, 2].set(xlabel="physical lag time", ylabel="track area MSD")
    for axis in axes.ravel():
        axis.legend()
    figure.suptitle(f"Event fit lag {outcome.lag}: long-time behavior")
    _save(path, figure)


def _increment_statistics(
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    increments = []
    conservative = []
    totals = []
    for area, mask, slots, _, _ in records:
        if len(area) < 2:
            continue
        valid = mask[1:] & mask[:-1] & (slots[1:] == slots[:-1])
        increment = np.where(valid, np.diff(area, axis=0), 0.0)
        counts = valid.sum(axis=1)
        projected = increment - np.divide(
            increment.sum(axis=1), counts,
            out=np.zeros(len(increment)), where=counts > 0,
        )[:, None] * valid
        increments.append(increment)
        conservative.append(projected)
        totals.append(np.diff(np.sum(area * mask, axis=1)))
    values = np.concatenate(increments)
    projected = np.concatenate(conservative)
    total = np.concatenate(totals)
    band_covariance = np.atleast_2d(np.cov(values, rowvar=False))
    conservative_covariance = np.atleast_2d(np.cov(projected, rowvar=False))
    total_covariance = np.asarray(
        [np.cov(values[:, index], total)[0, 1] for index in range(values.shape[1])]
    )
    return band_covariance.ravel(), conservative_covariance.ravel(), total_covariance


def _covariance(outcome: LagOutcome, arrays: dict[str, np.ndarray], path: Path) -> None:
    n_max = int(outcome.metadata["data"]["n_max"])
    observed = _increment_statistics(_path_records(arrays, "observed_", n_max))
    simulated = _increment_statistics(_path_records(arrays, "", n_max))
    titles = ("surviving-track increments", "conservative increments", r"increment vs. $\Delta A_T$")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, first, second, title in zip(axes, observed, simulated, titles, strict=True):
        axis.scatter(first, second, s=10, alpha=0.5)
        limit = max(np.max(np.abs(first), initial=0.0), np.max(np.abs(second), initial=0.0), 1e-12)
        axis.plot((-limit, limit), (-limit, limit), color="black", linewidth=0.8)
        axis.set(xlabel="observed covariance", ylabel="simulated covariance", title=title)
    figure.suptitle(f"Event fit lag {outcome.lag}: covariance check")
    _save(path, figure)


def plot_event_validation(outcome: LagOutcome, directory: Path) -> list[str]:
    paths = (
        "one_step.png",
        "whitened_residual.png",
        "predictive.png",
        "trajectory.png",
        "long_time.png",
        "covariance.png",
        "observation_rows.png",
        "negative_area.png",
    )
    arrays = _arrays(outcome)
    if not outcome.accepted or not arrays:
        message = "Lag rejected by MCMC criteria; predictive conclusions excluded."
        for filename in paths:
            _placeholder(directory / filename, f"Event fit lag {outcome.lag}", message)
    else:
        _one_step(outcome, arrays, directory / paths[0])
        _whitened(outcome, arrays, directory / paths[1])
        _predictive(outcome, arrays, directory / paths[2])
        _trajectories(outcome, arrays, directory / paths[3])
        _long_time(outcome, arrays, directory / paths[4])
        _covariance(outcome, arrays, directory / paths[5])
        _event_rows(outcome, arrays, directory / paths[6])
        _negative(
            outcome,
            outcome.metadata["validation"]["training"],
            directory / paths[7],
        )
    return list(paths)


def _diagnostic_values(chains: list[EventChain]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.concatenate([chain.n_active for chain in chains])
    totals = np.concatenate(
        [np.sum(chain.areas * chain.mask, axis=1) for chain in chains]
    )
    return counts, totals


def plot_total_area_by_band_count(
    training: list[EventChain],
    holdouts: dict[str, list[EventChain]],
    output_dir: Path,
) -> Path:
    """Plot the post-hoc diagnostic for a band-count-independent ``A_T_star``."""
    path = output_dir / "event" / "plots" / "total_area_by_band_count.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    groups = [("training", training, "tab:blue")]
    held_out = [chain for chains in holdouts.values() for chain in chains]
    if held_out:
        groups.append(("holdout", held_out, "tab:orange"))
    for label, chains, color in groups:
        counts, totals = _diagnostic_values(chains)
        selected = np.linspace(0, len(counts) - 1, min(len(counts), 50_000)).astype(int)
        axis.scatter(
            counts[selected], totals[selected], s=4, alpha=0.08, color=color, label=label
        )
        unique = np.unique(counts)
        medians = np.asarray([np.median(totals[counts == count]) for count in unique])
        quartiles = np.asarray(
            [np.quantile(totals[counts == count], (0.25, 0.75)) for count in unique]
        )
        axis.errorbar(
            unique,
            medians,
            yerr=(medians - quartiles[:, 0], quartiles[:, 1] - medians),
            fmt="o-",
            color=color,
            capsize=3,
        )
    axis.set(xlabel=r"active band count $n_k$", ylabel=r"total dilute area $A_T$")
    axis.set_xticks(np.unique(_diagnostic_values(training)[0]))
    axis.legend()
    axis.set_title(r"Post-hoc check of band-count-independent $A_T^*$")
    _save(path, figure)
    return path
