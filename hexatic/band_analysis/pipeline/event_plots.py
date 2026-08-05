"""Diagnostics specific to masked event-chain inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    for start, stop in zip(offsets[:20], offsets[1:21], strict=True):
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


def plot_event_validation(outcome: LagOutcome, directory: Path) -> list[str]:
    paths = ("one_step.png", "trajectory.png", "observation_rows.png", "negative_area.png")
    arrays = _arrays(outcome)
    if not outcome.accepted or not arrays:
        message = "Lag rejected by MCMC criteria; predictive conclusions excluded."
        for filename in paths:
            _placeholder(directory / filename, f"Event fit lag {outcome.lag}", message)
    else:
        _one_step(outcome, arrays, directory / paths[0])
        _trajectories(outcome, arrays, directory / paths[1])
        _event_rows(outcome, arrays, directory / paths[2])
        _negative(
            outcome,
            outcome.metadata["validation"]["training"],
            directory / paths[3],
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
