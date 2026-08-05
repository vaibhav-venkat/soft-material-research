"""Render posterior, predictive, residual, and holdout diagnostics."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .workflow import LagOutcome


PARAMETER_LABELS = {
    "gamma": r"$\gamma$",
    "gamma_o": r"$\gamma_o$",
    "delta": r"$\delta$",
    "omega": r"$\omega$",
    "amplitude_f": r"$\sigma_f^2$",
    "amplitude_r": r"$\sigma_o^2/\sigma_f^2$",
    "tau_p": r"$\tau_p$",
    "kappa_T": r"$\kappa_T$",
    "D_u": r"$D_u$",
    "D_T": r"$D_T$",
    "A_T_star": r"$A_T^*$",
    "sigma_b": r"$\sigma_b$",
    "sigma_E": r"$\sigma_E$",
}


def _save(path: Path, figure: plt.Figure | None = None) -> None:
    figure = figure or plt.gcf()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _placeholder(path: Path, title: str, message: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 3.5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    _save(path, figure)


def _arviz_plot(
    path: Path,
    title: str,
    plotter: Callable[[], Any],
) -> None:
    try:
        axes = np.asarray(plotter(), dtype=object).ravel()
        figure = next(
            (axis.figure for axis in axes if hasattr(axis, "figure")), plt.gcf()
        )
        figure.suptitle(title)
        _save(path, figure)
    except (ValueError, KeyError, TypeError, RuntimeError) as error:
        plt.close("all")
        _placeholder(path, title, f"Diagnostic unavailable: {error}")


def _posterior_plots(outcome: LagOutcome, directory: Path) -> None:
    names = list(outcome.parameter_names)
    _arviz_plot(
        directory / "posterior_trace.png",
        f"Lag {outcome.lag}: posterior and trace",
        lambda: az.plot_trace(outcome.idata, var_names=names, compact=False),
    )
    _arviz_plot(
        directory / "rank.png",
        f"Lag {outcome.lag}: chain ranks",
        lambda: az.plot_rank(outcome.idata, var_names=names),
    )
    _arviz_plot(
        directory / "pair.png",
        f"Lag {outcome.lag}: posterior pairs",
        lambda: az.plot_pair(
            outcome.idata,
            var_names=names,
            kind="scatter",
            marginals=True,
            divergences=True,
        ),
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    energy = np.asarray(
        outcome.idata.sample_stats.get("energy", np.asarray([]))
    ).ravel()
    if energy.size:
        axes[0].hist(energy, bins=40, density=True, alpha=0.8)
        axes[0].set_xlabel("energy")
    else:
        axes[0].text(0.5, 0.5, "energy unavailable", ha="center")
    try:
        bfmi = np.asarray(az.bfmi(outcome.idata)).ravel()
    except (ValueError, KeyError, TypeError):
        bfmi = np.empty(0)
    if bfmi.size:
        axes[1].bar(np.arange(1, len(bfmi) + 1), bfmi)
        axes[1].axhline(0.3, color="tab:red", linestyle="--", linewidth=1)
        axes[1].set_xlabel("chain")
        axes[1].set_ylabel("BFMI")
    else:
        axes[1].text(0.5, 0.5, "BFMI unavailable", ha="center")
    figure.suptitle(f"Lag {outcome.lag}: energy diagnostics")
    _save(directory / "energy_bfmi.png", figure)


def _hessian_plot(outcome: LagOutcome, path: Path) -> None:
    matrix = outcome.arrays.get("hessian")
    eigenvalues = outcome.arrays.get("hessian_eigenvalues")
    if matrix is None or eigenvalues is None:
        _placeholder(path, f"Lag {outcome.lag}: Hessian", "Hessian unavailable")
        return
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = outcome.parameter_names
    image = axes[0].imshow(matrix, cmap="coolwarm", aspect="auto")
    axes[0].set_xticks(
        range(len(names)),
        [PARAMETER_LABELS[name] for name in names],
    )
    axes[0].set_yticks(
        range(len(names)),
        [PARAMETER_LABELS[name] for name in names],
    )
    figure.colorbar(image, ax=axes[0], shrink=0.8)
    axes[1].plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, "o-")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("ordered mode")
    axes[1].set_ylabel("raw-space eigenvalue")
    figure.suptitle(f"Lag {outcome.lag}: local identifiability")
    _save(path, figure)


def _validation_arrays(outcome: LagOutcome, prefix: str) -> dict[str, np.ndarray]:
    return {
        key[len(prefix) :]: value
        for key, value in outcome.arrays.items()
        if key.startswith(prefix)
    }


def _whitened_plot(outcome: LagOutcome, arrays: dict[str, np.ndarray], path: Path) -> None:
    values = arrays["whitened_residual"]
    acf = arrays["whitened_residual_acf"]
    covariance = arrays["whitened_residual_covariance"]
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].hist(values, bins=40, density=True, alpha=0.7)
    x = np.linspace(-4.0, 4.0, 200)
    axes[0, 0].plot(x, stats.norm.pdf(x), color="black")
    axes[0, 0].set_title("whitened residual distribution")
    if values.size:
        stats.probplot(values, dist="norm", plot=axes[0, 1])
    dependencies = (
        ("residual_total_area", r"$A_T$"),
        ("residual_area_deviation", r"$A_i-\bar A$"),
        ("residual_band_count", "band count"),
        ("residual_dt", r"$\Delta\tau$"),
    )
    for axis, (key, label) in zip(axes[:, 2:].ravel(), dependencies, strict=True):
        predictor = arrays.get(key, np.empty(0))
        if len(predictor) == len(values):
            axis.scatter(predictor, values, s=3, alpha=0.2)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(xlabel=label, ylabel="whitened residual")
    axes[1, 0].stem(np.arange(min(50, len(acf))), acf[:50])
    axes[1, 0].set(
        title="non-overlapping residual ACF",
        xlabel="separation in fitted transitions",
    )
    offsets = arrays["residual_covariance_offset"]
    band_counts = arrays["residual_covariance_band_count"]
    diagonal: list[np.ndarray] = []
    off_diagonal: list[np.ndarray] = []
    for index, n_bands in enumerate(band_counts):
        matrix = covariance[offsets[index] : offsets[index + 1]].reshape(
            int(n_bands), int(n_bands)
        )
        diagonal.append(np.diag(matrix))
        off_diagonal.append(matrix[~np.eye(int(n_bands), dtype=bool)])
    if diagonal:
        axes[1, 1].hist(np.concatenate(diagonal), bins=25, alpha=0.6, label="diagonal")
        nonempty = [values for values in off_diagonal if len(values)]
        if nonempty:
            axes[1, 1].hist(
                np.concatenate(nonempty), bins=25, alpha=0.6, label="off-diagonal"
            )
        axes[1, 1].legend()
    axes[1, 1].set_title("component covariance entries")
    figure.suptitle(f"Lag {outcome.lag}: whitened residuals")
    _save(path, figure)


def _predictive_plot(outcome: LagOutcome, arrays: dict[str, np.ndarray], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    comparisons = (
        ("delta_total", r"$\Delta A_T$"),
        ("delta_area", r"$\Delta A_i$"),
        ("area_deviation", r"$A_i-\bar A$"),
        ("conservative_increment", "conservative increment"),
    )
    for axis, (suffix, label) in zip(axes.ravel(), comparisons, strict=True):
        observed_values = arrays[f"observed_{suffix}"]
        predicted_values = arrays[f"predicted_{suffix}"]
        axis.hist(observed_values, bins=40, density=True, alpha=0.55, label="observed")
        axis.hist(
            predicted_values,
            bins=40,
            density=True,
            alpha=0.55,
            label="posterior predictive",
        )
        axis.set_xlabel(label)
        axis.legend()
    figure.suptitle(f"Lag {outcome.lag}: one-step predictive check")
    _save(path, figure)


def _path_totals(
    arrays: dict[str, np.ndarray], maximum: int = 30
) -> list[tuple[np.ndarray, np.ndarray]]:
    values = arrays["path_area"]
    offsets = arrays["path_offset"]
    times = arrays["path_tau"]
    time_offsets = arrays["path_time_offset"]
    counts = arrays["path_band_count"]
    totals = []
    for index, n_bands in enumerate(counts[:maximum]):
        path = values[offsets[index] : offsets[index + 1]].reshape(-1, int(n_bands))
        tau = times[time_offsets[index] : time_offsets[index + 1]]
        totals.append((tau - tau[0], path.sum(axis=1)))
    return totals


def _trajectory_plot(outcome: LagOutcome, arrays: dict[str, np.ndarray], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 4))
    for tau, total in _path_totals(arrays):
        axis.plot(tau, total, color="tab:blue", alpha=0.12)
    observed_counts = arrays["observed_path_band_count"]
    stop = int(arrays["observed_path_offset"][1])
    observed = (
        arrays["observed_path_area"][:stop]
        .reshape(-1, int(observed_counts[0]))
        .sum(axis=1)
    )
    observed_time_stop = int(arrays["observed_path_time_offset"][1])
    observed_tau = arrays["observed_path_tau"][:observed_time_stop]
    observed_tau = observed_tau - observed_tau[0]
    axis.plot(
        observed_tau,
        observed,
        color="black",
        linewidth=2,
        label="observed segment 0",
    )
    axis.set(xlabel=r"physical time from segment start", ylabel=r"$A_T$")
    axis.legend()
    axis.set_title(f"Lag {outcome.lag}: posterior trajectories")
    _save(path, figure)


def _long_time_plot(outcome: LagOutcome, arrays: dict[str, np.ndarray], path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].hist(arrays["observed_total"], bins=40, density=True, alpha=0.6, label="observed")
    axes[0, 0].hist(arrays["simulated_total"], bins=40, density=True, alpha=0.5, label="simulated")
    axes[0, 0].set_xlabel(r"$A_T$")
    axes[0, 0].legend()
    for prefix, color in (("observed", "black"), ("simulated", "tab:blue")):
        lag_time = arrays[f"{prefix}_lag_time"]
        for target, key in (
            (axes[0, 1], f"{prefix}_total_acf"),
            (axes[0, 2], f"{prefix}_conservative_acf"),
        ):
            values = arrays[key]
            stop = min(800, len(values), len(lag_time))
            target.plot(lag_time[:stop], values[:stop], color=color, label=prefix)
    observed_lag_time = arrays["observed_lag_time"]
    simulated_lag_time = arrays["simulated_lag_time"]
    axis = axes[1, 2]
    observed_msd = arrays["observed_area_msd"]
    simulated_msd = arrays["simulated_area_msd"]
    msd_stop = min(
        800,
        len(observed_msd),
        len(simulated_msd),
        len(observed_lag_time),
        len(simulated_lag_time),
    )
    axis.plot(
        observed_lag_time[:msd_stop],
        observed_msd[:msd_stop],
        color="black",
        label="observed",
    )
    axis.plot(
        simulated_lag_time[:msd_stop],
        simulated_msd[:msd_stop],
        color="tab:blue",
        label="simulated",
    )
    envelope = arrays["simulated_area_msd_envelope"]
    if envelope.size:
        stop = min(msd_stop, envelope.shape[1])
        axis.fill_between(
            simulated_lag_time[:stop],
            envelope[0, :stop],
            envelope[1, :stop],
            color="tab:blue",
            alpha=0.2,
            linewidth=0,
            label="simulated 95%",
        )
    axis.set(
        xlabel="physical lag time",
        ylabel="band area MSD",
        xscale="log",
        yscale="log",
    )
    axis.legend()
    axes[0, 1].set(xlabel="physical lag time", ylabel=r"$A_T$ ACF")
    axes[0, 1].legend()
    for axis, suffix, label in (
        (axes[1, 0], "area", r"$A_i$"),
        (axes[1, 1], "conservative", r"$A_i-\bar A$"),
    ):
        observed_values = arrays[f"observed_{suffix}"]
        simulated_values = arrays[f"simulated_{suffix}"]
        if observed_values.size:
            axis.hist(observed_values, bins=40, density=True, alpha=0.6, label="observed")
            axis.hist(simulated_values, bins=40, density=True, alpha=0.5, label="simulated")
            axis.legend()
        axis.set_xlabel(label)
    axes[0, 2].set(xlabel="physical lag time", ylabel="conservative-mode ACF")
    axes[0, 2].legend()
    figure.suptitle(f"Lag {outcome.lag}: long-time behavior")
    _save(path, figure)


def _covariance_plot(outcome: LagOutcome, arrays: dict[str, np.ndarray], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    comparisons = (
        ("increment_covariance", "band increments"),
        ("conservative_increment_covariance", "conservative modes"),
        ("increment_total_covariance", r"band increment vs. $\Delta A_T$"),
    )
    for axis, (suffix, title) in zip(axes, comparisons, strict=True):
        observed = arrays[f"observed_{suffix}"]
        simulated = arrays[f"simulated_{suffix}"]
        axis.scatter(observed, simulated, s=10, alpha=0.5)
        limit = max(
            np.max(np.abs(observed), initial=0.0),
            np.max(np.abs(simulated), initial=0.0),
            1e-12,
        )
        axis.plot((-limit, limit), (-limit, limit), color="black", linewidth=0.8)
        axis.set(
            xlabel="observed covariance",
            ylabel="simulated covariance",
            title=title,
        )
    figure.suptitle(f"Lag {outcome.lag}: covariance check")
    _save(path, figure)


def _transfer_rate_metrics(outcome: LagOutcome) -> dict[str, Any]:
    validation = outcome.metadata.get("validation", {})
    return validation.get("training") or {}


def _transfer_rate_acf_plot(
    outcome: LagOutcome, arrays: dict[str, np.ndarray], path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    observed = arrays["observed_transfer_rate_acf"]
    simulated = arrays["simulated_transfer_rate_acf"]
    observed_time = arrays["observed_transfer_rate_lag_time"]
    simulated_time = arrays["simulated_transfer_rate_lag_time"]
    stop = min(
        800,
        len(observed),
        len(simulated),
        len(observed_time),
        len(simulated_time),
    )
    envelope = arrays["simulated_transfer_rate_acf_envelope"]
    if envelope.size:
        band_stop = min(stop, envelope.shape[1])
        axis.fill_between(
            simulated_time[:band_stop],
            envelope[0, :band_stop],
            envelope[1, :band_stop],
            color="tab:blue",
            alpha=0.2,
            linewidth=0,
            label="same-geometry 95%",
        )
    axis.plot(observed_time[:stop], observed[:stop], color="black", label="observed")
    axis.plot(
        simulated_time[:stop], simulated[:stop], color="tab:blue", label="simulated"
    )
    axis.axhline(0.0, color="0.5", linewidth=0.8)
    metrics = _transfer_rate_metrics(outcome)
    negative = metrics.get("transfer_rate_negative_area", float("nan"))
    negative_p = metrics.get("transfer_rate_negative_area_p", float("nan"))
    shape = metrics.get("transfer_rate_shape_chi2", float("nan"))
    shape_p = metrics.get("transfer_rate_shape_chi2_p", float("nan"))
    start = metrics.get("transfer_rate_window_start", float("nan"))
    negative_stop = metrics.get("transfer_rate_negative_window_stop", float("nan"))
    shape_stop = metrics.get("transfer_rate_shape_window_stop", float("nan"))
    title = f"Lag {outcome.lag}: transfer-rate ACF"
    if np.isfinite(shape_p):
        axis.axvspan(start, shape_stop, color="0.88", alpha=0.5, zorder=0)
        axis.axvspan(start, negative_stop, color="0.78", alpha=0.5, zorder=0)
        title += (
            f"\nnegative area [{start:.0f}, {negative_stop:.0f}]: "
            f"{negative:.3g}, p={negative_p:.3f}"
            f"   |   shape $\\chi^2$ [{start:.0f}, {shape_stop:.0f}]: "
            f"{shape:.4g}, p={shape_p:.3f}"
        )
    axis.set(xlabel="physical lag time", ylabel=r"transfer-rate ACF", title=title)
    axis.legend(loc="upper right")
    counts = arrays["observed_transfer_rate_segment_count"]
    if counts.size:
        count_stop = min(stop, len(counts))
        counted = axis.twinx()
        counted.step(
            observed_time[:count_stop],
            counts[:count_stop],
            where="post",
            color="0.6",
            linewidth=1,
            linestyle="--",
        )
        counted.set_ylabel("segments contributing", color="0.5")
        counted.tick_params(axis="y", colors="0.5")
        counted.set_ylim(0, max(counts[:count_stop]) * 1.05)
    _save(path, figure)


def _negative_plot(outcome: LagOutcome, metrics: dict[str, Any], path: Path) -> None:
    count = float(metrics.get("negative_area_count", 0))
    generated = float(metrics.get("generated_area_count", 0))
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(("negative", "nonnegative"), (count, max(generated - count, 0.0)))
    axis.set_yscale("symlog", linthresh=1.0)
    axis.set_ylabel("generated area values")
    axis.set_title(
        f"Lag {outcome.lag}: negative-area fraction "
        f"{metrics.get('negative_area_fraction', float('nan')):.3g}"
    )
    _save(path, figure)


def plot_lag(outcome: LagOutcome, output_dir: Path) -> list[str]:
    directory = output_dir / outcome.fit / f"lag_{outcome.lag}" / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    _posterior_plots(outcome, directory)
    _hessian_plot(outcome, directory / "hessian.png")
    if outcome.fit == "event":
        from .event_plots import plot_event_validation

        plot_event_validation(outcome, directory)
        return sorted(path.name for path in directory.glob("*.png"))
    validation_paths = {
        "whitened_residual.png": _whitened_plot,
        "predictive.png": _predictive_plot,
        "trajectory.png": _trajectory_plot,
        "long_time.png": _long_time_plot,
        "transfer_rate_acf.png": _transfer_rate_acf_plot,
        "covariance.png": _covariance_plot,
    }
    arrays = _validation_arrays(outcome, "training_")
    if not outcome.accepted or not arrays:
        message = "Lag rejected by MCMC criteria; predictive conclusions excluded."
        for filename in validation_paths:
            _placeholder(directory / filename, f"Lag {outcome.lag}", message)
        _placeholder(directory / "negative_area.png", f"Lag {outcome.lag}", message)
    else:
        for filename, plotter in validation_paths.items():
            plotter(outcome, arrays, directory / filename)
        metrics = outcome.metadata["validation"]["training"]
        _negative_plot(outcome, metrics, directory / "negative_area.png")
    return sorted(path.name for path in directory.glob("*.png"))


def _cross_lag_parameters(outcomes: list[LagOutcome], path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    accepted = [outcome for outcome in outcomes if outcome.accepted]
    names = outcomes[0].parameter_names
    for axis, name in zip(axes.ravel(), names, strict=False):
        if accepted:
            lag = np.asarray([outcome.lag for outcome in accepted])
            intervals = [outcome.metadata["posterior"][name] for outcome in accepted]
            median = np.asarray([item["median"] for item in intervals])
            low = np.asarray([item["hdi_low"] for item in intervals])
            high = np.asarray([item["hdi_high"] for item in intervals])
            axis.errorbar(lag, median, yerr=(median - low, high - median), fmt="o")
        axis.set(xlabel="frame lag", ylabel=PARAMETER_LABELS[name])
    for axis in axes.ravel()[len(names) :]:
        axis.axis("off")
    figure.suptitle("Accepted posterior medians and 95% HDIs")
    _save(path, figure)


def _physical_dt(outcomes: list[LagOutcome], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4))
    lag = np.asarray([outcome.lag for outcome in outcomes])
    medians = np.asarray(
        [outcome.metadata["data"]["physical_dt_median"] for outcome in outcomes]
    )
    lower = np.asarray(
        [outcome.metadata["data"]["physical_dt_min"] for outcome in outcomes]
    )
    upper = np.asarray(
        [outcome.metadata["data"]["physical_dt_max"] for outcome in outcomes]
    )
    colors = ["tab:blue" if outcome.accepted else "tab:red" for outcome in outcomes]
    for x, median, low, high, color in zip(
        lag, medians, lower, upper, colors, strict=True
    ):
        axis.errorbar(
            x,
            median,
            yerr=np.asarray([[median - low], [high - median]]),
            fmt="none",
            ecolor=color,
            capsize=3,
        )
    axis.scatter(lag, medians, c=colors)
    for x, y, outcome in zip(lag, medians, outcomes, strict=True):
        axis.annotate("accepted" if outcome.accepted else "rejected", (x, y), xytext=(3, 4), textcoords="offset points")
    axis.set(xlabel="frame lag", ylabel=r"physical $\Delta\tau$ (min/median/max)")
    axis.set_title("Inference cadence")
    _save(path, figure)


def _holdout_panel(
    outcomes: list[LagOutcome], path: Path, selector: Callable[[dict[str, Any]], Any], title: str
) -> None:
    rows = []
    for outcome in outcomes:
        if not outcome.accepted:
            continue
        metrics = selector(outcome.metadata["validation"]["holdout"])
        if metrics is not None:
            rows.append((outcome.lag, metrics))
    if not rows:
        _placeholder(path, title, "No accepted holdout validation is available.")
        return
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    names = ("predictive_log_score", "coverage_95", "negative_area_fraction")
    labels = ("predictive log score", "95% coverage", "negative-area fraction")
    for axis, name, label in zip(axes, names, labels, strict=True):
        axis.plot([row[0] for row in rows], [row[1][name] for row in rows], "o-")
        axis.set(xlabel="frame lag", ylabel=label)
    figure.suptitle(title)
    _save(path, figure)


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "seed"


def plot_summary(outcomes: list[LagOutcome], output_dir: Path) -> list[str]:
    plots = output_dir / outcomes[0].fit / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    _cross_lag_parameters(outcomes, plots / "cross_lag_parameters.png")
    _physical_dt(outcomes, plots / "physical_dt.png")
    _holdout_panel(
        outcomes,
        plots / "holdout_aggregate.png",
        lambda holdout: holdout["aggregate"],
        "Aggregate holdout validation",
    )
    seed_ids = sorted(
        {
            seed_id
            for outcome in outcomes
            for seed_id in outcome.metadata.get("validation", {})
            .get("holdout", {})
            .get("per_seed", {})
        }
    )
    for seed_id in seed_ids:
        _holdout_panel(
            outcomes,
            plots / f"holdout_{_slug(seed_id)}.png",
            lambda holdout, selected=seed_id: holdout["per_seed"].get(selected),
            f"Holdout validation: {seed_id}",
        )
    return sorted(path.name for path in plots.glob("*.png"))
