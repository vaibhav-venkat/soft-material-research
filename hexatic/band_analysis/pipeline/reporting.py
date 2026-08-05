"""Write side-by-side clean-segment and event-chain inference reports."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ..fitting.model import PARAMETER_NAMES
from .storage import write_json, write_text
from .workflow import AnalysisConfig, FitPath, LagOutcome, stable_lags


FitOutcomes = dict[FitPath, list[LagOutcome]]


def _base_and_stable(
    outcomes: list[LagOutcome], config: AnalysisConfig
) -> tuple[LagOutcome | None, tuple[int, ...]]:
    base = next(
        (outcome for outcome in outcomes if outcome.lag == config.base_lag), None
    )
    return base, stable_lags(outcomes, config.base_lag)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_configuration(output_dir: Path, configuration: dict[str, Any]) -> Path:
    path = output_dir / "configuration.json"
    write_json(path, _json_value(configuration))
    return path


def _fit_metrics(
    outcomes: list[LagOutcome], config: AnalysisConfig
) -> dict[str, Any]:
    base, stable = _base_and_stable(outcomes, config)
    return {
        "base_lag": config.base_lag,
        "base_lag_accepted": bool(base and base.accepted),
        "accepted_lags": [outcome.lag for outcome in outcomes if outcome.accepted],
        "rejected_lags": [
            outcome.lag for outcome in outcomes if not outcome.accepted
        ],
        "stable_lags": list(stable),
        "stable_range": [min(stable), max(stable)] if stable else None,
        "lags": {
            str(outcome.lag): {
                key: outcome.metadata.get(key, {})
                for key in (
                    "accepted",
                    "posterior",
                    "diagnostics",
                    "optimization",
                    "scaling",
                    "conservative_slope",
                    "data",
                    "validation",
                )
            }
            | {"cached": outcome.cached}
            for outcome in outcomes
        },
    }


def consolidated_metrics(
    outcomes: FitOutcomes, config: AnalysisConfig
) -> dict[str, Any]:
    return {
        "fits": {
            fit: _fit_metrics(fit_outcomes, config)
            for fit, fit_outcomes in outcomes.items()
        }
    }


def write_metrics(
    output_dir: Path, outcomes: FitOutcomes, config: AnalysisConfig
) -> Path:
    path = output_dir / "metrics.json"
    write_json(path, _json_value(consolidated_metrics(outcomes, config)))
    return path


def _number(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.5g}" if math.isfinite(number) else "—"


def _interval(outcome: LagOutcome | None, name: str) -> str:
    if outcome is None or name not in outcome.metadata.get("posterior", {}):
        return "—"
    values = outcome.metadata["posterior"][name]
    return (
        f"{_number(values['median'])} "
        f"[{_number(values['hdi_low'])}, {_number(values['hdi_high'])}]"
    )


def _parameter_table(outcome: LagOutcome) -> list[str]:
    lines = [
        "| Parameter | Median | 95% HDI | R-hat | ESS bulk | ESS tail |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    diagnostics = outcome.metadata["diagnostics"]
    for name in outcome.parameter_names:
        posterior = outcome.metadata["posterior"][name]
        interval = (
            f"[{_number(posterior['hdi_low'])}, "
            f"{_number(posterior['hdi_high'])}]"
        )
        lines.append(
            f"| {name} | {_number(posterior['median'])} | {interval} | "
            f"{_number(diagnostics['rhat'].get(name))} | "
            f"{_number(diagnostics['ess_bulk'].get(name))} | "
            f"{_number(diagnostics['ess_tail'].get(name))} |"
        )
    return lines


def _validation_lines(title: str, metrics: dict[str, Any] | None) -> list[str]:
    if metrics is None:
        return [f"#### {title}", "", "Not available."]
    keys = (
        "predictive_log_score",
        "coverage_95",
        "one_step_rmse",
        "residual_mean",
        "residual_variance",
        "residual_lag1_autocorrelation",
        "inactive_prediction_max_abs",
        "negative_area_fraction",
        "terminated_path_count",
        "observed_total_mean",
        "simulated_total_mean",
        "observed_total_variance",
        "simulated_total_variance",
        "observed_total_relaxation_time",
        "simulated_total_relaxation_time",
        "observed_conservative_relaxation_time",
        "simulated_conservative_relaxation_time",
    )
    available = [key for key in keys if key in metrics]
    lines = [f"#### {title}", "", "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {key} | {_number(metrics[key])} |" for key in available)
    return lines


def _stability(fit: FitPath, outcomes: list[LagOutcome], config: AnalysisConfig) -> str:
    base, stable = _base_and_stable(outcomes, config)
    if base is None or not base.accepted:
        return f"{fit.title()}: base lag {config.base_lag} failed MCMC acceptance."
    if stable:
        return (
            f"{fit.title()}: lags {', '.join(map(str, stable))} satisfy the "
            f"median-in-HDI rule (span {min(stable)}–{max(stable)})."
        )
    return f"{fit.title()}: no lag is stable against base lag {config.base_lag}."


def _comparison_table(outcomes: FitOutcomes, lag: int) -> list[str]:
    clean = next((item for item in outcomes.get("clean", ()) if item.lag == lag), None)
    event = next((item for item in outcomes.get("event", ()) if item.lag == lag), None)
    lines = [
        f"### Lag {lag}",
        "",
        "| Parameter | Clean segments: median [95% HDI] | Event chains: median [95% HDI] |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {name} | {_interval(clean, name)} | {_interval(event, name)} |"
        for name in PARAMETER_NAMES
    )
    return lines


def _lag_overview(outcomes: FitOutcomes) -> list[str]:
    lines = [
        "| Fit | Lag | Median physical Δτ | Status | Units | Transitions | "
        "Events | N_max | Divergences | Cached |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for fit, fit_outcomes in outcomes.items():
        for outcome in fit_outcomes:
            data = outcome.metadata["data"]
            lines.append(
                f"| {fit} | {outcome.lag} | {_number(data['physical_dt_median'])} | "
                f"{'accepted' if outcome.accepted else 'rejected'} | "
                f"{data['training_units']} | {data['training_transitions']} | "
                f"{_number(data.get('weighted_event_transitions'))} | "
                f"{_number(data.get('n_max'))} | "
                f"{outcome.metadata['diagnostics']['divergences']} | "
                f"{'yes' if outcome.cached else 'no'} |"
            )
    return lines


def write_report(
    output_dir: Path, outcomes: FitOutcomes, config: AnalysisConfig
) -> Path:
    fits = tuple(outcomes)
    lines = [
        "# Coupled Band-Area Inference",
        "",
        "Requested fits: " + ", ".join(fits) + ".",
        "",
        *(_stability(fit, outcomes[fit], config) for fit in fits),
        "",
        "## Fit overview",
        "",
        *_lag_overview(outcomes),
        "",
    ]
    if len(fits) == 2:
        lines.extend(["## Side-by-side posterior estimates", ""])
        for lag in config.lags:
            lines.extend([*_comparison_table(outcomes, lag), ""])

    for fit, fit_outcomes in outcomes.items():
        lines.extend([f"## {fit.title()} fit details", ""])
        for outcome in fit_outcomes:
            optimization = outcome.metadata["optimization"]
            lines.extend(
                [
                    f"### Lag {outcome.lag}: "
                    f"{'accepted' if outcome.accepted else 'rejected'}",
                    "",
                    *_parameter_table(outcome),
                    "",
                    "Direct conservative-slope variance: "
                    "`sigma_b^2 = "
                    f"{_number(outcome.metadata['conservative_slope']['sigma_b_squared'])}` "
                    "(area²/time²).",
                    "",
                    f"BFGS objective {_number(optimization['best_objective'])}; "
                    f"gradient norm {_number(optimization['best_gradient_norm'])}; "
                    f"Hessian condition number {_number(optimization['condition_number'])}; "
                    f"NUTS retry {outcome.metadata['retried']}.",
                ]
            )
            if outcome.accepted:
                validation = outcome.metadata["validation"]
                lines.extend(
                    ["", *_validation_lines("Training", validation["training"])]
                )
                holdout = validation["holdout"]
                lines.extend(
                    [
                        "",
                        *_validation_lines("Aggregate holdout", holdout["aggregate"]),
                    ]
                )
                for seed_id, metrics in holdout["per_seed"].items():
                    lines.extend(
                        ["", *_validation_lines(f"Holdout `{seed_id}`", metrics)]
                    )
            else:
                lines.extend(["", "Predictive validation was excluded for this lag."])
            lines.extend(
                [
                    "",
                    f"Plots: [`{fit}/lag_{outcome.lag}/plots/`]"
                    f"({fit}/lag_{outcome.lag}/plots/)",
                    "",
                ]
            )

    lines.extend(["## Diagnostics", ""])
    if "event" in outcomes:
        lines.extend(
            [
                "The event diagnostic [`A_T` against `n_k`]"
                "(event/plots/total_area_by_band_count.png) checks the global, "
                "band-count-independent `A_T_star` assumption post hoc.",
                "",
            ]
        )
    lines.extend(["Cross-lag and holdout plots live under each fit's `plots/` directory.", ""])
    path = output_dir / "report.md"
    write_text(path, "\n".join(lines))
    return path
