"""Write consolidated metrics and Markdown analysis reports."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .model import PARAMETER_NAMES
from .workflow import AnalysisConfig, LagOutcome, stable_lags


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


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
    _atomic_text(
        path,
        json.dumps(_json_value(configuration), indent=2, sort_keys=True) + "\n",
    )
    return path


def consolidated_metrics(
    outcomes: list[LagOutcome], config: AnalysisConfig
) -> dict[str, Any]:
    accepted = [outcome.lag for outcome in outcomes if outcome.accepted]
    stable = stable_lags(outcomes, config.base_lag)
    base = next(
        (outcome for outcome in outcomes if outcome.lag == config.base_lag), None
    )
    return {
        "schema": "hexatic.band_analysis.metrics.v1",
        "base_lag": config.base_lag,
        "base_lag_accepted": bool(base and base.accepted),
        "accepted_lags": accepted,
        "rejected_lags": [
            outcome.lag for outcome in outcomes if not outcome.accepted
        ],
        "stable_lags": list(stable),
        "stable_range": [min(stable), max(stable)] if stable else None,
        "lags": {
            str(outcome.lag): {
                "accepted": outcome.accepted,
                "cached": outcome.cached,
                "posterior": outcome.metadata["posterior"],
                "diagnostics": outcome.metadata["diagnostics"],
                "optimization": outcome.metadata["optimization"],
                "scaling": outcome.metadata["scaling"],
                "data": outcome.metadata.get("data", {}),
                "validation": outcome.metadata.get("validation", {}),
            }
            for outcome in outcomes
        },
    }


def write_metrics(
    output_dir: Path, outcomes: list[LagOutcome], config: AnalysisConfig
) -> Path:
    path = output_dir / "metrics.json"
    value = consolidated_metrics(outcomes, config)
    _atomic_text(path, json.dumps(_json_value(value), indent=2, sort_keys=True) + "\n")
    return path


def _number(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.5g}" if math.isfinite(number) else "—"


def _parameter_table(outcome: LagOutcome) -> list[str]:
    lines = [
        "| Parameter | Median | 95% HDI | R-hat | ESS bulk | ESS tail |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    diagnostics = outcome.metadata["diagnostics"]
    for name in PARAMETER_NAMES:
        posterior = outcome.metadata["posterior"][name]
        interval = (
            f"[{_number(posterior['hdi_low'])}, "
            f"{_number(posterior['hdi_high'])}]"
        )
        lines.append(
            f"| {name} | {_number(posterior['median'])} | {interval} | "
            f"{_number(diagnostics['rhat'][name])} | "
            f"{_number(diagnostics['ess_bulk'][name])} | "
            f"{_number(diagnostics['ess_tail'][name])} |"
        )
    return lines


def _validation_lines(title: str, metrics: dict[str, Any] | None) -> list[str]:
    if metrics is None:
        return [f"#### {title}", "", "Not available."]
    keys = (
        "predictive_log_score",
        "coverage_95",
        "residual_mean",
        "residual_variance",
        "residual_lag1_autocorrelation",
        "one_step_rmse",
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
    lines = [f"#### {title}", "", "| Metric | Value |", "|---|---:|"]
    lines.extend(
        f"| {key} | {_number(metrics.get(key))} |" for key in keys
    )
    return lines


def _matrix_table(matrix: np.ndarray) -> list[str]:
    headings = " | ".join(str(index + 1) for index in range(matrix.shape[1]))
    lines = [f"| Mode | {headings} |", "|---|" + "---:|" * matrix.shape[1]]
    for index, row in enumerate(matrix):
        lines.append(
            f"| {index + 1} | "
            + " | ".join(_number(value) for value in row)
            + " |"
        )
    return lines


def write_report(
    output_dir: Path, outcomes: list[LagOutcome], config: AnalysisConfig
) -> Path:
    stable = stable_lags(outcomes, config.base_lag)
    base = next(
        (outcome for outcome in outcomes if outcome.lag == config.base_lag), None
    )
    if base is None or not base.accepted:
        stability = (
            f"No stable range is reported because base lag {config.base_lag} "
            "failed MCMC acceptance."
        )
    elif stable:
        stability = (
            f"Lags {', '.join(map(str, stable))} satisfy the symmetric "
            f"median-in-HDI rule relative to base lag {config.base_lag}. "
            f"The stable span is {min(stable)}–{max(stable)}."
        )
    else:
        stability = (
            f"No lag satisfies the stability rule relative to accepted base lag "
            f"{config.base_lag}."
        )

    lines = [
        "# Coupled Stable-Band Area Inference",
        "",
        stability,
        "",
        "Rejected lags retain posterior and sampler diagnostics, but are excluded "
        "from predictive, holdout, and lag-stability conclusions.",
        "",
        "## Lag overview",
        "",
        "| Lag | Median physical Δτ | Status | Segments | Transitions | Seeds | Divergences | Cached |",
        "|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for outcome in outcomes:
        data = outcome.metadata.get("data", {})
        diagnostics = outcome.metadata["diagnostics"]
        physical_dt = data.get(
            "physical_dt_median",
            outcome.lag * outcome.metadata["scaling"]["time"],
        )
        lines.append(
            f"| {outcome.lag} | {_number(physical_dt)} | "
            f"{'accepted' if outcome.accepted else 'rejected'} | "
            f"{data.get('training_segments', '—')} | "
            f"{data.get('training_transitions', '—')} | "
            f"{data.get('training_seeds', '—')} | "
            f"{diagnostics['divergences']} | "
            f"{'yes' if outcome.cached else 'no'} |"
        )

    for outcome in outcomes:
        optimization = outcome.metadata["optimization"]
        diagnostics = outcome.metadata["diagnostics"]
        lines.extend(
            [
                "",
                f"## Lag {outcome.lag}: "
                f"{'accepted' if outcome.accepted else 'rejected'}",
                "",
                *(_parameter_table(outcome)),
                "",
                f"BFGS objective: {_number(optimization['best_objective'])}; "
                f"gradient norm: {_number(optimization['best_gradient_norm'])}; "
                f"raw-Hessian condition number: "
                f"{_number(optimization['condition_number'])}. "
                f"Nonpositive eigenvalue: {optimization['nonpositive_hessian']}; "
                f"weak mode: {optimization['weak_hessian']}. "
                f"Profile-best kappa_c: "
                f"{_number(optimization['profile_best_kappa_c'])}; "
                f"profile delta-log-likelihood at zero: "
                f"{_number(optimization['profile_delta_log_likelihood_at_zero'])}. "
                f"NUTS retry used: {outcome.metadata['retried']}; "
                f"divergences: {diagnostics['divergences']}.",
            ]
        )
        hessian = outcome.arrays.get("hessian")
        eigenvalues = outcome.arrays.get("hessian_eigenvalues")
        eigenvectors = outcome.arrays.get("hessian_eigenvectors")
        if hessian is not None and eigenvalues is not None and eigenvectors is not None:
            lines.extend(
                [
                    "",
                    "<details><summary>Raw-parameter Hessian diagnostics</summary>",
                    "",
                    "Hessian:",
                    "",
                    *_matrix_table(hessian),
                    "",
                    "Eigenvalues: "
                    + ", ".join(_number(value) for value in eigenvalues),
                    "",
                    "Eigenvectors (columns are ordered modes):",
                    "",
                    *_matrix_table(eigenvectors),
                    "",
                    "</details>",
                ]
            )
        if outcome.accepted:
            validation = outcome.metadata["validation"]
            lines.extend(["", *_validation_lines("Training", validation["training"])])
            holdout = validation["holdout"]
            lines.extend(
                ["", *_validation_lines("Aggregate holdout", holdout["aggregate"])]
            )
            for seed_id, seed_metrics in holdout["per_seed"].items():
                lines.extend(
                    ["", *_validation_lines(f"Holdout `{seed_id}`", seed_metrics)]
                )
        else:
            lines.extend(
                [
                    "",
                    "Predictive and holdout validation were not used for this lag.",
                ]
            )
        lines.extend(
            [
                "",
                f"Plots: [`lag_{outcome.lag}/plots/`](lag_{outcome.lag}/plots/)",
            ]
        )

    lines.extend(
        [
            "",
            "## Cross-lag and holdout plots",
            "",
            "See [`plots/`](plots/) for accepted posterior HDIs, physical cadence, "
            "and aggregate/per-seed holdout panels.",
            "",
        ]
    )
    path = output_dir / "report.md"
    _atomic_text(path, "\n".join(lines))
    return path
