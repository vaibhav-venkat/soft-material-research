"""Small exact-OU dataset and full clean-analysis smoke run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hexatic.band_analysis.band_analysis_id.segments import StableSegment
from hexatic.band_analysis.band_analysis_fitting.bayesian import MCMCConfig
from hexatic.band_analysis.band_analysis_fitting.model import (
    helmert_basis,
    ou_interval_moments,
    prepare_training_transitions,
)
from hexatic.band_analysis.band_analysis_fitting.validation import validate_posterior
from hexatic.band_analysis.band_analysis_fitting.plots import plot_lag, plot_summary
from hexatic.band_analysis.band_analysis_fitting.reporting import write_metrics, write_report
from hexatic.band_analysis.band_analysis_fitting.workflow import (
    AnalysisConfig,
    LagOutcome,
    run_analysis,
)


PARAMETERS = np.asarray([0.7, 0.25, 0.025, 0.04, 12.0, 0.015])


def synthetic_segments(
    *, count: int = 3, frames: int = 70, dt: float = 0.15, seed: int = 19
) -> list[StableSegment]:
    generator = np.random.default_rng(seed)
    gamma, kappa, diffusion_u, diffusion_total, area_star, sigma_b = PARAMETERS
    redistribution = ou_interval_moments(dt, gamma, diffusion_u)
    joint_covariance = np.asarray(
        [
            [
                redistribution.process_variance,
                redistribution.process_integral_covariance,
            ],
            [
                redistribution.process_integral_covariance,
                redistribution.integral_process_variance,
            ],
        ]
    )
    joint_cholesky = np.linalg.cholesky(joint_covariance)
    total_decay = np.exp(-kappa * dt)
    total_noise = np.sqrt(diffusion_total / kappa * (1.0 - total_decay**2))
    segments = []
    for index in range(count):
        basis = helmert_basis(3)
        areas = np.empty((frames, 3), dtype=np.float64)
        areas[0] = area_star / 3.0
        rates = generator.normal(size=2) * np.sqrt(diffusion_u / gamma)
        slope = generator.normal(size=2) * sigma_b
        for frame in range(frames - 1):
            noise = generator.normal(size=(2, 2)) @ joint_cholesky.T
            total = areas[frame].sum()
            following_total = (
                area_star
                + (total - area_star) * total_decay
                + total_noise * generator.normal()
            )
            integral = (
                float(redistribution.coefficient) * rates + noise[:, 1] + slope * dt
            )
            weights = areas[frame] / total
            areas[frame + 1] = (
                areas[frame]
                + weights * (following_total - total)
                + integral @ basis.T
            )
            rates = float(redistribution.transition) * rates + noise[:, 0]
        segments.append(
            StableSegment(
                seed_id=f"synthetic_{index}",
                track_ids=(0, 1, 2),
                frame_indices=np.arange(frames, dtype=np.int64),
                tau=np.arange(frames, dtype=np.float64) * dt,
                areas=areas,
            )
        )
    return segments


def run_smoke(output_dir: Path) -> Path:
    segments = synthetic_segments()
    prepared = prepare_training_transitions(segments, 1)
    normalized_parameters = PARAMETERS / prepared.scaling.parameter_factors
    validation = validate_posterior(
        segments,
        1,
        prepared.scaling,
        np.broadcast_to(normalized_parameters, (8, len(normalized_parameters))),
        predictive_draws=4,
        paths_per_segment=4,
        seed=31,
    )
    config = AnalysisConfig(
        lags=(1,),
        base_lag=1,
        optimizer_starts=2,
        optimizer_max_steps=500,
        mcmc=MCMCConfig(
            chains=2,
            warmup=30,
            draws=30,
            retry_warmup=40,
            retry_draws=40,
        ),
        predictive_draws=4,
        paths_per_segment=4,
    )
    outcomes = run_analysis(
        "clean", segments, {}, output_dir, config=config, overwrite=True
    )
    for outcome in outcomes:
        render_outcome = outcome
        if not outcome.accepted:
            render_outcome = LagOutcome(
                fit=outcome.fit,
                lag=outcome.lag,
                accepted=True,
                cached=outcome.cached,
                metadata={
                    **outcome.metadata,
                    "validation": {
                        "training": validation.metrics,
                        "holdout": {"aggregate": None, "per_seed": {}},
                    },
                },
                arrays={
                    **outcome.arrays,
                    **{
                        f"training_{name}": values
                        for name, values in validation.arrays.items()
                    },
                },
                idata=outcome.idata,
            )
        plot_lag(render_outcome, output_dir)
    plot_summary(outcomes, output_dir)
    write_metrics(output_dir, {"clean": outcomes}, config)
    return write_report(output_dir, {"clean": outcomes}, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    print(run_smoke(parser.parse_args().output_dir))
