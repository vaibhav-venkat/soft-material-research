"""Mathematical checks for the clean one-mode OU band model."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from hexatic.band_analysis.band_analysis_id.segments import StableSegment
from hexatic.band_analysis.band_analysis_fitting.model import (
    Scaling,
    StateSpaceSequence,
    TrainingTransitions,
    TransitionBlock,
    build_sequence_batches,
    build_state_space_sequences,
    helmert_basis,
    integrated_ou_variance,
    kalman_log_likelihood,
    ou_interval_moments,
    ou_process_variance,
    parameter_negative_log_likelihood,
    persistent_innovations,
    transition,
)


class TestOUModel(unittest.TestCase):
    @staticmethod
    def _total_block() -> TransitionBlock:
        generator = np.random.default_rng(17)
        count, dt, kappa, diffusion = 80, 0.12, 0.4, 0.03
        decay = np.exp(-kappa * dt)
        current_total = 1.0 + 0.08 * np.sin(np.arange(count) / 5.0)
        following_total = (
            1.0
            + (current_total - 1.0) * decay
            + np.sqrt(diffusion / kappa * (1.0 - decay**2))
            * generator.normal(size=count)
        )
        return TransitionBlock(
            jnp.asarray(np.column_stack((0.5 * current_total, 0.5 * current_total))),
            jnp.asarray(np.column_stack((0.5 * following_total, 0.5 * following_total))),
            jnp.full(count, dt),
            jnp.ones(count),
        )

    def test_redistribution_uses_left_endpoint_area_fractions(self) -> None:
        segment = StableSegment(
            seed_id="weighted",
            track_ids=(1, 2),
            frame_indices=np.asarray([0, 1]),
            tau=np.asarray([0.0, 1.0]),
            areas=np.asarray([[1.0, 3.0], [2.0, 5.0]]),
        )
        sequence = build_state_space_sequences([segment], 1)[0]
        increment = segment.areas[1] - segment.areas[0]
        weights = segment.areas[0] / segment.areas[0].sum()
        redistribution = np.asarray(sequence.y) @ helmert_basis(2).T
        np.testing.assert_allclose(
            weights * increment.sum() + redistribution.ravel(), increment
        )

    def test_exact_ou_moments(self) -> None:
        dt, gamma, diffusion = 0.31, 0.8, 0.2
        moments = ou_interval_moments(dt, gamma, diffusion)
        stationary_variance = diffusion / gamma
        self.assertAlmostEqual(float(moments.transition), np.exp(-gamma * dt))
        self.assertAlmostEqual(
            float(moments.process_variance),
            diffusion / gamma * (1.0 - np.exp(-2.0 * gamma * dt)),
        )
        reconstructed = (
            stationary_variance * float(moments.coefficient) ** 2
            + float(moments.integral_process_variance)
        )
        self.assertAlmostEqual(
            reconstructed, float(integrated_ou_variance(dt, gamma, diffusion))
        )
        self.assertAlmostEqual(float(ou_process_variance(0.0, gamma, diffusion)), 0.0)
        self.assertAlmostEqual(float(integrated_ou_variance(0.0, gamma, diffusion)), 0.0)

    def test_interval_moments_vectorize(self) -> None:
        dt = jnp.asarray([0.07, 0.12, 0.21])
        gamma = jnp.asarray([0.4, 0.8, 1.3])
        diffusion = jnp.asarray([0.02, 0.04, 0.07])
        batched = ou_interval_moments(dt, gamma, diffusion)
        for path in range(len(dt)):
            scalar = ou_interval_moments(dt[path], gamma[path], diffusion[path])
            for actual, expected in zip(batched, scalar, strict=True):
                np.testing.assert_allclose(np.asarray(actual)[path], expected)

    def test_exact_simulation_has_exponential_autocorrelation(self) -> None:
        generator = np.random.default_rng(4)
        gamma, diffusion, dt = 0.6, 0.08, 0.1
        decay = np.exp(-gamma * dt)
        innovation_scale = np.sqrt(
            diffusion / gamma * (1.0 - decay**2)
        )
        values = np.empty(40_000)
        values[0] = generator.normal(scale=np.sqrt(diffusion / gamma))
        for index in range(1, len(values)):
            values[index] = (
                decay * values[index - 1] + innovation_scale * generator.normal()
            )
        correlation = np.corrcoef(values[:-5], values[5:])[0, 1]
        self.assertAlmostEqual(correlation, np.exp(-gamma * 5 * dt), delta=0.025)

    def test_total_and_redistribution_rates_are_separate(self) -> None:
        area = jnp.asarray([0.4, 0.6])
        parameters = jnp.asarray([0.3, 0.25, 0.04, 0.03, 0.9, 0.02])
        mean, covariance = transition(area, jnp.asarray(0.2), parameters)
        changed_gamma = parameters.at[0].set(1.1)
        changed_kappa = parameters.at[1].set(1.1)
        changed_mean, _ = transition(area, jnp.asarray(0.2), changed_kappa)
        _, changed_covariance = transition(area, jnp.asarray(0.2), changed_gamma)
        self.assertFalse(np.allclose(mean, changed_mean))
        self.assertFalse(np.allclose(covariance, changed_covariance))

    def test_kalman_likelihood_and_gradients_are_finite(self) -> None:
        sequence = StateSpaceSequence(
            y=jnp.asarray(
                [[0.02, -0.01], [0.01, -0.015], [-0.01, 0.02], [0.015, -0.01]]
            ),
            dt=jnp.full(4, 0.2),
            tau=jnp.arange(5, dtype=jnp.float64) * 0.2,
            weight=jnp.asarray(1.0),
            segment_id=0,
        )
        parameters = jnp.asarray([0.3, 0.4, 0.02, 0.03, 1.0, 0.05])
        data = TrainingTransitions(
            Scaling(1.0, 1.0),
            (self._total_block(),),
            build_sequence_batches((sequence,), 1),
        )
        value = parameter_negative_log_likelihood(parameters, data)
        gradient = jax.grad(parameter_negative_log_likelihood)(parameters, data)
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(np.all(np.isfinite(np.asarray(gradient))))
        self.assertTrue(np.all(np.isfinite(np.asarray(persistent_innovations(sequence, parameters)))))

        redistribution_only = TrainingTransitions(
            Scaling(1.0, 1.0), (), build_sequence_batches((sequence,), 1)
        )
        self.assertAlmostEqual(
            float(parameter_negative_log_likelihood(parameters, redistribution_only)),
            -float(kalman_log_likelihood(sequence, parameters)),
        )


if __name__ == "__main__":
    unittest.main()
