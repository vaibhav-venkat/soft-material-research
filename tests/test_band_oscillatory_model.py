"""Small deterministic checks for the clean oscillatory band model."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from hexatic.band_analysis.fitting.model import (
    Scaling,
    StateSpaceSequence,
    TrainingTransitions,
    TransitionBlock,
    build_sequence_batches,
    build_state_space_sequences,
    helmert_basis,
    integrated_oscillatory_variance,
    kalman_log_likelihood,
    oscillatory_transition,
    oscillatory_transition_matrix,
    oscillatory_process_variance,
    parameter_negative_log_likelihood,
    persistent_innovations,
    transition,
)
from hexatic.band_analysis.detection.segments import StableSegment


class TestOscillatoryModel(unittest.TestCase):
    @staticmethod
    def _total_block(
        *, count: int = 40, kappa: float = 0.4, diffusion: float = 0.03
    ) -> TransitionBlock:
        generator = np.random.default_rng(17)
        dt = 0.12
        current_total = 1.0 + 0.08 * np.sin(np.arange(count) / 5.0)
        following_total = (
            current_total
            - kappa * (current_total - 1.0) * dt
            + np.sqrt(2.0 * diffusion * dt)
            * generator.normal(size=count)
        )
        current = np.column_stack((0.5 * current_total, 0.5 * current_total))
        following = np.column_stack(
            (0.5 * following_total, 0.5 * following_total)
        )
        return TransitionBlock(
            jnp.asarray(current),
            jnp.asarray(following),
            jnp.full(count, dt),
            jnp.ones(count),
        )

    def test_analytic_transition_matches_matrix_exponential(self) -> None:
        gamma, omega, dt = 0.73, 1.41, 0.27
        generator = np.block(
            [
                [-gamma * np.eye(1), omega * np.eye(1)],
                [-omega * np.eye(1), -gamma * np.eye(1)],
            ]
        )
        analytic = np.asarray(oscillatory_transition_matrix(dt, gamma, omega))
        expected = expm(generator * dt)
        np.testing.assert_allclose(analytic, expected, atol=1e-12)

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
        reconstructed = weights * increment.sum() + redistribution.ravel()
        np.testing.assert_allclose(reconstructed, increment, atol=1e-12)

    def test_zero_frequency_is_nonoscillatory_ou(self) -> None:
        u = jnp.asarray([0.7, -0.2])
        r = jnp.asarray([1.3, 0.4])
        dt, gamma = 0.31, 0.8
        next_u, next_r = oscillatory_transition(u, r, dt, gamma, 0.0)
        decay = np.exp(-gamma * dt)
        np.testing.assert_allclose(next_u, decay * u, atol=1e-12)
        np.testing.assert_allclose(next_r, decay * r, atol=1e-12)

    def test_exact_process_variance(self) -> None:
        dt, gamma, diffusion = 0.31, 0.8, 0.2
        expected = diffusion / gamma * (1.0 - np.exp(-2.0 * gamma * dt))
        self.assertAlmostEqual(
            float(oscillatory_process_variance(dt, gamma, diffusion)), expected
        )

    def test_integrated_variance_and_total_drift_use_different_rates(self) -> None:
        area = jnp.asarray([0.4, 0.6])
        dt = jnp.asarray(0.2)
        base = jnp.asarray([0.8, 1.3, 0.25, 0.04, 0.03, 0.9, 0.02])
        changed_omega = base.at[1].set(2.1)
        changed_kappa = base.at[2].set(1.1)
        _, base_covariance = transition(area, dt, base)
        _, omega_covariance = transition(area, dt, changed_omega)
        base_mean, _ = transition(area, dt, base)
        kappa_mean, _ = transition(area, dt, changed_kappa)
        self.assertFalse(np.allclose(base_covariance, omega_covariance))
        self.assertFalse(np.allclose(base_mean, kappa_mean))
        self.assertAlmostEqual(
            float(integrated_oscillatory_variance(0.0, 0.8, 1.3, 0.2)), 0.0
        )

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
        data = TrainingTransitions(
            Scaling(1.0, 1.0),
            (self._total_block(),),
            build_sequence_batches((sequence,), 1),
        )
        parameters = jnp.asarray([0.8, 1.1, 0.4, 0.2, 0.1, 2.0, 0.05])
        value = parameter_negative_log_likelihood(parameters, data)
        gradient = jax.grad(parameter_negative_log_likelihood)(parameters, data)
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(np.all(np.isfinite(np.asarray(gradient))))
        self.assertEqual(persistent_innovations(sequence, parameters).shape, (4, 2))
        self.assertTrue(np.isfinite(float(kalman_log_likelihood(sequence, parameters))))

    def test_synthetic_oscillatory_parameters_are_recoverable(self) -> None:
        generator = np.random.default_rng(13)
        true = np.asarray([0.7, 1.4, 0.4, 0.04, 0.03, 1.0, 0.03])
        sequences = []
        for segment_id in range(6):
            dt = 0.12
            beta = generator.normal(size=2) * true[6]
            for _ in range(2):
                u = generator.normal(size=2) * np.sqrt(true[3] / true[0])
                r = generator.normal(size=2) * np.sqrt(true[3] / true[0])
                values = []
                for _ in range(120):
                    values.append((u + beta) * dt)
                    decay = np.exp(-true[0] * dt)
                    cosine = np.cos(true[1] * dt)
                    sine = np.sin(true[1] * dt)
                    noise = np.sqrt(
                        true[3] / true[0] * (1.0 - decay**2)
                    )
                    u, r = (
                        decay * (cosine * u + sine * r)
                        + noise * generator.normal(size=2),
                        decay * (-sine * u + cosine * r)
                        + noise * generator.normal(size=2),
                    )
                sequences.append(
                    StateSpaceSequence(
                        jnp.asarray(values),
                        jnp.full(120, dt),
                        jnp.arange(121, dtype=jnp.float64) * dt,
                        jnp.asarray(0.5),
                        segment_id,
                    )
                )
        data = TrainingTransitions(
            Scaling(1.0, 1.0),
            (self._total_block(count=80, kappa=true[2]),),
            build_sequence_batches(tuple(sequences), 6),
        )
        value_and_grad = jax.jit(
            jax.value_and_grad(
                lambda raw: parameter_negative_log_likelihood(jnp.exp(raw), data)
            )
        )

        def objective(raw: np.ndarray) -> tuple[float, np.ndarray]:
            value, gradient = value_and_grad(jnp.asarray(raw))
            return float(value), np.asarray(gradient)

        result = minimize(
            objective,
            np.log([0.8, 1.0, 0.3, 0.03, 0.02, 1.0, 0.02]),
            jac=True,
            method="BFGS",
            options={"gtol": 1e-5, "maxiter": 200},
        )
        self.assertTrue(result.success, result.message)
        fitted = np.exp(result.x)
        np.testing.assert_allclose(
            fitted[[0, 1, 3, 6]], true[[0, 1, 3, 6]], rtol=0.3
        )


if __name__ == "__main__":
    unittest.main()
