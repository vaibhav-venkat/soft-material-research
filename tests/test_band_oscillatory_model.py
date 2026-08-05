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
    oscillatory_interval_moments,
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

    def test_interval_moments_reconstruct_integrated_variance(self) -> None:
        dt, gamma, omega, diffusion = 0.31, 0.8, 1.2, 0.2
        _, coefficients, _, _, process_variance = oscillatory_interval_moments(
            dt, gamma, omega, diffusion
        )
        stationary_variance = diffusion / gamma
        reconstructed = stationary_variance * np.dot(
            np.asarray(coefficients), np.asarray(coefficients)
        ) + float(process_variance)
        np.testing.assert_allclose(
            reconstructed,
            float(integrated_oscillatory_variance(dt, gamma, omega, diffusion)),
            atol=1e-12,
        )

    def test_interval_moments_broadcast_over_paths(self) -> None:
        dt = jnp.asarray([0.07, 0.12, 0.21])
        gamma = jnp.asarray([0.4, 0.8, 1.3])
        omega = jnp.asarray([0.9, 1.5, 2.2])
        diffusion = jnp.asarray([0.02, 0.04, 0.07])
        batched = tuple(
            np.asarray(value)
            for value in oscillatory_interval_moments(
                dt, gamma, omega, diffusion
            )
        )
        self.assertEqual(batched[0].shape, (3, 2, 2))
        self.assertEqual(batched[1].shape, (3, 2))
        self.assertEqual(batched[3].shape, (3, 2))
        for path in range(3):
            scalar = tuple(
                np.asarray(value)
                for value in oscillatory_interval_moments(
                    dt[path], gamma[path], omega[path], diffusion[path]
                )
            )
            for actual, expected in zip(batched, scalar, strict=True):
                np.testing.assert_allclose(actual[path], expected, atol=1e-12)

    def test_integrated_variance_and_total_drift_use_different_rates(self) -> None:
        area = jnp.asarray([0.4, 0.6])
        dt = jnp.asarray(0.2)
        base = jnp.asarray([0.3, 3.0, 1.3, 0.04, 0.4, 0.25, 0.03, 0.9, 0.02])
        changed_omega = base.at[2].set(2.1)
        changed_kappa = base.at[5].set(1.1)
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
        parameters = jnp.asarray([0.3, 3.0, 1.1, 0.2, 0.5, 0.4, 0.1, 2.0, 0.05])
        value = parameter_negative_log_likelihood(parameters, data)
        gradient = jax.grad(parameter_negative_log_likelihood)(parameters, data)
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(np.all(np.isfinite(np.asarray(gradient))))
        self.assertEqual(persistent_innovations(sequence, parameters).shape, (4, 2))
        self.assertTrue(np.isfinite(float(kalman_log_likelihood(sequence, parameters))))
        likelihood_data = TrainingTransitions(
            Scaling(1.0, 1.0), (), build_sequence_batches((sequence,), 1)
        )
        np.testing.assert_allclose(
            float(parameter_negative_log_likelihood(parameters, likelihood_data)),
            -float(kalman_log_likelihood(sequence, parameters)),
            atol=1e-10,
        )

    def test_two_modes_beat_one_on_two_mode_data(self) -> None:
        """Fit two-mode data with both models and check the second mode earns its keep.

        Data come from fine-grained Euler integration of the two coupled SDEs, so
        the simulator never touches the analytic interval moments the likelihood
        uses. The restricted fit pins the fast variance to a negligible constant,
        leaving the slow mode to carry everything -- that is the single
        oscillatory mode this model replaced.
        """
        true = np.asarray([0.25, 7.0, 1.0, 0.040, 0.25, 0.40, 0.030, 1.0, 0.020])
        rate_f = true[0] * (1.0 + true[1])
        variance_f, variance_o = true[3], true[3] * true[4]
        rows, components, steps, dt, substeps = 12, 2, 600, 0.12, 40
        generator = np.random.default_rng(11)

        shape = (rows, components)
        u_f = generator.normal(size=shape) * np.sqrt(variance_f)
        u_o = generator.normal(size=shape) * np.sqrt(variance_o)
        r_o = generator.normal(size=shape) * np.sqrt(variance_o)
        beta = np.repeat(
            generator.normal(size=(rows // 2, components)) * true[8], 2, axis=0
        )
        h = dt / substeps
        root_f = np.sqrt(2.0 * rate_f * variance_f * h)
        root_o = np.sqrt(2.0 * true[0] * variance_o * h)
        observations = np.empty((rows, steps, components))
        for step in range(steps):
            integral = np.zeros(shape)
            for _ in range(substeps):
                integral += 0.5 * h * (u_f + u_o)
                next_f = u_f - rate_f * u_f * h + root_f * generator.normal(size=shape)
                next_o = u_o + (
                    -true[0] * u_o + true[2] * r_o
                ) * h + root_o * generator.normal(size=shape)
                r_o = r_o + (
                    -true[2] * u_o - true[0] * r_o
                ) * h + root_o * generator.normal(size=shape)
                u_f, u_o = next_f, next_o
                integral += 0.5 * h * (u_f + u_o)
            observations[:, step] = integral + beta * dt

        decay = np.exp(-true[5] * dt)
        stationary = np.sqrt(true[6] / true[5])
        current = true[7] + stationary * generator.normal(size=4_000)
        following = (
            true[7]
            + (current - true[7]) * decay
            + stationary * np.sqrt(1.0 - decay**2) * generator.normal(size=4_000)
        )
        block = TransitionBlock(
            jnp.asarray(np.column_stack((0.5 * current, 0.5 * current))),
            jnp.asarray(np.column_stack((0.5 * following, 0.5 * following))),
            jnp.full(4_000, dt),
            jnp.ones(4_000),
        )
        sequences = tuple(
            StateSpaceSequence(
                y=jnp.asarray(observations[row]),
                dt=jnp.full(steps, dt),
                tau=jnp.arange(steps + 1, dtype=jnp.float64) * dt,
                weight=jnp.asarray(0.5),
                segment_id=row // 2,
            )
            for row in range(rows)
        )
        data = TrainingTransitions(
            Scaling(1.0, 1.0), (block,), build_sequence_batches(sequences, rows // 2)
        )

        def fit(start: np.ndarray, pinned: float | None) -> tuple[np.ndarray, float]:
            free = [index for index in range(9) if index != 3 or pinned is None]
            selection = jnp.asarray(free)
            fixed = jnp.zeros(9).at[3].set(pinned or 0.0)
            mask = jnp.asarray([pinned is not None and i == 3 for i in range(9)])
            value_and_grad = jax.jit(
                jax.value_and_grad(
                    lambda raw: parameter_negative_log_likelihood(
                        jnp.where(
                            mask, fixed, jnp.zeros(9).at[selection].set(jnp.exp(raw))
                        ),
                        data,
                    )
                )
            )
            result = minimize(
                lambda raw: tuple(
                    np.asarray(value, dtype=np.float64)
                    for value in value_and_grad(jnp.asarray(raw))
                ),
                np.log(start[free]),
                jac=True,
                method="L-BFGS-B",
                bounds=[(-25.0, 25.0)] * len(free),
                options={"gtol": 1e-8, "ftol": 1e-14, "maxiter": 3_000},
            )
            parameters = np.zeros(9)
            parameters[free] = np.exp(result.x)
            if pinned is not None:
                parameters[3] = pinned
            return parameters, float(result.fun)

        start = np.asarray([0.3, 5.0, 0.8, 0.03, 0.2, 0.3, 0.02, 1.0, 0.015])
        two_mode, two_mode_objective = fit(start, None)

        pinned = 1e-4
        restricted_start = start.copy()
        restricted_start[3] = pinned
        restricted_start[4] = (variance_f + variance_o) / pinned
        restricted_start[[5, 6, 7]] = true[[5, 6, 7]]
        one_mode, one_mode_objective = fit(restricted_start, pinned)

        # The two-mode fit recovers the rates and the variance split.
        self.assertAlmostEqual(
            two_mode[0] * (1.0 + two_mode[1]), rate_f, delta=0.3 * rate_f
        )
        self.assertAlmostEqual(two_mode[2], true[2], delta=0.3 * true[2])
        self.assertAlmostEqual(two_mode[3], variance_f, delta=0.4 * variance_f)
        self.assertAlmostEqual(
            two_mode[3] * two_mode[4], variance_o, delta=0.5 * variance_o
        )

        # One mode cannot hold both timescales: it locks onto the fast decay and
        # abandons the oscillation, collapsing to a small omega/gamma ratio.
        self.assertGreater(one_mode[0], 3.0 * two_mode[0])
        self.assertLess(one_mode[2] / one_mode[0], 0.3 * two_mode[2] / two_mode[0])

        # The extra mode is decisively preferred and tracks the true ACF better.
        self.assertGreater(one_mode_objective - two_mode_objective, 8.0)
        lags = np.linspace(0.0, 12.0, 200)

        def autocorrelation(parameters: np.ndarray) -> np.ndarray:
            return parameters[3] * np.exp(
                -parameters[0] * (1.0 + parameters[1]) * lags
            ) + parameters[3] * parameters[4] * np.exp(
                -parameters[0] * lags
            ) * np.cos(parameters[2] * lags)

        truth = autocorrelation(true)
        self.assertLess(
            np.sqrt(np.mean((autocorrelation(two_mode) - truth) ** 2)),
            np.sqrt(np.mean((autocorrelation(one_mode) - truth) ** 2)),
        )


if __name__ == "__main__":
    unittest.main()
