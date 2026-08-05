"""Mathematical validation of the masked event-chain area model."""

from __future__ import annotations

from collections.abc import Callable
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from hexatic.band_analysis.detection.chains import (
    EventChain,
    build_knot_grids,
    event_transitions,
    plan_chains,
)
from hexatic.band_analysis.detection.characterization import DetectedBand
from hexatic.band_analysis.detection.events import (
    EventStep,
    build_event_step,
    initial_slots,
    no_event_step,
)
from hexatic.band_analysis.detection.tracking import (
    BandTracker,
    DetectionFrame,
    EventCode,
    EventEdges,
)
from hexatic.band_analysis.fitting.event_validation import (
    _segment_breaks,
    simulate_event_chain,
)
from hexatic.band_analysis.fitting.inference import empirical_parameters
from hexatic.band_analysis.fitting.masked_model import (
    MaskedTrainingTransitions,
    parameter_negative_log_likelihood as event_nll,
    prepare_training_transitions as prepare_event,
    transition_log_density,
)
from hexatic.band_analysis.fitting.model import (
    JITTER,
    parameter_negative_log_likelihood as clean_nll,
    prepare_training_transitions as prepare_clean,
)
from hexatic.band_analysis.pipeline.event_plots import (
    _pooled_msd,
    _shared_msd_limit,
    _track_series,
)

from tests.synthetic_band_events import ignored_event_segments, scripted_schedule


TRUE_PARAMETERS = np.asarray([0.4, 0.22, 0.025, 0.045, 12.0, 0.1])


def _simultaneous_step() -> tuple[EventStep, dict[int, int]]:
    edges = EventEdges(
        edges=np.asarray(
            [
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        ),
        old_track_ids=(10, 11, 12, 13, 14),
        new_track_ids=(20, 21, 22, 23, 24, 25),
    )
    observed = {20: 2.0, 21: 3.0, 22: 5.0, 23: 7.0, 24: 8.0, 25: 4.0}
    return build_event_step(edges, initial_slots(edges.old_track_ids), observed, 7)


def _minimize_log_parameters(
    objective: Callable[[jax.Array], jax.Array], start: np.ndarray
) -> np.ndarray:
    value_and_grad = jax.jit(jax.value_and_grad(objective))

    def wrapped(log_parameters: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_grad(jnp.asarray(log_parameters))
        return float(value), np.asarray(gradient, dtype=np.float64)

    result = minimize(
        wrapped,
        np.log(start),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": 1_000},
    )
    if not result.success and (
        not np.isfinite(result.fun) or np.linalg.norm(result.jac) >= 2e-5
    ):
        raise AssertionError(result.message)
    return np.asarray(result.x)


def _laplace_interval(
    data: MaskedTrainingTransitions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    empirical = empirical_parameters(data)
    prior_scales = jnp.asarray([1.0, 1.0, 1.0, 1.0, 0.5])

    def posterior(log_parameters: jax.Array) -> jax.Array:
        parameters = jnp.exp(log_parameters)
        lognormal = 0.5 * jnp.sum(
            ((log_parameters[:5] - jnp.log(empirical[:5])) / prior_scales) ** 2
        )
        halfnormal = 0.5 * (parameters[5] / empirical[5]) ** 2 - log_parameters[5]
        return event_nll(parameters, data) + lognormal + halfnormal

    center = _minimize_log_parameters(posterior, empirical)
    hessian = np.asarray(jax.hessian(posterior)(jnp.asarray(center)))
    standard_error = np.sqrt(np.diag(np.linalg.inv(hessian)))
    factors = data.scaling.event_parameter_factors
    # Six 99.5% marginal bands form a simultaneous rectangle with at least 97%
    # posterior probability by the union bound.
    quantile = 2.807033768343811
    return (
        np.exp(center) * factors,
        np.exp(center - quantile * standard_error) * factors,
        np.exp(center + quantile * standard_error) * factors,
    )


def _density(
    step: EventStep,
    y: np.ndarray,
    area: np.ndarray,
    mask: np.ndarray,
    parameters: np.ndarray | jax.Array,
) -> jax.Array:
    return transition_log_density(
        jnp.asarray(y),
        jnp.asarray(area),
        jnp.asarray(mask),
        jnp.asarray(0.2),
        jnp.asarray(step.C),
        jnp.asarray(step.g),
        jnp.asarray(step.o),
        jnp.asarray(step.sigma_e_rows),
        jnp.asarray(parameters),
    )


class TestEventMapsAndLikelihood(unittest.TestCase):
    def test_long_time_msd_uses_shared_uninterrupted_segments(self) -> None:
        area = np.column_stack((np.arange(6.0), 10.0 + np.arange(6.0)))
        mask = np.ones_like(area, dtype=bool)
        slots = np.broadcast_to(np.asarray([10, 11]), area.shape).copy()
        tau = np.arange(6.0)
        segment_break = np.asarray([False, False, False, True, False, False])
        record = (area, mask, slots, tau, segment_break)

        observed, observed_times = _track_series([record], conservative=False)
        simulated, simulated_times = _track_series(
            [record, record], conservative=False
        )

        self.assertEqual([len(values) for values in observed], [3, 3, 3, 3])
        self.assertEqual([len(values) for values in observed_times], [3, 3, 3, 3])
        limit = _shared_msd_limit(observed, simulated)
        self.assertEqual(limit, 3)
        self.assertEqual(len(_pooled_msd(observed, limit)), limit)
        self.assertEqual(len(_pooled_msd(simulated, limit)), limit)
        self.assertEqual([len(values) for values in simulated_times], [3] * 8)

        gap_chain = EventChain(
            seed_id="gap",
            slot_ids=np.zeros((4, 1), dtype=np.int64),
            frame_indices=np.asarray([0, 1, 4, 5]),
            tau=np.asarray([0.0, 1.0, 4.0, 5.0]),
            areas=np.arange(4.0)[:, None],
            mask=np.ones((4, 1), dtype=bool),
            events=tuple(
                no_event_step(np.ones(1, dtype=bool), np.asarray([value]))
                for value in range(1, 4)
            ),
        )
        np.testing.assert_array_equal(
            _segment_breaks(gap_chain), [False, False, True, False]
        )

    def test_clean_oscillatory_likelihood_is_finite(self) -> None:
        tau = np.arange(30, dtype=np.float64) * 0.2
        areas = np.column_stack(
            (
                4.0 + 0.15 * np.sin(tau),
                3.5 + 0.10 * np.cos(0.7 * tau),
                4.5 - 0.15 * np.sin(tau) - 0.10 * np.cos(0.7 * tau),
            )
        )
        mask = np.ones_like(areas, dtype=bool)
        events = tuple(
            no_event_step(mask[index], areas[index + 1])
            for index in range(len(tau) - 1)
        )
        chain = EventChain(
            seed_id="continuous",
            slot_ids=np.broadcast_to(np.arange(3), areas.shape).copy(),
            frame_indices=np.arange(len(tau), dtype=np.int64),
            tau=tau,
            areas=areas,
            mask=mask,
            events=events,
        )
        segment = ignored_event_segments(chain)[0]
        clean_data = prepare_clean([segment], 1)
        parameters = jnp.asarray([0.8, 0.4, 0.5, 0.02, 0.03, 1.0, 0.05])
        self.assertTrue(np.isfinite(float(clean_nll(parameters, clean_data))))

    def test_event_density_conditions_on_run_slope(self) -> None:
        simulation = simulate_event_chain(
            scripted_schedule(cycles=1, continuous_steps=10).chain,
            TRUE_PARAMETERS,
            seed=22,
        )
        data = prepare_event([simulation.chain], 1)
        block = data.blocks[0]
        self.assertGreater(float(jnp.linalg.norm(block.slope[block.event])), 0.0)
        without_slope = data.__class__(
            scaling=data.scaling,
            blocks=(block._replace(slope=jnp.zeros_like(block.slope)),),
            rate_blocks=data.rate_blocks,
            conservative_slopes=data.conservative_slopes,
            slope_masks=data.slope_masks,
            sigma_b_squared=data.sigma_b_squared,
        )
        self.assertGreater(
            abs(
                float(event_nll(jnp.asarray(TRUE_PARAMETERS), data))
                - float(event_nll(jnp.asarray(TRUE_PARAMETERS), without_slope))
            ),
            1e-6,
        )

    def test_simultaneous_birth_death_survives_tracking(self) -> None:
        old_a = np.asarray([[1, 0], [0, 0]], dtype=bool)
        old_b = np.asarray([[0, 1], [0, 0]], dtype=bool)
        newborn = np.asarray([[0, 0], [1, 0]], dtype=bool)
        frames = [
            DetectionFrame(
                0,
                0,
                1.0,
                (DetectedBand(1.0, 0.0, old_a), DetectedBand(1.0, 1.0, old_b)),
            ),
            DetectionFrame(
                1,
                1,
                1.0,
                (DetectedBand(1.0, 0.0, old_a), DetectedBand(1.0, 1.0, newborn)),
            ),
        ]
        tracked = BandTracker(
            overlap_threshold=0.5, persistence_frames=1
        ).track(frames)

        self.assertEqual(tracked[1].events[0].code, EventCode.BIRTH)
        edges = tracked[1].edges
        self.assertIsNotNone(edges)
        assert edges is not None
        np.testing.assert_array_equal(
            edges.edges, np.asarray([[1, 0], [0, 0]], dtype=bool)
        )
        self.assertEqual(len(plan_chains(tracked, seed_id="simultaneous")), 1)

    def test_k_way_simultaneous_map_identities(self) -> None:
        step, slots = _simultaneous_step()
        before = np.asarray([10.0, 4.0, 6.0, 8.0, 2.0, 0.0, 0.0])
        after = step.H @ before + step.g

        daughter_slots = [slots[track_id] for track_id in (20, 21, 22)]
        self.assertAlmostEqual(after[daughter_slots].sum(), before[0])
        self.assertAlmostEqual(after[slots[23]], before[1] + before[2])
        self.assertAlmostEqual(after.sum(), before.sum() + 4.0 - before[4])
        self.assertAlmostEqual(after[slots[24]], before[3])
        self.assertAlmostEqual(after[slots[25]], 4.0)
        np.testing.assert_allclose(after[~step.mask_next], 0.0)
        self.assertEqual(int(step.sigma_e_rows.sum()), 2)
        self.assertFalse(step.o[slots[25]])

        birth_edges = EventEdges(
            np.asarray([[1, 0, 0], [0, 1, 0]], dtype=bool),
            (1, 2),
            (3, 4, 5),
        )
        birth, _ = build_event_step(
            birth_edges, initial_slots((1, 2)), {3: 4.0, 4: 6.0, 5: 3.0}, 4
        )
        birth_before = np.asarray([4.0, 6.0, 0.0, 0.0])
        self.assertAlmostEqual(
            float((birth.H @ birth_before + birth.g).sum()),
            float(birth_before.sum() + 3.0),
        )

        death_edges = EventEdges(
            np.asarray([[1, 0], [0, 1], [0, 0]], dtype=bool),
            (1, 2, 3),
            (4, 5),
        )
        death, _ = build_event_step(
            death_edges, initial_slots((1, 2, 3)), {4: 4.0, 5: 6.0}, 4
        )
        death_before = np.asarray([4.0, 6.0, 3.0, 0.0])
        self.assertAlmostEqual(
            float((death.H @ death_before + death.g).sum()),
            float(death_before.sum() - death_before[2]),
        )

    def test_observation_rows_are_marginalized_and_event_noisy(self) -> None:
        step, _ = _simultaneous_step()
        before = np.asarray([10.0, 4.0, 6.0, 8.0, 2.0, 0.0, 0.0])
        mask = np.asarray([1, 1, 1, 1, 1, 0, 0], dtype=float)
        parameters = jnp.asarray(TRUE_PARAMETERS)

        baseline = _density(step, step.y, before, mask, parameters)
        changed_silent = step.y.copy()
        changed_silent[~step.o] = 1.0e6
        silent = _density(step, changed_silent, before, mask, parameters)
        self.assertAlmostEqual(float(baseline), float(silent), places=11)

        changed_event_scale = parameters.at[5].set(0.3)
        widened = _density(step, step.y, before, mask, changed_event_scale)
        self.assertGreater(abs(float(baseline - widened)), 1e-3)

        active = mask / mask.sum()
        projection = np.diag(mask) - np.outer(mask, mask) / mask.sum()
        covariance = (
            TRUE_PARAMETERS[2] / TRUE_PARAMETERS[0] * 0.2**2 * projection
            + 2.0 * TRUE_PARAMETERS[3] * 0.2 * np.outer(active, active)
        )
        mean = mask * (
            before
            - TRUE_PARAMETERS[1]
            * (np.dot(mask, before) - TRUE_PARAMETERS[4])
            * 0.2
            * active
        )
        rows = np.flatnonzero(step.o)
        C = step.C[rows]
        explicit_covariance = C @ covariance @ C.T
        explicit_covariance += np.diag(
            step.sigma_e_rows[rows] * TRUE_PARAMETERS[5] ** 2
        )
        explicit_covariance += JITTER * np.eye(len(rows))
        residual = step.y[rows] - (C @ mean + step.g[rows])
        sign, logdet = np.linalg.slogdet(explicit_covariance)
        self.assertEqual(sign, 1.0)
        explicit = -0.5 * (
            len(rows) * np.log(2.0 * np.pi)
            + logdet
            + residual @ np.linalg.solve(explicit_covariance, residual)
        )
        self.assertAlmostEqual(float(baseline), float(explicit), places=8)

    def test_padding_contribution_is_parameter_independent(self) -> None:
        compact_mask = np.ones(2)
        compact_y = np.asarray([3.1, 4.2])
        compact_area = np.asarray([3.0, 4.0])
        compact = no_event_step(compact_mask.astype(bool), compact_y)
        padded_mask = np.asarray([1, 1, 0, 0, 0], dtype=float)
        padded_y = np.pad(compact_y, (0, 3))
        padded_area = np.pad(compact_area, (0, 3))
        padded = no_event_step(padded_mask.astype(bool), padded_y)

        differences = []
        for index in range(6):
            parameters = TRUE_PARAMETERS.copy()
            parameters[index] *= 1.7
            compact_density = _density(
                compact, compact.y, compact_area, compact_mask, parameters
            )
            padded_density = _density(
                padded, padded.y, padded_area, padded_mask, parameters
            )
            differences.append(float(padded_density - compact_density))
        np.testing.assert_allclose(differences, differences[0], atol=2e-12)

    def test_simulator_and_knot_grid_cover_scripted_events(self) -> None:
        scripted = scripted_schedule(cycles=2, continuous_steps=3)
        noiseless_parameters = TRUE_PARAMETERS.copy()
        noiseless_parameters[5] = 0.0
        simulation = simulate_event_chain(
            scripted.chain, noiseless_parameters, sigma_b_squared=0.0, seed=7
        )
        chain = simulation.chain
        np.testing.assert_allclose(chain.areas[~chain.mask], 0.0)
        self.assertEqual(
            {kind for kind in scripted.event_kinds if kind != "continuous"},
            {"split3_merge2", "merge3_split2", "birth5_death2", "birth2_death3"},
        )
        for index, kind in enumerate(scripted.event_kinds):
            if kind == "continuous":
                continue
            expected = chain.events[index].H @ simulation.pre_event_areas[index]
            expected += chain.events[index].g
            np.testing.assert_allclose(chain.areas[index + 1], expected, atol=1e-11)
            if "split" in kind:
                self.assertAlmostEqual(
                    float(chain.areas[index + 1].sum()),
                    float(simulation.pre_event_areas[index].sum()),
                    places=10,
                )

        flags = event_transitions(chain)
        for grid in build_knot_grids(chain, (1, 2, 3, 5)):
            for start, end in zip(grid.start, grid.end, strict=True):
                boundaries = np.flatnonzero(flags[int(start) : int(end)]) + start
                if boundaries.size:
                    self.assertEqual(boundaries.tolist(), [int(start)])
                    self.assertEqual(int(end), int(start) + 1)


class TestSyntheticRecoveryAndBias(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        simulations = [
            simulate_event_chain(
                scripted_schedule(
                    cycles=4, continuous_steps=50, seed_id=f"recovery_{seed}"
                ).chain,
                TRUE_PARAMETERS,
                seed=400 + seed,
            )
            for seed in range(4)
        ]
        cls.chains = [simulation.chain for simulation in simulations]
        cls.event_data = prepare_event(cls.chains, 1)
        cls.event_center, cls.event_low, cls.event_high = _laplace_interval(
            cls.event_data
        )

        segments = [
            segment
            for chain in cls.chains
            for segment in ignored_event_segments(chain)
        ]
        cls.clean_data = prepare_clean(segments, 1)
        empirical = empirical_parameters(cls.clean_data)

        def objective(log_parameters: jax.Array) -> jax.Array:
            return clean_nll(jnp.exp(log_parameters), cls.clean_data)

        clean_log = _minimize_log_parameters(objective, empirical)
        cls.clean_center = (
            np.exp(clean_log) * cls.clean_data.scaling.parameter_factors
        )

    def test_known_parameters_lie_in_laplace_credible_rectangle(self) -> None:
        np.testing.assert_array_less(self.event_low, TRUE_PARAMETERS)
        np.testing.assert_array_less(TRUE_PARAMETERS, self.event_high)
        relative_error = np.abs(self.event_center - TRUE_PARAMETERS) / TRUE_PARAMETERS
        self.assertLess(float(relative_error.max()), 0.12)

    def test_clean_fit_uses_oscillatory_parameterization(self) -> None:
        self.assertEqual(self.clean_center.shape, (7,))
        self.assertTrue(np.all(np.isfinite(self.clean_center)))


if __name__ == "__main__":
    unittest.main()
