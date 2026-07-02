from __future__ import annotations

import unittest

import numpy as np

from hexatic.rho_fitting import _rho_fitting_core
from hexatic.rho_fitting.fit import mechanical_report_lines
from hexatic.rho_fitting.regression import StabilityResult


class RhoFittingMechanicsTests(unittest.TestCase):
    def test_j_p_uses_one_velocity_direction_outer_product(self) -> None:
        coords = np.array([[[0.0, 0.0, 1.0]]], dtype=float)
        directions = np.array([[[2.0, 3.0, 4.0]]], dtype=float)
        velocities = np.array([[[5.0, 7.0]]], dtype=float)
        mask = np.array([[True]])
        sigma = 2.0

        fields = _rho_fitting_core.build_mechanical_fields(
            coords,
            directions,
            velocities,
            mask,
            np.array([0.0]),
            np.array([0.0]),
            10.0,
            10.0,
            1.0,
            sigma,
            1.0,
            100.0,
        )

        weight = 1.0 / (2.0 * np.pi * sigma**2)
        expected = weight * np.array([[10.0, 15.0, 20.0], [14.0, 21.0, 28.0]])
        np.testing.assert_allclose(np.asarray(fields["J_P"])[0, 0, 0], expected)

    def test_q_uses_3d_traceless_orientation_moment(self) -> None:
        coords = np.array([[[0.0, 0.0, 1.0]]], dtype=float)
        directions = np.array([[[1.0, 0.0, 0.0]]], dtype=float)
        velocities = np.array([[[0.0, 0.0]]], dtype=float)
        mask = np.array([[True]])
        sigma = 2.0

        fields = _rho_fitting_core.build_mechanical_fields(
            coords,
            directions,
            velocities,
            mask,
            np.array([0.0]),
            np.array([0.0]),
            10.0,
            10.0,
            1.0,
            sigma,
            1.0,
            100.0,
        )

        weight = 1.0 / (2.0 * np.pi * sigma**2)
        expected_q = weight * np.diag([2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0])
        expected_a = weight * np.diag([1.0, 0.0, 0.0])
        np.testing.assert_allclose(np.asarray(fields["Q"])[0, 0, 0], expected_q)
        np.testing.assert_allclose(np.trace(np.asarray(fields["Q"])[0, 0, 0]), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.asarray(fields["A"])[0, 0, 0], expected_a)

    def test_target_shapes_are_validated(self) -> None:
        p = np.zeros((1, 2, 3, 3))
        j_rho = np.zeros((1, 2, 3, 2))
        j_p = np.zeros((1, 2, 3, 2, 3))
        j_q = np.zeros((1, 2, 3, 2, 2, 2))

        with self.assertRaises(ValueError):
            _rho_fitting_core.build_mechanical_targets(p, j_rho, j_p, j_q, 1.0, 100.0)

    def test_old_2d_p_shape_is_rejected(self) -> None:
        p = np.zeros((1, 2, 3, 2))
        j_rho = np.zeros_like(p)
        j_p = np.zeros((1, 2, 3, 2, 2))
        j_q = np.zeros((1, 2, 3, 2, 2, 2))

        with self.assertRaises(ValueError):
            _rho_fitting_core.build_mechanical_targets(p, j_rho, j_p, j_q, 1.0, 100.0)

    def test_regression_wrapper_returns_stability_result(self) -> None:
        from hexatic.rho_fitting.regression import stability_selection

        x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        fit = stability_selection(
            x,
            y,
            ("x",),
            ("x",),
            seed=7,
            tau_eps=1e-2,
            subsamples=8,
            importance_threshold=0.0,
            alpha=10.0,
            max_iter=20,
        )

        self.assertEqual(fit.coefficients.shape, (1,))
        self.assertEqual(fit.tau_index, 0)
        self.assertEqual(fit.tau_values.shape, (1,))
        self.assertEqual(fit.importance_path.shape, (1, 1))
        self.assertAlmostEqual(fit.coefficients[0], 2.0, places=8)

    def test_mechanical_report_formats_coefficients(self) -> None:
        fit = StabilityResult(
            names=("term",),
            labels=("term",),
            coefficients=np.array([1.0]),
            importance=np.array([1.0]),
            raw_correlations=np.array([np.nan]),
            importance_path=np.ones((1, 1)),
            tau_values=np.array([0.0]),
            active=np.array([True]),
            tau_index=0,
            y_pred=np.array([1.0]),
            rmse=0.0,
            r2=1.0,
        )

        lines = mechanical_report_lines(
            case_id="case",
            nd=1,
            frames=1,
            grid_shape=(1, 1),
            sigma=1.0,
            cheb_cutoff=1,
            fits={"Y_rho": fit},
        )

        self.assertTrue(any("| term | 1 | 1 | 1.0000 | nan |" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
