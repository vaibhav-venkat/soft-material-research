from __future__ import annotations

import unittest

import numpy as np

from hexatic.rho_fitting import _rho_fitting_core
from hexatic.rho_fitting.fit import mechanical_report_lines
from hexatic.rho_fitting.regression import StabilityResult


class RhoFittingMechanicsTests(unittest.TestCase):
    def test_j_p_uses_one_velocity_direction_outer_product(self) -> None:
        coords = np.array([[[0.0, 0.0, 1.0]]], dtype=float)
        directions = np.array([[[2.0, 3.0]]], dtype=float)
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
        expected = weight * np.array([[10.0, 15.0], [14.0, 21.0]])
        np.testing.assert_allclose(np.asarray(fields["J_P"])[0, 0, 0], expected)

    def test_target_shapes_are_validated(self) -> None:
        p = np.zeros((1, 2, 3, 2))
        j_rho = np.zeros_like(p)
        j_p = np.zeros((1, 2, 3, 2))
        j_q = np.zeros((1, 2, 3, 2, 2, 2))

        with self.assertRaises(ValueError):
            _rho_fitting_core.build_mechanical_targets(p, j_rho, j_p, j_q, 1.0, 100.0)

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
            warnings=(),
        )

        self.assertTrue(any("| term | 1 | 1 | 1.0000 | nan |" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
