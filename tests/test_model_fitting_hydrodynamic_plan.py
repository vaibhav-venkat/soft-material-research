from __future__ import annotations

import unittest

import numpy as np

from hexatic.model_fitting.fitting import fields as fields_module


class HydrodynamicPlanTests(unittest.TestCase):
    def test_chirality_grid_is_sampled_to_particles_before_gaussian_smoothing(self):
        grid = np.asarray(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ],
            dtype=float,
        )
        coords = np.asarray(
            [
                [
                    [0.25, 0.25 * np.pi, 1.0],
                    [0.75, 1.25 * np.pi, 1.0],
                ]
            ],
            dtype=float,
        )

        values = fields_module._particle_values_from_grid_field(
            grid,
            coords,
            np.asarray([0.0, 0.5, 1.0]),
            np.asarray([0.0, np.pi, 2.0 * np.pi]),
        )

        np.testing.assert_allclose(values, np.asarray([[1.0, 4.0]]))

    def test_shared_mask_rejects_invalid_density_and_polarization_rows(self):
        scalar = np.ones((1, 2, 2), dtype=float)
        vector = np.ones((1, 2, 2, 2), dtype=float)
        partial_t_rho = scalar.copy()
        partial_t_rho[0, 0, 0] = np.nan
        s_cross = scalar.copy()
        s_cross[0, 0, 1] = np.inf
        partial_t_P = vector.copy()
        partial_t_P[0, 1, 0, 1] = np.nan

        mask = fields_module._shared_valid_mask(
            density_threshold=0.0,
            mid_rho=scalar,
            scalar_fields=(partial_t_rho, s_cross),
            vector_fields=(partial_t_P,),
        )

        expected = np.asarray(
            [[[False, False], [False, True]]],
            dtype=bool,
        )
        np.testing.assert_array_equal(mask, expected)


if __name__ == "__main__":
    unittest.main()
