from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hexatic.model_fitting.fitting import fields as fields_module
from hexatic.model_fitting.fitting.fit import FittingResult
from hexatic.model_fitting.fitting.regression import RegressionResult


class HydrodynamicPlanTests(unittest.TestCase):
    def test_summary_prints_normalized_residual_percentages(self):
        scalar = np.ones((1, 1, 1), dtype=float)
        vector = np.ones((1, 1, 1, 2), dtype=float)
        density = RegressionResult(
            names=("term",),
            labels=("term",),
            coefficients=np.asarray([1.0]),
            prediction=scalar,
            residual=scalar,
            metrics={"r2": 0.5, "normalized_mae": 0.1234},
            scales=np.asarray([1.0]),
            active=np.asarray([True]),
            rows_used=1,
        )
        polarization = RegressionResult(
            names=("term",),
            labels=("term",),
            coefficients=np.asarray([1.0]),
            prediction=vector,
            residual=vector,
            metrics={
                "r2_x": 0.6,
                "r2_y": 0.7,
                "normalized_mae_x": 0.08,
                "normalized_mae_y": 0.09,
            },
            scales=np.asarray([1.0]),
            active=np.asarray([True]),
            rows_used=2,
        )
        result = FittingResult(
            transition_steps=np.asarray([[0, 1]]),
            dt=1.0,
            cylinder_radius=1.0,
            lx=1.0,
            x_edges=np.asarray([0.0, 1.0]),
            x_centers=np.asarray([0.5]),
            theta_edges=np.asarray([0.0, 2.0 * np.pi]),
            theta_centers=np.asarray([np.pi]),
            fields=None,
            mask=np.ones((1, 1, 1), dtype=bool),
            density_target=scalar,
            polarization_target=vector,
            density=density,
            polarization=polarization,
            density_contributions=scalar[..., None],
            polarization_contributions=vector[..., None, :],
            curl_residual=scalar,
        )

        summary = result.summary()

        self.assertIn("density_nmae=12.34%", summary)
        self.assertIn("polarization_nmae=(8%, 9%)", summary)

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

    def test_smoothed_chirality_refreshes_after_cache_miss(self):
        active = SimpleNamespace(
            steps=np.asarray([0]),
            coords=np.asarray([[[0.25, 0.25 * np.pi, 1.0]]], dtype=float),
            shell_mask=np.asarray([[True]]),
        )
        scalars = SimpleNamespace(
            x_edges=np.asarray([0.0, 1.0]),
            x_centers=np.asarray([0.5]),
            theta_edges=np.asarray([0.0, 2.0 * np.pi]),
            theta_centers=np.asarray([np.pi]),
            lx=1.0,
            cylinder_radius=1.0,
        )
        config = SimpleNamespace(
            gaussian_fields_cache_path=Path("unused.npz"),
            hexatic_order_table_path=Path("unused_hexatic.txt"),
            neighbor_count_table_path=Path("unused_neighbors.txt"),
        )
        active_arrays = {
            "polar_mean": np.asarray([[[2.0, 0.0, 0.0]]]),
            "polar_cylindrical": np.asarray([[[0.0, 0.0, 3.0]]]),
            "force_density_cylindrical": np.asarray([[[4.0, 0.0, 5.0]]]),
        }

        originals = (
            fields_module._load_gaussian_field_cache,
            fields_module._load_hexatic_order_frames,
            fields_module._load_neighbor_count_frames,
            fields_module._load_or_compute_chirality_frames,
            fields_module._gaussian_scalar_field_frames,
            fields_module._save_gaussian_field_cache,
        )
        try:
            fields_module._load_gaussian_field_cache = lambda *args: {}
            fields_module._load_hexatic_order_frames = lambda *args: np.asarray([[5.0]])
            fields_module._load_neighbor_count_frames = lambda *args: np.asarray([[6.0]])
            fields_module._load_or_compute_chirality_frames = lambda *args: np.asarray([[[7.0]]])
            fields_module._gaussian_scalar_field_frames = lambda *args: {
                "rho_gaussian": np.asarray([[[1.0]]]),
                "hexatic_order_numerator": np.asarray([[[5.0]]]),
                "D_numerator": np.asarray([[[0.0]]]),
                "P_x_numerator": np.asarray([[[2.0]]]),
                "P_y_numerator": np.asarray([[[3.0]]]),
                "chirality_numerator": np.asarray([[[7.0]]]),
                "force_density_x_numerator": np.asarray([[[4.0]]]),
                "force_density_y_numerator": np.asarray([[[5.0]]]),
            }
            fields_module._save_gaussian_field_cache = lambda *args: None

            rho, hexatic_order, D, P, chirality, force_density = fields_module._load_smoothed_scalars(
                config, active, scalars, active_arrays, pocket_radius=1.0,
            )
        finally:
            (
                fields_module._load_gaussian_field_cache,
                fields_module._load_hexatic_order_frames,
                fields_module._load_neighbor_count_frames,
                fields_module._load_or_compute_chirality_frames,
                fields_module._gaussian_scalar_field_frames,
                fields_module._save_gaussian_field_cache,
            ) = originals

        np.testing.assert_allclose(chirality, np.asarray([[[7.0]]]))
        np.testing.assert_allclose(force_density, np.asarray([[[[4.0, 5.0]]]]))

    def test_gaussian_crossing_source_matches_gaussian_density_change_for_entry(self):
        coords = np.asarray(
            [
                [[0.0, 0.0, 1.0], [0.0, np.pi, 1.0]],
                [[0.0, 0.0, 1.0], [0.0, np.pi, 1.0]],
            ],
            dtype=float,
        )
        shell_mask = np.asarray([[True, False], [False, True]])
        x_centers = np.asarray([0.0])
        theta_centers = np.asarray([0.0, np.pi])
        dt = 0.5

        rho = fields_module._gaussian_scalar_field_frames(
            coords,
            shell_mask,
            {"rho": np.ones(coords.shape[:2], dtype=float)},
            x_centers,
            theta_centers,
            lx=10.0,
            cylinder_radius=1.0,
            pocket_radius=1.0,
        )["rho"]
        source = fields_module._gaussian_crossing_source(
            coords,
            shell_mask,
            x_centers,
            theta_centers,
            lx=10.0,
            cylinder_radius=1.0,
            pocket_radius=1.0,
            dt=dt,
        )

        np.testing.assert_allclose(source, (rho[1:] - rho[:-1]) / dt)


if __name__ == "__main__":
    unittest.main()
