from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hexatic.model_fitting.fitting import fields as fields_module
from hexatic.model_fitting.fitting import fit as fit_module
from hexatic.model_fitting.fitting.library import build_current_library
from hexatic.model_fitting.fitting.fit import FittingResult
from hexatic.model_fitting.fitting.regression import RegressionResult
from hexatic.model_fitting.fitting.write_report import write_model_report


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
        source = RegressionResult(
            names=("constant",),
            labels=("1",),
            coefficients=np.asarray([1.0]),
            prediction=scalar,
            residual=np.zeros_like(scalar),
            metrics={"r2": 1.0, "mae": 0.0, "normalized_mae": 0.0},
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
            source=source,
            density=density,
            polarization=polarization,
            source_contributions=scalar[..., None],
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
            "polar_cylindrical": np.asarray([[[0.0, 9.0, 3.0]]]),
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
            fields_module._load_gaussian_field_cache = lambda *args, **kwargs: {}
            fields_module._load_hexatic_order_frames = lambda *args: np.asarray([[5.0]])
            fields_module._load_neighbor_count_frames = lambda *args: np.asarray([[6.0]])
            fields_module._load_or_compute_chirality_frames = lambda *args: np.asarray([[[7.0]]])
            fields_module._gaussian_scalar_field_frames = lambda *args: {
                "rho_gaussian": np.asarray([[[1.0]]]),
                "hexatic_order_numerator": np.asarray([[[5.0]]]),
                "D_numerator": np.asarray([[[0.0]]]),
                "h_numerator": np.asarray([[[1.0]]]),
                "P_r_numerator": np.asarray([[[9.0]]]),
                "disclination_numerator": np.asarray([[[1.0]]]),
                "P_x_numerator": np.asarray([[[2.0]]]),
                "P_y_numerator": np.asarray([[[3.0]]]),
                "chirality_numerator": np.asarray([[[7.0]]]),
                "force_density_x_numerator": np.asarray([[[4.0]]]),
                "force_density_y_numerator": np.asarray([[[5.0]]]),
            }
            fields_module._save_gaussian_field_cache = lambda *args, **kwargs: None

            (
                rho,
                hexatic_order,
                D,
                h,
                P_r,
                disclination_particle_mask,
                P,
                chirality,
                force_density,
            ) = fields_module._load_smoothed_scalars(
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

        np.testing.assert_allclose(h, np.asarray([[[1.0]]]))
        np.testing.assert_allclose(P_r, np.asarray([[[9.0]]]))
        np.testing.assert_array_equal(
            disclination_particle_mask,
            np.asarray([[False]]),
        )
        np.testing.assert_allclose(chirality, np.asarray([[[7.0]]]))
        np.testing.assert_allclose(force_density, np.asarray([[[[4.0, 5.0]]]]))

    def test_effective_fit_mask_can_restrict_to_disclinations(self):
        fields = SimpleNamespace(
            mask=np.asarray([[[True, True], [False, True]]], dtype=bool),
            disclination_mask=np.asarray([[[True, False], [True, True]]], dtype=bool),
        )
        original = fit_module.DISCLINATIONS_ONLY
        try:
            fit_module.DISCLINATIONS_ONLY = False
            np.testing.assert_array_equal(
                fit_module._effective_fit_mask(fields),
                fields.mask,
            )
            fit_module.DISCLINATIONS_ONLY = True
            np.testing.assert_array_equal(
                fit_module._effective_fit_mask(fields),
                np.asarray([[[True, False], [False, True]]], dtype=bool),
            )
        finally:
            fit_module.DISCLINATIONS_ONLY = original

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

    def test_current_library_uses_p_not_rho_times_p_for_polar_terms(self):
        scalar = np.asarray([[[2.0]]], dtype=float)
        vector = np.asarray([[[[3.0, 5.0]]]], dtype=float)
        force = np.asarray([[[[7.0, 11.0]]]], dtype=float)
        fields = SimpleNamespace(
            mid_rho=scalar,
            mid_P=vector,
            mid_chirality=np.asarray([[[13.0]]], dtype=float),
            mid_D=np.asarray([[[17.0]]], dtype=float),
            mid_force_density=force,
            grad_rho=np.asarray([[[[19.0, 23.0]]]], dtype=float),
            grad_hexatic_order=np.asarray([[[[29.0, 31.0]]]], dtype=float),
            grad_D=np.asarray([[[[37.0, 41.0]]]], dtype=float),
        )

        library = build_current_library(fields)

        self.assertEqual(library.names[0], "P")
        self.assertEqual(library.names[1], "chiral_P_perp")
        self.assertEqual(library.names[3], "D_P")
        self.assertEqual(library.names[4], "D_chiral_P_perp")
        np.testing.assert_allclose(library.values[..., 0, :], vector)
        np.testing.assert_allclose(library.values[..., 1, :], [[[[ -65.0, 39.0 ]]]])
        np.testing.assert_allclose(library.values[..., 3, :], 17.0 * vector)

    def test_report_writes_three_stochastic_density_models_without_polarization(self):
        with self.subTest("report formats"):
            import tempfile

            scalar = np.ones((1, 1, 1), dtype=float)
            vector = np.ones((1, 1, 1, 2), dtype=float)
            density = RegressionResult(
                names=("P",),
                labels=("P",),
                coefficients=np.asarray([1.0]),
                prediction=scalar,
                residual=scalar,
                metrics={"correlation": 1.0, "r2": 1.0, "normalized_mae": 0.0},
                scales=np.asarray([1.0]),
                active=np.asarray([True]),
                rows_used=2,
            )
            source = RegressionResult(
                names=("constant",),
                labels=("1",),
                coefficients=np.asarray([0.0]),
                prediction=np.zeros_like(scalar),
                residual=np.zeros_like(scalar),
                metrics={
                    "correlation": 1.0,
                    "r2": 1.0,
                    "mae": 0.0,
                    "normalized_mae": 0.0,
                },
                scales=np.asarray([1.0]),
                active=np.asarray([True]),
                rows_used=2,
            )
            polarization = RegressionResult(
                names=("P",),
                labels=("P",),
                coefficients=np.asarray([2.0]),
                prediction=vector,
                residual=vector,
                metrics={
                    "correlation": 1.0,
                    "r2_x": 0.2,
                    "r2_y": 0.3,
                    "normalized_mae_x": 0.4,
                    "normalized_mae_y": 0.5,
                },
                scales=np.asarray([1.0]),
                active=np.asarray([True]),
                rows_used=2,
            )
            fields = SimpleNamespace(
                cylinder_radius=1.0,
                theta_edges=np.asarray([0.0, 2.0 * np.pi]),
                theta_centers=np.asarray([np.pi]),
                x_centers=np.asarray([0.5]),
                lx=1.0,
                material_current=vector,
                partial_t_rho=scalar,
                S_cross=np.zeros_like(scalar),
                partial_t_P=vector,
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
                fields=fields,
                mask=np.ones((1, 1, 1), dtype=bool),
                density_target=scalar,
                polarization_target=vector,
                source=source,
                density=density,
                polarization=polarization,
                source_contributions=scalar[..., None],
                density_contributions=scalar[..., None],
                polarization_contributions=vector[..., None, :],
                curl_residual=scalar,
            )
            with tempfile.TemporaryDirectory() as tmp:
                txt_path = write_model_report(result, tmp, case_id="case")
                md_path = Path(tmp) / "case_model_report.md"

                self.assertTrue(txt_path.exists())
                self.assertTrue(md_path.exists())
                txt = txt_path.read_text()
                md = md_path.read_text()
                self.assertIn("Three Full Stochastic Density Models", txt)
                self.assertIn("Model 1: J_fit residual split", txt)
                self.assertIn("Model 2: J_EOM residual split", txt)
                self.assertIn("Model 3: J_fit without force_density residual split", txt)
                self.assertNotIn("Polarization", txt)
                self.assertIn("## Three Full Stochastic Density Models", md)
                self.assertNotIn("Polarization", md)


if __name__ == "__main__":
    unittest.main()
