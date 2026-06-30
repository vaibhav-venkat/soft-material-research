from __future__ import annotations

from pathlib import Path
import warnings
import unittest

import numpy as np

from hexatic.model_fitting import run_fitting
from hexatic.model_fitting.fitting import fit as fit_module
from hexatic.model_fitting.fitting.config import FittingConfig
from hexatic.model_fitting.fitting.fields import FittingFields, compute_scalar_modifiers


class ModelFittingRefactorTests(unittest.TestCase):
    def test_compute_fitting_empty_mask_returns_zero_fit_and_nan_diagnostics(self):
        J = np.ones((1, 2, 2, 2), dtype=float)
        vector = np.ones_like(J)
        scalar = np.ones((1, 2, 2), dtype=float)
        fields = FittingFields(
            transition_steps=np.array([[0, 1]]),
            dt=1.0,
            cylinder_radius=1.0,
            lx=1.0,
            x_edges=np.array([0.0, 0.5, 1.0]),
            x_centers=np.array([0.25, 0.75]),
            theta_edges=np.array([0.0, np.pi, 2.0 * np.pi]),
            theta_centers=np.array([0.5 * np.pi, 1.5 * np.pi]),
            J=J,
            frame_fields={"rho": scalar},
            mid_fields={
                "rho": scalar.copy(),
                "D": scalar.copy(),
                "hexatic_order": scalar.copy(),
                "P_density": vector.copy(),
            },
            counts=np.zeros((1, 2, 2), dtype=int),
        )
        original_loader = fit_module.load_or_compute_fields
        try:
            fit_module.load_or_compute_fields = lambda config: fields
            result = fit_module.compute_fitting(
                FittingConfig(candidate_names=("P_density",), smoothing_bins=1.0)
            )
        finally:
            fit_module.load_or_compute_fields = original_loader

        self.assertFalse(np.any(result.mask))
        np.testing.assert_allclose(result.fitted, 0.0)
        for coefficients in result.coef_map.values():
            np.testing.assert_allclose(coefficients, 0.0)
        self.assertTrue(np.isnan(result.residual_x_mean_abs))
        self.assertTrue(np.isnan(result.residual_y_mean_abs))

    def test_unknown_drop_fit_candidate_raises_before_computing(self):
        original_drop = run_fitting.DROP_FIT_CANDIDATE
        original_compute = run_fitting.compute_fitting
        try:
            run_fitting.DROP_FIT_CANDIDATE = ("not_a_candidate",)
            run_fitting.compute_fitting = lambda config: self.fail(
                "compute_fitting should not run"
            )

            with self.assertRaisesRegex(ValueError, "unknown candidates"):
                run_fitting.main(["--case", "drop_validation_test", "--no-plot"])
        finally:
            run_fitting.DROP_FIT_CANDIDATE = original_drop
            run_fitting.compute_fitting = original_compute

    def test_unknown_drop_g_raises_before_computing(self):
        original_drop = run_fitting.DROP_G
        original_compute = run_fitting.compute_fitting
        try:
            run_fitting.DROP_G = ("G_99",)
            run_fitting.compute_fitting = lambda config: self.fail(
                "compute_fitting should not run"
            )

            with self.assertRaisesRegex(ValueError, "unknown G modifier"):
                run_fitting.main(["--case", "drop_g_validation", "--no-plot"])
        finally:
            run_fitting.DROP_G = original_drop
            run_fitting.compute_fitting = original_compute

    def test_drop_g_reduces_coefficient_count(self):
        measured = np.ones((1, 1, 1, 2), dtype=float)
        mid_fields = {
            "rho": np.ones((1, 1, 1), dtype=float),
            "D": np.ones((1, 1, 1), dtype=float),
            "hexatic_order": np.ones((1, 1, 1), dtype=float),
            "P_density": np.ones((1, 1, 1, 2), dtype=float),
        }
        mask = np.ones((1, 1, 1), dtype=bool)
        num, modifiers = compute_scalar_modifiers(mid_fields, mask)

        # Drop G_2 and G_4 -> indices (0, 1, 3)
        kept = (0, 1, 3)
        coef_map = fit_module._global_coefficients(
            mid_fields,
            measured,
            mask,
            ("P_density",),
            g_modifier_indices=kept,
            G_modifiers=modifiers,
            threshold=0.0,
            max_iter=1,
        )

        self.assertEqual(set(coef_map), {f"P_density_m{idx}" for idx in kept})
        for coefficients in coef_map.values():
            self.assertEqual(coefficients.shape, (2,))

    def test_scalar_modifiers_all_false_mask_has_no_runtime_warning(self):
        scalar = np.ones((1, 2, 2), dtype=float)
        mask = np.zeros((1, 2, 2), dtype=bool)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            num, modifiers = compute_scalar_modifiers(
                {"rho": scalar, "D": scalar, "hexatic_order": scalar},
                mask,
            )

        runtime_warnings = [
            warning for warning in caught if issubclass(warning.category, RuntimeWarning)
        ]
        self.assertEqual(runtime_warnings, [])
        self.assertEqual(set(modifiers), {"G_0", "G_1", "G_2", "G_3", "G_4"})

    def test_model_fitting_film_continuity_numba_does_not_use_disk_cache(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / "hexatic" / "model_fitting" / "film_continuity").glob("*.py"):
            source = path.read_text()
            self.assertNotIn("cache=True", source, msg=f"stale Numba cache risk in {path}")

    def test_global_coefficients_nonempty_mask_returns_component_coefficients(self):
        measured = np.ones((1, 1, 1, 2), dtype=float)
        mid_fields = {
            "rho": np.ones((1, 1, 1), dtype=float),
            "D": np.ones((1, 1, 1), dtype=float),
            "hexatic_order": np.ones((1, 1, 1), dtype=float),
            "P_density": np.ones((1, 1, 1, 2), dtype=float),
        }
        mask = np.ones((1, 1, 1), dtype=bool)
        num, modifiers = compute_scalar_modifiers(mid_fields, mask)

        coef_map = fit_module._global_coefficients(
            mid_fields,
            measured,
            mask,
            ("P_density",),
            g_modifier_indices=(0, 1, 2, 3, 4),
            G_modifiers=modifiers,
            threshold=0.0,
            max_iter=1,
        )

        self.assertEqual(set(coef_map), {f"P_density_m{idx}" for idx in range(5)})
        for coefficients in coef_map.values():
            self.assertEqual(coefficients.shape, (2,))


if __name__ == "__main__":
    unittest.main()
