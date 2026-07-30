from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from safetensors.numpy import save_file

from hexatic.new_sims_analysis.isf import (
    IsfResult,
    component_seed,
    gaussian_reference,
    isotropic_3d_seed,
    lag_origin_counts,
    select_particle_ids,
    validate_results,
)
from hexatic.new_sims_analysis.loading import (
    CYLINDRICAL_COORDINATE_ORDER,
    _load_frame_series,
    _load_manifest,
)
from hexatic.new_sims_analysis.plot_correlation import _isf_geometry_route
from hexatic.new_sims_analysis import plot_correlation


class IsfNumericsTest(unittest.TestCase):
    def test_zero_lag_and_ballistic_estimators(self) -> None:
        frames = np.arange(6, dtype=np.float64)
        particle_count = 4
        velocity = np.asarray([0.5, -0.25, 0.75])
        com = frames[:, None] * velocity[None, :] / np.sqrt(particle_count)
        particles = np.repeat(
            frames[:, None, None] * velocity[None, None, :], 3, axis=1
        )
        k_values = np.asarray([0.2, 0.7])

        isotropic = isotropic_3d_seed(
            com, particles, k_values, particle_count, max_lag=4
        )
        speed = np.linalg.norm(velocity)
        expected = np.stack(
            [np.sinc(k_values * speed * lag / np.pi) for lag in range(5)]
        )
        np.testing.assert_allclose(isotropic.com.real, expected)
        np.testing.assert_allclose(isotropic.single.real, expected)
        np.testing.assert_allclose(isotropic.com.imag, 0.0)
        np.testing.assert_array_equal(
            isotropic.origin_counts, np.asarray([6, 5, 4, 3, 2])
        )

        component = component_seed(
            com[:, 0], particles[:, :, 0], k_values, particle_count, 4
        )
        expected_component = np.stack(
            [np.exp(-1j * k_values * velocity[0] * lag) for lag in range(5)]
        )
        np.testing.assert_allclose(component.com, expected_component)
        np.testing.assert_allclose(component.single, expected_component)
        np.testing.assert_allclose(component.com_msd, (velocity[0] * np.arange(5)) ** 2)

    def test_gaussian_reference_sampling_and_counts(self) -> None:
        msd = np.asarray([0.0, 2.0, 8.0])
        k = np.asarray([0.5, 1.0])
        np.testing.assert_allclose(
            gaussian_reference(msd, k, 2),
            np.exp(-msd[:, None] * k[None, :] ** 2 / 4.0),
        )
        first = select_particle_ids(100, 12)
        second = select_particle_ids(100, 12)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(len(np.unique(first)), 12)
        self.assertTrue(np.all(np.diff(first) > 0))
        np.testing.assert_array_equal(lag_origin_counts(5, 3), [5, 4, 3, 2])


class IsfValidationTest(unittest.TestCase):
    @staticmethod
    def _valid_result() -> tuple[IsfResult, np.ndarray]:
        k = np.asarray([0.1, 0.3])
        msd = np.asarray([0.0, 0.01, 0.05, 0.1])
        values = np.exp(-msd[:, None] * k[None, :] ** 2 / 2.0).astype(
            np.complex128
        )
        return (
            IsfResult(
                key="x",
                title="x",
                dimensions=1,
                com=values,
                single=values.copy(),
                gaussian=values.real.copy(),
                com_msd=msd,
                origin_counts=np.asarray([100, 99, 98, 97]),
            ),
            k,
        )

    def test_validation_passes(self) -> None:
        result, k = self._valid_result()
        diagnostics = validate_results([result], k)
        self.assertLess(diagnostics.maximum_small_k_msd_error, 0.01)

    def test_imaginary_leakage_fails(self) -> None:
        result, k = self._valid_result()
        single = result.single.copy()
        single[1, 0] = 0.8 + 0.2j
        changed = replace(result, single=single)
        with self.assertRaisesRegex(ValueError, "imaginary leakage"):
            validate_results([changed], k)

    def test_small_k_msd_fails(self) -> None:
        result, k = self._valid_result()
        changed = replace(result, com_msd=result.com_msd * 2.0)
        with self.assertRaisesRegex(ValueError, "ISF/MSD relative error"):
            validate_results([changed], k)


class CylinderLoadingTest(unittest.TestCase):
    def test_seams_unwrap_across_shards_with_fixed_radius(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = np.asarray(
                [
                    [[4.5, 3.0, 1.8], [-4.5, -3.0, 1.9]],
                    [[-4.5, -3.0, 1.7], [4.5, 3.0, 1.8]],
                    [[-3.5, -2.8, 1.6], [3.5, 2.8, 1.7]],
                ],
                dtype=np.float32,
            )
            polarization = np.zeros((3, 2, 3), dtype=np.float32)
            polarization[:, :, 0] = 1.0
            save_file({"lx": np.asarray(10.0)}, root / "static.safetensors")
            save_file(
                {
                    "coords": frames[:2],
                    "step": np.asarray([0, 1], dtype=np.int64),
                    "polarization": polarization[:2],
                },
                root / "frames_000000_000002.safetensors",
            )
            save_file(
                {
                    "coords": frames[2:],
                    "step": np.asarray([2], dtype=np.int64),
                    "polarization": polarization[2:],
                },
                root / "frames_000002_000003.safetensors",
            )
            manifest = {
                "schema": "hexatic.new_sims.analysis.v1",
                "complete": True,
                "coordinate_order": CYLINDRICAL_COORDINATE_ORDER,
                "case": {
                    "case_id": "synthetic_cylinder",
                    "dimensions": 3,
                    "radius": 2.0,
                },
                "shards": [
                    {
                        "file": "frames_000000_000002.safetensors",
                        "frame_start": 0,
                        "frame_stop": 2,
                    },
                    {
                        "file": "frames_000002_000003.safetensors",
                        "frame_start": 2,
                        "frame_stop": 3,
                    },
                ],
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            loaded_manifest = _load_manifest(root, allow_2d=False)
            com, _, _, tracks = _load_frame_series(
                root, loaded_manifest, 3, np.asarray([0, 1], dtype=np.int64)
            )
            np.testing.assert_allclose(tracks[:, 0, 0], [4.5, 5.5, 6.5])
            np.testing.assert_allclose(
                tracks[:, 0, 1], 2.0 * np.asarray([3.0, 2 * np.pi - 3.0, 2 * np.pi - 2.8]),
                rtol=1e-6,
            )
            np.testing.assert_allclose(com[:, 0], 0.0)
            np.testing.assert_allclose(com[:, 1], 0.0)


class GeometryRoutingTest(unittest.TestCase):
    def test_routes_and_skips(self) -> None:
        bulk = {"geometry_kind": "periodic_3d_bulk", "n_particles": 10}
        cylinder = {"geometry_kind": "ideal_abp_cylinder", "n_particles": 10}
        prism = {"geometry_kind": "x_walled_prism", "n_particles": 10}
        self.assertEqual(
            _isf_geometry_route(bulk, ["x", "y", "z"], disabled=False), "bulk"
        )
        self.assertEqual(
            _isf_geometry_route(
                cylinder, CYLINDRICAL_COORDINATE_ORDER, disabled=False
            ),
            "cylinder",
        )
        self.assertIsNone(
            _isf_geometry_route(prism, ["x", "y", "z"], disabled=False)
        )
        self.assertIsNone(
            _isf_geometry_route(
                {"geometry_kind": "periodic_2d_plane"},
                ["x", "y"],
                disabled=False,
            )
        )

    def test_tracer_fails_unless_disabled(self) -> None:
        tracer = {
            "geometry_kind": "single_active_tracer_cylinder",
            "n_particles": 10,
            "active_count": 1,
        }
        with self.assertRaisesRegex(ValueError, "tracer"):
            _isf_geometry_route(
                tracer, CYLINDRICAL_COORDINATE_ORDER, disabled=False
            )
        self.assertIsNone(
            _isf_geometry_route(
                tracer, CYLINDRICAL_COORDINATE_ORDER, disabled=True
            )
        )


class CliSmokeTest(unittest.TestCase):
    def test_bulk_cli_writes_isf_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            generator = np.random.default_rng(91)
            n_frames = 80
            n_particles = 16
            increments = generator.normal(scale=0.03, size=(n_frames - 1, 3))
            center = np.vstack([np.zeros(3), np.cumsum(increments, axis=0)])
            offsets = generator.normal(scale=0.2, size=(n_particles, 3))
            offsets -= np.mean(offsets, axis=0)
            base = center[:, None, :] + offsets[None, :, :]
            polarization = np.zeros((n_frames, n_particles, 3), dtype=np.float32)
            polarization[:, :, 0] = 1.0
            input_dirs: list[Path] = []
            for seed_index, sign in enumerate((1.0, -1.0)):
                input_dir = root / f"seed_{seed_index}"
                input_dir.mkdir()
                filename = "frames_000000_000080.safetensors"
                save_file(
                    {
                        "coords": np.asarray(sign * base, dtype=np.float32),
                        "step": np.arange(n_frames, dtype=np.int64),
                        "polarization": polarization,
                    },
                    input_dir / filename,
                )
                manifest = {
                    "schema": "hexatic.new_sims.analysis.v1",
                    "complete": True,
                    "coordinate_order": ["x", "y", "z"],
                    "case": {
                        "case_id": "synthetic_bulk",
                        "label": "synthetic bulk",
                        "geometry_kind": "periodic_3d_bulk",
                        "dimensions": 3,
                        "n_particles": n_particles,
                        "active_count": n_particles,
                        "timestep": 1.0,
                        "u0": 1.0,
                        "tau_r": 1.0,
                    },
                    "shards": [
                        {
                            "file": filename,
                            "frame_start": 0,
                            "frame_stop": n_frames,
                        }
                    ],
                }
                (input_dir / "manifest.json").write_text(json.dumps(manifest))
                input_dirs.append(input_dir)
            arguments = [
                "new-sims-plot",
                "--input-dir", str(input_dirs[0]),
                "--input-dir", str(input_dirs[1]),
                "--output-dir", str(output),
                "--frames", str(n_frames),
                "--max-lag", "20",
                "--tau-min", "1",
                "--laplace-r-points", "3",
                "--laplace-omega-points", "3",
                "--isf-particles", "8",
            ]
            with mock.patch("sys.argv", arguments):
                plot_correlation.main()
            for stem in ("isf_summary_synthetic_bulk", "isf_heatmap_synthetic_bulk"):
                self.assertTrue((output / f"{stem}.svg").is_file())
                self.assertTrue((output / f"{stem}.png").is_file())


if __name__ == "__main__":
    unittest.main()
