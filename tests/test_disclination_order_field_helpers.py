from __future__ import annotations

import numpy as np
import pytest

from hexatic.multiple_sim_analysis.disclination_order_fields.events.geometry import (
    annulus_mask,
    cylindrical_to_cartesian,
    cylinder_distances,
    minimum_image_x_delta,
)
from hexatic.multiple_sim_analysis.disclination_order_fields.events.validation import (
    validate_frame_particle_shape,
    validate_step_alignment,
)


def test_cylindrical_to_cartesian_preserves_x_and_reconstructs_cross_section():
    coords = np.asarray([[[2.0, 0.0, 3.0], [4.0, np.pi / 2.0, 5.0]]])

    cartesian = cylindrical_to_cartesian(coords)

    np.testing.assert_allclose(cartesian[0, :, 0], [2.0, 4.0])
    np.testing.assert_allclose(cartesian[0, :, 1], [0.0, 5.0])
    np.testing.assert_allclose(cartesian[0, :, 2], [3.0, 0.0], atol=1e-12)


def test_cylinder_distances_use_minimum_image_x():
    points = np.asarray([[9.0, 0.0, 0.0]])
    centers = np.asarray([[1.0, 0.0, 0.0]])

    np.testing.assert_allclose(minimum_image_x_delta(8.0, 10.0), -2.0)
    np.testing.assert_allclose(cylinder_distances(points, centers, 10.0), [[2.0]])


def test_annulus_mask_excludes_disclination_particles():
    particles = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    defects = np.asarray([[0.0, 0.0, 0.0]])
    disclinations = np.asarray([True, True, False, False])

    mask = annulus_mask(particles, defects, 10.0, 1.0, 3.0, disclinations)

    np.testing.assert_array_equal(mask, [False, False, True, False])


def test_validation_helpers_raise_clear_shape_errors():
    validate_frame_particle_shape("coords", np.zeros((2, 3, 3)), (2, 3))
    validate_step_alignment("fields", np.asarray([10, 20]), np.asarray([10, 20]))

    with pytest.raises(ValueError, match="frame/particle shape"):
        validate_frame_particle_shape("coords", np.zeros((2, 4, 3)), (2, 3))

    with pytest.raises(ValueError, match="steps do not align"):
        validate_step_alignment("fields", np.asarray([10, 30]), np.asarray([10, 20]))
