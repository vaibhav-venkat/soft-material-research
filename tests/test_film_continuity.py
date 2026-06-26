import numpy as np

from hexatic.density_analysis.film_continuity.binning import accumulate_counts_and_sums
from hexatic.density_analysis.film_continuity.config import (
    FilmContinuityConfig,
    FilmContinuityScalars,
)
from hexatic.density_analysis.film_continuity.continuity import residual
from hexatic.density_analysis.film_continuity.fields import (
    J_film_from_face_crossings,
    S_cross,
    neg_div_J_from_face_crossings,
    partial_t_rho,
    rho_film,
)
from hexatic.density_analysis.film_continuity.velocity import compute_velocities


def test_film_continuity_config_default_paths():
    config = FilmContinuityConfig()

    assert config.active_matter_path.name == "radius_15D_active_matter_fields.npz"
    assert config.cache_path.name == "radius_15D_film_continuity.npz"


def test_film_continuity_scalars_from_edges_area_shape():
    scalars = FilmContinuityScalars.from_edges(
        cylinder_radius=2.0,
        lx=4.0,
        dt=0.1,
        x_edges=[0.0, 1.0, 3.0],
        theta_edges=[0.0, 0.5, 1.0, 2.0],
    )

    assert scalars.area_bin.shape == (2, 3)


def test_compute_velocities_minimum_image_wraps_x_and_theta():
    coords = np.array(
        [
            [[1.9, 6.2, 0.0], [-1.8, 0.1, 0.0]],
            [[-1.9, 0.1, 0.0], [1.8, 6.2, 0.0]],
        ],
        dtype=float,
    )

    vx, vy = compute_velocities(
        coords,
        shell_mask=None,
        lx=4.0,
        cylinder_radius=2.0,
        dt=0.5,
    )

    np.testing.assert_allclose(vx[0], [0.4, -0.8])
    expected_dtheta = np.array([0.1 - 6.2 + 2.0 * np.pi, 6.2 - 0.1 - 2.0 * np.pi])
    np.testing.assert_allclose(vy[0], 2.0 * expected_dtheta / 0.5)


def test_source_only_transition_matches_density_change():
    x_edges = np.array([0.0, 1.0, 2.0])
    theta_edges = np.array([0.0, np.pi, 2.0 * np.pi])
    area_bin = np.diff(x_edges)[:, None] * np.diff(theta_edges)[None, :]
    coords = np.array(
        [
            [[0.25, 0.25 * np.pi, 0.0], [1.25, 1.25 * np.pi, 0.0]],
            [[0.25, 0.25 * np.pi, 0.0], [1.25, 1.25 * np.pi, 0.0]],
        ],
        dtype=float,
    )
    shell_mask = np.array([[True, False], [False, True]])
    vx, vy = compute_velocities(coords, shell_mask, lx=2.0, cylinder_radius=1.0, dt=1.0)

    binned = accumulate_counts_and_sums(
        coords,
        shell_mask,
        vx,
        vy,
        x_edges,
        theta_edges,
    )
    rho_frames = rho_film(binned.film_count_per_bin_frame, area_bin)
    partial_t = partial_t_rho(rho_frames, dt=1.0)
    source = S_cross(binned.n_in, binned.n_out, area_bin, dt=1.0)
    neg_div = neg_div_J_from_face_crossings(
        binned.x_face_crossings,
        binned.theta_face_crossings,
        area_bin,
        dt=1.0,
    )

    np.testing.assert_allclose(neg_div, 0.0)
    np.testing.assert_allclose(partial_t, source)
    np.testing.assert_allclose(residual(partial_t, neg_div, source), 0.0)


def test_finite_volume_crossing_flux_matches_lateral_density_change():
    x_edges = np.array([0.0, 1.0, 2.0])
    theta_edges = np.array([0.0, 1.0, 2.0])
    area_bin = np.ones((2, 2))
    coords = np.array(
        [
            [[0.25, 0.25, 0.0]],
            [[1.25, 1.25, 0.0]],
        ],
        dtype=float,
    )
    shell_mask = np.array([[True], [True]])
    vx, vy = compute_velocities(coords, shell_mask, lx=2.0, cylinder_radius=1.0, dt=1.0)

    binned = accumulate_counts_and_sums(
        coords,
        shell_mask,
        vx,
        vy,
        x_edges,
        theta_edges,
    )
    rho_frames = rho_film(binned.film_count_per_bin_frame, area_bin)
    partial_t = partial_t_rho(rho_frames, dt=1.0)
    source = S_cross(binned.n_in, binned.n_out, area_bin, dt=1.0)
    neg_div = neg_div_J_from_face_crossings(
        binned.x_face_crossings,
        binned.theta_face_crossings,
        area_bin,
        dt=1.0,
    )
    j_film = J_film_from_face_crossings(
        binned.x_face_crossings,
        binned.theta_face_crossings,
        x_edges,
        theta_edges,
        cylinder_radius=1.0,
        dt=1.0,
    )

    assert j_film.shape == (1, 2, 2, 2)
    np.testing.assert_allclose(source, 0.0)
    np.testing.assert_allclose(partial_t, neg_div)
    np.testing.assert_allclose(residual(partial_t, neg_div, source), 0.0)
