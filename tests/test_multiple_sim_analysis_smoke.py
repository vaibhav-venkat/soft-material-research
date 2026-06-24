import numpy as np

from hexatic.multiple_sim_analysis.best_fit import fit_exponential_radius_trend
from hexatic.multiple_sim_analysis.common import frame_indices
from hexatic.multiple_sim_analysis.density_profile import _density_values_from_shell_counts


def test_frame_indices_uses_stop_exclusive_overlap():
    assert np.array_equal(frame_indices(100), np.arange(50, 100))
    assert np.array_equal(frame_indices(75), np.arange(50, 75))
    assert frame_indices(25).size == 0


def test_exponential_fit_smoke():
    radii = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    values = 2.0 + 3.0 * np.exp(-radii / 2.0)
    fit = fit_exponential_radius_trend(radii, values)
    assert fit is not None
    assert np.isfinite(fit.y_inf)
    assert np.isfinite(fit.amplitude)
    assert fit.length_scale > 0.0


def test_density_profile_uses_count_density_difference():
    values = _density_values_from_shell_counts(
        [np.asarray([True, True, False, False, False])],
        shell_volume=2.0,
        core_volume=6.0,
    )
    assert values["shell"] == 1.0
    assert values["core"] == 0.5
    assert values["shell_minus_core"] == 0.5
