from __future__ import annotations

import unittest

import numpy as np
from numpy.polynomial import chebyshev as cheb

from hexatic.rho_fitting.basis import chebyshev_filter_and_derivative, temporal_power_spectrum


class RhoFittingChebyshevTests(unittest.TestCase):
    def test_diagnostic_coefficients_do_not_alternate_for_low_mode(self) -> None:
        frame_count = 100
        mode = 7
        scaled = np.linspace(-1.0, 1.0, frame_count)
        coeffs = np.zeros(frame_count)
        coeffs[mode] = 1.0
        values = cheb.chebval(scaled, coeffs)[:, np.newaxis, np.newaxis]

        result = chebyshev_filter_and_derivative(
            values,
            np.arange(frame_count, dtype=np.int64),
            1.0,
            cutoff=mode + 1,
        )
        recovered = result.coefficients[:, 0, 0]

        self.assertAlmostEqual(recovered[mode], 1.0, places=3)
        self.assertLess(np.max(np.abs(np.delete(recovered, mode))), 1.0e-3)

    def test_filter_and_derivative_match_low_order_polynomial(self) -> None:
        frame_count = 41
        steps = np.arange(frame_count, dtype=np.int64)
        scaled = np.linspace(-1.0, 1.0, frame_count)
        values = (2.0 + 0.5 * scaled + 0.25 * (2.0 * scaled**2 - 1.0))[:, np.newaxis]

        result = chebyshev_filter_and_derivative(values, steps, timestep=1.0, cutoff=3)

        np.testing.assert_allclose(result.filtered[:, 0], values[:, 0], atol=1.0e-12)
        expected_derivative = (0.5 + scaled) / (0.5 * (frame_count - 1))
        np.testing.assert_allclose(result.derivative[:, 0], expected_derivative, atol=1.0e-12)

    def test_single_frame_diagnostic_power(self) -> None:
        values = np.array([[[3.0, 4.0]]])

        result = chebyshev_filter_and_derivative(values, np.array([10]), timestep=1.0, cutoff=3)
        power = temporal_power_spectrum(result.coefficients)

        self.assertEqual(result.coefficients.shape, (1, 1, 2))
        np.testing.assert_allclose(result.derivative, np.zeros_like(values))
        np.testing.assert_allclose(power, np.array([25.0]))


if __name__ == "__main__":
    unittest.main()
