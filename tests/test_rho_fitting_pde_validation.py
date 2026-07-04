from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from hexatic.rho_fitting.pde_validation.cache import ValidationInputs
from hexatic.rho_fitting.pde_validation.model import (
    RhoFitPDE,
    ValidationOptions,
    interpolated_p_q,
    make_grid,
    pack_state,
    run_validation,
)
from hexatic.rho_fitting.pde_validation.operators import closure_fields


def synthetic_inputs() -> ValidationInputs:
    nx, ny = 4, 5
    times = np.array([0.0, 0.01, 0.02])
    x = np.linspace(0.0, 1.0, nx, endpoint=False)[:, None]
    y = np.linspace(0.0, 1.0, ny, endpoint=False)[None, :]
    rho0 = 1.0 + 0.01 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    rho = np.repeat(rho0[None, ...], times.size, axis=0)
    p = np.zeros((times.size, nx, ny, 3), dtype=np.float64)
    p[..., 0] = 0.02
    q = np.zeros((times.size, nx, ny, 3, 3), dtype=np.float64)
    q[..., 0, 0] = 0.01
    q[..., 1, 1] = -0.005
    q[..., 2, 2] = -0.005
    identity = np.eye(3, dtype=np.float64)
    a = q + rho[..., None, None] * identity / 3.0
    y_p = 0.2 * a[..., :2, :]
    psi6_sq = np.repeat((0.5 + 0.01 * x + 0.02 * y)[None, ...], times.size, axis=0)
    return ValidationInputs(
        cache_path=Path("synthetic.npz"),
        metadata={},
        rho=rho,
        p=p,
        q=q,
        a=a,
        psi6_sq=psi6_sq,
        y_p=y_p,
        times=times,
        y_rho_coefficients=np.array([0.1, 0.01, 0.2]),
        y_p_coefficients=np.array([0.3, -0.1, 0.05, 0.02, -0.01, 0.001]),
        y_q_coefficients=np.array([0.4, -0.03, 0.02, -0.01, 0.001]),
        lx=1.0,
        ly=1.0,
        radius=1.0,
        u0=2.0,
        gamma=1.0,
        tau_r=1.0,
    )


class RhoFittingPdeValidationTests(unittest.TestCase):
    def test_closure_shapes_match_targets(self) -> None:
        inputs = synthetic_inputs()
        closures = closure_fields(
            inputs.rho[0],
            inputs.p[0],
            inputs.q[0],
            inputs.psi6_sq[0],
            inputs.y_rho_coefficients,
            inputs.y_p_coefficients,
            inputs.y_q_coefficients,
            inputs.lx / inputs.rho.shape[1],
            inputs.ly / inputs.rho.shape[2],
        )

        self.assertEqual(closures.f_rho.shape, inputs.rho.shape[1:] + (2,))
        self.assertEqual(closures.f_p.shape, inputs.rho.shape[1:] + (2, 3))
        self.assertEqual(closures.f_q.shape, inputs.rho.shape[1:] + (2, 3, 3))
        self.assertEqual(closures.ubar.shape, inputs.rho.shape[1:])

    def test_rhs_preserves_state_shape(self) -> None:
        inputs = synthetic_inputs()
        grid = make_grid(inputs)
        state = pack_state(grid, inputs.rho[0], inputs.p[0], inputs.q[0])
        rate = RhoFitPDE(inputs).evolution_rate(state)

        self.assertEqual(rate.data.shape, state.data.shape)
        self.assertTrue(np.all(np.isfinite(rate.data)))

    def test_run_validation_returns_finite_rho_series(self) -> None:
        inputs = synthetic_inputs()
        result = run_validation(inputs)

        self.assertEqual(result.rho_fit.shape, inputs.rho.shape)
        self.assertEqual(result.rmse_t.shape, inputs.times.shape)
        self.assertTrue(np.all(np.isfinite(result.rho_fit)))
        self.assertTrue(np.all(np.isfinite(result.rmse_t)))

    def test_rho_only_validation_uses_hard_p_q_fields(self) -> None:
        inputs = synthetic_inputs()
        result = run_validation(inputs, ValidationOptions(mode="rho-only"))
        p_mid, q_mid = interpolated_p_q(inputs, 0.005)

        self.assertEqual(result.rho_fit.shape, inputs.rho.shape)
        np.testing.assert_allclose(p_mid, inputs.p[0])
        np.testing.assert_allclose(q_mid, inputs.q[0])
        self.assertTrue(np.all(np.isfinite(result.rho_fit)))

    def test_single_field_validation_modes_return_field_trajectories(self) -> None:
        inputs = synthetic_inputs()
        p_result = run_validation(inputs, ValidationOptions(mode="p-only"))
        q_result = run_validation(inputs, ValidationOptions(mode="q-only"))

        self.assertEqual(p_result.p_fit.shape, inputs.p.shape)
        self.assertEqual(q_result.q_fit.shape, inputs.q.shape)
        self.assertTrue(np.all(np.isfinite(p_result.p_fit)))
        self.assertTrue(np.all(np.isfinite(q_result.q_fit)))


if __name__ == "__main__":
    unittest.main()
