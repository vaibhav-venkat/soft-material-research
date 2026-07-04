# Plan: py-pde validation for `rho_fitting`

## Goal

Add a new validation workflow under:

```text
hexatic/rho_fitting/pde_validation/
```

The workflow loads the existing fitted mechanical fields, evolves the coupled
`rho`, `P`, and `Q` equations with `py-pde`, and compares the simulated
`rho_fit(t)` against the measured `rho(t)` on the cylinder. The only required
visual output is a Plotly 3D animation showing fitted density and measured
density in the same time axis.

Use the Pixi `sap` environment for all commands. `py-pde` and `plotly` are
already in `pixi.toml`, so no dependency change is needed.

## Equations

Use the three equations together as one coupled system:

```text
partial_t rho = -div(U0 P_surface + F_rho / gamma)
partial_t P_i = -div(U0 F_P[:, i]) - P_i * (d - 1) / tau_r
partial_t Q_ij = -div(F_Q[:, i, j]) - (2d / tau_r) * Q_ij
```

with:

```text
d = 3
A_ij = Q_ij + rho * delta_ij / 3
P_surface = (P_0, P_1)
```

Important coupling rule: `A` is not loaded as an independent cached state during
rollout. It is recomputed inside every PDE RHS call from the currently evolved
`rho` and `Q`, so the `P` equation depends on the simulated `Q` dynamics.

`psi6_sq` is a hard auxiliary value/field because there is no fitted
`partial_t psi6` yet. It is not evolved. Default to a frozen field from the
initial validation frame, with a CLI option to use a scalar hard value if needed.

## Data Sources

Read from the existing fit cache by default:

```text
hexatic/rho_fitting/output/radius_15D_fit_result.npz
```

Required arrays:

- `rho`: measured density used for comparison, shape `(T, Nx, Ny)`.
- `P`: initial polar density, shape `(T, Nx, Ny, 3)`.
- `Q`: initial nematic tensor density, shape `(T, Nx, Ny, 3, 3)`.
- `psi6_sq`: fixed auxiliary field source.
- `Y_rho_names`, `Y_P_names`, `Y_Q_names`.
- `Y_rho_coefficients`, `Y_P_coefficients`, `Y_Q_coefficients`.
- `cheb_times`: physical output times.

Also load grid metadata from `CasePaths.active_fields_path`:

- `x_edges`, `x_centers`.
- `theta_edges`, `theta_centers`.
- `cylinder_radius`.

Use `surface_lengths` from `hexatic/rho_fitting/geometry.py` so the periodic
unwrapped grid remains `(x, y = R theta)`.

## New Package Layout

```text
hexatic/rho_fitting/pde_validation/
  __init__.py
  __main__.py
  config.py
  data.py
  state.py
  forces.py
  equations.py
  rollout.py
  metrics.py
  plotly_animation.py
  README.md
  output/
```

Responsibilities:

- `config.py`: dataclasses for case id, cache path, output path, frame window,
  `tau_r`, fixed `psi6` policy, solver settings, and Plotly settings.
- `data.py`: load fit cache and active-field grid metadata; validate shapes.
- `state.py`: pack/unpack `rho`, `P`, and `Q` between NumPy arrays and a
  `py-pde` `FieldCollection` of scalar fields.
- `forces.py`: evaluate fitted `F_rho`, `F_P`, and `F_Q` from the current PDE
  state and fit coefficients.
- `equations.py`: `RhoMomentPDE(PDEBase)` with the coupled RHS.
- `rollout.py`: run the solver from the initial frame and store `rho_fit(t)`.
- `metrics.py`: RMSE, relative RMSE, correlation, mass drift, and framewise
  error fields.
- `plotly_animation.py`: write interactive 3D HTML animations.
- `README.md`: short command examples and output descriptions.

## py-pde Representation

Use a 2D periodic Cartesian grid over the unwrapped cylinder surface:

```python
CartesianGrid([(0.0, lx), (0.0, ly)], [nx, ny], periodic=[True, True])
```

Represent the 13 state components as scalar fields in a `FieldCollection`:

```text
rho
P0, P1, P2
Q00, Q01, Q02, Q10, Q11, Q12, Q20, Q21, Q22
```

Do not use `VectorField` or `Tensor2Field` for `P`/`Q` themselves because
`py-pde` vector/tensor dimensions follow the 2D grid, while this model needs
3D orientational components. Use temporary 2D `VectorField`s only for surface
fluxes before taking divergence.

## Fitted Force Evaluation

`forces.py` should reconstruct the same fitted equations used by
`rho_fitting`, but evaluated on the current PDE state rather than on cached
time slices.

For `F_rho`, support the current names:

```text
grad_rho
grad_lap_rho
Q_dot_grad_rho
```

where:

```text
(Q_dot_grad_rho)_k = Q_k0 * grad_rho_0 + Q_k1 * grad_rho_1
k in {0, 1}
```

For `F_P`, support:

```text
A
rho_delta_psi6sq_A
```

Use only surface rows of `A` for the surface flux:

```text
F_P[:, i] = c_A * A_surface[:, i]
          + c_psi * rho * (psi6_sq_hard - mean_psi6_sq_hard) * A_surface[:, i]
```

Because `A = Q + rho I / 3` is recomputed from the PDE state, this term couples
the `P` rollout to the evolved `Q` rollout.

For `F_Q`, support:

```text
Ubar_P_dot_alpha_traceless
```

Build the traceless tensor exactly like the Rust helper:

```text
alpha_kij = P_i * delta_kj + P_j * delta_ki - (2/3) * P_k * delta_ij
```

Then:

```text
F_Q[k, i, j] = c_Q * Ubar * alpha_kij
```

Compute `Ubar` inside each RHS call from the current `F_P` and current `A` using
the same projection as the Rust fit code:

```text
Ubar = sum_k,i F_P[k, i] * A_surface[k, i] / sum_k,i A_surface[k, i]^2
```

Use `0.0` where the denominator is zero. This keeps the Q equation tied to the
current fitted P flux instead of replaying a cached target.

## PDE RHS

`RhoMomentPDE.evolution_rate(state, t)` should:

1. Unpack `rho`, `P`, and `Q`.
2. Recompute `A = Q + rho I / 3`.
3. Evaluate `F_rho`, `F_P`, and `F_Q`.
4. Build density current:

   ```text
   J_rho_fit = U0 * P_surface + F_rho / gamma
   ```

5. Return:

   ```text
   partial_t rho = -div(J_rho_fit)
   partial_t P_i = -div(U0 * F_P[:, i]) - 2 * P_i / tau_r
   partial_t Q_ij = -div(F_Q[:, i, j]) - 6 * Q_ij / tau_r
   ```

since `d = 3`.

Use periodic boundary conditions for all gradients, Laplacians, and divergences.
Prefer `py-pde` operators first. If component-wise flux divergence is awkward,
add a small local NumPy FFT helper that mirrors the existing `rho_fitting`
periodic derivative convention.

## Rollout Policy

Initial condition:

```text
rho_fit(t0) = rho_data[t0]
P_fit(t0) = P_data[t0]
Q_fit(t0) = Q_data[t0]
psi6_sq_hard = psi6_sq_data[t0] by default
```

Run one coupled simulation over `cheb_times`. Store the simulated state at the
same frame times as the data. Use internal substeps if the solver requires a
smaller stable step.

CLI sketch:

```text
pixi run python -m hexatic.rho_fitting.pde_validation \
  --case radius_15D \
  --fit-cache hexatic/rho_fitting/output/radius_15D_fit_result.npz \
  --output-dir hexatic/rho_fitting/pde_validation/output \
  --tau-r <value> \
  --overwrite
```

`tau_r` must be explicit unless there is already a trusted constant in
`hexatic/constants/`. Inspect constants before choosing a default.

Useful options:

- `--start-frame`, `--stop-frame`: validate a shorter window.
- `--dt`: internal solver step.
- `--solver`: start with explicit Euler or RK; keep the interface open for
  py-pde solver choices.
- `--psi6-policy initial-field|scalar`.
- `--psi6-value`: scalar hard value when `--psi6-policy scalar`.
- `--max-frames-plot`: downsample the animation if HTML gets too large.

## Outputs

Write outputs under:

```text
hexatic/rho_fitting/pde_validation/output/
```

Files:

- `{case_id}_pde_validation.npz`
  - `rho_fit`
  - `rho_data`
  - `P_fit_final`
  - `Q_fit_final`
  - `times`
  - `x_centers`
  - `theta_centers`
  - `radius`
  - error metrics
- `{case_id}_rho_fit_vs_data_3d.html`
- `{case_id}_rho_fit_vs_data_metrics.md`

The NPZ should never overwrite unless `--overwrite` is passed.

## Plotly 3D Animation

Convert the unwrapped grid back to cylinder coordinates:

```text
X = x
Y = R cos(theta)
Z = R sin(theta)
```

Render two animated cylinder surfaces:

- left: `rho_fit(t)`.
- right: measured `rho(t)`.

Use the same color scale and fixed color range for both surfaces. Add a third
optional animation mode for error:

```text
rho_fit(t) - rho_data(t)
```

Keep the primary deliverable as one HTML file with a time slider and play/pause
buttons.

## Validation Metrics

Compute:

- framewise RMSE of `rho_fit - rho_data`.
- relative RMSE using RMS of `rho_data`.
- Pearson correlation per frame.
- total mass in `rho_fit` and `rho_data`.
- mass drift from the initial frame.
- final-frame error heatmap data in the NPZ.

Report whether mass drift is small enough to trust the rollout before reading
the fit-vs-data error too strongly.

## Tests

Add focused tests in:

```text
tests/test_rho_fitting_pde_validation.py
```

Tests:

- state pack/unpack preserves shapes and component order.
- `A` is recomputed from the current `rho` and `Q`.
- fixed `psi6_sq` remains unchanged across RHS calls.
- zero coefficients and zero initial fields produce zero RHS.
- constant `rho`, `P`, and `Q` have zero divergence terms on a periodic grid.
- `F_rho_prediction` reconstructed from cached coefficients matches the cache
  on a small fixture or sampled frames.

Run:

```text
pixi run python -m compileall hexatic/rho_fitting/pde_validation
pixi run python -m unittest tests.test_rho_fitting_pde_validation
```

Do not run long simulations in tests.

## Implementation Order

1. Create `hexatic/rho_fitting/pde_validation/` with config, data loading, and
   state packing.
2. Implement fitted force reconstruction and verify it against cached fitted
   fields where possible.
3. Implement the coupled `py-pde` RHS with `A` recomputed from evolved `Q`.
4. Add a short rollout CLI for a small frame window.
5. Add metrics and NPZ output.
6. Add Plotly 3D animation for `rho_fit(t)` vs measured `rho(t)`.
7. Add tests and compile checks.
8. Run a short validation first, then the full `radius_15D` rollout only when
   the short run is stable.

## Non-Goals

- Do not rerun HOOMD simulations.
- Do not overwrite GSD or source NPZ simulation data.
- Do not evolve `psi6_sq` until a real `partial_t psi6` model exists.
- Do not treat cached `A` as the PDE state. It is derived from current `rho`
  and current `Q`.
