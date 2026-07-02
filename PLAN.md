# Plan: Burn-backed Mechanical Fits in `hexatic/rho_fitting`

## Goal

Extend `hexatic/rho_fitting/` so it fits:

- `Y_rho = gamma * (J_rho - U0 * P)`
- `Y_P = J_P / U0`
- `Y_Q = J_Q`

The workflow stays self-contained in `hexatic/rho_fitting/` plus its dedicated Rust crate `rust/rho_fitting_core`. Do not import from `hexatic/model_fitting/`. Constants and neutral shared utilities are fine.

## Decisions

- Use all particles for all new fields and fits. No shell-specific `S_cross`, shell entry/exit source, or disclination-only logic.
- Treat current `rho_fitting` `J_density` as `J_rho`.
- Use the `rho_fitting` current convention:
  - centered particle surface velocities for interior frames;
  - one-sided velocities at endpoints;
  - frame-time Gaussian coarse-graining;
  - Chebyshev filtering for time alignment.
- Use the unwrapped cylinder basis `(x, y)` with `y = R * theta`.
- Use density-weighted moments:
  - `P = sum_i p_i W_i`
  - `Q = sum_i (p_i p_i - I/d) W_i`
  - `A = Q + rho I/d`
- Use 2D surface tensors for this implementation:
  - `P: (..., 2)`
  - `Q, A, J_P: (..., 2, 2)`
  - `J_Q: (..., 2, 2, 2)`

## Paper Alignment

From Omar et al. 2023 Materials and Methods:

- Eq. 25: `partial_t rho + div J_rho = 0`.
- Eq. 26 motivates `gamma * (J_rho - U0 * P)` with `P` as polar density.
- Eq. 27b motivates fitting `J_P / U0` against density powers of `A = Q + rho I/d`.
- Eq. 28b motivates `J_Q` terms from `P dot alpha`, `P dot II`, and a force-like `F_rho I` term.

The implementation follows the paper's density-weighted moment convention, not normalized local averages.

## Ownership Boundary

Python code lives in:

```text
hexatic/rho_fitting/
  __main__.py
  config.py
  io.py
  geometry.py
  basis.py
  fields.py
  library.py
  regression.py
  fit.py
  report.py
  plots.py
  cache.py
```

Rust numerical code lives in:

```text
rust/rho_fitting_core/
```

Python responsibilities:

- load GSD/NPZ data;
- resolve config, case paths, constants, and output paths;
- call `_rho_fitting_core`;
- run or orchestrate stability selection;
- write cache, reports, and plots.

Rust/Burn responsibilities:

- compute particle surface velocities;
- compute Gaussian weights and coarse-grained fields;
- compute all tensor moments and fluxes;
- compute spatial derivatives and candidate libraries;
- build sampled regression rows for scalar, vector, rank-2, and rank-3 targets.

## Current Baseline

Existing `J_density` review:

- `hexatic/rho_fitting/fit.py` computes `J_density` by coarse-graining particle surface velocities.
- It maps angular displacement to `R * dtheta`, uses minimum-image wrapping, and uses saved steps times `settings.timestep`.
- It already uses `all_particles = np.ones_like(active.shell_mask)`, which matches the new global requirement.
- Keep `partial_t rho + div J_rho` as a diagnostic. Do not change the current convention unless this residual exposes a real problem.

Current Rust-core gap:

- `rho_fitting_core` currently uses `ndarray` plus explicit loops for density/vector coarse-graining.
- It has density-only candidate flux helpers.
- It does not compute `P`, `Q`, `A`, `J_P`, `J_Q`, tensor candidate libraries, or the actual STLSQ regression.

## Burn-backed Rust Core

Use Burn as the tensor engine inside `rho_fitting_core`.

Why Burn here:

- The next workflow is tensor-heavy, with repeated vector/rank-2/rank-3 operations.
- The same code should eventually run on CPU/GPU without rewriting the tensor logic.
- Python should receive NumPy arrays, but Python should not own the expensive tensor loops.

Cargo plan:

- Add Burn dependencies to `rust/rho_fitting_core/Cargo.toml`.
- Keep `pyo3`, `numpy`, and `ndarray` for Python boundary conversion.
- Keep `rustfft` unless Burn-backed derivative support is proven equivalent. Better: use both Burn and rustfft together.
- Start with a reliable CPU and GPU via `metal` backend .
- Here is ONE way to convert from the numpy directly into the `burn`, without redundant double-copying (best):
```rust
let shared = bytes::Bytes::copy_from_slice(bytemuck::cast_slice::<f32, u8>(slice));
let bytes = Bytes::from_shared(shared, AllocationProperty::Managed);
let data = TensorData::from_bytes(bytes, [rows, cols], DType::F32);
```
This is just an example
- There can be alternatives. Aim to use mostly Burn, and also use the GPU metal building.

Proposed Rust modules:

```text
src/
  lib.rs
  python.rs
  errors.rs
  arrays.rs
  burn_backend.rs
  geometry.rs
  particles.rs
  gaussian.rs
  moments.rs
  fluxes.rs
  derivatives.rs
  tensors.rs
  library.rs
  sampling.rs
  regression_rows.rs
```

Module responsibilities:

- `burn_backend.rs`: backend aliases, device selection, feature flags.
- `particles.rs`: surface velocities and tangent direction preparation.
- `gaussian.rs`: minimum-image pair distances and Gaussian weights.
- `moments.rs`: `rho`, `P`, `Q`, `A`.
- `fluxes.rs`: `J_rho`, `J_P`, `J_Q`.
- `derivatives.rs`: gradient, divergence, laplacian, `grad(lap rho)`, `grad(|grad rho|^2)`.
- `tensors.rs`: `alpha`, `II`, contractions, `F_rho I`.
- `library.rs`: `Y_rho`, `Y_P`, `Y_Q` candidate fields.
- `regression_rows.rs`: flatten target/library arrays into sampled rows with shared coefficients across components. This just calculates the rows, not the actual regression.

Python API from `_rho_fitting_core`:

```python
build_mechanical_fields(...) -> dict
build_mechanical_libraries(...) -> dict
sample_component_rows(...) -> dict
build_density_fluxes(...) -> dict  # keep during migration
```

`build_mechanical_fields` returns NumPy arrays:

- `rho`
- `P`
- `Q`
- `A`
- `J_rho`
- `J_P`
- `J_Q`
- `Y_rho`
- `Y_P`
- `Y_Q`
- `continuity_residual`

All Burn tensors stay inside Rust.

## Field Construction

1. Python loads active fields with `load_active_matter_npz`.
2. Python builds or loads tangent directions `p_i = (p_x, p_y)`:
   - prefer `direction_cylindrical`;
   - otherwise use `active_direction`;
   - otherwise derive from GSD orientation locally in `rho_fitting`.
3. Python passes contiguous arrays and scalar settings to `_rho_fitting_core`.
4. Rust/Burn computes surface velocities using the `rho_fitting` convention.
5. Rust/Burn computes Gaussian all-particle moments:
   - `rho = sum W_i`
   - `P = sum p_i W_i`
   - `Q = sum (p_i p_i - I/d) W_i`
   - `A = Q + rho I/d`
6. Rust/Burn computes Gaussian all-particle fluxes:
   - `J_rho = sum v_i W_i`
   - `J_P = sum v_i outer p_i W_i`
   - `J_Q = sum v_i outer (p_i p_i - I/d) W_i`
7. Chebyshev filtering may initially remain in Python through `basis.py`.
8. Targets are computed after filtering:
   - `Y_rho = gamma * (J_rho - U0 * P)`
   - `Y_P = J_P / U0`
   - `Y_Q = J_Q`

Do not implement coarse-graining in Python, do it in Rust.

## Candidate Libraries

### `Y_rho`

Target shape: `(frames, nx, ny, 2)`.

Candidates:

- `grad_rho`
- `rho * grad_rho`
- `rho^2 * grad_rho`
- `grad(lap_rho)`
- `lap_rho * grad_rho`
- `|grad_rho|^2 * grad_rho`
- `grad(|grad_rho|^2)`

Compute in `rho_fitting_core`, using Burn for tensor algebra and the existing derivative backend until replaced.

### `Y_P`

Target shape: `(frames, nx, ny, 2, 2)`.

Candidates:

- `A`
- `rho * A`
- `rho^2 * A`
- `rho^3 * A`

Compute in `rho_fitting_core`.

### `Y_Q`

Target shape: `(frames, nx, ny, 2, 2, 2)`.

Let `F_rho` default to the fitted prediction for `Y_rho`.

Candidates:

- `P dot alpha`
- `rho * P dot alpha`
- `rho^2 * P dot alpha`
- `P dot II`
- `F_rho I`


`alpha_ijkl = delta_ij delta_kl + delta_ik delta_jl + delta_il delta_jk` is a constant 2D tensor. It is not coarse-grained. `delta` is the Kronecker delta, not the Gaussian kernel.

`P dot II` is a tensor contraction, so `(P dot II)_kij = P_k delta_ij` where `delta_ij` is the identity matrix.

ALl other fields should follow a similar index ordering based on the shape.

## Regression

Keep Python stability selection first. Move only row construction into Rust.

Plan:

1. Rust builds candidate fields.
2. Rust samples valid `(frame, x, y)` rows.
3. Rust flattens component axes so every component shares the same candidate coefficients.
4. Python runs the existing STLSQ/stability-selection code on returned `X`, `y`, names, and labels. 
  5. It's important to leave this relatively unchanged unless there is an issue
5. Python sends `F_rho` back into Rust to build the `Y_Q` library.

Default: coefficients are shared across components, matching the existing vector-fit philosophy.

## Cache and Outputs

Write outputs under:

```text
hexatic/rho_fitting/output/
```

Cache:

- filtered `rho`, `P`, `Q`, `A`;
- filtered `J_rho`, `J_P`, `J_Q`;
- `Y_rho`, `Y_P`, `Y_Q`;
- candidate names/labels per target;
- sampled rows per target;
- coefficients, active masks, predictions, residuals, metrics;
- `partial_t rho + div J_rho` diagnostic;
- metadata: case id, radius, `lx`, `ly`, sigma, timestep, Chebyshev cutoff, `gamma`, `U0`, Burn backend, cache version.

Do not overwrite generated outputs unless `overwrite=True`.

## Tests

No formal tests, but frequent smaller parity tests with the data.

## Open Questions

1. What exact contraction should `P dot II` use? **answer**: It is a tensor contraction, so `(P dot II)_kij = P_k delta_ij` where `delta_ij` is the identity matrix.
2. Should `F_rho` in `Y_Q` remain the fitted `Y_rho` prediction? **answer** yes.
3. Are `gamma=1` and `U0=100` always correct for these saved datasets, or should they be loaded from per-case metadata? **answer** load them from the constants file present within the code.
4. Should first implementation target only `radius_15D` or all `rho_fitting` case paths? **answer** Accept a flag `--case` which specifies. I.e `--case radius_15D`

## Edge Cases and Potential Issues

Data and metadata:

- Missing radius metadata can silently corrupt `y = R theta`; fail early unless fallback is explicit.
- Constants may not match old generated data if simulations were run with different `gamma`, `U0`, or timestep. 
  - **This applies more for `Lx`, which already handled in the rho_fitting. Don't worry about the others**
- NPZ direction arrays may be normalized directions, means, or density-weighted quantities; verify before using as particle `p_i`.
- GSD/NPZ step mismatch must block orientation fallback.
- Old density-only caches must be invalidated with a cache-version bump.

Geometry:

- `theta` wrapping near `0`/`2*pi` must use minimum-image deltas.
- `x` wrapping must use the period from `x_edges`, not global constants.
- FFT derivatives assume periodicity in both `x` and `y`.
- Radial drift is ignored; all fields are projected to the cylinder surface.
- Radius changes alter `dy`, `ly`, and derivative scales; recompute per case.

Coarse-graining and tensors:

- Large `sigma` can wash out gradients and make libraries rank-deficient. For now, keep `sigma = 1`
- Small `sigma` can create sparse/noisy fields and unstable derivatives.
- Low-density cells can dominate errors; sample with a finite/density mask.
- `Q` should be symmetric and traceless per particle contribution; small numerical trace drift is possible after summation.
- `A = Q + rho I/d` should equal `sum p_i p_i W_i`; use this as a sanity check.
- Endpoint velocities are one-sided and may be noisier than centered interior frames.
- Do not mix midpoint displacement currents with frame-time `rho_fitting` fields.

Burn/Rust:

- Burn tensor axes can be semantically wrong even if shapes compile; add explicit shape comments and small hand-computed checks during implementation.
- Backend differences can create small numeric drift; tolerances should be explicit.
- Converting large tensors back to NumPy can dominate runtime/memory; return only needed arrays.
- Full dense libraries for rank-3 targets can be large; prefer sampled row construction in Rust when possible.
- Keep old `ndarray` density/current code as a reference until Burn parity is stable.
- Adding Burn may require Pixi/Cargo lockfile updates.

Libraries and regression:

- Candidate signs must match the plan; do not silently reuse old `neg_div_*` names.
- `P dot alpha`, `P dot II`, and `F_rho I` must all match `J_Q` shape.
- `F_rho` makes `Y_Q` depend on first-stage fit quality; report that dependency.
- Density powers of `A` may be highly correlated.
- Zero-scale columns should be inactive and reported.
- If sampled rows are all invalid or `tau_max` is zero, return a clear no-fit result.
- Sampling must be deterministic for a fixed seed.

Outputs:

- Never overwrite simulation data.
- Write cache files atomically.
- Reports should include assumptions and any disabled/uncertain terms.
- Plot labels should not imply shell-local physics; this workflow is all-particle/global.

## Small Guide

1. Add all depencies required like `burn` for Rust.
2. Add/Edit Rust modules for GPU and CPU support, particles, Gaussian weights, moments, fluxes, tensors, and libraries.
3. Use Burn as specified within the code for the tensors.
5. Add Python dataclasses and stricter types, that aren't verbose, however.
6. Add target metadata in `hexatic/rho_fitting/library.py`; numerical field construction happens in Rust.
7. Rework `hexatic/rho_fitting/fit.py` to fit `Y_rho`, then `Y_P`, then `Y_Q`.
  8. Keep the old regression functionality within the Python through `PySINDY`, not Rust. Rust will do the row computation only, not the regresion.
8. Add cache/report/plot support for the three targets.
9. Add `partial_t rho + div J_rho` diagnostic.
10. Add the same plots and report, which should report `R^2` and other statistics directly for `fit vs target`.

**Big idea**: Most of the functionality is already built-in, but you should still verify everything. If i'm missing something, you probably can pick it up. This plan is not exhaustive so don't strictly follow the guide, you should be able to pick up on the gaps (i.e i forgot to include something in the guide) or see existing functionality to understand what is necessary.
