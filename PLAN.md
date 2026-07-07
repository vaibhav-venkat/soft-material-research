# Plan: Burn Cylindrical Coarse-Graining Rewrite

## Goal

Replace the current Gaussian, grid-chunk Burn coarse-graining path with a GPU-only finite-volume cylindrical pipeline:

- Use Burn CUDA on CUDA systems and Burn MLX/Metal on Apple systems.
- Remove the non-GPU rho-fitting coarse-graining implementation and all silent CPU fallback behavior.
- Deposit particles conservatively with TSC onto a cylindrical grid.
- Preserve rho-fitting/PDE contracts for 3D orientation moments and 2D surface fluxes.
- Keep the implementation small, direct, and easy to audit.

## Non-Goals

- Do not run HOOMD simulations or the full radius workflow.
- Do not refactor regression, reporting, plotting, or PDE validation unless a shape contract truly requires it.
- Do not treat flux fit quality as validation of PDE closure quality.
- Do not add broad helper layers or large class-style architecture.

## Target Structure

Prefer the smallest structure that stays readable. If `rust/rho_fitting_core/src/coarse_grain_burn.rs` becomes too large, split it into a module folder with only the pieces that earn their keep:

- `coarse_grain_burn/mod.rs`: public API, validation, backend dispatch.
- `coarse_grain_burn/backend.rs`: CUDA vs MLX device setup and panic handling.
- `coarse_grain_burn/grid.rs`: cylindrical grid edges, centers, volumes, periodic indexing.
- `coarse_grain_burn/tsc.rs`: TSC stencil and local per-particle normalization.
- `coarse_grain_burn/deposit.rs`: conservative mass and weighted-moment deposition.
- `coarse_grain_burn/smooth.rs`: optional conservative grid smoothing.
- `coarse_grain_burn/mechanical.rs`: mechanical fields, currents, tensor moments, output packing.
- `coarse_grain_burn/tensor.rs`: tiny Burn tensor/readback helpers only if still needed.

Keep module names flexible. Avoid both one giant file and a broad helper layer that obscures the math.

## Phase 1: Baseline And Contracts

1. Record the current public Rust entry points:
   - `coarse_grain_fields`
   - `build_mechanical_fields`

2. Lock down expected array shapes:
   - `P`: `(T, Nx, Ny, 3)`
   - `Q`, `A`: `(T, Nx, Ny, 3, 3)`
   - `J_rho`: `(T, Nx, Ny, 2)`
   - `J_P`: `(T, Nx, Ny, 2, 3)`
   - `J_Q`: `(T, Nx, Ny, 2, 3, 3)`

3. Confirm how the existing radial dimension is reduced or selected for the final surface fields.

4. Keep radius `15D` conventions intact, especially the fixed `Lx` assumption used by this dataset.

## Phase 2: Backend Migration

1. Replace `burn-wgpu` Metal usage with Burn MLX/Metal.

2. Keep CUDA support:
   - Prefer CUDA when a CUDA device is available.
   - Use MLX/Metal on Apple systems.
   - If the APIs diverge too much, keep the CUDA and MLX execution paths separate.

3. Remove coarse-graining CPU fallback:
   - GPU requested but unavailable should be a clear error.
   - GPU initialization failure should be surfaced, not silently rerouted to CPU.

4. Update feature/dependency planning:
   - Replace `gpu-metal` internals with MLX.
   - Keep feature names stable if possible to avoid Python-side churn.
   - Only rename features if Burn MLX forces it.

## Phase 3: Cylindrical Grid

1. Build finite-volume cells in `(x, r, theta)`.

2. Compute per-cell volume from:
   - `dx`
   - `dtheta`
   - radial shell volume contribution

3. Treat theta as periodic.

4. Treat x according to the existing periodic cylinder convention.

5. Mark invalid cells outside the physical shell before normalization or smoothing.

6. Avoid assuming equal cell volume; cylindrical cells do not all have the same volume.

## Phase 4: TSC Deposition

1. For each valid particle, find the local `3 x 3 x 3` TSC stencil.

2. Compute separable TSC weights in `x`, `r`, and `theta`.

3. Drop invalid cells from the stencil.

4. Normalize each particle's remaining stencil so total deposited mass is one.

5. Deposit mass first.

6. Convert mass to density only after deposition, using cell volume.

7. Preserve global particle count:
   - `sum(mass) == valid_particle_count`
   - `sum(rho * volume) == valid_particle_count`

## Phase 5: Moment And Tensor Deposition

1. Deposit weighted numerators for all particle fields instead of averaging tensor values directly.

2. Deposit mass alongside each numerator.

3. Divide numerator by mass only where mass is positive.

4. Decide and preserve the current downstream-compatible empty-cell policy before changing output values:
   - use `NaN` only where downstream code already accepts it
   - otherwise keep the existing finite sentinel/zero behavior

5. Construct:
   - density
   - `P`
   - `Q`
   - `A`
   - `psi6_sq`
   - `J_rho`
   - `J_P`
   - `J_Q`

6. Keep orientation components 3D and final surface flux directions 2D.

## Phase 6: Conservative Smoothing

1. Make smoothing optional.

2. Smooth mass fields, not density fields.

3. Smooth tensor/moment numerators and mass with the same conservative stencil.

4. Start with a small separable stencil such as `1, 2, 1` in each dimension.

5. Normalize smoothing weights over valid target cells for every source cell.

6. Recompute density and averages after smoothing.

7. Verify mass conservation before and after smoothing.

## Phase 7: Copy Reduction

1. Stop building full repeated grid coordinate vectors when a stencil-based particle loop is enough.

2. Prefer one upload per frame for particle arrays.

3. Keep intermediate fields on device until final output packing.

4. Avoid repeated per-component readbacks inside nested loops.

5. Use host readback only for final ndarray outputs or unavoidable API boundaries.

6. Keep the code readable even if a few small copies remain.

## Phase 8: Python Boundary

1. Keep Python function names and output dictionary keys stable.

2. Remove silent CPU fallback from Python wrappers once the Rust GPU path is mandatory.

3. Ensure failure messages explain whether the problem is:
   - missing GPU feature
   - missing CUDA/MLX device
   - Burn initialization failure
   - invalid input geometry or shapes

4. Avoid changing rho-fitting workflow code unless shape compatibility requires it.

## Phase 9: Validation

Use Pixi for all checks.

1. Rust compile checks:
   - `pixi run cargo check --manifest-path rust/rho_fitting_core/Cargo.toml --features gpu-metal`
   - `pixi run cargo check --manifest-path rust/rho_fitting_core/Cargo.toml --features gpu-cuda`
   - `pixi run cargo check --manifest-path rust/rho_fitting_core/Cargo.toml --features gpu`
   - Run the no-feature cargo check only if non-coarse-graining Rust APIs are intentionally kept buildable without GPU support.

2. Build check:
   - `pixi run rho-fitting-build`

3. Python syntax/type checks:
   - `pixi run python -m compileall hexatic`
   - `pixi run ty check hexatic/rho_fitting`

4. Data sanity checks on cached `radius_15D` inputs:
   - shape checks for all mechanical fields
   - finite/NaN policy checks for empty cells
   - mass conservation before and after smoothing
   - no accidental 3D surface flux output where PDE expects 2D

5. PDE validation:
   - Run only after cached field generation is sane.
   - Compare divergence metrics and rollout metrics, not only flux R2.

## Assumptions

- The active target is still the cached `radius_15D` rho-fitting data, not a new HOOMD run or the full radius workflow.
- For this dataset, `Lx` follows the fixed `15D` convention from the repository notes, not a generic radius helper.
- Particle inputs can be mapped to `(x, r, theta)` before deposition, and final rho-fitting outputs still collapse/select the radial coordinate into 2D surface fields.
- `x` and `theta` are periodic for stencil indexing; `r` is not periodic and is restricted to the physical shell.
- Feature names should stay stable if possible: `gpu-metal` should mean the Apple Metal path even if its implementation changes from `burn-wgpu` to `burn-mlx`.
- If both CUDA and MLX/Metal features are compiled, backend selection must be deterministic and documented.
- Optional smoothing is off unless explicitly enabled by the workflow or configuration.

## Edge Cases

- Particles exactly on `x` or `theta` boundaries must wrap consistently and not double-count boundary cells.
- Particles outside the valid radial shell, or with a TSC stencil whose valid-weight sum is zero, must be skipped with a diagnostic count rather than normalized by zero.
- TSC weights near non-periodic radial boundaries must be renormalized over valid cells so each valid particle deposits unit mass.
- Invalid grid cells must not receive mass, moment numerators, smoothing mass, or smoothing numerators.
- Cells with zero volume or non-finite volume are invalid and should fail validation before deposition.
- Empty cells must not divide moment/tensor numerators by zero; apply the documented empty-cell policy consistently to `P`, `Q`, `A`, `psi6_sq`, `J_rho`, `J_P`, and `J_Q`.
- Smoothing must conserve mass over valid cells even near radial shell boundaries and periodic seams.
- Surface flux outputs must never expose a radial flux component to PDE validation; only the `(x, R theta)` directions belong in `J_rho`, `J_P`, and `J_Q`.
- GPU initialization failures, missing compiled features, missing devices, and invalid input shapes should produce distinct errors rather than falling back to CPU.
- Near-zero-copy is a performance goal, but correctness and readable direct code take priority over contorted tensor plumbing.

## Risks

- Burn MLX API may not match CUDA closely enough for a single generic backend path.
- TSC deposition may be harder to vectorize than the current grid-chunk Gaussian method.
- Radial-grid handling must not leak an unwanted radial flux component into PDE validation.
- Conservative smoothing can preserve mass while still changing closure quality, so reports must distinguish numerical conservation from PDE usefulness.

## Follow-Ups

- Remove the old CPU coarse-graining path from rho-fitting rather than preserving it as a fallback.
- Leave smoothing default selection to the user after implementation; do not run comparison studies as part of this plan.
