# Hydrodynamic Model Fitting Plan

## Goal

Refactor `hexatic/model_fitting/fitting/` to fit the requested hydrodynamic smoothed-field model for density and polarization.

The workflow should use one smoothed field representation, one Fourier derivative operator, one shared fit mask, and a simpler module structure.

Adding or removing a model term should mostly require changing one model-library location plus reporting labels.

## Constraints

- Work in `hexatic/model_fitting/`, especially `hexatic/model_fitting/fitting/`.
- Do not run long simulations or radius workflows.
- Use hydrodynamic smoothed fields for the main model.
- Do not mix finite-volume exact fields with FFT/smoothed fields in a regression.
- Use the same `(x, y = Rθ)` grid for all fitted quantities.
- Use the same Fourier operator for every derivative.
- Use one shared mask for density and polarization fits.
- Do not include force density in the density equation.
- Fit polarization as a vector PDE with shared scalar coefficients.
- Preserve radius-aware parameters.
- Avoid over-complicated classes, config, and type layers.
- Avoid over-modularizing into tiny files.

## Coordinate and Field Conventions

- `∇s = (∂x, ∂y)`.
- `y = Rθ`.
- `P = (P_x, P_y)`.
- `P_y` is the surface arclength component.
- `P_perp = (-P_y, P_x)`.
- `hexatic_order` means `|psi6|`.
- Fourier `ky` scaling must use the case-specific cylinder radius.

For every frame, compute or load Gaussian-smoothed fields on the same grid:

- `rho`.
- `P_x`.
- `P_y`.
- `chirality`.
- `D`.
- `hexatic_order`.
- `S_cross` as a transition/source field.

These fields should be the only representation used by the regressions.

Finite-volume machinery can still provide the measured crossing source, but finite-volume fluxes/divergences should not become features.

model_fitting/film_continuity computes some of these values already, along with the fitting folder, but everythign should be self-contained in `model_fitting/fitting`. You can look at the other folders for help, if needed.

## Derived Quantities

Compute these from the smoothed representation:

- `partial_t rho`.
- `partial_t P_x`.
- `partial_t P_y`.
- `grad rho`.
- `grad D`.
- `grad hexatic_order`.
- `div P`.
- `div(chirality * P_perp)`.
- `(P · grad)P`.
- `(P_perp · grad)P`.
- `laplacian P`.
- `laplacian P_perp`.

Time derivatives should come from adjacent smoothed frame fields.

RHS terms should be evaluated on transition-midpoint fields.

All derivatives should route through the same operator module.

This, along with the previous fields, should be saved into an appropriate *.npz within output/npz/[file].npz

## Density Fit

Target:

- `Y_rho = partial_t rho - S_cross_measured`.

Features, in stable order:

- `-div P`.
- `-div(chirality * P_perp)`.
- `laplacian rho`.
- `laplacian hexatic_order`.
- `laplacian D`.

The density coefficient vector should contain five scalar values.

Do not include force-density terms, nor keep local coefficient maps; flatten the mask.

## Polarization Fit

Targets:

- `partial_t P_x`.
- `partial_t P_y`.

Features:

- `P`.
- `chirality * P_perp`.
- `grad rho`.
- `grad D`.
- `grad hexatic_order`.
- `D * P`.
- `D * chirality * P_perp`.
- `|P|^2 * P`.
- `(P · grad)P`.
- `(P_perp · grad)P`.
- `laplacian P`.
- `laplacian P_perp`.

The polarization coefficient vector should contain twelve scalar values.

Stack both vector components into one regression so each term has one shared coefficient.

Do not fit separate `P_x` and `P_y` coefficient vectors.

## Shared Mask

Use one valid-sample mask for all fits, having a required valid density, valid polarization components.

If extra validity checks are needed, keep them common unless there is a strong reason not to.

## Regression Workflow

Use the same regression method for density and polarization:

- flatten valid space-time samples.
- remove invalid rows safely.
- RMS-normalize every feature column.
- fit ridge first.
- apply STLSQ sparsification after ridge.
- convert coefficients back to physical units.
- reconstruct predictions on the full transition grid.

Density should flatten to one scalar row per valid sample.

Polarization should flatten to two rows per valid sample with twelve shared columns.

Degenerate columns should not produce NaNs or crashes.

Empty masks should return zero coefficients and clear diagnostics.

## Reports and Plots

Density reporting should include:

- correlation between predicted and target `Y_rho`.
- `R²` for `Y_rho`.
- normalized MAE for `Y_rho`.
- coefficient table with term names.
- term contribution maps

Polarization reporting should include:

- correlation between predicted and target `partial_t P`.
- `R²` for `partial_t P_x`.
- `R²` for `partial_t P_y`.
- normalized MAE for `partial_t P_x`.
- normalized MAE for `partial_t P_y`.
- coefficient table with term names.
- curl-related residual structure.

Plots should include:

- true versus predicted density target.
- true versus predicted `partial_t P_x`.
- true versus predicted `partial_t P_y`.
- density residual maps.
- polarization residual maps.
- density term contribution maps.
- curl residual structure.

Report ridge and sparsification settings alongside coefficients. Use safe divisions for contribution maps and mark invalid ratios as NaN.

## Cache and Result Shape

Increment the cache version because the result schema changes substantially.

The final result should include:

- case id.
- geometry and grid metadata.
- steps and transition steps.
- smoothing settings.
- regression settings.
- shared mask.
- density target, prediction, residual, metrics, names, and coefficients.
- polarization target, prediction, residual, metrics, names, and coefficients.
- density contribution maps.
- curl residual structure.

## Proposed File Responsibilities

### `config.py`

Keep practical workflow settings: case id, optional paths, output directory, smoothing scale, density threshold, ridge alpha, sparsification method/parameters, and cache path helpers.

Remove old G-modifier and candidate-drop concepts from the main path.

### `operators.py`

Centralize spectral derivative logic: gradient, divergence, laplacian, vector laplacian, directional derivative, and shape validation.

No other module should duplicate Fourier derivative logic.

### `fields.py`

Own field construction: active-matter loading, chirality loading/computation, hexatic-order loading, `D` construction, measured `S_cross`, smoothed frame fields, transition fields, time derivatives, geometry metadata, and shared mask.

Avoid putting regression or model-term policy here.

### `library.py`

Define the density and polarization model libraries: term registry, stable ordering, labels, and feature-array construction from transition fields.

This should be the main file edited when changing the PDE library.

### `regression.py`

Own regression mechanics: flattening helpers if generic, finite-row filtering, RMS normalization, ridge fitting, STLSQsparsification, physical coefficient recovery, and degenerate-regression handling.

This module should not know the physics of density or polarization.

### `fit.py`

Orchestrate the complete fit: assemble fields, build libraries, call regression, reconstruct predictions, compute residuals/metrics, assemble result objects, and serialize/load caches.

Keep low-level operators, plotting, and field-loading details out of this file where practical.

### `plots.py`

Generate visual outputs from the result object: true-predicted views, residual maps, contribution maps, curl residual maps, and output naming.

### `run_fitting.py`

Keep the CLI simple: case selection, overwrite behavior, skip plotting, smoothing scale, mask thresholds, ridge strength, sparsification method, sparsification threshold, and iteration count.

Remove old runner constants tied to dropped candidate/G-modifier behavior.

## Implementation Phases

### Phase 1: Clean Old Model Assumptions

### Phase 2: Centralize Operators

### Phase 3: Build Hydrodynamic Fields

### Phase 4: Build Model Libraries

### Phase 5: Fit and Predict

### Phase 6: Diagnostics and Outputs

### Phase 7: Runner and Plots

### Phase 8: Cleanup **IMPORTANT**

- Reduce long-file complexity **IMPORTANT**.
- Remove stale imports and unused helpers.
- Keep comments focused on physics or numerical assumptions.
- Keep naming consistent across fields, terms, coefficients, cache keys, and plots.
- Run a Pixi syntax check after implementation.

## Assumptions

- Existing active-matter NPZ files are the preferred source for coordinates and polarization inputs.
- Existing chirality files should be loaded when available.
- Existing hexatic output tables are the preferred source for `hexatic_order`.
- Existing neighbor-count or disclination data can provide `D`.
- Existing film-continuity code can provide the measured crossing source.
- Cached Gaussian fields may be reused if metadata matches the new workflow.

## Edge Cases

- Empty mask.
- Missing required input files or arrays.
- Mismatched frame counts.
- Mismatched grid shapes.
- Invalid `dt`, `lx`, or cylinder radius.
- All-zero or non-finite feature columns.
- Near-zero target scale for normalized errors.
- Near-zero density target values in contribution maps.
- Stale cache files from the old fitting workflow.

Handle these explicitly rather than relying on downstream NumPy errors. Prefer simple `assert`'s over long error messages.

## Potential Issues

- Fourier periodicity in `x` may bias boundary-adjacent behavior.
- Gaussian smoothing can change target/source amplitudes.
- `S_cross` may remain noisy after smoothing.
- Polarization terms may be strongly collinear.
- Shared polarization coefficients may reveal coordinate artifacts.
- Contribution maps may be unstable where targets are small.
- Too much refactoring could obscure the physics; keep boundaries practical.

## Final Checklist for verification

- Density target is `partial_t rho - S_cross_measured`.
- Density library has five requested terms.
- Density library has no force-density term.
- Polarization library has exactly twelve requested terms.
- Polarization coefficients are shared across vector components.
- All derivatives use one Fourier operator module.
- All fitted fields come from one smoothed representation.
- One mask is used for both equations.
- Regression uses RMS normalization, ridge, sparsification, and physical coefficient recovery.
- Density reports include correlation, `R²`, normalized MAE, and contribution maps.
- Polarization reports include joint correlation, per-component `R²`, per-component normalized MAE, and curl residual structure.
- Plots include requested true-predicted views and residual maps.
- Cache schema is updated and stale caches are rejected.
- Adding/removing terms is localized and straightforward.
- The final fitting package is simpler than the current one.
