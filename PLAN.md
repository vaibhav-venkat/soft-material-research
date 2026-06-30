# Rho Fitting Workflow Plan

## 1. Goal And Boundaries

Build a new workflow to reproduce and fit `partial_t rho` from particle simulations using the hydrodynamic equation learning approach in Supekar et al. 2023 and its SI. The implementation should be similar in spirit to `hexatic/model_fitting`, but cleaner, smaller, and isolated.

Hard boundaries:

- New Python code goes under `hexatic/rho_fitting/`.
- New Rust code goes under root-level `rust/`.
- `hexatic/rho_fitting` must not import `hexatic.model_fitting`.
- The Rust extension is for this workflow only, even if some routines look reusable elsewhere.
- Python loads `.gsd` and `.npz` data, then passes NumPy arrays into Rust through PyO3.
- Rust does the expensive raw numerical work: Gaussian coarse graining, spatial spectral derivatives, random sample generation, and sampled candidate-library construction.
- Python does orchestration, IO, Chebyshev time processing, STLSQ/stability selection, caching, plotting, and reports.

Primary target equation:

```text
partial_t rho(t, x, y) = sum_l a_l C_l(rho, p)
```

The paper also fits polarization dynamics. Implement that in the same workflow because the candidate-library and validation path depend on it:

```text
partial_t p(t, x, y) = sum_l b_l Cvec_l(rho, p)
```

For cylinder data, always use:

```text
x = x
y = R * theta
```

where `R` is the case-specific cylinder radius and `theta` is periodic over `[0, 2*pi)`.

## 2. Paper Method To Preserve

Use the Supekar et al. method as the source of truth when it conflicts with local habits.

Coarse graining follows the paper's Eqs. 2a and 2b:

```text
rho[t, X, Y] = sum_i K(grid[X, Y] - x_i[t])
p[t, X, Y, :] = sum_i K(grid[X, Y] - x_i[t]) * p_i[t]
```

Use a normalized 2D Gaussian kernel on the unwrapped cylinder surface:

```text
K(delta_x, delta_y) = (2*pi*sigma^2)^(-1)
                      * exp(-(delta_x^2 + delta_y^2) / (2*sigma^2))
```

Important distinction:

- `rho` is particle number density.
- `p` is polarization density, not normalized mean polarization.
- Do not regress on `polar_mean`, `polar_cylindrical`, or any `p/rho` representation unless it is clearly labeled as a diagnostic.

Use the paper's spectral representation from Eqs. 3a and 3b:

```text
rho(t, x) = sum_{n, q} rho_hat[n, q] * T_n(t) * F_q(x)
p(t, x)   = sum_{n, q} p_hat[n, q]   * T_n(t) * F_q(x)
```

where:

- `T_n(t)` is the Chebyshev polynomial in time.
- `F_q(x)` is the Fourier spatial mode.
- Spatial derivatives come from Fourier modes.
- Time derivatives come from Chebyshev differentiation.
- Fast temporal modes above the configured Chebyshev cutoff are filtered before derivative evaluation.
- Cutoff:
  - Choose an recommended cutoff `n` at first. But also add to the plots, very important, a tepmoral spectral power verus sChebyshev mode number `n`:
  ```python
  power_n = (
      np.sum(np.abs(rho_hat)**2, axis=(1, 2))
      + np.sum(np.abs(px_hat)**2, axis=(1, 2))
      + np.sum(np.abs(py_hat)**2, axis=(1, 2))
  )
  ```

Build regression systems as in SI Eq. A2:

```text
U_t_density    = Theta_density * a
U_t_polarized  = Theta_polarization * b
```

For polarization, stack component rows:

```text
[partial_t p_x rows]
[partial_t p_y rows]
```

and stack the corresponding x/y candidate rows so both components use the same coefficient vector `b`. This enforces rotationally invariant shared coefficients.

Use the paper's sampling and stability selection:

- Randomly sample `N_d = 5e5` valid time-space points.
- Use the same sampled rows/mask for target fields and candidate fields.
- Use 40 `tau` values on a log10 path from `tau_max` to `eps * tau_max`.
- Use `eps = 1e-2`.
- For each `tau`, use random 50% subsamples.
- The SI used 200 subsamples; make that the default but configurable.
- A term's importance score is the fraction of subsamples where STLSQ keeps it.
- Keep terms with importance score above `0.6`.
- Refit kept terms on full, unnormalized data.

STLSQ settings:

```text
alpha = 1e-6
max_iter = 20
normalize_columns = True
fit_intercept = False
```

Compute `tau_max` from normalized columns:

```python
coef0, *_ = np.linalg.lstsq(Xn, yn, rcond=None)
tau_max = np.max(np.abs(coef0)) * 1.01
```

Before the final refit, preprocessing should match the SI:

- Candidate matrix columns are centered and scaled for the threshold path.
- Target vector is centered for the threshold path.
- Final coefficients are refit on full unnormalized data without an intercept.

Also, within Rust, still create a placeholder STLSQ function like (with least_squres from ndarray-linalg):
```
function STLSQ(Θ, U_t, τ, max_iter):

    active_terms = [true, true, ..., true]
    ξ = zeros(number_of_terms)

    repeat up to max_iter times:

        Θ_active = columns of Θ where active_terms is true

        ξ_active = least_squares(Θ_active, U_t)

        ξ_new = zeros(number_of_terms)
        put ξ_active back into ξ_new at active term locations

        new_active_terms = abs(ξ_new) >= τ

        if new_active_terms == active_terms:
            ξ = ξ_new
            stop

        active_terms = new_active_terms
        ξ = ξ_new

    return ξ, active_terms
```
And run the same sort of thing as python did, but just replace the pysindy.STLSQ with the new function. It won't be used, yet, but allow it to be used in the future very easily.

## 3. Data Flow

The main workflow should be:
1. Python resolves case paths, parameters, and configs.

2. Python loads GSD/NPZ metadata and raw particle arrays.
   - `.gsd` for particle positions and orientations when needed.
   - `.npz` for `coords`, `shell_mask`, `steps`, `x_edges`, `theta_edges`, and radius metadata where available.
3. Python constructs particle tangential vectors `p_i`
   - Use active orientation from GSD quaternions or equivalent NPZ `active_direction`.
   - coords = (x, theta, r)
   - p_particles = (p_x, p_y)
   Define y = R theta for grid operations.

4. Python passes compact contiguous arrays to Rust:
     coords: (frames, particles, 3)
     p_particles: (frames, particles, 2)
     shell_mask: (frames, particles)
     x_centers
     y_centers
     Lx, Ly, radius, sigma

5. Rust computes `rho` and `P_density` on the `(x, y = R * theta)` grid:
```
rho[t, x, y]
P_x[t, x, y]
P_y[t, x, y]
```
6. Python performs Chebyshev time projection/filtering and `partial_t` evaluation.
7. Rust performs Fourier-space spatial derivatives and requested composite fields.
8. Rust samples `N_d` valid rows and builds candidate matrices from Python term names.
9. Python runs density and polarization STLSQ stability selection.
10. Python writes cache files, reports, plots, and validation diagnostics.
Expected core array shapes:

```text
rho                  (T, Nx, Ny)
P_density            (T, Nx, Ny, 2)
partial_t_rho        (T_eval, Nx, Ny)
partial_t_P_density  (T_eval, Nx, Ny, 2)
sample_indices       (N_d, 3) as (t_index, x_index, y_index)
X_density            (N_d, n_density_terms)
y_density            (N_d,)
X_polarization       (2 * N_d, n_polarization_terms)
y_polarization       (2 * N_d,)
```

`T_eval` should match the time grid on which the spectrally filtered fields are evaluated. If the implementation evaluates derivatives at original frame times, use `T_eval = T`. If it evaluates at transition midpoints, document and cache that convention explicitly.

## 4. Python Package Structure

Create this compact package:

```text
hexatic/rho_fitting/
  __init__.py
  __main__.py
  config.py
  io.py
  geometry.py
  basis.py
  library.py
  regression.py
  fit.py
  plots.py
  report.py
  cache.py
  output/
```

Responsibilities:

- `__main__.py`: CLI entrypoint, argument parsing, calls `fit.run`.
- `config.py`: dataclasses for case paths and numerical settings.
- `io.py`: `.gsd`/`.npz` loading and validation. No `model_fitting` imports.
- `geometry.py`: quaternion orientation conversion, cylindrical basis conversion, `x/theta -> x/y`, periodic wrapping helpers.
- `basis.py`: Chebyshev projection, temporal filtering, `partial_t` evaluation, cutoff diagnostics.
- `library.py`: term registries, term validation, labels, and mapping from public term names to Rust term names.
- `regression.py`: STLSQ wrapper, stability selection, normalization/refit logic, importance scores.
- `fit.py`: orchestration dataclasses and end-to-end workflow.
- `plots.py`: predicted-vs-true, importance-score curves, coefficient bars, kymographs, heatmaps, snapshots, vector fields.
- `report.py`: text/markdown summary with chosen terms, coefficients, settings, metrics, and caveats.
- `cache.py`: `.npz` cache read/write and cache metadata validation.

Keep each file small. If a file grows because of a real boundary, split only that boundary.

## 5. Rust Folder Structure

Use one root-level Rust crate for now:

```text
rust/
  rho_fitting_core/
    Cargo.toml
    pyproject.toml
    README.md
    src/
      lib.rs
      python.rs
      arrays.rs
      geometry.rs
      coarse_grain.rs
      fft_ops.rs
      sampling.rs
      library.rs
      errors.rs
      regression.rs
```

Minimal module responsibilities:

- `lib.rs`: crate module declarations and re-exports only.
- `python.rs`: PyO3 module definition and Python-callable wrappers.
- `arrays.rs`: NumPy/PyArray to ndarray conversion, shape checks, contiguous-copy handling.
- `geometry.rs`: minimum image wrapping, `theta` periodic deltas, cylinder surface coordinates, Gaussian kernel utilities.
- `coarse_grain.rs`: `rho` and `P_density` Gaussian sums.
- `fft_ops.rs`: Fourier derivatives with `rustfft`: gradient, divergence, laplacian, bilaplacian, vector laplacian, `grad_div`.
- `sampling.rs`: deterministic valid-mask sampling, sample-index packing, seed handling.
- `library.rs`: sampled candidate term assembly from requested term names.
- `errors.rs`: typed errors converted to Python `ValueError`/`RuntimeError`.
- `regression.rs`: creates STLSQ function and the algorithm method, not used currently.


Rust dependencies in `Cargo.toml`:

```text
pyo3
numpy
ndarray
num-complex
rustfft
rand
rand_chacha
thiserror
```

Optional only if useful after profiling:

```text
rayon
```

`pyproject.toml` should use maturin and name the Python extension so it imports as:

```python
from hexatic.rho_fitting import _rho_fitting_core
```

Pixi changes:

- Add `rust`, `cargo`, and `maturin` through conda-forge if available in this environment.
- Keep PySINDy as the Python STLSQ dependency.
- Add a task if useful:

```toml
[tasks]
rho-fitting-build = "maturin develop --manifest-path rust/rho_fitting_core/Cargo.toml"
rho-fitting-smoke = "python -m hexatic.rho_fitting --case radius_15D --nd 1000 --no-plot"
```

## 6. Rust Public API

Expose only a small API to Python:

```python
coarse_grain_fields(
    ...
) -> dict
```

Returns:

```text
rho        (T, Nx, Ny)
P_density  (T, Nx, Ny, 2)
```

```python
spatial_derivatives(
    ...
    
) -> dict
```

Returns only requested derivative/composite arrays to avoid cache bloat.

```python
sample_rows(
    ...
) -> ndarray
```

Returns `(n_sampled, 3)` integer indices. If fewer than `nd` valid rows exist and `replace=False`, return all valid rows and record a warning in Python.

```python
build_density_library(
    ...
) -> ndarray
```

Returns `(n_sampled, n_terms)`.

```python
build_polarization_library(
    ...
) -> ndarray
```

Returns `(2 * n_sampled, n_terms)` using shared x/y coefficients.

The Rust API should not know about file paths, case IDs, plot settings, or PySINDy.

## 7. Candidate Library

Define candidate terms in Python as data, not branches scattered through the workflow. Example:

```python
DENSITY_TERMS = (
    "div_p",
    "lap_rho",
    ...
)

POLARIZATION_TERMS = (
    "p",
    "rho_p",
    ...
)
```

Each term entry should include:

- `name`: stable cache/API name.
- `label`: plot/report label.
- `kind`: `"scalar"` or `"vector"`.
- `rust_name`: name passed to Rust.
- `requires`: base fields/derivatives needed.

Default density terms from the paper SI Fig. S3/Table SII:

```text
div_p                     = div(p)
lap_rho                   = lap(rho)
div_rho_p                 = div(rho * p)
lap_rho2                  = lap(rho^2)
lap_p_norm2               = lap(|p|^2)
div_rho2_p                = div(rho^2 * p)
lap_rho3                  = lap(rho^3)
div_p_norm2_p             = div(|p|^2 * p)
div_rho_grad_p_norm2      = div(rho * grad(|p|^2))
div_p_norm2_grad_rho      = div(|p|^2 * grad(rho))
div_p_perp                = div(p_perp)
div_rho_p_perp            = div(rho * p_perp)
div_rho2_p_perp           = div(rho^2 * p_perp)
div_p_norm2_p_perp        = div(|p|^2 * p_perp)
```
Must be calculated with fft and chebyshev for time.

Default polarization terms from the paper SI Fig. S4/Table SIII:

```text
p                         = p
rho_p                     = rho * p
p_perp                    = p_perp
rho_p_perp                = rho * p_perp
p_norm2_p                 = |p|^2 * p
p_norm2_p_perp            = |p|^2 * p_perp
grad_rho                  = grad(rho)
p_dot_grad_p              = (p . grad) p
p_dot_grad_p_perp         = (p . grad) p_perp
p_perp_dot_grad_p         = (p_perp . grad) p
grad_div_p                = grad(div(p))
grad_div_p_perp           = grad(div(p_perp))
lap_p                     = lap(p)
lap_p_perp                = lap(p_perp)
grad_p_norm2              = grad(|p|^2)
div_p_p                   = div(p) * p
div_p_p_perp              = div(p) * p_perp
bilap_p                   = lap(lap(p))
bilap_p_perp              = lap(lap(p_perp))
```

Use `p_perp = (-p_y, p_x)`.

Sign convention:

- Use the paper's direct terms, such as `div_p`, and let coefficients carry signs.
- Do not silently switch to `minus_div_*` names unless the term name says so.

Maintainability rule:

- Adding or removing a term should normally require editing only the Python registry and Rust `library.rs` term dispatcher.
- Unknown terms should fail before expensive gridding begins.

## 8. Spectral And Numerical Details

Spatial domain:

```text
Lx = x_edges[-1] - x_edges[0]
Ly = R * (theta_edges[-1] - theta_edges[0])
```

Spatial Fourier wave numbers:

```text
kx = 2*pi*fftfreq(Nx, d=Lx/Nx)
ky = 2*pi*fftfreq(Ny, d=Ly/Ny)
```

Derivatives:

```text
partial_x f = ifft(i*kx*fft(f))
partial_y f = ifft(i*ky*fft(f))
lap(f)      = ifft(-(kx^2 + ky^2)*fft(f))
bilap(f)   = lap(lap(f))
div(v)     = partial_x v_x + partial_y v_y
grad_div(v)= grad(div(v))
```

Chebyshev time handling:

- Map physical times to `[-1, 1]`.
- Use frame steps and simulation `dt` consistently.
- Fit Chebyshev coefficients per spatial grid location and component.
- Truncate modes above `cheb_cutoff`.
- Evaluate filtered fields and `partial_t` on a documented time grid.
- Convert derivative from Chebyshev coordinate back to physical time using the chain rule.

Cache the following metadata with every output:

- `case_id`
- input paths and mtimes
- `Nx`, `Ny`
- `sigma`
- `lx`, `radius`, `ly`
- Chebyshev cutoff
- spatial cutoff/filtering settings
- density/polarization term lists
- `N_d`, seed, replacement flag
- Rust crate version or git hash if available
- cache schema version

## 9. Validation And Outputs

Write outputs to:

```text
hexatic/rho_fitting/output/
```

Suggested files:

```text
{case_id}_rho_fields.npz
{case_id}_sampled_system.npz
{case_id}_fit_result.npz
{case_id}_rho_fitting_report.md
{case_id}_density_true_vs_predicted.png
{case_id}_polarization_x_true_vs_predicted.png
{case_id}_polarization_y_true_vs_predicted.png
{case_id}_temporal_map_chebyshev.png
{case_id}_density_importance_scores.png
{case_id}_polarization_importance_scores.png
{case_id}_density_coefficients.png
{case_id}_polarization_coefficients.png
{case_id}_rho_kymograph.png
{case_id}_partial_t_rho_kymograph.png
{case_id}_density_residual_heatmap.png
{case_id}_polarization_vector_snapshot.png
```

Report should include:

- Settings and input files.
- Number of valid rows and sampled rows.
- Candidate term lists.
- `tau_max`, `eps`, number of thresholds, number of subsamples.
- Terms above importance threshold.
- Final coefficients.
- Basic metrics: correlation, R2, MAE, normalized MAE.
- Notes about skipped terms, dropped rows, finite-mask issues, or reduced sample size.

Validation plots:

- Predicted vs true `partial_t rho`.
- Predicted vs true `partial_t p_x` and `partial_t p_y`.
- Importance score versus `-log10(tau/tau_max)` for each term.
- Coefficient bar charts after final refit.
- Kymographs for `rho`, `partial_t rho`, predicted `partial_t rho`, residual.
- Heatmaps of time-averaged residuals over `(x, theta)`.
- Vector field snapshots comparing true/predicted polarization dynamics.

## 10. Edge Cases And Issues To Account For

Data availability:

- `.npz` may contain normalized polarization-like arrays; do not trust them as `P_density` without checking.
- `.gsd` orientations may be missing or malformed; fail with a clear error.
- `.gsd` and `.npz` frame counts or steps may disagree; fail before gridding unless a deliberate alignment rule is implemented.
- Some particles may leave the shell; use `shell_mask` consistently in `rho` and `P_density`.
- `coords[..., 2]` can differ from the nominal radius; the unwrapped grid uses `R theta`, but particle-to-grid distances should use the intended surface convention documented in `geometry.py`.

Geometry and periodicity:

- `x` is periodic with length `Lx`. Use the edge case handling similar in `model_fitting` where `Lx` is calculated more over teh `npz` or `gsd` than the current way the constants file or anything else does it. Its technically a warning but its very important you implement this. 
- `theta` is periodic over `2*pi`; `y` periodic length is `2*pi*R`.
- Gaussian distances must use minimum-image deltas in both `x` and `y`.
- If `theta_edges` are not exactly `[0, 2*pi]`, use their actual span for `Ly` and wrapping.

Kernel/coarse graining:
- The 2D Gaussian normalization differs from existing 3D pocket-density helpers.
- Very small `sigma` can create noisy sparse fields and unstable derivatives.
- Very large `sigma` can erase relevant density variation.
- Floating-point underflow in far-away Gaussian weights is acceptable; overflow or nonfinite `sigma` is not.

Spectral processing:

- Chebyshev differentiation is sensitive to time-grid assumptions; document whether interpolation to Chebyshev extrema is used.
- Spatial and temporal filtering can change target alignment; all target and candidate arrays must share the same evaluated time grid.

Sampling:

- If valid rows are fewer than `N_d`, either sample all rows without replacement or allow replacement only when explicitly configured.
- Sampling must be deterministic for a fixed seed.
- Density and polarization must use identical sampled `(t, x, y)` points before polarization component stacking.
- Mask out rows where any target or candidate term is nonfinite.
- Record how many rows were rejected after candidate construction.

Candidate library:

- Some paper library terms are linearly dependent in theory; run an SVD/rank diagnostic on sampled `Theta`. Do this within python and within the rust `regression.rs` (even tho its not used yet) thru crate ndarray-linalg
- Near-constant or zero columns should be dropped or reported before STLSQ.
- Extremely different column scales are expected; normalize for the threshold path.
- Do not include hidden intercept terms because `fit_intercept=False`.
- Keep term sign conventions stable across cache loads.

Regression:

- `tau_max` can be zero if all normalized coefficients vanish; handle this as a no-fit case.
- Final refit on unnormalized data must use only selected terms and no intercept.
- For polarization, never fit separate coefficients for x/y unless explicitly adding a diagnostic mode.

Performance:

- Full `N_d=5e5`, 40 thresholds, and 200 subsamples is expensive.
- Avoid returning all possible derivative arrays from Rust when only a subset is needed.
- Cache coarse-grained fields separately from sampled regression systems.

Caching:

- Cache invalidation must include term lists and numerical settings.
- Existing output files should not be overwritten unless `--overwrite` is set.
- Failed partial cache writes should not leave files that look valid; write to a temporary path then rename.

## 11. Test Plan

No testing

## 12. Open Decisions For Manual Review

- Default `sigma`: the paper chose based on particle scale and spectral entropy. Pick a conservative configurable default from existing local pocket/grid scales, then tune manually.
- Chebyshev cutoff: expose as a setting and include spectra plots so it can be chosen from the mode-power dropoff. Already covered this.
- Time-grid convention: choose transition midpoints once and keep all arrays aligned to it.
- Whether to include only density fitting in the first runnable milestone or density plus polarization together. The final workflow should include both. **include both**
- Whether to add rank/SVD pruning before STLSQ or only report rank warnings at first. Only report rank warnings at first.
