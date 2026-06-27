# Plan - Fitting J to density gradients

## Goal
Within `hexatic/density_analysis/`, create `run_fitting.py` and a `fitting/` submodule. Fit the relationship `J_x ≈ c_x ∂_x ρ` and `J_y ≈ c_y ∂_y ρ` using FFT-based gradients and linear least squares.

## Structure
Mirror `film_continuity/` submodule layout:
- `fitting/config.py` — `FittingConfig` dataclass (case_id, npz_path, gsd_path, output_dir)
- `fitting/fields.py` — gaussian rho computation, FFT gradient helpers
- `fitting/fit.py` — core regression logic, returns `FittingResult` dataclass
- `fitting/plots.py` — residual heatmap plots
- `fitting/io_cache.py` — npz cache load/write (reuse pattern from film_continuity)
- `run_fitting.py` — CLI entrypoint, argparse with `--case`, `--overwrite`, `--no-cache`, `--plot`

## Steps

### 1. Load or compute ρ(x, y, t)
- Load from `{case_id}_active_matter_fields.npz` if the gaussian-smoothed density field exists there.
- If not present, recompute using `_density_sum` from `active_matter_cylinder/math_utils.py` (which uses `_gaussian_delta_weights`).
- The grid is the same (x_edges, theta_edges) used by film_continuity. Convert θ → y via `y = R * θ`.

### 2. Load J_x, J_y at t+1/2
- These should already exist in the active_matter_fields npz as flux arrays (computed from face crossings over the transition).
- If missing, compute from particle velocities using the same binning approach as film_continuity.
- J^{t+1/2} represents the flux over the transition t → t+1.

### 3. Compute FFT gradients
For each frame's ρ(x, y) on the (nx, nθ) grid:
```
ρ_hat = scipy.fft.fft2(ρ)
kx = 2π m / Lx       (m = 0..nx-1)
ky = 2π n / Ly       (n = 0..nθ-1), Ly = 2π R
(∂_x ρ)_hat = 1j * kx * ρ_hat
(∂_y ρ)_hat = 1j * ky * ρ_hat
∂_x ρ = scipy.fft.ifft2((∂_x ρ)_hat).real
∂_y ρ = scipy.fft.ifft2((∂_y ρ)_hat).real
```
- `Lx` from cached `x_edges[-1] - x_edges[0]`, or from `cylinder.lx_for_radius(case.radius)`.
- `R` = `case.radius`.

### 4. Linear least squares fit
For each transition (frame pair t → t+1):
1. Compute average gradient at t+1/2: `∂_x ρ^{t+1/2} = (∂_x ρ^t + ∂_x ρ^{t+1}) / 2`
2. Fit: `J_x ≈ c_x * ∂_x ρ^{t+1/2}` and `J_y ≈ c_y * ∂_y ρ^{t+1/2}`
3. Use `scipy.linalg.lstsq` on flattened arrays: `J_x_flat = c_x * (∂_x ρ^{t+1/2})_flat`

Store `c_x`, `c_y` per transition and as a global aggregate.

### 5. Plot residual heatmaps
- `residual_x = J_x_measured - c_x * ∂_x ρ` on the (x, θ) grid
- `residual_y = J_y_measured - c_y * ∂_y ρ` on the (x, θ) grid
- Also plot a value of `c_x` and `c_y`, two additional plots, on the (x, theta) grid
- Save as static plots (matplotlib imshow/pcolormesh) to `fitting/output/{case_id}_fit_residual_x.png` etc.

## Assumptions
- Grid is uniform and periodic in x (period Lx) and θ (period 2π).
- Frames are uniformly spaced in time (already enforced by film_continuity).
- The gaussian smoothing radius used for ρ matches the one in active_matter_fields npz. Default: `pocket_radius` from the npz, or fall back to `0.5 * cylinder_radius`.
- J arrays have shape `(n_transitions, nx, nθ, 2)` where last dim is (Jx, Jy).
- ρ arrays have shape `(n_frames, nx, nθ)`.

## Edge Cases
- **Missing fields in npz**: If `rho_gaussian` or `J_film` keys are absent, recompute from raw particle data (positions + velocities in the GSD).
- **Non-uniform frame spacing**: Raise `ValueError` with a clear message (same as film_continuity).
- **Empty or near-empty bins**: Mask bins with `count < min_count` (default 2) before fitting to avoid division noise.
- **Single frame**: Need at least 2 frames for a transition; raise if fewer.

## Constants & Config
- Import `RadiusCase` from `hexatic.radii_analysis.cases` (via `get_case(case_id)`).
- Import cylinder constants from `hexatic.constants.cylinder`.
- `DEFAULT_CASE_ID = "radius_15D"` (same default as film_continuity).
- Output dir: `density_analysis/output/fitting/`.

## Validation
- `pixi run python -m compileall hexatic/density_analysis` should pass.
- Running with `--case radius_15D --plot` should produce residual PNGs without error.
