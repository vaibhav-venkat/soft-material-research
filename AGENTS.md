# Repository Guidelines


## Current Work

The active simulation work is `hexatic/unwrapped_analysis`: the sweep now uses an **exact, twisted triangular-lattice supercell**. It is the replacement for the earlier experimentally sheared `simulate_case.py` route. The supercell gives a genuinely twisted cylinder while retaining perfect hexatic order at initialization, so it needs neither an overlap-prone coordinate shear nor a high-drag relaxation ramp.

`hexatic/rho_fitting` remains the established analysis workflow for the radius `15D` trajectory. It now fits 3D cylindrical mechanical moment closures, assesses the closure in divergence space, and validates it by rolling out the fitted PDE. Treat the simulation work and the `rho_fitting` analysis inputs as separate unless new supercell trajectories are deliberately promoted into the fitting pipeline.

## Unwrapped Twisted Supercell

Use `hexatic.unwrapped_analysis.simulate_case_perfect_hexatic` for all new unwrapped simulations. `run_sweep.py` already dispatches to it:

```bash
pixi run python -m hexatic.unwrapped_analysis.run_sweep --all --overwrite
pixi run python -m hexatic.unwrapped_analysis.simulate_case_perfect_hexatic --case circ_60_0D --overwrite
```

The lattice is indexed by triangular-basis integers `(j, i)`. Its circumference identification is

```text
c = (1, N_theta - 1),
```

so crossing the axial seam changes the azimuthal lattice index by one: `(0, 0)` identifies with `(1, N_theta - 1)`. `UnwrappedCase` constructs an integer axial supercell vector orthogonal to `c` in the triangular metric,

```text
p = ((2n + m)/g, -(2m + n)/g),  where c = (m, n), g = gcd(2n + m, 2m + n).
```

`simulate_case_perfect_hexatic.py` enumerates one fundamental parallelogram of `(c, p)`, maps its circumference fraction to `theta`, and maps its axial fraction to the periodic HOOMD `x` box. This preserves all triangular nearest-neighbour bonds under both periodic identifications. Consequently:

- `perfect_hexatic_a`, `perfect_hexatic_lx`, and `perfect_hexatic_n_particles` are the physical supercell values; do not substitute the legacy `a`, `lx`, or `n_particles` properties when writing the perfect-supercell GSD.
- The supercell generally has a different axial length and particle count from the old rectangular approximation. This is intentional and should be recorded when comparing cases.
- The old `simulate_case.py` remains only as a non-perfect, linearly sheared experimental implementation. Do not use it for the sweep or as the reference initial condition.
- Initial states, trajectories, and metadata retain the same per-case output paths. Running an alternative generator with `--overwrite` replaces those artifacts.

## Rho-Fitting: Current Architecture

The fitting workflow has grown from a surface-density fit into a 3D cylindrical mechanical-moment pipeline:

```text
GSD + active-matter NPZ + hexatic inputs
  -> GPU Gaussian coarse-graining on (x, theta, r)
  -> rho, P, Q, A and measured currents J_rho, J_P, J_Q
  -> temporal Chebyshev filtering + cylindrical spectral derivatives
  -> mechanical targets and candidate flux libraries
  -> divergence-primary constrained sparse regression
  -> cached fit/report/plots
  -> dealiased cylindrical PDE rollout and validation artifacts
```

The moment and current conventions are now volumetric and use the orthonormal cylindrical frame `(x, e_theta, e_r)`:

```text
rho:   (T, Nx, Ntheta, Nr)
P:     (T, Nx, Ntheta, Nr, 3)
Q, A:  (T, Nx, Ntheta, Nr, 3, 3)
J_rho: (T, Nx, Ntheta, Nr, 3)
J_P:   (T, Nx, Ntheta, Nr, 3, 3)
J_Q:   (T, Nx, Ntheta, Nr, 3, 3, 3)
```

The closures being fitted and rolled out are:

```text
partial_t rho = -div(U0 P + F_rho / gamma)
partial_t P   = -U0 div(F_P) + P relaxation
partial_t Q   = -div(F_Q) + Q relaxation
```

For every target, fitting and reporting distinguish the field/flux prediction from the divergence prediction. The divergence is the PDE-relevant primary target; a good flux R^2 alone is not evidence for a usable closure.

Important Python modules:

- `fit.py`: orchestrates loading, GPU coarse-graining, spectral fields, target/library construction, and regression output.
- `config.py`: canonical case paths plus numerical controls, including radial bins/range and mechanical flux weighting.
- `spectral.py`: cached Shenfun cylindrical `(x, theta, r)` operators, radial Chebyshev interpolation, and 2/3 filtering.
- `basis.py`: Chebyshev temporal filtering/derivatives and temporal power diagnostics.
- `regression.py`: constrained SR3/L1 fitting and coefficient-path/importance diagnostics.
- `pde_validation/`: cache validation, IMEX rollout modes (`full`, `rho-only`, `p-only`, `q-only`), reports, and HTML plots.

## Rust Numerical Core

`packages/rho-fitting-core` is the PyO3/maturin adapter exposed as `hexatic.rho_fitting._rho_fitting_core`. The reusable implementation is split across the Rust workspace crates `rho-fitting-types`, `rho-fitting-numerics`, and `rho-fitting-gpu`. It is no longer just a small candidate-library helper. Their responsibilities are:

- GPU Gaussian deposition of 3D cylindrical mechanical fields through Burn (`coarse_grain_burn/`), with Metal/WGPU and CUDA feature variants;
- construction and validation of `rho`, `P`, `Q`, `A`, `psi6_sq`, and all current tensors;
- conversion of measured currents into `Y_rho`, `Y_P`, and `Y_Q` mechanical targets;
- finite-row/component sampling for regression; and
- Python bindings and shape/error translation in `python.rs`.

The CPU mechanics implementation exists for core numerical logic, but the Python coarse-graining entry point requires a GPU-enabled build. Build the extension after changing Rust code or when the local ABI/Python version changes:

```bash
pixi run rho-fitting-build        # Metal/WGPU
pixi run rho-fitting-build-cuda   # CUDA
```

Keep physical-frame component order and array shapes synchronized across the Rust core, `fit.py`, `spectral.py`, and `pde_validation`. A mismatch can produce plausible arrays but incorrect cylindrical divergences.

## Data and Outputs

The canonical `radius_15D` fitting inputs are local to `hexatic/rho_fitting/`:

- `gsd/trajectory_radius_15D.gsd`: positions, orientations, and velocities.
- `npz/radius_15D_active_matter_fields.npz`: active-matter grid/trajectory inputs.
- `hexatic_output/`: particle hexatic order, velocity, and neighbour-count data.
- `output/radius_15D_fit_result.npz`: cached mechanical fields, targets, libraries, and fit coefficients.
- `output/radius_15D_rho_fitting_report.md`: regression and mechanical-fit report.
- `output/radius_15D_pde_validation_*.npz`, report, and HTML: rollout results.

The current radius `15D` data has 100 frames and 9870 particles, starting at simulation step 100000. Its axial length is a dataset-specific fixed value; do not infer it from generic cylinder helpers.

Generated artifacts are not source changes. Never overwrite them without explicit `--overwrite` and user authorization.

## Commands

Use Pixi for Python and Rust commands.

```bash
# Lightweight source checks
pixi run python -m compileall hexatic/unwrapped_analysis hexatic/rho_fitting
pixi run ty check hexatic/rho_fitting

# Rust extension checks/builds
pixi run cargo check --workspace
pixi run cargo check --workspace --all-features
pixi run rho-fitting-build

# Zig dense-linear-algebra checks (macOS Accelerate)
pixi run linalg-check
pixi run linalg-test

# Cached rho-fitting and PDE workflows (can be expensive)
pixi run rho-fitting-lite --case radius_15D --fit-only --overwrite
pixi run python -m hexatic.rho_fitting.pde_validation --case radius_15D --mode all --no-plot --overwrite
```

Do not run full HOOMD sweeps, `radii_analysis.sh`, full coarse-graining, or PDE rollouts casually. They are expensive and write generated data.

## Working Rules

- Read `hexatic/constants/` before changing any simulation geometry, density, interaction, or time-step parameter.
- Preserve the supercell’s integer lattice-vector construction; do not reintroduce a fractional axial shear as a substitute for the twisted periodic identification.
- Prefer cached NPZ/GSD inputs to recomputation.
- Use `rg` for repository searches and `apply_patch` for edits.
- Keep Python functions direct, with standard 4-space indentation and type hints where useful. Prefer small dataclasses for structured configuration/results.
- For rho-fitting changes, preserve the `(x, theta, r)` grid order and physical `(x, e_theta, e_r)` component order throughout.
- Do not commit generated GSD, NPZ, plots, HTML, logs, compiled extension binaries, or unrelated user changes.
