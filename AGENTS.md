# Repository Guidelines
**All tests have been removed. No more tests**
## Current Goal

The active work is `hexatic/rho_fitting`: fit and validate mechanical moment closures for the radius `15D` cylinder data. The current focus is not the HOOMD simulation itself; it is the cached/coarse-grained fields, regression libraries, divergence-aware fitting, and PDE validation rollouts.

Key equations being validated:

```text
partial_t rho = -div(U0 P + F_rho / gamma)
partial_t P   = -div(U0 F_P) + relaxation
partial_t Q   = -div(F_Q) + relaxation
```

Use 3D orientation moments with 2D surface flux directions:

```text
P:     (T, Nx, Ny, 3)
Q, A:  (T, Nx, Ny, 3, 3)
J_rho: (T, Nx, Ny, 2)
J_P:   (T, Nx, Ny, 2, 3)
J_Q:   (T, Nx, Ny, 2, 3, 3)
```

The report distinguishes:

```text
flux R^2:       fit F itself
divergence R^2: fit div(F), the PDE-relevant target
```

Do not treat a good flux R^2 as proof of a good PDE closure.

## Architecture Notes

Keep this project concise. Avoid overbuilt classes, verbose helper layers, and broad refactors. Prefer small, direct functions.

High-level flow:

```text
User -> plots/fits dynamics and fields -> analysis/rho_fitting workflows -> GSD, NPZ, reports, HTML
```

Important components:

- Pixi is the default environment runner.
- Numba is welcome where it keeps code simple.
- SciPy is preferred for mathematically heavy operations when useful.
- Rust is used for `rho_fitting_core`, especially mechanical field construction and candidate libraries.
- Burn is used for GPU coarse-graining support. `faer`/CPU linear algebra does not need GPU.

Use the Pixi environment from `pixi.toml`.

## File Tree

```text
.
├── AGENTS.md
│   └── This guide for agents working in the repository.
├── ARCHITECTURE.md
│   └── Original architecture/data notes and conventions.
├── pixi.toml
│   └── Pixi tasks and dependencies. Includes `rho-fitting-build`.
├── radii_analysis.sh
│   └── Full radius workflow. Expensive; do not run casually.
├── hexatic/
│   ├── constants/
│   │   └── Simulation constants and radius-aware geometry helpers.
│   ├── hoomd_cylinder*.py, hoomd_spherical_cavity.py
│   │   └── HOOMD simulation entrypoints. Expensive; avoid unless requested.
│   ├── analysis/
│   │   └── General plotting/analysis utilities.
│   ├── active_matter_cylinder/
│   │   ├── cartesian/
│   │   │   └── Cartesian-grid active matter comparisons and stress/flux diagnostics.
│   │   ├── fields/
│   │   │   └── Field construction helpers.
│   │   └── shear/
│   │       └── Shear and stress decomposition logic.
│   ├── chirality/
│   │   └── Translation/bond chirality calculations.
│   ├── cylinder_dynamics/
│   │   └── Cylinder dynamical diagnostics, relaxation plots, and shell analyses.
│   ├── model_fitting/
│   │   ├── fitting/
│   │   │   └── Older/general model-fitting infrastructure.
│   │   ├── film_continuity/
│   │   │   └── Continuity-equation style film fitting utilities.
│   │   └── output/
│   │       └── Generated model-fitting outputs.
│   ├── radii_analysis/
│   │   ├── gsd/
│   │   ├── npz_fields/
│   │   ├── hexatic_output/
│   │   ├── metadata/
│   │   └── logs/
│   │       └── Radius sweep simulations, derived fields, and logs.
│   ├── rho_fitting/
│   │   ├── fit.py
│   │   │   └── Main rho-fitting orchestration, caching, regression calls.
│   │   ├── regression.py
│   │   │   └── PySINDy SR3/L1 regression and stability path reporting.
│   │   ├── outputs.py, report.py, plots.py
│   │   │   └── Fit caches, markdown reports, and plots.
│   │   ├── cache.py, config.py, library.py
│   │   │   └── IO/config/candidate labels.
│   │   ├── pde_validation/
│   │   │   ├── model.py
│   │   │   │   └── PDE rollout model and validation modes.
│   │   │   ├── operators.py
│   │   │   │   └── Surface derivatives and closure evaluation.
│   │   │   ├── report.py
│   │   │   │   └── Text report for RMSE/R^2 at selected rollout steps.
│   │   │   ├── plot.py
│   │   │   │   └── HTML visualizations.
│   │   │   └── cache.py
│   │   │       └── Validation cache loading and shape/name checks.
│   │   ├── gsd/
│   │   │   └── Local trajectory inputs for rho fitting.
│   │   ├── hexatic_output/
│   │   │   └── Hexatic order and neighbor-count text files.
│   │   ├── npz/
│   │   │   └── Input NPZ fields.
│   │   └── output/
│   │       └── Fit results, PDE validation NPZ/HTML/TXT reports.
│   └── output/
│       └── Generated simulation/analysis outputs.
├── rust/
│   └── rho_fitting_core/
│       ├── src/mechanics/
│       │   └── Rust mechanical coarse-graining targets and candidate libraries.
│       ├── src/coarse_grain_burn.rs
│       │   └── Burn-backed GPU coarse-graining support.
│       ├── src/fft_ops.rs
│       │   └── Spectral derivatives for candidate construction.
│       └── Cargo.toml
│           └── Native extension configuration.
├── tests/
│   └── Lightweight tests for fitting, mechanics, and PDE validation.
└── outputs/, logs/
    └── Generated scratch/agent outputs; usually unrelated to source changes.
```

## Important Data Sources

Current radius `15D` data conventions from the architecture notes:

```text
100 frames
9870 particles
steps start at 100000
radius = 15D
```

For this simulation, `Lx` is fixed by:

```text
FIXED_LX = 4000 / (RHO * pi * BASELINE_CYLINDER_RADIUS**2)
```

Do not assume `Lx` follows the helper function in `hexatic/constants/` for this dataset.

Useful sources:

- `hexatic/rho_fitting/gsd/trajectory_radius_15D.gsd`: particle positions, orientations, velocities.
- `hexatic/rho_fitting/hexatic_output/radius_15D_hexatic_order.txt`: per-particle `psi6`.
- `hexatic/rho_fitting/hexatic_output/radius_15D_neighbor_counts.txt`: neighbor counts; disclination charge can be inferred from `6 - neighbor_count`.
- `hexatic/rho_fitting/output/radius_15D_fit_result.npz`: current rho-fitting cache.
- `hexatic/rho_fitting/output/radius_15D_rho_fitting_report.md`: current fit report.
- `hexatic/rho_fitting/output/radius_15D_pde_validation_report.txt`: PDE rollout RMSE/R^2 report.

Older/general analysis NPZ files may live under `hexatic/density_analysis/npz/` or radius-analysis output folders. Prefer existing `.npz`/`.gsd` data over recomputation.

## Array Conventions

Particle-local arrays usually use:

```text
(frame, particle, component)
```

Gridded arrays usually use:

```text
(frame, Nx, Ny, ...)
```

For unwrapped cylinder surface fields:

```text
x direction      -> axis 0 / surface component 0
y = R * theta    -> axis 1 / surface component 1
radial direction -> orientation component 2
```

Most PDE/rho-fitting plots should use either `(x, R theta)` as a 2D unwrapped cylinder or convert carefully back to `(x, y, z)`.

## Build, Test, And Run Commands

Use Pixi for all Python and Rust commands.

Common checks:

```bash
pixi run python -m compileall hexatic
pixi run ty check hexatic/rho_fitting
pixi run python -m unittest tests.test_rho_fitting_pde_validation
pixi run cargo check --manifest-path rust/rho_fitting_core/Cargo.toml
pixi run cargo check --manifest-path rust/rho_fitting_core/Cargo.toml --features gpu-metal
```

Build the rho-fitting extension after Rust candidate/core changes:

```bash
pixi run rho-fitting-build
```

Fit from cached coarse-graining:

```bash
pixi run rho-fitting-lite --case radius_15D --fit-only --overwrite
```

Run PDE validation and write the text report:

```bash
pixi run python -m hexatic.rho_fitting.pde_validation --case radius_15D --mode all --no-plot --overwrite
```

Avoid expensive workflows unless explicitly requested:

```bash
bash radii_analysis.sh
```

## Coding Style

- Use standard Python style, 4-space indentation, and type hints where they clarify interfaces.
- Use dataclasses for structured result/config objects.
- Keep functions small and direct.
- Prefer existing local patterns over new abstractions.
- Keep comments minimal; add them only for non-obvious physics, numerical assumptions, or file-format details.
- Use `snake_case` for functions, variables, modules, and output filenames.
- Use `PascalCase` for classes.
- Use `assert` for fail-fast validation in this codebase unless a public API needs a specific exception.

## Data And Output Practices

- Prefer loading cached `.npz`/`.gsd` files over recomputing expensive intermediates.
- Do not overwrite generated simulation data unless the command has an explicit `--overwrite` and the user requested it.
- Keep generated reports/plots in package-specific output folders, especially `hexatic/rho_fitting/output/`.
- Do not commit unrelated generated artifacts.

## Agent Instructions

- Read existing constants in `hexatic/constants/` before changing simulation geometry, density, time step, or radius handling.
- Preserve radius-aware parameters; avoid global default radius assumptions.
- Use `rg` for searches.
- Use `apply_patch` for edits.
- Run frequent lightweight lint/syntax checks after feature edits.
- Do not run full simulations or expensive analysis scripts unless explicitly requested.
- If touching rho-fitting Rust code, run `cargo check` and `cargo check --features gpu-metal` through Pixi.
- If touching `pde_validation`, run compileall, `ty`, and `tests.test_rho_fitting_pde_validation`.
