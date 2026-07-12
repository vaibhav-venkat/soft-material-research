# Structures and Functions

Scientific Python workflows for the hexatic-cylinder simulations and rho-fitting
analysis. Python orchestration remains under `hexatic/`; reusable native code
is organized as a Cargo workspace.

Build or check the native pipeline with Pixi:

```bash
pixi run cargo-check
pixi run rho-fitting-build
```

The PyO3 module remains available as
`hexatic.rho_fitting._rho_fitting_core`. Numerical algorithms are kept in the
Python-free crates under `crates/`, and the adapter is under
`packages/rho-fitting-core/`.
