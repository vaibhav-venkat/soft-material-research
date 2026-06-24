# Repository Guidelines

## Project Structure & Module Organization

This repository contains Python tools for HOOMD-based particle simulations and cylinder/sphere analysis. Core source code lives in `hexatic/`. Simulation entrypoints include `hexatic/hoomd_cylinder.py`, `hexatic/hoomd_cylinder_new_rad.py`, and radius-sweep tooling in `hexatic/radii_analysis/`. Analysis modules are grouped by domain: `hexatic/analysis/`, `hexatic/chirality/`, `hexatic/active_matter_cylinder/`, `hexatic/cylinder_dynamics/`, and `hexatic/multiple_sim_analysis/`. Generated trajectories and fields are usually under `hexatic/output/` or `hexatic/radii_analysis/{gsd,npz_fields,hexatic_output}/`. Lightweight tests live in `tests/`.

## Build, Test, and Development Commands

Use the Pixi workspace named `sap` for all Python commands.

- `pixi shell`: enter the project environment.
- `pixi run python -m compileall hexatic`: syntax-check the package without running simulations.
- `pixi run python -m compileall hexatic/multiple_sim_analysis`: quick check for the aggregate radius-analysis package.
- `pixi run python -m hexatic.multiple_sim_analysis --help`: inspect the multi-simulation analysis CLI.
- `bash radii_analysis.sh`: run the full radius workflow; this launches long GPU simulations and should not be run casually.

Avoid running expensive simulations or analysis scripts unless explicitly requested.

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, type hints where they clarify interfaces, and dataclasses for structured result containers. Keep functions small and prefer reusable helpers over repeated array logic. Use `snake_case` for modules, functions, variables, and output filenames; use `PascalCase` for classes. Keep comments minimal: add them only to explain non-obvious physics, numerical assumptions, or file-format details.

## Data & Output Practices

Prefer loading existing `.npz` or `.gsd` outputs instead of recomputing expensive intermediates. When writing new outputs, place them in a clearly named package output directory such as `hexatic/multiple_sim_analysis/output/`. Do not overwrite generated simulation data unless an explicit `--overwrite` style option is provided and requested.

## Agent-Specific Instructions

Inspect existing constants in `hexatic/constants/` before changing simulation geometry, density, or time-step behavior. Preserve radius-aware parameters when adding analysis code; avoid relying on global default radius values for sweep cases. Use `rg` for searching, `apply_patch` for edits, and keep unrelated generated artifacts out of commits.
