# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Scientific workflows for characterizing dynamics and structural features of active Brownian particles (ABPs) in cylindrical confinement (current focus: MIPS). Simulations run in HOOMD-blue (a forked submodule, `hoomd-blue` → `hoomd-blue-polydisperse`); analysis is Python orchestration on top of native Rust and Zig compute kernels.

Right now, we are are analyzing the stochastic fitting for the area of dilute bands within a MIPS system with ABPs on a crystalline cylindrical confinement. All work we are doing currently is in `band_analysis`, where we are fitting PDE's first for the Area.

## Environment and commands

Everything runs inside the pixi environment (`pixi shell`, or `pixi run <task>`). Tasks are defined in `pixi.toml`; prefer them over raw commands.

```bash
pixi run cargo-check                    # cargo check --workspace --all-features
pixi run cargo-fmt-check                # cargo fmt --all --check
pixi run rho-fitting-build              # maturin develop, Metal GPU (macOS)
pixi run rho-fitting-build-cuda         # maturin develop, CUDA
pixi run <pkg>-check / <pkg>-test / <pkg>-build   # per Zig package, e.g. linalg-test
pixi run hoomd-install                  # configure+build+install the HOOMD fork (CUDA)
pixi run hoomd-test-import              # verify HOOMD 7.0.1 + GPU
```


Python type checking uses `ty` (installed in the env). Rust is pinned to 1.96.0 via `rust-toolchain.toml`; Zig comes from the system (`zvm`), not pixi.

Analysis/simulation entry points are all `pixi run` tasks wrapping `python -um hexatic.<module>` — e.g. `band-analysis`, `rho-fitting`, `big-lx-pipeline`, `new-sims-pipeline`, `confinement-pipeline`. Long sweeps have `--dry-run` variants (`big-lx-dry-run`, `confinement-dry-run`, `new-sims-dry-run`); use those first.

Large data artifacts (`gsd/`, `npz/`, `output/`) are DVC-tracked against a DagsHub S3 remote, and `.gitignore` excludes `*.gsd`, `*.npz`, `*.png`, `*.json`, `*.csv`, `*.log`. Do not commit generated data.

## Tests
We will be using synthetic-data tests from now in, in the root structures_and_functions/tests/. These will test mathemtical and physical validitiy specified by me or implied to be useful.

## Architecture

### Three-language layering

Python (`hexatic/`) owns configuration, orchestration, plotting, and I/O. Heavy numerics are pushed into:

- **Rust** (`crates/`, `packages/rho-fitting-core/`) — the rho-fitting path, a Cargo workspace with a strictly one-way dependency direction:
  `packages/rho-fitting-core` (PyO3/NumPy adapter) → `rho_fitting_numerics` + `rho_fitting_gpu` → `rho_fitting_types`.
  `rho_fitting_types` centrally owns the cylindrical grid, the physical component order `(x, e_theta, e_r)`, mechanical field containers, and validation errors — encode conventions there so Python and Rust cannot drift. `rho_fitting_gpu` (Burn backend, Gaussian deposition) is the preferred path; CPU is a reference fallback. `ArrayD` belongs at the PyO3 boundary only; use fixed-rank arrays inside Rust. The extension imports as `hexatic.rho_fitting._rho_fitting_core` and Python guards the import (`_rho_fitting_core_import_error`) so an unbuilt extension fails loudly at call sites, not import time. See `ARCHITECTURE.md` and `INSTRUCTIONS.md`.

- **Zig** (`packages/`) — standalone analysis libraries loaded from Python via ctypes. Leaf libraries `linalg`, `safetensors`, `gsd`, `hdf5` are consumed by `dynamics_analysis`, `cluster_analysis`, `laplacian_analysis`, `simulation_analysis`. Python loads them through `hexatic/big_lx_analysis/_native.py::load_library`, which resolves `packages/<name>/zig-out/lib/lib<name>.<ext>` or the `HEXATIC_<NAME>_LIBRARY` override — so a Zig change requires `pixi run <pkg>-build` before Python sees it.

### Simulation-study pattern

Each study directory (`hexatic/big_lx/`, `confinement_comparison/`, `new_sims/`, `unwrapped_analysis/`, `radii_analysis/`) repeats the same module layout; follow it when adding a study:

- `cases.py` — frozen dataclass case definitions with validating `__post_init__` and derived geometry as properties; case IDs are the filesystem keys.
- `validate_cases.py` / `gpu_check.py` — preflight.
- `simulate_case.py` → `analyze_case.py` — single-case work, invoked as subprocesses.
- `scheduler.py` — worker pool that pins jobs to GPU IDs and tees per-case logs.
- `run_sweep.py` / `run_analysis.py` / `run_pipeline.py` — the pipeline runs the sweep, barriers, then analysis; it forwards a shared arg set (`--all`, `--case`, `--output-root`, `--overwrite`, `--dry-run`) to both stages.
- `storage.py` — all writes are atomic (write `.tmp`, then `replace`), safetensors for arrays and JSON for metadata.

Physical constants are centralized in `hexatic/constants/` (`cylinder.py`, `sphere.py`, `project.py`) — take `PARTICLE_DIAMETER`, `RHO`, `SEED`, `TAU_R`, etc. from there rather than redefining. Note the exception documented in `ARCHITECTURE.md`: the `radius_15D` rho-fitting dataset uses `FIXED_LX = 4000 / (RHO * pi * BASELINE_CYLINDER_RADIUS**2)`, not `lx_for_radius`.

`ARCHITECTURE.md` also documents the on-disk data schemas (GSD fields, NPZ array names, which file is the best source for each physical quantity) and the array-shape conventions: particle-local arrays are `(frame, particle, component)`, gridded arrays `(grid_point, component)` or `(frame, grid_point, component)`.

## Style

From `ARCHITECTURE.md`, applied repo-wide: **be concise, limit comments, keep it clean and simple.** Do not create overcomplicated files, classes, or verbose functions. Prefer SciPy over hand-rolled NumPy for mathematically heavy work, and Numba where it does not add much complexity. Python uses `from __future__ import annotations`, frozen dataclasses, and `pathlib`. Rust lints: `unsafe_code = "warn"`, `clippy::all = "warn"`.

AI-assisted contributions are disclosed in `DISCLOSURE.md`; update it if that changes.
