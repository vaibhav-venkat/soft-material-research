# Copilot Instructions

## Environment

- **Conda environment**: `omar-group` (`/opt/homebrew/anaconda3/envs/omar-group`)
- **Type checker**: `ty` (configured in `pyproject.toml`)
- **Key dependencies**: `hoomd`, `gsd`, `numpy`, `matplotlib`

Run simulations with:
```bash
conda run -n omar-group python setup_vesicle.py
conda run -n omar-group python setup_membrane.py
conda run -n omar-group python structure_calculation.py
```

Run the type checker:
```bash
conda run -n omar-group ty check
```

## Architecture

The project has two phases: **simulation** and **analysis**.

**Simulation** (`setup_vesicle.py`, `setup_membrane.py`):
- Each script builds an initial GSD snapshot, writes it to disk, then runs a HOOMD-blue MD simulation that appends frames to a `.gsd` trajectory file.
- Both scripts use the same two particle types: `vertex` (membrane nodes, index 0) and `abp` (Active Brownian Particles, index 1).
- The `ACTIVE` flag at the top of each script toggles between active (ABP propulsion + rotational diffusion) and passive runs; output filenames encode this (`-active.gsd` / `-passive.gsd`).
- A `PerimeterConservation` custom force class (subclasses `hoomd.md.force.Custom`) is defined in both setup files but is currently commented out.

**Analysis** (`structure_calculation.py` + `analysis/` package):
- `structure_calculation.py` is the entry point; set `MODE = "vesicle"` or `MODE = "membrane"` at the top to switch targets.
- `analysis/analyzers.py` defines the abstract `ModeAnalyzer` base class with two concrete implementations: `Vesicle` (extracts radial coordinates from a ring) and `Membrane` (extracts x-positions from a flat line).
- `analysis/calculate.py` reads GSD trajectories, skips `EQUILIBRIUM_FRAMES` frames, then computes averaged structure factors via FFT (`np.fft.fft` of interpolated vertex positions).
- `analysis/plot.py` plots structure factors on a log-log scale (`log(|h(k)|²)` vs `k`).

## Key Conventions

- **All simulations are 2D**: box is always `[Lx, Ly, 0, 0, 0, 0]` (Lz=0); `moment_inertia = [(0, 0, 1)]` for all particles; active forces and rotations are in the xy-plane only.
- **Vertex ordering matters**: bonds and angles are built as `(i, (i+1) % N_vertex)` rings; the `PerimeterConservation` force relies on this sequential ordering.
- **Equilibrium bond length**: always `l0 = 4 * sigma_vertex`. LJ cutoffs use `2^(1/6) * sigma` (WCA / purely repulsive).
- **`vertex-abp` LJ repulsion uses the mixed sigma**: `(sigma_vertex + sigma_abp) / 2`; `abp-abp` LJ epsilon is set to 0 (no ABP self-interaction).
- **`analysis/` uses numpy type annotations** (`npt.NDArray[np.float64]`) throughout; new analysis code should follow this convention.
- **`get_analyzer(mode, sigma_vertex)`** is the factory for `ModeAnalyzer` instances; `sigma_vertex` is only used by `Vesicle`, not `Membrane`.
- **GSD filenames** follow the pattern: `<geometry>_CPU_<bond-type>-<active|passive>.gsd` (e.g. `perimeter-conserved-2D-vesicle_CPU_harmonic_bonds-active.gsd`).
