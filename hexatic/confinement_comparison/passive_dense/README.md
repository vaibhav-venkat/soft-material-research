# Passive cylinder and dense 2D simulations

This independent simulation-only workflow contains five cases:

- `passive_cylinder_60_5D`: the exact twisted `C = 60.5D`, `Lx = 1x`
  cylinder initial condition with unconstrained Brownian dynamics at `kT = 1`,
  no active force, and repulsive pair/wall stiffness `epsilon = 50 kT`.
- `dense_2d_60D`: an active 2D channel containing an exact `88 x 60`
  untwisted triangular patch. Its nearest-neighbor spacing is exactly `a = D`,
  and its initial orientations point toward the nearest y wall.
- `dense_2d_60D_center_vacancy`: the same crystal with the closest available
  site to `(x, y) = (0, 0)` removed. The even-row lattice has no site exactly
  at `y = 0`; the selected site has `|y| = D/4`.
- `dense_2d_60D_wall_vacancy`: the same crystal with one site in the outer
  `+y` wall layer removed.
- `dense_2d_60D_opposite_wall_vacancies`: the same crystal with two vacancies
  at inversion-related coordinates `(x, y)` and `(-x, -y)` near the two walls.

The three vacancy cases default to 300 frames: `3e7` steps with the existing
`1e5`-step trajectory period.

The package does not add these cases to the parent seven-case confinement
pipeline.

Run only the three vacancy cases concurrently on three physical GPUs:

```bash
CONDA_OVERRIDE_CUDA=12.9 \
pixi run -e big-lx-cuda12 python -u -m \
  hexatic.confinement_comparison.passive_dense.run_pipeline \
  --sim \
    dense_2d_60D_center_vacancy \
    dense_2d_60D_wall_vacancy \
    dense_2d_60D_opposite_wall_vacancies \
  --gpu-ids 2,3,4 \
  --workers 3 \
  --output-root /mnt/drive3/vaibhav_data/passive_dense_vacancies \
  --overwrite
```

Replace `2,3,4` with three currently free physical GPU IDs. The pipeline maps
the listed cases to those IDs in order.

## Production pipeline

Production defaults to GPU and runs both jobs concurrently across the selected
physical devices:

```bash
CONDA_OVERRIDE_CUDA=12.9 \
pixi run -e big-lx-cuda12 python -u -m \
  hexatic.confinement_comparison.passive_dense.run_pipeline \
  --all \
  --gpu-ids 0,1 \
  --workers 2 \
  --output-root /mnt/drive3/vaibhav_data/passive_dense_production \
  --overwrite
```

Use inclusive selection when only one case is wanted:

```bash
pixi run python -m hexatic.confinement_comparison.passive_dense.run_pipeline \
  --sim dense_2d_60D --dry-run
```

## Validation and frame-0 preview

Validate both deterministic initial conditions without integrating:

```bash
pixi run passive-dense-validate
```

Write and CPU-load only the dense 2D frame-0 GSD:

```bash
pixi run python -m hexatic.confinement_comparison.passive_dense.simulate_case \
  --case dense_2d_60D --device cpu --initial-only
```

The preview is written to `initial/initial_dense_2d_60D.gsd`. A later
production run using the same output root must pass `--overwrite` to replace
the preview metadata and initial file.
