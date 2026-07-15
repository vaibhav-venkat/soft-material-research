# Confinement comparison

This workflow compares three `C = 60.5D`, `Lx = 1x` geometries derived from
`hexatic.big_lx`:

- `prism_volume`: equal-volume square prism with four plane walls.
- `cylinder_rattle`: exact twisted film constrained to a cylinder.
- `cylinder_rattle_tangent`: the same constraint with tangent active force.

All production cases contain 7082 particles and write 1000 frames over
`1e8` steps. The cylinder manifold in HOOMD 7.0.1 is fixed along the stored
z-axis, so cylinder GSD files use a rigid axis permutation. Analysis outputs
are mapped back to the logical x-axis convention used by `big_lx`.

## Local validation and CPU diagnostic

```bash
pixi run confinement-validate
pixi run confinement-diagnostic
```

The diagnostic runs one `1e5`-step write interval on the CPU and writes frames
at steps 0 and 100000 under `diagnostics/`. Pass `--overwrite` explicitly to
the module command when replacing generated diagnostics:

```bash
pixi run python -m hexatic.confinement_comparison.diagnostic --all --overwrite
```

## Remote two-GPU production pipeline

```bash
pixi run -e big-lx-cuda12 confinement-gpu-check
pixi run -e big-lx-cuda12 confinement-dry-run
pixi run -e big-lx-cuda12 confinement-pipeline --all --gpu-ids 0,1 --require-gpu
```

The pipeline launches all three cases concurrently. The prism is pinned to
physical GPU 1 and both cylinder cases are pinned to physical GPU 0. It waits
for all three simulation completion markers, then applies the same affinity to
resumable JAX/safetensor analysis.
Use `--overwrite` only when replacing existing production outputs, or
`--resume-analysis` to continue incomplete analysis shards.
