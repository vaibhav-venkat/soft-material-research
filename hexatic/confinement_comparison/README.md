# Confinement comparison

This workflow compares seven `C = 60.5D`, `Lx = 1x` geometries derived from
`hexatic.big_lx`:

- `prism_volume`: equal-volume square prism with four plane walls.
- `prism_surface_area`: square prism whose four wall areas sum to the cylinder
  lateral area.
- `sandwich_volume`: equal-volume box with periodic x/y and two z walls.
- `sandwich_surface_area`: periodic x/y box whose two z-wall areas sum to the
  cylinder lateral area.
- `two_dimension`: HOOMD 2D channel with periodic x, two y walls, and a GSD box
  with `Lz = 0`.
- `cylinder_rattle`: exact twisted film constrained to a cylinder.
- `cylinder_rattle_tangent`: the same constraint with tangent active force.

The original cases retain 7082 particles. New 3D cases preserve the original
cylinder bulk density, and the 2D case preserves its surface density. All cases
write 1000 frames over `1e8` steps. The cylinder manifold in HOOMD 7.0.1 is
fixed along the stored z-axis, so cylinder GSD files use a rigid axis
permutation. Analysis outputs are mapped back to the logical x-axis convention
used by `big_lx`.

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

The pipeline schedules the selected cases across the two physical GPUs, waits
for all selected simulation completion markers, and then applies the configured
analysis affinity to resumable JAX/safetensor analysis.
Use `--overwrite` only when replacing existing production outputs, or
`--resume-analysis` to continue incomplete analysis shards.

## Four-case planar production pipeline

`--all` selects all seven registered cases. Use the inclusive `--sim` selector
to run only the four new planar cases. The prism-surface and 2D jobs use GPU 0;
both sandwich jobs use GPU 1 during simulation and analysis.

```bash
CONDA_OVERRIDE_CUDA=12.9 \
pixi run python -u -m hexatic.confinement_comparison.run_pipeline \
  --sim \
    prism_surface_area \
    sandwich_volume \
    sandwich_surface_area \
    two_dimension \
  --gpu-ids 0,1 \
  --simulation-workers 4 \
  --analysis-workers 4 \
  --backend auto \
  --output-root /mnt/drive3/vaibhav_data/confinement_comparison_production_run1 \
  --require-gpu \
  --overwrite
```
