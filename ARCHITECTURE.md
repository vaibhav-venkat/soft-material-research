# Architecture Overview

**DO NOT BE VERBOSE** **BE CONCISE** **LIMIT COMMENTS** **BE CLEAN AND SIMPLE**

## Rust workspace

The rho-fitting native path is a Cargo workspace with a one-way dependency
direction:

```text
packages/rho-fitting-core       PyO3 + NumPy adapter
             |              \
             v               v
crates/rho_fitting_numerics   crates/rho_fitting_gpu
             |
             v
      crates/rho_fitting_types
```

`rho-fitting-types` owns the cylindrical grid, physical component order
`(x, e_theta, e_r)`, mechanical field containers, validation errors, and
targets. `rho-fitting-numerics` owns Python-free fitting, spectral, temporal,
interpolation, particle, regression, and PDE algorithms. `rho-fitting-gpu`
owns Burn backend selection and Gaussian deposition. The package under
`packages/rho-fitting-core` only validates/converts NumPy arrays and translates
Rust errors into Python exceptions, while preserving the import path
`hexatic.rho_fitting._rho_fitting_core`.

## 2. High-Level System Diagram
 
[User] <--> [Plots/fits dynamics and fields] <--> [Runs multiple_sim_analyis] <--> [GSD, NPZ, and archive files]

## 3. Unique Components

- Numba: aim to use this extensively unless it increases too much complexity.
- Scipy: aim to use this instead of numpy for more mathematically heavy data structures
- DO not make overcomplicated files, classes, and verbose functions
- Pixi:
  - env_name: sap, default
  - command: `pixi shell`
  - contained in [pixi.toml](~/structures_and_functions/pixi.toml)
- **NEW**: Rust. TBD

# ANALYSIS + PLOTS
Focus on [density analysis](~/structures_and_functions/hexatic/density_analysis).

Summary of the structure:
**GSD**

| Path | Contains | Main fields |
|---|---|---|
| `gsd/trajectory_radius_15D.gsd` | Original HOOMD trajectory | `position (N,3)`, `orientation (N,4)`, `velocity (N,3)`, `typeid`, `box`, `step` |
| `hexatic_output/radius_15D_hexatic_velocity.gsd` | Trajectory with hexatic-related velocity/output stream | same GSD particle fields: `position`, `orientation`, `velocity`, `typeid`, `box`, `step` |

Both currently have:

```text
100 frames
9870 particles
steps starting at 100000
radius = 15D
```
IMPORTANT. in this sim, Lx is not a function, unlike within the constants folder, its this:
FIXED_LX = 4000 / (RHO * np.pi * BASELINE_CYLINDER_RADIUS**2)

**Hexatic Output**

| Path | Contains | Columns |
|---|---|---|
| `hexatic_output/radius_15D_hexatic_order.txt` | Per-particle hexatic order | `frame`, `step`, `particle`, `psi_real`, `psi_imag`, `psi_abs` |
| `hexatic_output/radius_15D_neighbor_counts.txt` | Per-particle neighbor count | `frame`, `step`, `particle`, `neighbor_count` |

Useful derived meaning:

```text
psi_abs      -> local |psi6|
neighbor_count -> disclination charge via 6 - neighbor_count
```

**NPZ Files**

All are in:

```text
hexatic/density_analysis/npz/
```

| File | What It Stores | Important Arrays |
|---|---|---|
| `radius_15D_active_matter_fields.npz` | Particle-local active matter fields over time | `steps`, `coords (frame, particle, x/theta/r)`, `shell_mask`, `rho`, `active_direction`, `direction_cylindrical`, `polar_mean`, `polar_cylindrical`, `flux_cylindrical`, `force_density`, `force_density_cylindrical`, grid edges/centers |
| `radius_15D_translation_chirality_fields.npz` | Particle-local translation chirality | `steps`, `chirality (frame, particle)`, `neighborhood_radius` |
| `radius_15D_shell_bond_translation_chirality.npz` | Frame-level shell/bond chirality summary | `steps`, `mean_abs_bond_translation_chirality`, `bond_counts`, `neighborhood_radius`, `cylinder_radius` |
| `radius_15D_shear_flux_decomposition.npz` | One-frame gridded stress/flux decomposition | `grid_coords`, `grid_points`, `rho_density`, `sigma_full`, `sigma_normal`, `sigma_shear`, `div_sigma_*`, `polar_density`, `pair_force_density`, `wall_force_density`, `j_active`, `j_normal`, `j_shear`, `j_wall`, `j_total`, correlations/slopes |
| `radius_15D_shear_flux_decomposition_series.npz` | Time series version of shear/flux decomposition | same fields as above, but with leading `frame` dimension, plus `steps`, `frame_indices` |
| `radius_15D_cartesian_flux_comparison.npz` | Cartesian grid comparison of instantaneous vs finite-difference flux/stress | `grid_points`, `rho_density`, `polar_density`, `force_density`, `instantaneous_flux_density`, `finite_difference_flux_density`, `virial_stress_density`, `virial_divergence_density`, finite-time stress/flux fields |

**Most Useful Field Sources**

| Quantity | Best Source |
|---|---|
| Particle positions / orientations / velocities | `gsd/trajectory_radius_15D.gsd` |
| Cylindrical particle coords `(x, theta, r)` | `npz/radius_15D_active_matter_fields.npz -> coords` |
| Density near particles | `active_matter_fields.npz -> rho` |
| Active direction / polarization | `active_matter_fields.npz -> active_direction`, `polar_*` |
| Flux/current | `active_matter_fields.npz -> flux_cylindrical`; gridded flux from `shear_flux_decomposition*.npz -> j_*` |
| Force density | `active_matter_fields.npz -> force_density*`; gridded from `shear_flux_decomposition*.npz` |
| Chirality | `translation_chirality_fields.npz -> chirality` |
| Shell chirality summary | `shell_bond_translation_chirality.npz` |
| Hexatic order `psi6` | `hexatic_output/radius_15D_hexatic_order.txt` |
| Neighbor counts / disclinations | `hexatic_output/radius_15D_neighbor_counts.txt` |
| Stress tensors | `shear_flux_decomposition*.npz -> sigma_*` |
| Stress divergence | `shear_flux_decomposition*.npz -> div_sigma_*` |

**Key Convention**

Most particle-local arrays use:

```text
(frame, particle, component)
```

Most gridded arrays use either:

```text
(grid_point, component)
```

or, for series files:

```text
(frame, grid_point, component)
```
