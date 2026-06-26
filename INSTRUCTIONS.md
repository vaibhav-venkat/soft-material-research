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

## Calculation:
Use the cylindrical coordinates already stored in:

`hexatic/density_analysis/npz/radius_15D_active_matter_fields.npz`

with:
``
coords[frame, particle] = (x, theta, r)
shell_mask[frame, particle]
``
Use shell_mask with the definition of the shell_thickness within the constants folder, or another folder and file may already contain it.

Use `y = R θ` so `(x, y) = (x, Rθ)`. You will do everything over the film:
- Choose bins in x and theta, with surface area `A = delta x * R * delta theta`
- For a bin b, `ρ_film,b(t) = N_b(t) / A_bin`
  - Also calculate over discrete frame-difference: `partial_t rho_film_b`
- Calculate `J` from the particles: 
  - `J_x,b(t) = (1 / A_bin) Σ_{i in bin b} v_x,i(t)`
  - `J_y,b(t) = (1 / A_bin) Σ_{i in bin b} v_y,i(t)`
  - `v_y = R dθ/dt`
- Use the velocity not directly from the *.gsd, but rather by calculating the `delta x` and `delta y` over each frame, so its discrete.
  - `v_x,i(t) = [x_i(t+1) - x_i(t)] / Δt`
  - `v_y,i(t) = R [θ_i(t+1) - θ_i(t)] / Δt`

- Do the same for rho_film, b
- Next compute -div * J_film:
  - `neg_div_J_film,b(t) = -dJx_dx - dJy_dy`
    - Where `y = R theta` and both are calculated over finite-difference 1 frame. 
## S_cross
Define membership:
`m_i(t) = 1` if i is in the film, `0` if not, (just the shell_mask)
``
m_i(t) = 0 and m_i(t+1) = 1  → entry event
m_i(t) = 1 and m_i(t+1) = 0  → exit event
``
Then for bin `b`:
`S_cross,b(t) = [N_in, b(t) - N_out,b(t)] / [A_bin delta t]`
``
N_in,b  = number of particles entering the film in bin b
N_out,b = number of particles leaving the film in bin b
A_bin   = Δx · R Δθ
Δt      = time between saved frames
``
For an entry event bin it using the position at `t+1`, exit is at `t`


- Store J_film_b, rho_film_b, neg_div_J_film_b, S_cross, partial_t rho_film_b, across all frames and bins, preferably in another *.npz cache than you can use for subsequent runs
- Plot these values in a (x, theta) map, with color corresopnding to value:
  - partial_t rho_film_b, neg_div_J_film_b, S_cross, and then the sum of those. 
  - 4 maps total. Plot the magnitude as the color, and for (x, theta) there should be an arrow telling about the direction
- Use PLOTLY, not MATPLOTLIB
