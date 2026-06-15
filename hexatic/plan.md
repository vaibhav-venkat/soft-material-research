# Active-Matter Cylinder Output And Analysis Plan

## Summary
Add the missing simulation outputs in `hoomd_cylinder.py`, then extend `dynamic_info_cylinder.py` to compute active-matter local density, polar field, axial flux `J_x(x,t)`, and `rho_dot = -dJ_x/dx`. Ignore the PDF.

## Key Implementation Changes
- In `hoomd_cylinder.py`, keep particle orientations as HOOMD quaternions `(w, x, y, z)`, store them as a normalized `(N, 4)` array, and pass `dynamic=["property", "particles/orientation"]` to `hoomd.write.GSD` so updated orientations are written every frame.
- Add a HOOMD logger for the pair LJ object only:
  - `logger = hoomd.logging.Logger(categories=["particle"])`
  - log `lj` quantities `["forces", "virials"]`
  - pass `logger=logger` to the GSD writer.
- Add cylinder constants for active analysis:
  - `LOCAL_POCKET_RADIUS = 2.0 * PARTICLE_DIAMETER`
  - `ACTIVE_FIELD_X_BINS = 100`
  - output paths under `output/cylinder` for `.npz` data and under `output/images/active` for plots.

## Dynamic Analysis Behavior
- Read positions, orientations, LJ forces, and virials from the GSD. If LJ force/virial logs are missing, raise a clear error naming the expected logged quantities.
- Convert quaternion orientation to active direction `q_i` by rotating the body-frame x-axis:
  - `q = (1 - 2(y^2 + z^2), 2(xy + wz), 2(xz - wy))`
  - normalize quaternions before conversion.
- Compute both domains:
  - `all`: every particle, pocket distance is 3D Euclidean with periodic minimum image in x.
  - `shell`: existing outer-wall mask, pocket distance is cylinder-surface distance using `(x, R theta)` with periodic x and theta.
- Replace each ideal Dirac delta in local particle fields with an indicator kernel:
  - `chi_a(d) = 1` when `d <= LOCAL_POCKET_RADIUS`, otherwise `0`.
  - `d(i, core)` is the distance between particle `i` and the core particle defining the local pocket.
  - for `all`, `d(i, core)` is minimum-image 3D distance.
  - for `shell`, `d(i, core) = sqrt(dx_periodic^2 + (R * dtheta_periodic)^2)`.
- For each core particle pocket, apply that indicator kernel:
  - `rho_count(core,t) = sum_i chi_a(d(i, core))`
  - `polar_sum(core,t) = sum_i q_i(t) chi_a(d(i, core))`
  - `polar_mean = polar_sum / rho_count`, with zeros or NaNs only when a pocket is empty.
- For axial flux:
  - bin particles by periodic x into `ACTIVE_FIELD_X_BINS`; this x-bin indicator is the discrete delta approximation in `J_x(x,t)`.
  - compute `rdot_i = U0 * q_i + (1 / gamma) * F_LJ_i`.
  - compute `J_x_sum[x_bin,t] = sum_i rdot_i,x(t) chi_bin(x_bin - x_i(t))`.
  - `chi_bin = 1` when particle `i` lies in the x bin, otherwise `0`.
  - compute `J_x_density = J_x_sum / dx`.
  - compute `rho_dot = -dJ_x_density/dx` using periodic centered finite differences.

## Outputs
- Save compressed numeric outputs for both `all` and `shell` domains:
  - local pocket fields: steps, particle/core ids, positions or x/theta/r, `rho_count`, `polar_sum`, `polar_mean`.
  - axial fields: steps, x-bin edges/centers, `J_x_sum`, `J_x_density`, `rho_dot`.
- Save plots for both domains:
  - x-time heatmap of binned average local density.
  - x-time heatmap of binned polar magnitude, plus optional x/y/z component line summaries.
  - x-time heatmap of `J_x_density`.
  - x-time heatmap of `rho_dot`.

## Test Plan
- Run a short HOOMD smoke trajectory and confirm every frame contains non-default `particles/orientation` and logged LJ `forces`/`virials` with shapes `(N, 3)` and `(N, 6)`.
- Unit-check quaternion conversion with identity and simple 90-degree rotations.
- Unit-check pocket density/polar fields on a tiny synthetic particle set with known neighbors.
- Unit-check `rho_dot` using a known periodic `J_x = sin(2πx/L)` profile.
- Run `dynamic_info_cylinder.py` on the smoke GSD and verify all `.npz` files and active-field plots are produced.

## Assumptions
- Use pair LJ forces only for `F_ij`, not wall forces and not total forces.
- Use 1D axial flux `J_x(x,t)` and `rho_dot = -dJ_x/dx`.
- Save both all-particle and outer-shell analyses.
- Treat all delta approximations as unnormalized indicator kernels: `chi_a` for local pockets and `chi_bin` for axial flux bins. Density normalization can be derived later from saved counts if needed.
