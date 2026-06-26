# Implementation Plan: Disclination Birth/Death Dynamics & Event-Centered Plots

This plan implements `INSTRUCTIONS.md` for `hexatic/multiple_sim_analysis/disclination_order_fields/`.
It reuses the existing infrastructure described in `ARCHITECTURE.md` and follows `AGENTS.md`
(Pixi env `sap`, numba where reasonable, scipy for heavier math, snake_case, dataclasses,
load existing `.npz`/`.gsd` instead of recomputing, radius-aware parameters, no overwriting
generated data without `--overwrite`).

Reference hypothesis: defect pattern changes are **source/reaction dominated** (birth / death /
annihilation / cluster rearrangement) rather than smooth defect drift.

---

## 0. Scope, inputs, and shared conventions

### 0.1 What exists and is reused (do not rewrite)
- `hexatic/multiple_sim_analysis/common.py`: `FRAME_START`/`FRAME_STOP`, `frame_indices`,
  `finite_nanmean`, `minimum_image_delta`, `unwrapped_x_positions`, `active_fields_path`,
  `neighbor_counts_path`, `hexatic_velocity_gsd_path`, `save_metric_npz`,
  `load_cached_metric_values`, `load_metric_values`, `load_active_fields`,
  `trajectory_frame_count`, output dir tree (`NPZ_OUTPUT_DIR`, `PLOT_OUTPUT_DIR`,
  `FIT_OUTPUT_DIR`).
- `hexatic/disclination.py`: `_load_neighbor_counts` (frame×particle count table).
- `hexatic/radii_analysis/cases.py`: `RadiusCase` (carries `radius`, `lx`,
  `trajectory_write_period`, `run_steps`), `get_case`, `all_cases`, the `_20_0D`
  / `radius_*` case identifiers, `HEXATIC_OUTPUT_DIR`, `NPZ_FIELDS_DIR`.
- `hexatic/constants/cylinder.py`: `NEIGHBORS`, `PARTICLE_DIAMETER`,
  `ANALYSIS.neighbor_count_radius`, `ANALYSIS.wall_cutoff`, `RHO`,
  `lx_for_radius`, `n_particles_for_radius`.
- `hexatic/multiple_sim_analysis/plotting.py`: `plot_radius_values`,
  `plot_for_cases`, `plots_missing`, `companion_circumference_plot_path`.
- `hexatic/multiple_sim_analysis/disclination_order_fields/shared.py`:
  cell-list helpers (`_cell_index`), `_disclination_mask`,
  `_load_hexatic_abs`, `_validate_particle_frame_shape`,
  `CIRCUMFERENCE_REFERENCE_CASE_ID`, `LOCAL_CONTRAST_LENGTH`,
  `local_disclination_field_contrasts`, `local_disclination_field_profiles`.
- Existing field producers already cached on disk as `.npz`:
  `chirality.py`, `nematic.py`, `radial_exchange_current.py`, `force_density.py`,
  `density_profile.py`, plus the active-matter fields `.npz` produced by
  `active_matter_cylinder`. The plan loads these; it does **not** regenerate them.
- Important cache caveat: most existing `multiple_sim_analysis/output/npz/*.npz`
  files are aggregate summaries, not frame×particle local fields. For event-centered
  local sampling, recompute or load particle-local arrays from primary sources
  (`active_matter_fields`, hexatic/neighbor-count tables, trajectory GSDs) rather
  than assuming aggregate `.npz` files can provide per-particle values.

### 0.2 Output layout (new)
All new outputs go under the existing analysis tree so caching/overwrite logic stays uniform:

```
hexatic/multiple_sim_analysis/disclination_order_fields/
  events/                       # new package: defect tracking + events
    __init__.py
    tracking.py                 # frame-to-frame defect identity matching
    events.py                   # birth/death/annihilation classification
    fields.py                   # event-centered local-field samplers
    maps.py                     # representative-frame map panels (section 14)
    plotting.py                 # event-centered + probability plots
    runner.py                   # orchestrates the whole pipeline + CLI hook
  output/                       # already exists
    npz/
      defect_tracks_<case>.npz
      defect_events_<case>.npz
      event_fields_<case>.npz
      cluster_fields_<case>.npz
      radial_summary_R20D.npz   # section 5 / 9 quantities vs R
      probability_summary_R20D.npz
    plots/
      events/
      maps/
      vs_R/
      probability/
      radial_exchange/
      chirality/
```

### 0.3 Time convention
- `Δt` (real time between two *saved* frames) =
  `(steps[t+1] - steps[t]) * timestep`. `steps` come from the `.gsd`/`.npz`
  trajectory; `trajectory_write_period` lives on `RadiusCase`. Frames in the
  event tables are stored **both** as frame index and as `step` number so the
  `t_b − τ, t_b, t_b + τ` windows are unambiguous (per INSTRUCTIONS §2.3).
- `τ` for event-centered plots is expressed in saved frames (default `τ = 2`
  frames) and converted to steps when writing the figure axis labels.

### 0.4 Key length scales (radius-aware, from constants)
- `a = cylinder.ANALYSIS.neighbor_count_radius`.
- Annihilation distance `d_ann = 2.5 * a`.
- Annihilation time window `≈ 2` saved frames.
- Persistence threshold `k = 2` consecutive frames for a birth/death to count.
- Cluster bond length `ℓ_cluster = 4.5 * a`.
- Annulus: `core = d < a`, `annulus = a < d < 3a`, excluding particles that are
  themselves disclinations.

---

## 1. Cleanup (INSTRUCTIONS §2.1)

1. Read `hexatic/multiple_sim_analysis/disclination_order_fields/` to inventory the
   current contents (already done during planning: `__init__.py`, `contrast.py`,
   `moving.py`, `runner.py`, `shared.py`, `shell_profiles.py`,
   `velocity_summary.py`).
2. Delete the files inside this directory (their public entry points are summarized
   in `__init__.py`). Keep the directory itself.
3. Recreate the package with the new `events/` subpackage and a new top-level
   `runner.py` + `shared.py` (shared must retain the helpers from §0.1: cell list,
   `_disclination_mask`, `_load_hexatic_abs`, `_validate_particle_frame_shape`,
   `local_disclination_field_contrasts`, `local_disclination_field_profiles`,
   `LOCAL_CONTRAST_LENGTH`). These helpers are imported by sibling modules outside
   the focus directory? — confirm before deleting; if anything in
   `multiple_sim_analysis` still imports them, keep `shared.py` intact and only
   remove the obsolete driver modules (`contrast.py`, `shell_profiles.py`,
   `velocity_summary.py`) whose jobs are subsumed by the new pipeline.
4. Do **not** touch `hexatic/output/` simulation GSDs or the radii_analysis
   metadata. Never delete generated data without an explicit flag.

---

## 2. Persistent defect tracking (INSTRUCTIONS §2.2)

**File:** `events/tracking.py`

### 2.1 Per-frame defect lists
- Load `neighbor_counts` via `_load_neighbor_counts(neighbor_counts_path(case))`.
- `charges = cylinder.NEIGHBORS - counts` → `+1` / `−1` / `0`.
- Build per-frame arrays of defect particle indices, their charge, and their
  particle coordinates. Coordinates are taken from the cached coords array used
  by the active-matter fields (frame×particle×(x,θ,r)); reconstruct Euclidean
  `(x, y=r·sinθ, z=r·cosθ)` for distance math. Use `minimum_image_delta` on `x`
  (`box_length_x = case.lx`).
- Before tracking, assert active-field coords frame/particle order exactly matches
  neighbor-count tables and GSD particle indices: same frame count, same particle
  count, matching step numbers, and stable particle-index ordering. Abort with a
  clear error if this fails.

### 2.2 Greedy nearest-neighbour matching
Implement `_match_defects_frame(prev, cur, box_length_x)` in regular Python/NumPy:
- Build a residual/candidate list over previous-frame defects per charge sign.
- For each previous defect, find nearest same-sign defect in the current frame
  within `match_tol = a * 1.5` (configurable; record ambiguous candidates).
- Greedy assignment by ascending distance (scipy `linear_sum_assignment` for the
  remaining ambiguous pairs if costs are small).
- A match is **confident** if the nearest distance is `< match_tol` and the
  second-nearest is at least `1.5×` the nearest (reject ambiguous identity swaps).
- Output: arrays of `(prev_idx, cur_idx, distance)` pairs, plus unmatched lists on
  each side → these feed birth/death.
- Do not use `@njit` for the matching function if it calls scipy
  `linear_sum_assignment`; scipy cannot run inside numba. Keep the scipy assignment
  path in Python, or replace it with a numba-compatible pure greedy assignment.
- `match_tol = 1.5a` may be fragile at saved-frame spacing; too small creates false
  births/deaths, too large causes swaps in clusters. Make it configurable and report
  ambiguity/rejection counts.
- Rejecting ambiguous identity swaps is good, but the downstream birth/death
  classifier must not count every rejected swap as a physical event.

### 2.3 Track objects
Dataclass `DefectTrack`:
```
track_id, charge, frame_start, frame_stop, particle_index[frame],
positions (x,θ,r per frame), steps per frame
matched_confident[frame bool], velocity (x,θ,r per frame)  # NaN where not confident
annihilation_flag, annihilation_partner_id
```
- Velocity only computed for confidently matched consecutive frames:
  `v_i(t) = (X_i(t+Δt) − X_i(t)) / Δt`, with `Δt` in real units.
- New-born / dying / ambiguous frames get `NaN` velocity and are flagged.
- Persist to `output/npz/defect_tracks_<case>.npz` via `save_metric_npz`
  (store arrays, not the dataclass, plus `frame_start`/`frame_stop`/case metadata).

---

## 3. Birth / death / annihilation events (INSTRUCTIONS §2.3–2.4)

**File:** `events/events.py`

### 3.1 Birth
- A defect appearing at `t_b` is a birth iff it persists `≥ k=2` frames
  (including origin frame) — i.e. a new track with `frame_stop − frame_start ≥ 1`
  in saved-frame units.
- Record per birth event:
  `birth_pos (x,θ,r)`, `birth_frame`, `birth_step`, `charge`,
  `nearest_opposite_defect_index + distance`, `nearest_same_defect_index + distance`.
- The "nearest opposite/same" are computed *within the birth frame* using the
  defect-excluded annulus rule (section 3 of INSTRUCTIONS) so clustered defects
  don't bias each other.

### 3.2 Death
- A defect that existed `≥ k=2` frames then disappears (unmatched on the
  "previous side" with no continuation). Deaths cannot be newly born defects that
  fail the persistence rule; they can only be defects that were "birthed", i.e.
  survived `k` consecutive frames including their origin frame.
- Record: `death_pos`, `death_frame`, `death_step`, `charge`,
  `nearest_opposite_defect_index + distance` *in the frame before death*,
  `annihilation_bool`.

### 3.3 Annihilation classifier
- For each death, scan opposite-charge defects in the ~2 frames around death:
  pair is annihilating iff `min distance over the window ≤ d_ann = 2.5*a`
  **and** both disappear within the 2-saved-frame window.
- Mark both member tracks' `annihilation_flag` and store partner id.
- Deaths that pair with a birth (a `+` and `−` appearing simultaneously close)
  are *not* annihilations; keep them as birth events.
- Annihilation matching can be many-to-one in dense clusters; the plan needs
  tie-breaking by track ID/distance/time.
- "Deaths that pair with a birth are not annihilations" may hide
  replacement/reaction events unless separately labeled.

### 3.4 Output
`output/npz/defect_events_<case>.npz` with parallel arrays:
`birth_frames, birth_steps, birth_x, birth_theta, birth_r, birth_charge,
 birth_nearest_opp_dist, birth_nearest_same_dist,
 death_frames, death_steps, death_pos..., death_charge,
 death_nearest_opp_dist, death_annihilated, death_partner_idx`.

---

## 4. Clean local neighborhoods (INSTRUCTIONS §3)

**File:** `events/fields.py`

### 4.1 Defect-excluded annulus
For a target defect `i`:
```
core_mask     : d_ij < a
annulus_mask  : a < d_ij < 3a
d_ij          : Euclidean/cylinder-aware distance from target defect to particle j
exclude       : particles that are themselves defects (charge != 0) from the annulus
```
Use the existing cell list (`shared._cell_index`) with `cell_size = a`.
If the annulus has zero non-disclination particles after exclusion, record the
annulus average as `NaN` unless a downstream count/rate explicitly requires zero;
always persist the contributing count so empty-neighborhood cases are visible.

### 4.2 Background velocity
For each annulus compute `u_mean = |mean(v_j)|`, `u_rms = mean(|v_j|)`,
`u_fluct = sqrt(mean(|v_j − mean(v_j)|²))`. Existing `velocity.py` `.npz` outputs
are aggregate summaries, not frame×particle velocity arrays, so derive
frame×particle velocities from active-field `coords`/`steps` or the GSD as needed.

### 4.3 Local-field samplers
Provide a single sampler that, given `(case, frame, target_xy)` returns a
dict of annulus-averaged scalars (reuse particle-local field arrays loaded once
per case in the runner; recompute from primary sources where only aggregate `.npz`
files exist):
- `rho`, `J_r` (radial exchange current), `D²_min`,
- `S` (tangent nematic order), `Q` tensor components,
- `|ψ6|` (hexatic order), `χ` (chirality, from `chirality.py` cache),
- `stress/flow`: `F_density` (from `force_density.py` + wall term),
  `u_rms`, `u_fluct`, `nearest_defect_distance`.
- `strain` for the birth window is taken as cumulative `D²_min` over the lag.
- `rho`, `J_r`, `S`, `Q`, `F_density`, etc. mix particle-local arrays, radial-bin
  arrays, and aggregate arrays. The sampler needs a clear interpolation/binning
  convention.

### 4.4 D²_min (per INSTRUCTIONS §4.5)
New computation if not already cached: for particle `i`, neighbors `j` at `t`,
compare to `t+Δt`, solve least squares for `F_i` (scipy `scipy.linalg.lstsq` on the
per-neighbor displacement vectors), `D²_min(i) = (1/n) Σ |Δr_ij − F_i r_ij|²`.
Wrap the inner summation in numba (`@njit`) for the per-frame traversal, with
the 3×3 least-squares solved by a small closed-form normal-equation matrix
(numba-compatible; avoid scipy inside the hot loop). Cache the resulting
frame×particle array to `output/npz/D2min_<case>.npz` (separate file so it can be
reused across plots without recompute gated on `--overwrite`).
- Only calculate `D²_min` up to the second-to-last frame because it compares frame
  `t` to `t+Δt`.
- Computing `D²_min` requires actual neighbor identities, not just neighbor counts.
- `D²_min` is undefined on the final frame and can be unstable for too few or
  nearly collinear neighbors.
**UPDATE**:
For shell particles (defects), compute D^2_minin local physical coordinates: `X = (x, Rθ)`
Also normalize `D²_min / a²`

---

## 5. Aggregate per-frame and per-radius quantities (INSTRUCTIONS §4–5)

**File:** `events/runner.py` (case-level) + top-level `runner.py` (R sweep).

### 5.1 Per-frame per-case (sect. 4)
For each `(R)` and frame compute:
- `N_+, N_−, N_total` and binned densities `n_+(x,θ,t), n_−(x,θ,t)`
  (use the active-fields grid `(x_centers, θ_centers)` and bin defects).
- Event counts per frame: `B_+, B_−, D_+, D_−, A_+, A_−`, `A_top`,
  `a_top = A_top / N_total` with a zero-defect guard (`NaN` when `N_total == 0`,
  unless a plot explicitly requires zero).
- Defect motion fields: `|v_defect|, v_x, v_θ, net displacement`, plus
  `v_parallel = v·û_local`, `v_perp`, `cos(v_defect, u_local)` (local direction
  from cached active direction field).
- Clusters (next subsection).
- Active-field `rho` is currently particle-local pocket density, not an `(x, theta)`
  grid, so defect density binning needs its own grid arrays.
- `v_theta` must be converted consistently, likely `R * dtheta/dt`, before dot
  products with tangent directions.

### 5.2 Clusters (sect. 4.4)
- Union-find over same-frame defects with bond `dist < ℓ_cluster = 4.5a` (numba
  implementation, periodic in `x`).
- Per cluster: `size, charge = N_+ − N_−, total = N_+ + N_−, COM, velocity`.
- Persist `output/npz/cluster_fields_<case>.npz`.
- Cluster COM on periodic `x` and angular `theta` needs circular/periodic
  averaging; naive means will fail across boundaries.

### 5.3 Radius plots vs R (sect. 5.1–5.2)
Reuse `plot_radius_values` over all `radius_*` cases plus the `circ_60` reference
excluded case as needed (`CIRCUMFERENCE_REFERENCE_CASE_ID`). Plots:
- `mean N_+(R), mean N_−(R)`, `B(R), D(R), A(R)`, `A_top(R)`, `mean lifetime(R)`.
- `A_top(R)`, `median |v_defect|(R)`, `median track lifetime(R)`,
  `median displacement before death(R)`.
- `motion_activity(R) = median(|v_defect|) / a` and the combined
  `a_top(R) / motion_activity(R)` panel.

Persist the R-summary to `output/npz/radial_summary_R20D.npz`.

---

## 6. Event-centered plots (INSTRUCTIONS §6) — primary deliverable

**File:** `events/plotting.py` (R = 20D case only, per INSTRUCTIONS §6.1).

### 6.1 Birth-triggered averages
For each birth at `t_b`, collect annulus scalars at frames `t_b − τ .. t_b .. t_b + τ`
(`τ` frames; default 2) at the future birth location X_b using local spatial interpolation or nearest local particles.
Average over all births. Stitch timesteps/frames via the sampler in §4.3.
Event windows near frame 0 or the last frame need truncation or exclusion rules.

Plots (relative time on x):
`ρ, J_r, u_rms, u_fluct, S, |ψ6|, χ, D²_min, strain` vs `t − t_b`.
Mandatory examples from INSTRUCTIONS: `|ψ6|, χ, D²_min, u_rms, J_r, ρ`.

### 6.2 Death-triggered averages
Same machinery at the dying-defect position. Mandatory plots:
`nearest opposite-charge distance, |ψ6|, χ, D²_min, u_rms, density` vs `t − t_d`.

### 6.3 Separation of annihilation vs non-annihilation death
Split death curves into `annihilated` and `non-annihilated` subpopulations
(classified in §3.3); plot both with distinct colors so the source/reaction
hypothesis is testable.

---

## 7. Radial exchange / shell-build-up plots (INSTRUCTIONS §9, R = 20D)

- `J_r = ρ v_r`, split into `J_in` (shell→core), `J_out` (core→shell), `J_abs`.
- Compute four event-aligned series: `J_r near future birth`, `near future death`,
  `near stable defects`, `far from defects` (control).
- Plots:
  `birth rate vs |J_r|`, `birth rate vs J_in`, `birth rate vs J_out`,
  `death rate vs |J_r|`, `defect density vs shell density`,
  `birth rate vs shell thickness`.
- Shell mask from `common.shell_mask_for_positions`; shell thickness computed as
  `mean(r_outer − r_inner)` via existing density/thickness profile utility or, if
  missing, a light helper added to `density_profile.py`.
- `common.shell_mask_for_positions` exists, but "shell thickness" utility may not;
  this may become new implementation work.

---

## 8. Probability plots (INSTRUCTIONS §10.2, R = 20D)

Bucket birth/death/annihilation events by the local pre-event field value at the
event position in the frame before the event; normalize by total time-exposure
of all particles in that same bucket (control) → probability.

Plots:
`birth prob vs |ψ6|, vs 1−|ψ6|, vs |∇ψ6|`,
`death prob vs |ψ6|`, `annihilation prob vs |ψ6|`.
`|∇ψ6|` computed by central differences of the cached `|ψ6|` frame grid.
Probability normalization by "all particles in same bucket" needs care to avoid
mixing particle-local fields with event-location fields.
Gradients need periodic handling in `x` and `theta`, plus smoothing/NaN handling.

Output: `output/npz/probability_summary_R20D.npz` + plots under `plots/probability/`.

---

## 9. Chirality plots (INSTRUCTIONS §11, R = 20D)

Same probability machinery with chirality fields:
`birth prob vs χ, vs |χ|, vs ∂tχ, vs |∇χ|`,
`death prob vs χ`, `annihilation prob vs χ`.
Event-centered: `χ(t−t_birth), χ(t−t_death), χ_annulus(t−t_birth), χ_annulus(t−t_death)`.

---

## 10. Maps for representative frames (INSTRUCTIONS §14)

**File:** `events/maps.py`. For `R = 20D` (and a couple representative frames):
panels `particles+defects`, `|ψ6|+defects`, `χ+defects`, `u_rms or D²_min+defects`,
`J_r+defects`, `density+defects`, and an overlay of birth/death events for that frame.
Use matplotlib scatter on unwrapped `(x, y)` with colorbars; reuse plotting style
already present in `multiple_sim_analysis/plotting.py`.

---

## 11. Wiring & CLI

- The new top-level `disclination_order_fields/runner.py` exposes
  `run(cases, frame_start, frame_stop, overwrite)` returning a summary dict,
  following the existing `run` signature expected by `multiple_sim_analysis/script.py`.
- Extend `script.py`/`__main__.py` CLI with flags:
  `--overwrite`, `--case`, `--all`, `--frame-start`, `--frame-stop`,
  `--tau` (event half-window in frames), `--event-case radius_20D`
  (the case used for event-centered plots; configurable).
- Default event case = `radius_20D` (verify exact `case_id` in `cases.py`;
  grep confirms radius cases are formatted `radius_<N>D`).
- Caching discipline: each `.npz` write goes through `save_metric_npz`; reads go
  through `load_cached_metric_values`/`load_metric_values`; plots gated via
  `plots_missing` so re-runs are cheap (consistent with `disclination.py` pattern).

---

## 12. Testing (light, in `tests/`)

- `test_defect_tracking.py`: synthetic 2-frame neighbor counts + positions,
  assert confident matches, birth/death classification, annihilation flag when a
  `+`/`−` pair co-annihilates within `d_ann`.
- `test_d2min.py`: known affine shear → `D²_min ≈ 0`; random displacement → `> 0`.
- `test_annulus.py`: defect-excluded annulus uses `d`, excludes defect particles,
  and records an empty annulus as `NaN` plus zero contributing count.
- `test_clusters.py`: union-find bond length gating over periodic `x`.
- Run with `pixi run python -m pytest tests/` and gate syntax with
  `pixi run python -m compileall hexatic/multiple_sim_analysis`.

---

## 13. Execution order (build increments)

1. **Cleanup** (sect. 1) — preserve `shared.py` helpers; remove obsolete drivers.
2. **Tracking** (sect. 2) — `defect_tracks_<case>.npz`; unit test.
3. **Events** (sect. 3) — annihilation classifier; unit test.
4. **D²_min** (sect. 4.4) — cached `.npz`; unit test.
5. **Local-field sampler** (sect. 4.1–4.3) — reuse existing field caches.
6. **Cluster + per-frame aggregate** (sect. 5.1–5.2).
7. **R-sweep plots** (sect. 5.3).
8. **Event-centered averages** (sect. 6) — headline plots.
9. **Radial exchange / shell plots** (sect. 7).
10. **Probability + chirality plots** (sect. 8, 9).
11. **Maps** (sect. 10).
12. **CLI wiring + caching** (sect. 11).
13. **Tests & `compileall`** (sect. 12).

Each step is independently cacheable and runnable; jobs that need only print
checks can run via `pixi run python -m hexatic.multiple_sim_analysis --help`
without launching GPU sims (per `AGENTS.md`).
