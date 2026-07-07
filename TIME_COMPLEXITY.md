# Time Complexity Plan: Localize Work and Avoid Global Scans

This plan is a companion to `PLAN.md`. Its single guiding principle is:
**stop doing global scans when the operation is local.** The target end state
pushes the pipeline from roughly `O(FN + FD^2 + EN)` toward:

```text
O(FN)  for basic scans
+ O(FD)  for tracking / clustering in sparse regimes
+ O(EK)  for event sampling
```

with `D2min` only as broad as actually needed.

Symbols:
- `F` = number of frames
- `N` = particles per frame
- `D` = defects per frame
- `T` = number of tracks
- `E` = number of events (birth/death/annihilation)
- `K` = local particle/defect count within the query neighborhood
- `a` = `cylinder.ANALYSIS.neighbor_count_radius` (key length scale)

---

## 1. Cell lists for defect matching and clustering

**Problem.** Current defect matching (`tracking.py`) and clustering (`runner.py`)
compare defects against all other defects, giving `O(FD^2)`.

**Change.** Bucket per-frame defects into spatial cells of size near
`match_tol` (matching) or `cluster_bond_length = 4.5a` (clustering). Only compare
defects in nearby cells. Honor periodic boundaries in `x` (case `lx`) and the
angular `theta` wrap.

```text
O(FD * local_defect_count)   # close to O(FD) when defects are sparse
```

**Scope.**
- `events/tracking.py :: _match_defects_frame` — replace all-pairs candidate
  building with a cell-list neighbor query sized to `match_tol = 1.5a`
  (configurable).
- `events/runner.py` cluster step (union-find, `dist < 4.5a`) — build the cell
  grid at `cluster_bond_length` and only bond defects in adjacent cells.

**Constraints.**
- Reuse `shared._cell_index`; do not introduce a second cell-list implementation.
- Cell geometry must be cylindrical-aware: `x` is periodic with period `case.lx`,
  `theta` is periodic with period `2π`. Distances use `minimum_image_delta` on `x`.
- Keep the existing `match_tol` configurability and ambiguity/rejection reporting
  from `PLAN.md §2.2`.

---

## 2. Sparse track storage

**Problem.** A `DefectTrackTable` with arrays shaped `tracks × frames` costs
`O(TF)` memory and write bandwidth even though tracks are sparse (most tracks
span only a few frames).

**Change.** Store tracks as ragged, per-track segments:

- `track_id[track]`
- `charge[track]`
- `frame_start[track]`, `frame_stop[track]`
- ragged `particle_index[track][seg]`, `positions[track][seg]` (x,θ,r),
  `steps[track][seg]`
- ragged `matched_confident[track][seg]`, `velocity[track][seg]`

```text
O(total track observations)   # ~ O(FD)
```

**Scope.**
- `events/tracking.py :: DefectTrack` / serialization to
  `output/npz/defect_tracks_<case>.npz`.
- Persist as flat arrays + offsets (or a single ragged object array) so
  `save_metric_npz` keeps the existing caching contract; readers must rebuild
  the ragged view. No change to the on-disk filename or overwrite discipline.

**Constraint.** Do not regress the existing `frame_start`/`frame_stop`/case
metadata fields that downstream `events.py` and plotting depend on.

---

## 3. Precompute per-frame defect lookup structures

**Problem.** Birth/death nearest-same and nearest-opposite scans
(`events/events.py`) are roughly `O(ED)` because they scan the defect list per
event.

**Change.** For each frame and charge sign, build one of:

- a cell grid sized to `match_tol` (consistent with §1), or
- a `scipy.spatial.cKDTree` (per-frame, per-charge) with periodic `x` handled
  by querying wrapped query points.

Nearest-neighbor then becomes:

```text
O(log D)        # with cKDTree
O(local cells)  # with cell list, near-constant when sparse
```

**Scope.**
- `events/events.py :: birth/death nearest same/opposite` lookups.
- `events/events.py :: annihilation window scan` (opposite-charge defects within
  ~2 frames of death) — same per-frame structure reused.

**Constraint.**
- The annulus rule from `PLAN.md §3.1` is unchanged: nearest same/opposite is
  computed *within the relevant frame* using the defect-excluded annulus rule.
- `cKDTree` is a Python/scipy object; keep it out of any `@njit` kernel.

---

## 4. Cell-list-based annulus sampling everywhere

**Problem.** Event-local field sampling (`events/fields.py`) can scan all `N`
particles per event: `O(EN)`.

**Change.** Build a particle cell list per frame (and per case) with
`cell_size = a` (the annulus inner radius). For an annulus query at radius up to
`3a`, only touch cells intersecting the `3a` disk. This holds for every sampler in
`PLAN.md §4.3` (`rho`, `J_r`, `D²_min`, `S`, `Q`, `|ψ6|`, `χ`, `F_density`,
`u_rms`, `u_fluct`, `nearest_defect_distance`).

```text
O(EK)   # K = particles in the 3a neighborhood
```

**Scope.**
- `events/fields.py` — all annulus-averaged scalars go through the cell list.
- `shared.py :: _cell_index` is the single source for cell indexing; add a
  per-frame particle cell-list builder if one does not already exist.

**Constraints (from `PLAN.md §4.1`).**
- `core_mask: d < a`, `annulus_mask: a < d < 3a`.
- Exclude particles that are themselves defects (`charge != 0`) from the annulus.
- If the annulus has zero non-disclination particles after exclusion, record the
  average as `NaN` and always persist the contributing count.

---

## 5. Cache reusable per-frame structures

**Problem.** Several steps independently rebuild the same per-frame arrays:
coordinates, defect masks, defect lists by frame/charge, particle cell lists,
neighbor identities for `D2min`.

**Change.** Build once per `case`/frame and pass the structure through:

- cylindrical → cartesian coordinates `(x, y=r·sinθ, z=r·cosθ)`;
- `charges = cylinder.NEIGHBORS - counts` and the `_disclination_mask`;
- defect lists split by frame and by charge sign;
- particle cell list (`cell_size = a`);
- defect cell list (`cell_size = match_tol` or `cluster_bond_length`);
- neighbor identity tables needed by `D2min`.

```text
no new asymptotic term; removes duplicate rebuilds (constant-factor win)
```

**Scope.**
- Introduce a `FrameCache` (dataclass) constructed once per case in the runner
  and threaded through `tracking.py`, `events.py`, `fields.py`, `runner.py`,
  and `plotting.py`.

**Constraints (from `PLAN.md`).**
- Coordinate/source consistency assertion (`PLAN.md §2.1`): same frame count,
  same particle count, matching step numbers, stable particle-index ordering.
  The cache builder performs this assert once; downstream code trusts it.
- Radius-aware parameters come from `RadiusCase` / `cases.py` and
  `constants/cylinder.py`; no global default radius assumptions in the cache.
- Cache lives in memory only (not written to disk) unless an existing
  `.npz` already provides the same particle-local data — in that case load it
  per `AGENTS.md`.

---

## 6. Limit `D2min` to needed windows

**Problem.** Full-trajectory `D2min` is near `O(FN)` and can dominate cost for
little gain when only event-centered plots consume it.

**Change.** Compute `D2min` only for:

- frames within `t_event ± τ` for each event (default `τ = 2` saved frames, same
  convention as `PLAN.md §0.3`);
- particles whose annulus intersects an event neighborhood;
- representative frames needed by the map panels (`PLAN.md §10`).

```text
event-window cost, not whole-trajectory cost
```

**Scope.**
- `events/fields.py :: D²_min` and its cache writer.
- Keep the cache file contract from `PLAN.md §4.4`:
  `output/npz/D2min_<case>.npz`, but store window-scoped arrays keyed by
  frame ranges with `save_metric_npz`. Recomputation gated on `--overwrite`.

**Constraints (from `PLAN.md §4.4`).**
- Use local physical coordinates `X = (x, Rθ)` for shell/defect particles.
- Normalize `D²_min / a²`.
- Only compute up to the second-to-last frame (needs frame `t` and `t+Δt`).
- `D²_min` is undefined on the final frame and unstable for too few or nearly
  collinear neighbors — record `NaN` and contributing count.
- Numba inner summation with a 3×3 closed-form normal-equation solve
  (numba-compatible); no scipy inside the hot loop.

---

## 7. Vectorized / numba kernels for hot local loops

**Good numba candidates** (inner loops only):

- cell-list neighborhood queries (cell index → occupant iteration);
- periodic-`x` distance calculations (`minimum_image_delta` core);
- annulus aggregation (`u_mean`, `u_rms`, `u_fluct`, scalar means + counts);
- `D2min` local least-squares accumulation.

**Do not wrap with `@njit`:**

- scipy `linear_sum_assignment` (cannot run inside numba);
- matplotlib plotting;
- file IO (`save_metric_npz` / loaders);
- `cKDTree` construction and queries.

**Constraint.** Keep the "scipy or numba, not both in one function" rule from
`PLAN.md §2.2`.

---

## 8. Cap or separate ambiguous assignment

**Problem.** Sending a large ambiguous matching subset to
`linear_sum_assignment` creates pathological `O(D^3)` frames in dense clusters.

**Change.** In `tracking.py :: _match_defects_frame`:

- Greedy cell-list match first (§1).
- Only send the small residual ambiguous subset to `linear_sum_assignment`.
- If the ambiguous component exceeds a configurable cap, label the frame as
  `ambiguous` / `reaction-zone` and **skip** cubic assignment entirely; those
  tracks get `matched_confident = False` and are not counted as confident
  matches. Birth/death classification must not treat a skipped ambiguous frame
  as a physical event (per `PLAN.md §2.2`).

```text
caps worst case at the small residual subset; no O(D^3) on dense frames
```

**Constraint.** Report ambiguity/rejection counts so the cap is observable, not
silent.

---

## Most valuable practical order

```text
cell-list defect matching/clustering   (§1)
sparse track storage                   (§2)
cell-list annulus sampling             (§4)
event-window-only D2min                (§6)
cached per-frame structures            (§5)
```

(§3, §7, §8 are supporting improvements layered on top of the above.)

## Target cost summary

```text
O(FN)  for basic scans (unchanged; per-frame particle passes)
+ O(FD) for tracking/clustering in sparse regimes (cell-list, §1 + §2)
+ O(EK) for event sampling (cell-list annulus, §4)
```

with `D2min` only as broad as actually needed (§6), per-frame lookups at
`O(log D)` or local-cell constant (§3), no `O(D^3)` dense frames (§8), and no
duplicate per-frame rebuilds (§5). Nothing besides this file is edited.
