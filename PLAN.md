# Programming Plan — Film Density / Flux Continuity Analysis on `radius_15D`

This plan implements the task in `INSTRUCTIONS.md`, scoped by `AGENTS.md` (repo conventions, Pixi workspace `sap`) and `ARCHITECTURE.md` (module layout, Numba/Scipy/Plotly preferences). It works entirely off existing cached outputs — **no simulations are run**.

---

## 0. Goal (restated)

Compute, on the unwrapped cylindrical *film* surface of the `radius_15D` cylinder, the following per-bin, per-transition quantities and verify the discrete continuity balance

```
∂_t ρ_film  +  div J_film   =   S_cross
```

i.e.  (∂_t ρ_film)_b  =  (−div J_film)_b  +  S_cross_b   ⇒   residual ≈ 0.

Quantities to compute & cache (per transition `t → t+1`, per (x, θ) bin `b`):

| Symbol stored name                  | Definition |
|---|---|
| `rho_film_b`      (frame)          | `N_b(t)/A_bin` — film-particle count density |
| `partial_t_rho_film_b`             | forward finite diff of `rho_film_b` over 1 frame |
| `J_film_b`      (Jx, Jy)           | mean film velocity × density: `(1/A_bin) Σ_{i∈film∩b} v_i` |
| `neg_div_J_film_b`                 | `−∂Jx/∂x − ∂Jy/∂y`, finite-diff over 1 frame, `y = Rθ` |
| `S_cross_b`                         | `(N_in,b − N_out,b)/(A_bin Δt)` shell entry/exit source |

Then produce **4 Plotly (x, θ) maps** (color = magnitude; overlaid arrow field):

1. `partial_t_rho_film_b`
2. `neg_div_J_film_b`
3. `S_cross_b`
4. **residual** = `partial_t_rho_film_b − neg_div_J_film_b − S_cross_b` (≈ 0 by continuity).

> See §6 decision points for sign convention and the meaning of map 4.

---

## 1. Sources & key numbers (verified against disk)

All inputs live in `hexatic/density_analysis/`:

- `npz/radius_15D_active_matter_fields.npz` — primary source.
  - `coords`     `(100, 9870, 3)`  float64  — `coords[f, i] = (x, θ, r)` (already unwrapped in θ ∈ [0, 2π)? see §6 Q1; to be verified)
  - `shell_mask` `(100, 9870)`     bool     — film membership `m_i(t)`
  - `steps`      `(100,)`          int64    — `[100000, 200000, …, 10000000]`; per-frame gap = `100000`
  - `x_edges`    `(101,)`  → **100 bins**, span `[−27.7054, +27.7054]` ⇒ **box `Lx = 55.41075`** (authoritative long-axis period)
  - `theta_edges` `(73,)`  → **72 bins**, span `[0, 2π)`
  - (aux) `rho`, `active_direction`, `flux_cylindrical`, etc. — not needed; velocities computed fresh (§3).
- `gsd/trajectory_radius_15D.gsd` — for box walls / sanity only. `box = [55.41075, 56.123104, 56.123104, 0,0,0]`.

### Constants (from `hexatic/constants/cylinder.py`, `hexatic/radii_analysis/cases.py`)
- `PARTICLE_DIAMETER = 2^(1/6) ≈ 1.122462` (`SIGMA = 1.0`).
- Cylinder radius for `radius_15D`: `R = 15 * PARTICLE_DIAMETER ≈ 16.83693` (from `cases.SCALED_RADIUS_CASES` / `radius_from_diameters(15.0)`). **This is the `R` used for `y = Rθ` and `A_bin = Δx·R·Δθ`.**
- `RHO = 0.2`, `TIMESTEP = 1e-6`, `TRAJECTORY_WRITE_PERIOD = 1e5` (radial case) → **per-frame `Δt = 100000 × 1e-6 = 0.1`** sim-time units. Also derive programmatically as `(steps[t+1] − steps[t]) * TIMESTEP` to stay radius-aware.

### Discrepancies to respect (per INSTRUCTIONS note)
- INSTRUCTIONS state `Lx = FIXED_LX = 4000/(RHO·π·BASELINE_CYLINDER_RADIUS²)` already gives ≈50.55 for the *baseline* 10-D cylinder, but the **actual `radius_15D` box `Lx = 55.41075`** (from GSD / `x_edges`). **Use the actual box `Lx`** stored in the npz/GSD; do **not** recompute via the baseline formula. Add the `FIXED_LX` constant for reference/cross-check only.
- `BASELINE_CYLINDER_RADIUS` (11.22) is for the default 10-D analysis; for `radius_15D` use `R = 15·D = 16.837`. Verify the `shell_mask` targets the **outer surface shell** (mean r where mask True ≈ 15.715, i.e. within one lattice spacing of `R`), consistent with `pocket_radius = 2·D` semantics in `active_matter_cylinder/fields/compute.py`. Confirm before computing (§6 Q2).

---

## 2. New module layout

All new code under a self-contained analysis package:

```
hexatic/density_analysis/
├── __init__.py                      (new, may already be a package marker)
├── film_continuity/
│   ├── __init__.py
│   ├── config.py                    # dataclasses: FilmContinuityConfig, paths, derived scalars (R, Lx, dt, A_bin)
│   ├── io_cache.py                  # load active_matter npz; write/read film continuity npz cache
│   ├── velocity.py                  # finite-difference, minimum-image-unwrapped v_x, v_y (Numba-kernel-ed)
│   ├── binning.py                   # (x,θ) bin index lookup + per-transition bin sums (Numba)
│   ├── fields.py                    # rho_film, J_film, neg_div_J, partial_t_rho, S_cross
│   ├── continuity.py                # assemble all fields into FilmContinuityResult dataclass
│   └── plots.py                     # Plotly heatmap + quiver overlay (4 maps)
├── run_film_continuity.py           # CLI entrypoint: compute → cache → plot
└── output/
    └── film_continuity/
        ├── radius_15D_film_continuity.npz     # cache (§5)
        └── radius_15D_film_continuity_map_*.html  # 4 Plotly maps
```

### CLI (`run_film_continuity.py`)
```
pixi run python -m hexatic.density_analysis.run_film_continuity \
    --case radius_15D \
    [--overwrite] [--no-cache] [--frame-idx N] [--plot]
```
`--overwrite` required to overwrite existing cache/figures (per repo data rule). Defaults are radius-aware: load case via `hexatic.radii_analysis.cases.get_case("radius_15D")` for `R`, `lx`, `n_particles`; override `lx` with the actual GSD box when present.

---

## 3. Physics / math specifications

### 3.1 Geometry & bins
- Reuse stored `x_edges` (100 → `N_x=100`), `theta_edges` (72 → `N_θ=72`).
- `bin b = (ix, iθ)`. Bin widths: `Δx[ix] = x_edges[ix+1]−x_edges[ix]`, `Δθ[iθ] = theta_edges[iθ+1]−theta_edges[iθ]` (uniform but keep array form for generality).
- Surface area of bin `b`: `A_bin[b] = Δx[ix] * R * Δθ[iθ]` (`y = Rθ`).
- Coordinates on film surface: `(x, y) = (x, R·θ)`. `θ` wrapped to `[0, 2π)` before binning; `x` wrapped to `[−Lx/2, Lx/2)`.

### 3.2 Velocities (finite-difference, NOT from GSD `velocity` field)
For each particle `i`, transition `t → t+1`:
```
dx_raw   = x_i(t+1) − x_i(t)
dx_i     = dx_raw  − Lx · round(dx_raw / Lx)          # minimum-image unwrap in long axis
dθ_raw   = θ_i(t+1) − θ_i(t)
dθ_i     = dθ_raw  − 2π · round(dθ_raw / 2π)          # unwrap θ across periodic seam
v_x,i(t) = dx_i / Δt
v_y,i(t) = R · dθ_i / Δt
```
Velocities are aligned to frame `t` (forward difference), giving **`n_frames − 1 = 99` transitions**, indexed `t = 0..98` mapping to the `t → t+1` interval. Use **Numba** `@njit` loops (per ARCHITECTURE.md) for the unwrap + binning kernel. Prefer `scipy.ndimage` / numpy for finite-differencing the gridded fields where simpler.

> If `coords[..., 0]` proves already absolute/not minimum-image-safe (§6 Q1), the `Lx·round` correction is still valid and idempotent for small per-frame displacements, so keep it unconditionally.

### 3.3 Per-bin field definitions (per transition `t`)
For each transition index `t` (between physical frames `t` and `t+1`):

**Counts/density** (film = particles with `shell_mask[phys_frame] == True`):
```
ρ_film,b(t)      = N_b(t) / A_bin[b]
  where N_b(t) = #{ i : shell_mask[i,t]==1  AND  bin(i, t)==b }
  bin(i, t) uses coords[t, i, 0] (x) and coords[t, i, 1] (θ) — position at the SAME physical frame t as the count.
```

**Current (mean velocity flux):**
```
Jx,b(t) = (1/A_bin[b]) Σ_{i∈film∩b at t} v_x,i(t)
Jy,b(t) = (1/A_bin[b]) Σ_{i∈film∩b at t} v_y,i(t)
```
(J uses the same membership/bin at frame `t` as the density; v is the forward-diff velocity starting at `t`.)

**Time derivative of density:**
```
∂_t ρ_film,b(t) = (ρ_film,b(t+1) − ρ_film,b(t)) / Δt     # forward diff over 1 frame
```
(Needed on the transition index; uses `ρ_film` at frames `t` and `t+1`, so `t = 0..98`.)

**Divergence of J (finite-diff over 1 frame across spatial bins), `y = Rθ`:**
```
∂Jx/∂x|_b   = (Jx_{b with ix+1} − Jx_{b with ix-1}) / (2 Δx)        # or forward diff; pick one (§6 Q3)
∂Jy/∂y|_b   = (1/R) · ∂Jy/∂θ|_b
            = (1/R) · (Jy_{b with iθ+1} − Jy_{b with iθ-1}) / (2 Δθ)
neg_div_J_film,b(t) = −∂Jx/∂x − ∂Jy/∂y
```
Wrap spatial indices cyclically (`ix → (ix±1) mod N_x`, `iθ → (iθ±1) mod N_θ`) — both axes are periodic.

Alternative per INSTRUCTIONS ("both calculated over finite-difference 1 frame"): interpret `div J` itself as a **time** finite difference too — but the standard reading is *spatial* divergence of `J(t)`. Decision §6 Q3; default = spatial central difference on each transition.

**Cross-shell source `S_cross`:**
```
m_i(t)   = shell_mask[i, t]
entry_e_i(t) = 1  if  m_i(t)==0  AND  m_i(t+1)==1   # particle joins film
exit_e_i(t)  = 1  if  m_i(t)==1  AND  m_i(t+1)==0   # particle leaves film

# bin an entry event by its position at t+1; an exit event by its position at t:
N_in,b(t)  = #{ entry events i with bin(i, t+1)==b }
N_out,b(t) = #{ exit  events i with bin(i, t  )==b }
S_cross,b(t) = (N_in,b(t) − N_out,b(t)) / (A_bin[b] · Δt)
```

**Continuity residual (verification/sanity, map 4):**
```
RHS_b(t)   = neg_div_J_film,b(t) + S_cross,b(t)        # = −div J + S_cross
residual_b = ∂_t ρ_film,b(t)  −  RHS_b(t)              # ≈ 0 (within discretization noise)
```
(Consistency check: `mean |residual| ≪ mean |∂_t ρ|`. If large, revisit unwrapping / sign convention — §6 Q4.)

### 3.4 Output cache (`output/film_continuity/radius_15D_film_continuity.npz`)
Arrays, all lead axis = transition index `t ∈ [0, 99]`:

| key | shape |
|---|---|
| `transition_steps` | `(99, 2)` — `(steps[t], steps[t+1])` |
| `dt` | `()` scalar |
| `cylinder_radius` `R`, `lx` `Lx`, `pocket_radius`, `particle_diameter` | scalars |
| `x_edges`, `x_centers`, `theta_edges`, `theta_centers` | binning grids |
| `A_bin` | `(100, 72)` |
| `rho_film_b` | `(99, 100, 72)` (frame-`t` density on each transition) |
| `partial_t_rho_film_b` | `(99, 100, 72)` |
| `J_film_b` | `(99, 100, 72, 2)` — components `(Jx, Jy)` |
| `neg_div_J_film_b` | `(99, 100, 72)` |
| `S_cross_b` | `(99, 100, 72)` |
| `residual_b` | `(99, 100, 72)` |
| `n_film_frame`, `film_count_per_bin_frame` | diagnostics |

Write only if missing or `--overwrite`.

---

## 4. Plotting (`plots.py`, Plotly — no matplotlib)

For each requested quantity Q ∈ {`partial_t_rho_film_b`, `neg_div_J_film_b`, `S_cross_b`, `residual_b`} produce an interactive HTML map (default: per-transition averaged over a configurable frame range, plus single-frame optional output):

- **Heatmap**: `plotly.graph_objects.Heatmap` with `x = x_centers` (µm in `x`), `y = theta_centers` (or `y = R·theta_centers` for a true flatted-surface aspect), `z = Q.mean(axis=0)` (over transitions in range), color = **magnitude** (`abs(Q)`). Use `diverging` colormap (`RdBu_r`) for signed `Q`; magnitude shown via a separate `colorscale='Viridis'` passed on `abs(Q)`.
  - Follow INSTRUCTIONS: "color corresponding to value" / "magnitude as the color". Provide selectable mode (signed vs magnitude) via buttons (`updatemenus`).
- **Arrow overlay**: `plotly.figure_factory.create_quiver` (or a manual `Scatter` with marker arrows) using the **`J_film_b`** field `(Jx, Jy)` sampled on a coarsened `(x, y=Rθ)` grid (e.g. every 4th bin) to avoid clutter, *independent* of the heatmap quantity (the arrow tells direction of the film current; per INSTRUCTIONS "for (x, theta) there should be an arrow telling about the direction").
- Axes labeled `x`, `θ` (and optionally `y = Rθ`; keep primary axis `x` vs `θ`). Title names the quantity and the transition/frame range.
- Save to `output/film_continuity/radius_15D_film_continuity_map_<quantity>.html`.

---

## 5. Implementation order (tasks)

1. **`config.py`** — `@dataclass(frozen=True) FilmContinuityConfig` with `case_id`, paths, and a `@dataclass FilmContinuityScalars` holding `R, Lx, dt, A_bin, x_edges, theta_edges`. Load case via `hexatic.radii_analysis.cases.get_case`. Resolve `lx` from GSD box if it disagrees with case (warn). Pin `TIMESTEP`, compute `dt`.
2. **`io_cache.py`** — `load_active_matter_fields(path)` returns the subset needed (`steps, coords, shell_mask, x_edges, theta_edges, pocket_radius`). `write_cache`/`load_cache` with existence + `--overwrite` guard.
3. **`velocity.py`** — Numba `compute_velocities(coords, shell_mask, Lx, R, dt)` → `vx, vy` shaped `(99, 9870)` aligned to transition `t`, with minimum-image unwrap in `x` and `θ`. Add a tiny fallback pure-numpy path for tests.
4. **`binning.py`** — Numba `particle_bin_indices(x, theta, x_edges, theta_edges)` (vectorized `searchsorted` + cyclic wrap). `accumulate_counts_and_sums` returning per-transition `counts`, `sum_vx`, `sum_vy` (shape `(99, 100, 72)`), restricting to `shell_mask[t]` membership; also entry/exit counting using `shell_mask[t] vs [t+1]`.
5. **`fields.py`** — pure functions:
   - `rho_film(counts, A_bin)`
   - `J_film(counts, sum_vx, sum_vy, A_bin)`
   - `partial_t_rho(rho_film, dt)`
   - `neg_div_J(J_film, dx, dtheta, R)` — central or forward diff (see §6 Q3); cyclic boundaries.
   - `S_cross(n_in, n_out, A_bin, dt)`
6. **`continuity.py`** — orchestrate into `FilmContinuityResult` dataclass; assemble all arrays + diagnostics (per-bin film counts, global residual stats).
7. **`plots.py`** — 4 Plotly maps as specified (+ an optional combined 2×2 subplot HTML).
8. **`run_film_continuity.py`** — argparse CLI; default behavior loads cache if present else computes. `pixi run python -m hexatic.density_analysis.run_film_continuity --case radius_15D --plot`.
9. **Tests** (`tests/test_film_continuity.py`): a 2-particle toy analytic case on a flat periodic strip asserting `∂_t ρ ≈ −div J + S_cross` to machine precision; a wrap-around displacement test for `velocity.py`.
10. **Sanity**: run `pixi run python -m compileall hexatic/density_analysis`; then the CLI on `radius_15D`; print residual statistics (mean abs residual / mean abs ∂_t ρ). Do **not** commit generated npz/html.

---

## 7. Compliance with repo rules (AGENTS.md)

- Pixi workspace `sap`; syntax-check via `pixi run python -m compileall hexatic/density_analysis`.
- No expensive simulations; load existing npz only.
- Outputs in a clearly named package dir (`hexatic/density_analysis/output/film_continuity/`); never overwrite cached sim data; cache writes guarded by `--overwrite` (the film-continuity cache is *analysis* output, recomputable, so allow regeneration but log clearly).
- Inspect `hexatic/constants/` first (done: `R, PARTICLE_DIAMETER, RHO, TIMESTEP` from `cylinder.py` + `cases.py`); radius-aware (`get_case("radius_15D")`) — no global default-radius reliance.
- Style: 4-space indent, type hints, `@dataclass` containers, `snake_case`, minimal comments only for non-obvious physics (unwrapping, sign convention, `y = Rθ`).
- Use Numba for hot loops (per ARCHITECTURE.md), Scipy where math-heavy (finite-diff/cyclic gradient could use `scipy.ndimage`/`np.gradient`); Plotly for plots (explicit requirement).

---

## 8. Edge cases & assumptions (verified against data)

Data probe of `radius_15D_active_matter_fields.npz` (run before drafting this section):

| Probe | Result | Implication |
| `x` range | `[−27.7053, +27.7053]` = `[−Lx/2, Lx/2]` | **x is wrapped** to the periodic cell (not an unwrapped trajectory). |
| raw `|Δx| > Lx/2` fraction | **1.28 %** of particle-transitions | Real seam crossings in x. Min-image unwrap in `velocity.py` is *required*, not optional. |
| raw `|Δθ| > π` fraction | **0.36 %** | Seam crossings in θ; min-image unwrap on θ *required*. |


. Edge cases the plan must explicitly handle
1. **Periodic seam crossings in x (~1.3%) and θ (~0.4%)** — min-image unwrap `dx − Lx·round(dx/Lx)` and `dθ − 2π·round(dθ/2π)` is *essential*. 
2. **Empty / near-empty bins** — a thin shell over a 100×72 grid spans ~4700 particles ⇒ mean ~0.65 particles/bin, so many bins are *empty*. Consequences:
   - `ρ_film_b = 0/A = 0` is fine, but **central-difference `div J`** at a 0-bin flanked by populated bins yields a *spurious large gradient* (artifact peaks at film/void boundaries).
   - Mitigation: mask/flag bins with `N_b < min_count` (`< 2`) in plots and in residual statistics 
3. **Cyclic spatial boundaries for `div J`** — wrap `ix±1 mod N_x`, `iθ±1 mod N_θ`; both axes are genuinely periodic. (Plan states this; ensure implementation uses `np.roll`/mod indexing, not edge-truncation.)
6. **`v` from `Δx`, not GSD `velocity`** — explicitly required. But note `coords` may already be *coarsened* relative to the integrator (HOOMD writes every `1e5` steps, `dt_sim = 1e-6` ⇒ `Δt = 0.1` sim time, ~1000 Brownian times per saved frame). Over such a hop, particle displacement is large; this is a **coarse-grained velocity** (a *current*), not the instantaneous velocity. 
7. **Last transition boundary** — `partial_t_rho` needs `ρ` at frame `t+1`; the final transition `t=98→99` is the last valid one. So all derived fields have `n_frames − 1 = 99` rows 
8. **`r` not used for binning** — binning is purely `(x, θ)` on the film. Particles whose `x`/`θ` lie in a bin while `r` is just outside the shell are correctly excluded by pre-filter on `shell_mask`. Make sure to bin **after** applying the film mask, not before, otherwise `J_film` includes interior-particle tangential drift. 
10. **Quiver density** — arrow overlay on a 100×72 grid = 7200 arrows. Must *coarsen* (every k-th bin, k≥4) and normalize arrow length, else the Plotly figure is unreadable

### D. Decision-prompt from probe
- **Non-zero baseline residual is expected** given entry/exit asymmetry and short Δt discretization. Define success criterion up front as e.g. `median |residual| / median |∂_t ρ| < 0.2` (configurable), *not* "residual ≈ 0 to machine precision". Adjust §5.6 to emit this normalized metric.
