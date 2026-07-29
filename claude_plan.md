# plot_all — cross-family Laplace grid and spectral entropy

## Context

`hexatic/new_sims_analysis/plot_correlation.py` reduces **one** case (one or more
seeds of the same `case_id`) into a correlation plot, MSD plots, and a
three-panel Laplace heatmap. The current focus is the Laplace transform of the
COM velocity autocorrelation, and the open question is how `L(r, ω)` changes
*across* geometries and across the axial box length `Lx`.

Answering that needs one figure that puts every case side by side on a shared
color scale, plus a scalar summary (spectral entropy of `L(0, ω)`) plotted
against `Lx`. `plot_correlation.py` cannot do this: it takes one case, and its
`r`/`ω` grid is derived per-case from the lag-time span, so two cases are not
directly comparable.

`plot_all` adds that cross-case view. It must read three *different* safetensor
families, which all describe `C = 60.5D` geometries but do not share a manifest
schema or a tensor layout:

| family | manifest schema | COM tensor | notes |
| --- | --- | --- | --- |
| `hexatic/new_sims` | `hexatic.new_sims.analysis.v1` | `coords` | `coordinate_order` in manifest; non-cylindrical coords are **already unwrapped** |
| `hexatic/big_lx` | `hexatic.big_lx.analysis.v1` | `coords` | always cylindrical `(x, θ, r)`, wrapped; the only family with `Lx > 1x` |
| `hexatic/confinement_comparison` | `hexatic.confinement_comparison.analysis.v1` | `coords` (cylinder + 2D) / `position_cartesian` (prism, sandwich) | no `coordinate_order`; wrapped; axial length is `logical_lx`, not `lx`/`box[0]` |

## Decisions already made

- **Correlation fed to `L`**: normalized `C_V/C_V(0)`, matching
  `plot_correlation.py:293-299`.
- **Components**: reuse the existing convention, and produce **two variants**, so
  **4 output images** total:
  - `full` — every meaningful component of `V`;
  - `axial` — `v_x` only.
  Each variant gets one Laplace grid image and one entropy image.
- **Confinement cases**: all seven (`prism_volume`, `prism_surface_area`,
  `sandwich_volume`, `sandwich_surface_area`, `two_dimension`,
  `cylinder_rattle`, `cylinder_rattle_tangent`).
- **Frames**: every case is truncated to 1000 frames / `--max-lag 998`. All three
  families write at `Δt = 1e5` steps with the same timestep, so after truncation
  the lag grid is byte-identical for every case and the existing
  `_laplace_transform` yields identical `r`/`ω` grids with no changes. Only
  `new_sims` `ideal_abp_cylinder` (3000 frames from `3e8` steps) is actually
  shortened.

`full` per coordinate kind — worth confirming when you see the output:

- cylindrical `(x, θ, r)` → `(ẋ, r̄ · θ̇)`: differentiate the raw COM columns and
  scale the azimuthal rate by the **mean radius**. Raw `(x, θ, r)` columns mix a
  length with radians, so they are not usable directly, but differentiating the
  *product* `rθ` (the `_unrolled` variant in `msd.py:35-58`) is also wrong — it
  adds a spurious `ṙ·θ` term, and `θ` is unwrapped so it grows without bound
  over the trajectory.
- planar `(x, y)` → both components.
- Cartesian `(x, y, z)` → all three.

`axial` is always column 0 of the COM, which is the logical axial axis in all
three families (confinement maps stored→logical axes during analysis, see
`confinement_comparison/analyze_case.py:308`).

## Files

Four new flat modules alongside the existing ones in
`hexatic/new_sims_analysis/` (the package is flat today; keep it that way):

### 1. `all_cases_loading.py` — the cross-family adapter

The one genuinely new piece of logic. Everything else reuses existing code.

```python
SUPPORTED_SCHEMAS: dict[str, str]      # schema -> family name
TARGET_CIRCUMFERENCE_DIAMETERS = 60.5

@dataclass(frozen=True)
class CaseSeries:
    family: str            # "new_sims" | "big_lx" | "confinement"
    case_id: str
    label: str
    geometry_kind: str     # metadata geometry_kind, or "cylinder" for big_lx
    lx_multiplier: int
    coordinate_kind: str   # "cylindrical" | "planar" | "cartesian"
    com: NDArray           # (frames, components)
    steps: NDArray         # (frames,)
    timestep: float
```

- `load_manifest(dir) -> tuple[dict, str]` — read `manifest.json`, dispatch on
  `schema` against `SUPPORTED_SCHEMAS`, require `complete is True`, require a
  dict `case`. Unknown schema → `ValueError` naming the directory.
- `_circumference_diameters(case)` — prefer `case["circumference_diameters"]`
  (present in `big_lx` and, via `base.as_metadata()`, `confinement`); otherwise
  `case["circumference"] / case["particle_diameter"]` (the `new_sims` path;
  both families use the same `cylinder.PARTICLE_DIAMETER`). Fail fast with a
  `ValueError` if it is not `60.5` within `1e-6`.
- `_lx_multiplier(case)` — `case["lx_multiplier"]` if present, else regex
  `_lx_(\d+)x$` against `case["base_case_id"]` (`new_sims` path, always `1`).
  Neither available → `ValueError`.
- `_coordinate_spec(manifest, family)` → `(tensor_name, coordinate_kind)`:
  - `new_sims`: `("coords", from manifest["coordinate_order"])` using the
    existing `PLANAR_/CARTESIAN_/CYLINDRICAL_COORDINATE_ORDER` constants in
    `loading.py:18-24`.
  - `big_lx`: `("coords", "cylindrical")`.
  - `confinement`: `geometry_kind` in `{cylinder_rattle,
    cylinder_rattle_tangent}` → `("coords", "cylindrical")`;
    `two_dimension` → `("coords", "planar")`;
    otherwise → `("position_cartesian", "cartesian")`.
- `_axial_period(dir, family)` — `logical_lx` for `confinement`; otherwise reuse
  `loading._load_lx` (which handles `lx` then `box[0]`). Do **not** fall back to
  the box for confinement: cylinder cases store a permuted box with the axial
  axis on `z`.
- `timestep` — always `cylinder.SIMULATION.timestep`, never the manifest. Only
  `new_sims` records a `timestep`; `big_lx` and `confinement_comparison` omit it,
  and letting them fall back means the shared lag grid rests on an unasserted
  coincidence between stored metadata and a live constant.
- `load_com_series(dir, manifest, family, frame_limit) -> (com, steps)` — a
  COM-only generalization of `loading._load_frame_series:152-287`, walking the
  same contiguous-shard validation but parameterized on the tensor name and
  skipping polarization entirely (`plot_all` never needs `q`, which also drops
  the frames×particles memory cost). Unwrapping, reusing
  `loading._PeriodicComUnwrapper`:
  - cylindrical → unwrap col 0 by `lx`, col 1 by `2π`, plain mean of col 2
    (identical to the existing cylindrical branch);
  - `new_sims` non-cylindrical → plain mean, because the manifest declares
    `coordinate_storage == "cartesian_unwrapped_on_periodic_axes"`; assert that
    string and fail fast if it changes;
  - `confinement` cartesian/planar → wrapped, so unwrap each axis listed in
    `case["periodic_axes"]` using `stored_box` for its period (`logical_lx` for
    `x`), plain mean for wall-bounded axes. Assert
    `logical_to_stored_axes == (0, 1, 2)` for non-cylinder cases so the logical
    and stored component orders agree.

  Fail fast, naming the directory, when a case yields fewer than `frame_limit`
  frames. A short case would otherwise get its own smaller lag grid and surface
  only as an opaque Laplace r-grid mismatch several layers later.
- `case_key(family, case_id) -> str` — `f"{family}:{case_id}"`. Directories
  sharing a key are seeds of the same case and get averaged; distinct keys are
  distinct panels. The prefix prevents `big_lx`'s `circ_60_5D_lx_1x` from
  colliding with a confinement `base_case_id`.

### 2. `laplacian_grid.py` — section 1

- Reuse `laplace._laplace_transform` **unchanged**. Its return is
  `(r, omega, values)` with `values` shaped `(omega_points, r_points)`, which is
  already what `pcolormesh(r, omega, field)` wants, and `r[-1] == 0.0` exactly
  (`np.linspace(-10/T, 0.0, n)`), so `values[:, -1]` is `L(0, ω)`.
- `validate_shared_grid(entries)` — `np.array_equal` every case's `r` and
  `omega` against the first; on mismatch raise a `ValueError` naming the
  offending case and the two shapes/endpoints. With the truncation above this
  should never trigger, which is the point: it is the fail-fast guard. Called
  **once** per variant, from `plot_all`, before either plot — it guards the
  shared color scale and the entropy bins equally, so `plot_grid` documents the
  precondition rather than re-checking it.
- `plot_grid(entries, variant_label, output)`:
  - `field = log10(maximum(abs(values), tiny))` per case, same expression as
    `laplace._plot_laplace:79`;
  - `vmax` = global max over all finite field values; `vmin` =
    `np.percentile(finite, 1)`, **not** the raw minimum — that minimum is one
    cell out of `r_points × omega_points × N`, so a single near-zero cell (or the
    `tiny` floor at `log10 ≈ -307`) would flatten every panel. The colorbar takes
    `extend="min"` when the clamp actually bites. Both are **plot-only**,
    computed after every transform, never fed back into the analysis;
  - `ncols = ceil(sqrt(N))`, `nrows = ceil(N/ncols)`, `squeeze=False`,
    `sharex=True, sharey=True`, `constrained_layout=True`; per-panel
    `pcolormesh(r, omega, field, vmin=vmin, vmax=vmax, cmap="viridis",
    shading="auto")`; unused axes `set_axis_off()`; one shared colorbar over the
    **used** axes only, shrunk by the filled fraction so a mostly-empty last row
    does not stretch it;
  - panel titles are `textwrap.fill(entry.case_id, 26)` — the human labels run to
    62 characters and overrun a 4-inch panel; `suptitle` carries the variant and
    the seed count;
  - axis labels: y on column 0; x on the bottom-most **visible** axis of each
    column, i.e. where `index + ncols >= N`, with `tick_params(labelbottom=True)`.
    `sharex` hides tick labels by subplot geometry, not visibility, so keying off
    `row == nrows - 1` silently strips the r axis from every column that sits
    above a switched-off slot;
  - `sns.set_theme(context="paper", style="ticks")` and the same
    `savefig` SVG + 300-dpi PNG pair as `laplace._plot_laplace:99-107`.

### 3. `spectral_entropy.py` — section 2

- `spectral_entropy(values_at_r0) -> float`:
  ```
  power = abs(L)**2
  total = power.sum()          # zero total -> ValueError
  x = power / total
  H = -(x[x > 0] * log(x[x > 0])).sum()
  ```
  The `x > 0` mask is exactly the "skip the zero terms" rule; `x log x → 0`
  there anyway.
- `plot_entropy(points, variant_label, output)`:
  - x = `lx_multiplier`, `xscale="log", base=2`, ticks at `1, 2, 4, 8, 16`;
    y = `H_spec`. No error bars (explicit requirement).
  - Encoding needs **three** channels, not two. **marker = geometry group**
    (`cylinder` / `cylinder_rattle` / `prism` / `sandwich` / `2d_plane` /
    `3d_bulk` / `tracer`, derived from `geometry_kind`), **color = interaction
    class** from `case["pair_interaction"]` plus a third color for the tracer
    cases (`active_count < n_particles`), **fill = case within its bucket**
    (`full` / `none` / `left` / `right` / `top` / `bottom`).

    Marker × color alone is *not* injective: the 23 real cases collapse into 14
    slots, and 15 of them share a slot with another case — including
    `cylinder_rattle` vs `cylinder_rattle_tangent`, the three `ideal_3d_bulk`
    diffusion periods, and the three prisms. Adding fill gives 23 distinct
    styles with zero collisions (verified against the case registries). Buckets
    are keyed on `(lx_multiplier, geometry_group, interaction_class)` so the fill
    channel stays as narrow as possible and the big-Lx sweep is drawn as one
    consistent series; raise if a bucket ever exceeds the six fill styles.
    Markers are drawn with `axis.plot` rather than `scatter`, since `fillstyle`
    is a `Line2D` property.
  - Legend: one entry per case, labelled by `case_id`, using that case's exact
    marker/color/fill, sorted by `(geometry_group, interaction_class, case_id)`
    and placed outside the axes on the right. Grouping remains readable at a
    glance while every point stays identifiable.
  - The five `big_lx` `circ_60_5D` points are the only Lx sweep, so join them
    with a thin line; every other case is a lone marker in the `Lx = 1` column.
  - Module-level `LX_JITTER = 0.0` knob: a deterministic horizontal offset that
    spreads points across slots taken **within the `Lx = 1` subset alone**, so no
    two crowded points can land on the same offset. Off by default, to turn on if
    that column reads as cluttered.
  - Same SVG + PNG output pair.

### 4. `plot_all.py` — the root file

- Module-level editable list, no `--input-dir` flag:
  ```python
  INPUT_DIRS: tuple[str, ...] = (
      # absolute paths; directories sharing a case_id are treated as seeds
      "/mnt/drive3/vaibhav_data/.../safetensors_output/ideal_abp_cylinder",
      ...
  )
  ```
  Seeded with placeholder paths and a comment; you fill them in.
- CLI: `--output-dir` (default `plot_all_output`), `--frames` (1000),
  `--max-lag` (998), `--laplace-r-points` (161), `--laplace-omega-points` (241),
  reusing the `_nonnegative_int` / `_positive_float` validators from
  `plot_correlation.py:30-41`.
- Flow:
  1. Load every dir → `CaseSeries`; the circumference check fails fast inside
     the loader. Group by `case_key`.
  2. Per case: assert all seeds share `steps` and `timestep` (same checks as
     `plot_correlation.py:192-204`).
  3. Per variant (`full`, `axial`): take
     `correlation._finite_difference_vector(com, loading._elapsed_time(...))`
     first, then select components — `velocity[:, :1]` for `axial`; for `full`,
     every column, except on a cylinder where it is `(ẋ, r̄·θ̇)`. Then
     `correlation._normalized_autocorrelation(..., normalize=False)`. Average the
     per-seed curves, divide by lag 0, and call `laplace._laplace_transform` on
     `correlation._lag_times(elapsed, max_lag)`.
  4. `validate_shared_grid`, then `plot_grid` →
     `laplacian_grid_{variant}.svg/.png`.
  5. `spectral_entropy(values[:, -1])` per case, then `plot_entropy` →
     `spectral_entropy_{variant}.svg/.png`.
- Progress prints with a `[plot_all]` prefix, matching the existing
  `[correlation]` style.
- Register a `pixi.toml` task mirroring however `plot_correlation` is invoked
  today, if one exists there.

## Notes / corrections to the original spec

- "The laplacian grid" is the **Laplace** transform `L(r, ω)`, not a Laplacian
  operator — the plan uses the existing `laplace.py` transform and keeps the
  filenames `laplacian_grid_*` as written in the spec.
- Fixing the `ω`/`r` range is not a new computation: truncating every case to
  1000 frames makes the lag grid identical, and the existing transform then
  derives identical grids. The validation step is therefore a real assertion
  rather than a reconciliation.
- The confinement `two_dimension` shard stores a *coarse-grained*
  `polarization`, not per-particle unit directions — irrelevant here because
  `plot_all` only reads `coords`.
- `new_sims` `x_walled_prism` has walls on **x** and periodic `y/z`, whereas
  confinement `prism_*` is periodic in **x** with `y/z` walls. Under the agreed
  "reuse the existing convention" rule the `axial` variant is `v_x` for both,
  which for `new_sims x_walled_prism` is a wall-bounded direction — expect a
  qualitatively different (non-diffusive) `L` there. Flagging it rather than
  special-casing it.
**USER**: I'll likely not even input x_walled_prism

## Verification

No `new_sims`/`big_lx`/`confinement` safetensor output exists in this checkout
(`find . -name manifest.json` over the analysis schemas returns nothing), so
verification has two stages.

1. **Synthetic smoke test** — a throwaway script under the scratchpad that
   writes three tiny fake analysis directories, one per schema (cylindrical
   `new_sims`, cylindrical `big_lx`, Cartesian confinement `prism_volume`), each
   with a `manifest.json`, a `static.safetensors`, and one shard of ~200 frames
   of synthetic damped-oscillatory COM. Run `plot_all.py` with `INPUT_DIRS`
   pointed at them and `--frames 200 --max-lag 198`. Confirms: schema dispatch,
   the `logical_lx` vs `lx` split, unwrapping, seed grouping, 4 output files,
   one shared colorbar, and identical `r`/`ω` grids.
2. **Negative checks** on the same fixtures — a `circ_60_0D` manifest must raise
   the circumference error; a manifest whose shard count gives a different frame
   count must trip `validate_shared_grid`.
3. **Real run** on the machine holding the production outputs: fill `INPUT_DIRS`
   with the real absolute paths and run
   `pixi run python -m hexatic.new_sims_analysis.plot_all
   --output-dir plot_all_output`. Sanity-check that the four images appear, that
   the shared color scale still leaves per-panel structure visible (if one case
   dominates the range, that is the signal to revisit the `vmin` floor), and that
   the entropy plot's `Lx = 1` column is legible before deciding on `LX_JITTER`.
4. Spot-check one case against `plot_correlation.py` run on the same directories:
   with `--frames 1000 --max-lag 998` the `axial` variant's `L(r, ω)` panel must
   match `plot_correlation`'s `log10|L|` panel for a cylindrical case.

The safetensors exist on a separate machine.
