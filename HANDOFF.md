# HANDOFF — event-aware band-area SDE

Implementation of `PLAN.md` ("Adding dynamic birth/death/split/merge events").
Steps 0–3 are done and committed. Steps 4–8 remain.

Read `PLAN.md` in full before continuing. This file records what landed, the
decisions that are not in `PLAN.md`, and the constraints on the remaining work.

## Status

| Step | Work | Commit |
|---|---|---|
| 0 | Reorganize `band_analysis/` by identification and fitting responsibility | `effcfec` |
| 1 | Expose bipartite edge structure per event frame | `4ae1313` |
| 2 | `events.py` — `EventStep`, maps, slot allocation | `082cca1` |
| 3 | `chains.py` — `EventChain`, global N_max, knot grids | `12e73ee` |
| 4 | Masked state space + `sigma_E` | **todo** |
| 5 | Six-parameter vector in inference/bayesian | **todo** |
| 6 | Event-aware posterior predictive + forward simulator | **todo** |
| 7 | Run/report clean-segment and event-chain fits side by side | **todo** |
| 8 | Synthetic tests (recovery, bias demo, checks 4–13) | **todo** |

## Layout

```
hexatic/band_analysis/
  __init__.py
  __main__.py
  band_analysis_id/       components, characterization, density, extraction,
                          tracking, segments, events, chains, io, storage
  band_analysis_fitting/  model, inference, bayesian, validation, workflow,
                          reporting, plots, event_plots, cli
```

Keep every file under 600 lines —
if something is heading past that, the content is wrong, not the file.

## What steps 1–3 built

### `band_analysis_id/tracking.py` — `EventEdges`

`BandTracker.track` used to collapse each event frame to one dominant
`EventCode`, discarding the overlap structure. `TrackedFrame.edges` now carries
`EventEdges | None`:

- `edges` — bool `(n_old, n_new)` overlap matrix
- `old_track_ids` — row → pre-event track id, ordered as `sorted(active)`
- `new_track_ids` — column → post-event track id, in `DetectionFrame.bands` order

**Gotcha:** column order is detection order, *not* `TrackedFrame.bands` order,
which is sorted by track id. That is why `new_track_ids` is stored explicitly
instead of implied. Non-`None` only on event frames — `None` for no-event,
`RESTORED`, and gap frames.

`EventRecord` / `EventCode` are untouched; the clean-segment path still uses them.

### `band_analysis_id/events.py` — the map builder (218 lines, NumPy)

`build_event_step(edges, slots, observed, n_max) -> (EventStep, next_slots)`.

Classification reads **degrees, never `EventCode`** (`PLAN.md` §Simultaneous
events): row degree > 1 = split, column degree > 1 = merge, unmatched column =
birth, unmatched row = death. Every row lands in exactly one class.

`EventStep` fields: `H`, `g`, `C`, `o`, `sigma_e_rows`, `mask_next`, `y`.

**Deviation from `PLAN.md`, deliberate:** the plan writes `R_ii = sigma_E**2`,
but `sigma_E` is a *fitted parameter*, so `R` cannot be a precomputed NumPy
matrix. `EventStep.sigma_e_rows` is a bool indicator of which diagonal rows
carry `sigma_E**2`; **the likelihood must multiply in the parameter itself.**
This is what the plan's own "precomputed in NumPy before entering the
JIT-compiled likelihood" requires. Step 4 must honour this.

Disjointness is asserted over three claim sets — pre-event slots (H columns),
post-event slots (H rows / `g` / `m'`), and observation rows. A violation raises
`EventCompositionError` naming the offending slot; the caller breaks the chain.
Same-step reuse of a dead slot by a birth is legal (pre and post claimed
separately), matching "a retired track frees its slot at k+1".

Verified numerically: split `(HA)[i] + (HA)[r] == A[i]`; merge
`(HA)[i] == A[i] + A[j]`; total preserved for both; birth total `+B` exactly;
death total `-A[i]`; unaffected rows keep their continuous prediction;
`m'=0 => A=0`.

### `band_analysis_id/chains.py` — `EventChain` (248 lines, NumPy)

Two passes, because N_max is global across **all** seeds:

1. `plan_chains(frames, seed_id=...) -> list[ChainPlan]` — segment into unbroken
   chains, allocating nothing.
2. `global_n_max(plans) -> int`, then
   `build_event_chains(plan, n_max) -> list[EventChain]`.
   `build_chains({seed_id: frames})` wraps both.

Chain breaks: `UNCERTAIN`, `TRANSIENT_GAP`, and `EventCompositionError` only.
`RESTORED` is **not** a break — the pre-gap row connects straight to the restored
row, so `tau` carries the full elapsed interval across the bridged gap
(verified: dtau 3.0 vs the ordinary 1.0). Gap frames are never chain rows.
Chains shorter than 2 rows are dropped (not specified in `PLAN.md`; they have no
transitions).

**Decision:** `EventChain.events` is uniform length `T-1`, one step per
transition, with `no_event_step` filling eventless ones — *not* only
event-carrying transitions as the plan's field comment says. The likelihood
needs a step for every transition, and a ragged tuple would force an
index-mapping side table and break the one-trace/vmap goal of §Storage.
`event_transitions(chain)` recovers the `(T-1,)` bool flags.

**Naming trap:** `EventChain.n_steps` returns the **row count T**, not `T-1`
transitions. Used consistently, but do not assume from the name.

**Bug found and fixed during step 3, do not reintroduce:** deriving "is this an
event" by comparing `H` to `diag(m)` silently misses a **pure birth**, because a
birth writes into `g` and leaves `H` identity on surviving slots. Births then
looked continuous and knot grids allowed transitions spanning them. `_is_event`
compares all six map fields against `no_event_step`.

`KnotGrid(lag, phase, knots, event_index)` with `weight = 1/lag`, `start`, `end`.
`event_index[i]` indexes `EventChain.events` for an event-carrying transition,
`-1` for continuous — consume it directly. Because both `k` and `k+1` are knots
they are adjacent after sorting, so no transition can span an event.

## Constraints on steps 4–8

### NumPy vs JAX boundary

**NumPy up to the block builder, `jnp` from there in.**

`events.py` and `chains.py` stay NumPy: `PLAN.md` says so explicitly, and that
code is Python loops, dict-based slot allocation, and exception control flow —
untraceable, undifferentiated, and slower under `jnp` per-op dispatch.

Everything JIT-compiled, vmap-ed, or differentiated uses `jnp`: transition and
covariance algebra, the Cholesky path, the likelihood, inference, the forward
simulator, posterior predictive. Convert **once**, where `TrainingTransitions` is
assembled — the same place `build_transition_blocks` already calls `jnp.asarray`.
Step 4 should also audit `band_analysis_fitting/model.py` for hand-rolled NumPy sitting inside
the traced path and move it to `jnp`.

### Step 4 — masked state space

New file `band_analysis_fitting/masked_model.py`, **not** an extension of `model.py`.
`PLAN.md` requires the clean-segment fit to remain independently runnable (its
validation step 3 fits both paths and compares the bias), so the separation
should be structural. `model.py` is 389 lines and would blow the 600-line limit
anyway. `segments.py` and `build_stable_segments` stay untouched.

Required by `PLAN.md` §Masked state space / §Numerics:

- Replace the Helmert basis with the mask projector
  `P_m = diag(m) - m m^T / n`, allocation `a_k = m_k / n_k`.
- `Sigma_k = c_k P_m + 2 D_T dtau_k a_k a_k^T`,
  `c_k = (sigma_b^2 + D_u/tau_p) dtau_k^2`.
- **Do not use `eigh`.** `P_m` has an `(n-1)`-fold degenerate eigenvalue and JAX
  `eigh` gradients are unstable at repeated eigenvalues. Pad the diagonal on
  inactive slots: `Sigma~ = Sigma + diag(1 - m) + JITTER * I`, then
  `jnp.linalg.cholesky` succeeds at fixed padded shape. Mask the residual to
  zero on inactive slots; each contributes a constant `0.5 log 2pi`. Subtract
  `(N_max - n_k) * 0.5 * log 2pi` when reporting absolute log-likelihoods only.
- Same `diag(1 - o)` padding trick for unobserved rows of the observation model.
- `PARAMETER_NAMES` becomes
  `("tau_p", "kappa_T", "D_u", "D_T", "A_T_star", "sigma_E")`;
  `Scaling.parameter_factors` gains `self.area` for `sigma_E`.
- Total-area residual is evaluated against the **pre-event** total `A_T^-`, never
  the post-event observed total — otherwise part of the residual is identically
  zero for an exact birth insertion and `D_T` is biased downward with the birth
  rate.
- Persistent-rate blocks skip event-crossing steps for their AR(1) term and
  restart from stationary `N(0, D_u/tau_p)` in created slots
  (`PersistentRateBlock.initial` / `initial_weight` already play this role).
  Re-project `u` after an event: `H_k u_k` is not zero-sum in general.

### Step 8 — the tests that actually matter

Per `CLAUDE.md`, synthetic-data tests live in `tests/` at the repo root.

`PLAN.md` checks 5–10 (conservation identities) would all pass for a sign error
in `H_k` that happens to preserve total area. The two that carry weight:

- **check 2, parameter recovery** — fit simulated data, assert known parameters
  fall inside their posterior credible intervals.
- **check 3, bias demonstration** — fit the *same* simulated data with events
  ignored (clean-segment path only) and assert a measurable bias in `kappa_T`
  and `D_T`. This is the only check that the event machinery *helps* rather than
  merely runs.

Also worth keeping: check 13, padded-slot contributions are
parameter-independent (perturb every parameter, assert the inactive-slot
contribution is unchanged).

## Working agreement

- One subagent per step, run sequentially, Opus. Verify each before the next.
- Commit after each step. Conventional Commits, **single-line subject, no body.**
- Do not commit generated data (`*.gsd`, `*.npz`, `*.png`, `*.json`, `*.csv`).
- Style per `CLAUDE.md`: concise, few comments, frozen dataclasses,
  `from __future__ import annotations`, no backwards-compat shims, no defensive
  validation of malformed input (validate in tests instead) — but *statistical*
  validation does belong in the code.
- Verification for each step so far has been a throwaway script in the job
  scratchpad asserting the relevant `PLAN.md` identities, not just an import
  check. Keep that bar.
