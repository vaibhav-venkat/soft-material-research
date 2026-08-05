# Adding dynamic birth/death/split/merge events within the code

We are not yet predicting when events occur. Event times, event types, and
participating bands are supplied from the observed data by `BandTracker`.

We are transitioning from a continuous SDE fitted only on clean fixed-identity
segments to an SDE with observed events.

## Scope

In scope:

- masked padded state in raw area space;
- observed birth, death, split, and merge maps, including k-way and
  simultaneous events;
- one new parameter `sigma_E` for event mass-conservation noise;
- an event-capable forward simulator and a parameter-recovery test.

Not in scope:

- inferring event rates or event times;
- non-conservative split/merge fractions as inferred physics (the daughter
  fraction is exogenous);
- any Kalman/particle filtering. Every band area is individually observed at
  every frame, so the chain still conditions on data at each step. Only
  *pre-event* quantities are ever latent, and those are handled by
  marginalization, not filtering.

## Storage

Choose a single global

\[
N_{\max}=\max_k n_k
\]

over all chains, so every chain has one padded shape, one JIT trace, and can be
`vmap`-ed. Store

\[
\mathbf A_k\in\mathbb R^{N_{\max}},
\qquad
\mathbf m_k\in\{0,1\}^{N_{\max}},
\]

where \(m_{i,k}=1\) means slot \(i\) contains an active band. Set inactive areas
to zero:

\[
A_{i,k}=0
\qquad\text{when}\qquad
m_{i,k}=0,
\]

so that

\[
A_{T,k}=\sum_{i=1}^{N_{\max}}m_{i,k}A_{i,k},
\qquad
n_k=\mathbf 1^{\mathsf T}\mathbf m_k .
\]

All drift, noise, covariance, and likelihood terms must be masked so padded
slots contribute nothing.

### `EventChain`

Add a new dataclass alongside `StableSegment`; do **not** modify
`StableSegment` or `build_stable_segments`. The clean-segment fit remains an
independently runnable reference path, and the recovery test below compares the
two directly.

```python
@dataclass(frozen=True)
class EventChain:
    seed_id: str
    slot_ids: np.ndarray      # (T, N_max) int64, track_id per slot, -1 if empty
    frame_indices: np.ndarray # (T,)
    tau: np.ndarray           # (T,)
    areas: np.ndarray         # (T, N_max), zero where inactive
    mask: np.ndarray          # (T, N_max) bool
    events: tuple[EventStep, ...]  # one per transition that carries an event
```

A chain is broken (and a new chain started with a fresh mask) only by
`EventCode.UNCERTAIN` and `EventCode.TRANSIENT_GAP`. Slot assignment is a new
bookkeeping layer: a surviving track keeps its slot; a created track takes the
lowest free slot; a retired track frees its slot at \(k+1\).

## Masked state space

Work in raw area space in \(\mathbb R^{N_{\max}}\). Replace the
dimension-specific Helmert basis with the mask projector

\[
\mathbf P_{\mathbf m}
=
\mathbf M-\frac{\mathbf m\mathbf m^{\mathsf T}}{n},
\qquad
\mathbf M=\operatorname{diag}(\mathbf m),
\qquad
n=\mathbf 1^{\mathsf T}\mathbf m ,
\]

and the allocation vector

\[
\mathbf a_k=\frac{\mathbf m_k}{n_k}
\qquad(\text{uniform allocation, as in the current \texttt{transition}}).
\]

The one-step covariance is then

\[
\boldsymbol\Sigma_k
=
c_k\mathbf P_{\mathbf m_k}
+
2D_T\,\Delta\tau_k\,\mathbf a_k\mathbf a_k^{\mathsf T},
\qquad
c_k=\left(\sigma_b^2+\frac{D_u}{\tau_p}\right)\Delta\tau_k^2 .
\]

This reproduces the existing factorization exactly: the conservative subspace
carries variance \(c_k\) and the total direction carries
\(\mathbf 1^{\mathsf T}\boldsymbol\Sigma_k\mathbf 1=2D_T\Delta\tau_k\).

### Numerics: do not use `eigh`

Restricted to active slots, \(\boldsymbol\Sigma_k\) has rank
\((n_k-1)+1=n_k\) and is therefore **full rank on the active block**. It is
singular only because of the padded rows. Do not eigendecompose:
\(\mathbf P_{\mathbf m}\) has an \((n-1)\)-fold degenerate eigenvalue, and JAX
`eigh` gradients are unstable at repeated eigenvalues.

Instead pad the diagonal on inactive slots:

\[
\widetilde{\boldsymbol\Sigma}_k
=
\boldsymbol\Sigma_k
+
\operatorname{diag}(\mathbf 1-\mathbf m_k)
+
\varepsilon\mathbf I ,
\]

with `JITTER` for \(\varepsilon\). Then `jnp.linalg.cholesky` succeeds at fixed
padded shape, the residual is masked to zero on inactive slots, and each
inactive slot contributes the parameter-independent constant
\(\tfrac12\log 2\pi\). Subtract
\((N_{\max}-n_k)\tfrac12\log 2\pi\) when reporting absolute log-likelihoods; it
does not affect optimization or the posterior.

## Continuous prediction

The active-band allocation weights are

\[
\widetilde{\mathbf w}_k
=
\frac{\mathbf m_k\odot\mathbf w(\mathbf A_k)}
{\sum_j m_{j,k}w_j(\mathbf A_k)},
\qquad
\mathbf 1^{\mathsf T}\widetilde{\mathbf w}_k=1,
\]

which for the current uniform allocation is \(\widetilde{\mathbf w}_k=\mathbf a_k\).

The continuous total-area increment is

\[
\Delta A_{T,k}^{\mathrm{cont}}
=
-\kappa_T\!\left(A_{T,k}-A_T^{\star}\right)\Delta\tau_k .
\]

`A_T_star` stays a single global, band-count-independent parameter. Add a
diagnostic plotting \(A_T\) against \(n_k\) so the assumption can be checked
post hoc.

The pre-event predictive mean is

\[
\boldsymbol\mu_k^{-}
=
\mathbf A_k
+
\widetilde{\mathbf w}_k\Delta A_{T,k}^{\mathrm{cont}}
+
\mathbf M_k\!\left[\mathbf b_k+\mathbf u_k\right]\Delta\tau_k ,
\]

with redistribution terms conserving total area over the active slots:

\[
\mathbf 1^{\mathsf T}\mathbf M_k\mathbf b_k=0,
\qquad
\mathbf 1^{\mathsf T}\mathbf M_k\mathbf u_k=0 .
\]

Inactive coordinates must satisfy

\[
m_{i,k}=0
\quad\Longrightarrow\quad
\mu_{i,k}^{-}=0 .
\]

### Latent conservative rate and segment slope across an event

The persistent rate \(\mathbf u\) is treated as an observable (as in the
current code) via masked conservative differences. Across an event step,
transform and then re-project:

\[
\mathbf u_{k+1}
=
\frac{1}{\Delta\tau_k}
\mathbf P_{\mathbf m_{k+1}}
\!\left[
\mathbf A_{k+1}^{\mathrm{obs}}
-
\left(\mathbf H_k\mathbf A_k^{\mathrm{obs}}+\mathbf g_k\right)
\right].
\]

Re-projection is required: \(\mathbf H_k\mathbf u_k\) is not zero-sum in
general, so the constraint \(\mathbf 1^{\mathsf T}\mathbf M\mathbf u=0\) breaks
without it. Slots created by a birth or a split start a fresh OU chain drawn
from the stationary \(\mathcal N(0,D_u/\tau_p)\); this is already the role of
`PersistentRateBlock.initial` / `initial_weight`.

The per-run slope \(\mathbf b_s\) stays a nuisance estimated per inter-event
run, with the Helmert projection replaced by \(\mathbf P_{\mathbf m}\) and
`estimate_slope_variance` counting \(n-1\) degrees of freedom per run using the
run's active count.

## Event transformation

Separate two objects that the previous draft conflated.

**Propagation.** \(\mathbf H_k,\mathbf g_k\) define the predictive mean after
the event,

\[
\boldsymbol\mu_{k+1}
=
\mathbf H_k\boldsymbol\mu_k^{-}+\mathbf g_k ,
\]

and are used for simulation, rollout, and posterior predictive draws.

**Conditioning.** The state carried into the next transition is the *observed*
post-event vector \(\mathbf A_{k+1}^{\mathrm{obs}}\) placed into slots, not
\(\mathbf H_k\mathbf A_k+\mathbf g_k\). This keeps the estimator a conditional
one-step transition model, exactly as `build_transition_blocks` is today.

**Observation.** \(\mathbf C_k\in\mathbb R^{N_{\max}\times N_{\max}}\) maps the
pre-event state to what is actually measured, \(\mathbf o_k\in\{0,1\}^{N_{\max}}\)
flags informative rows, and \(\mathbf R_k\) is the event noise. The likelihood
of one transition is

\[
\mathbf y_k
\sim
\mathcal N\!\left(
\mathbf C_k\boldsymbol\mu_k^{-}+\mathbf g_k^{\mathrm{obs}},
\;
\mathbf C_k\boldsymbol\Sigma_k\mathbf C_k^{\mathsf T}+\mathbf R_k
\right)
\quad\text{on the rows }o_{i,k}=1 ,
\]

with the same \(\operatorname{diag}(\mathbf 1-\mathbf o_k)\) padding trick for
the unobserved rows. Dropping a row *is* marginalization of the corresponding
latent pre-event quantity, which is what a death requires.

\(\mathbf H_k\), \(\mathbf g_k\), \(\mathbf C_k\), \(\mathbf o_k\),
\(\mathbf R_k\), and \(\mathbf m_k\) are all precomputed in NumPy before
entering the JIT-compiled likelihood.

### No event

\(\mathbf H_k=\mathbf C_k=\mathbf M_k\), \(\mathbf g_k=\mathbf 0\),
\(\mathbf o_k=\mathbf m_k\), \(\mathbf R_k=\mathbf 0\).
`EventCode.RESTORED` is treated as no event: the same track IDs are recovered,
so the preceding gap frames are bridged by one continuous transition with the
full elapsed \(\Delta\tau\).

### Birth

Place the observed new area into a free slot \(r\):

\[
A_{r,k+1}=A^{\mathrm{obs}}_{\mathrm{new},k+1},
\qquad
m_{r,k+1}=1 .
\]

All unaffected bands retain their continuous predictions.

Maps: \(g_{r,k}=B_k\); \(o_{r,k}=0\) — the birth area is inserted exactly and
carries no density, because we are not modelling where new area comes from.

### Death

For the death of band \(i\):

\[
A_{i,k+1}=0,
\qquad
m_{i,k+1}=0 .
\]

The removed area is the *predicted* pre-death area \(A_{i,k+1}^{-}\), which is
never observed.

Maps: \(H_{ii}=0\); row \(i\) of \(\mathbf C_k\) is zeroed and \(o_{i,k}=0\).
The scalar total-area information at a death step therefore enters through the
joint marginal over the surviving slots, not through a separate \(A_T\)
residual. Because the joint predictive is Gaussian, marginalizing the latent
\(A_{i,k+1}^{-}\) is exactly the row deletion above — no integral is needed. A
temporary fallback, acceptable only if the joint path is not yet wired up, is to
skip the total-area term for death steps alone.

### Split

Band \(i\) splits into daughters stored in slots \(i\) and \(r\). Use the
observed daughter fraction

\[
f_k=\frac{A^{\mathrm{obs}}_{i_1,k+1}}
{A^{\mathrm{obs}}_{i_1,k+1}+A^{\mathrm{obs}}_{i_2,k+1}}
\]

as an **exogenous constant**. Propagation:

\[
A_{i,k+1}=f_kA_{i,k+1}^{-},
\qquad
A_{r,k+1}=(1-f_k)A_{i,k+1}^{-},
\qquad
m_{r,k+1}=1 ,
\]

so \(H_{i,i}=f_k\), \(H_{r,i}=1-f_k\), \(m_{r,k+1}=1\).

Observation is **one-dimensional, on the daughter sum**:

\[
y_{i,k}=A^{\mathrm{obs}}_{i_1,k+1}+A^{\mathrm{obs}}_{i_2,k+1},
\qquad
\mathbf C_k\text{ row }i=\mathbf e_i,
\qquad
o_{i,k}=1,
\qquad
R_{ii}=\sigma_E^2,
\qquad
o_{r,k}=0 .
\]

Rationale: with \(f_k\) defined from the observations, the two-component
residual
\(\left(A_1-f_kA^{-},\,A_2-(1-f_k)A^{-}\right)
=(f_k,1-f_k)\left(A_1+A_2-A^{-}\right)\)
is exactly rank one. A full \(\boldsymbol\Sigma_S\) would have one
unidentifiable eigendirection. Splits and merges are therefore the same
statement: is mass conserved through a topology change?

k-way splits generalize with a simplex weight vector
\(\mathbf f_k\), \(\sum f_{k,d}=1\), taken from the observed daughter areas;
the observation row is unchanged.

### Merge

Bands \(i\) and \(j\) merge into slot \(i\):

\[
A_{i,k+1}=A_{i,k+1}^{-}+A_{j,k+1}^{-},
\qquad
A_{j,k+1}=0,
\qquad
m_{j,k+1}=0 .
\]

Merging is conservative: \(H_{i,i}=H_{i,j}=1\) and row \(j\) is zeroed. (The
earlier draft wrote \(\eta_M\) here while the text asserted conservation; there
is no \(\eta_M\) parameter — non-conservation is carried by \(\sigma_E\) in the
likelihood, not by a deterministic gain.)

Observation:

\[
y_{i,k}=A^{\mathrm{obs}}_{ij,k+1},
\qquad
\mathbf C_k\text{ row }i=\mathbf e_i+\mathbf e_j,
\qquad
o_{i,k}=1,
\qquad
R_{ii}=\sigma_E^2,
\qquad
o_{j,k}=0 .
\]

k-way merges sum the corresponding unit vectors into one row.

### Simultaneous events

One `TrackedFrame` can carry several events; `BandTracker.track` collapses them
to a single dominant `EventCode`, so **read the bipartite overlap edge
structure, not `EventRecord.code`**, when building the maps. Row degree \(>1\)
is a split, column degree \(>1\) is a merge, an unmatched column is a birth, an
unmatched row is a death. Disjoint events compose by writing into disjoint rows
and columns of the same \(\mathbf H_k,\mathbf g_k,\mathbf C_k\); assert
disjointness and fall back to a chain break if it fails.

## Time grid with mandatory event knots

For each lag \(L\) and phase \(p\), begin with the ordinary sampled indices
\(p,p+L,p+2L,\dots\); then for every event boundary insert the mandatory knots
\(k\) and \(k+1\); then sort and deduplicate. A transition that contains no
event uses the ordinary continuous form over its (possibly irregular) interval.
The exact \(k\to k+1\) step is one frame of continuous evolution producing
\(\mathbf A_{k+1}^{-}\), followed by the event map producing
\(\mathbf A_{k+1}\).

There must never be a transition such as \(k-3\to k+4\) spanning an event at
\(k\to k+1\). Transition weights stay \(1/L\) so lag averaging and
`stable_lags` keep their present meaning.

## Likelihood

\[
\log p(\text{data})
=
\log p_{\mathrm{cont}}
+
\log p_{\mathrm{event}} ,
\]

both realized by the single Gaussian above: unaffected active rows carry the
ordinary continuous residual, event-affected rows carry the \(\mathbf C_k\)
reduction with \(\mathbf R_k\), and unobservable rows are marginalized out.
Event-affected bands are neither treated as ordinary continuous residuals nor
removed from the fit.

The continuous total-area residual is always evaluated against the **pre-event**
total \(A_{T,k+1}^{-}\), never against the post-event observed total. Comparing
against the post-event total would make part of the residual identically zero
for an exact birth insertion and bias \(D_T\) downward with the birth rate.

The persistent-rate blocks skip event-crossing steps for their AR(1) term and
restart from stationary in created slots, per the \(\mathbf u\) rule above.

### Parameters

`PARAMETER_NAMES` becomes

```text
("tau_p", "kappa_T", "D_u", "D_T", "A_T_star", "sigma_E")
```

with `Scaling.parameter_factors` gaining `self.area` for `sigma_E`, softplus
positivity as for the rest, and a half-normal prior in `coupled_area_model`
scaled to the normalized area unit. Splits and merges share one `sigma_E`. If a
future dataset supplies at least 30 of each event type, splitting it into
`sigma_S` and `sigma_M` is a one-line change; below that threshold, keep them
tied.

## Workflow

\[
A_{T,k}
\rightarrow
\Delta A_{T,k}^{\mathrm{cont}}
\rightarrow
\mathbf A_{k+1}^{-}
\rightarrow
\mathbf A_{k+1}
\rightarrow
A_{T,k+1},
\qquad
A_{T,k+1}=\sum_i m_{i,k+1}A_{i,k+1}.
\]

Do not update \(A_T\) independently; the event-induced total-area change is
determined from the transformed vector. In *simulation* the next state is
\(\mathbf H_k\boldsymbol\mu_k^{-}+\mathbf g_k\) plus noise; in *inference* the
next state is the observation. The next transition starts from
\(\mathbf A_{k+1}\), \(\mathbf m_{k+1}\), \(A_{T,k+1}\), so events propagate
into all later drift, redistribution, covariance, and MSD behavior.

## Code changes

- `tracking.py`: expose the bipartite edge structure per event frame (currently
  discarded after `EventCode` collapse) so the map builder can read it.
- new `events.py`: `EventStep`, edge-structure to
  \((\mathbf H,\mathbf g,\mathbf C,\mathbf o,\mathbf R,\mathbf m')\), slot
  allocation, disjointness assertions.
- new `chains.py`: `EventChain` and its builder; global \(N_{\max}\); knot grid
  construction. `segments.py` untouched.
- `model.py`: masked-space `transition`/`transition_log_density` with the
  \(\operatorname{diag}(\mathbf 1-\mathbf m)\) padding; masked
  `conservative_projection`; masked slope and \(\sigma_b^2\); padded chain
  blocks replacing dimension-grouped blocks for the event path; `sigma_E` in
  the parameter vector.
- `inference.py` / `bayesian.py`: six-parameter vector, `sigma_E` start value
  and prior.
- `validation.py`: event-aware posterior predictive; the forward simulator.
- `workflow.py` / `reporting.py`: run and report both the clean-segment and
  event-chain fits side by side.

## Validation

1. Forward simulator: simulate the masked SDE in \(\mathbb R^{N_{\max}}\) with
   a scripted schedule of frequent births, deaths, splits, and merges.
2. **Parameter recovery**: fit simulated data and assert known parameters fall
   inside their posterior credible intervals.
3. **Bias demonstration**: fit the same simulated data with events ignored
   (clean-segment path only) and assert a measurable bias in `kappa_T` and
   `D_T`. Together with (2), this is the only check that the event machinery
   helps rather than merely runs; the identity checks below would all pass for
   a sign error in \(\mathbf H_k\) that happens to preserve total area.
4. Verify inactive slots satisfy \(m_{i,k}=0\Rightarrow A_{i,k}=0\) throughout.
5. Conservative splits: \(A_{i_1,k+1}+A_{i_2,k+1}=A_{i,k+1}^{-}\).
6. Conservative merges: \(A_{ij,k+1}=A_{i,k+1}^{-}+A_{j,k+1}^{-}\).
7. Hence \(A_{T,k+1}=A_{T,k+1}^{-}\) for splits and merges.
8. Births increase the total area by exactly the inserted birth area.
9. Deaths decrease the total area by the predicted pre-death area.
10. Unaffected bands retain their continuous predictions during an event.
11. Event-affected observations enter through \(\mathbf C_k,\mathbf R_k\)
    rather than as ordinary continuous residuals, and unobservable rows
    (\(o_i=0\)) contribute no parameter-dependent term.
12. No transition in any constructed grid spans an event boundary.
13. Padded-slot log-density contributions are parameter-independent: perturb
    every parameter and assert the inactive-slot contribution is unchanged.
