# Adding the possibility for the area transfer ACF to be negative

Goal: Improve the MSD fits. 

Location: `hexatic/band_analysis`

## Physical Model

General steps

1. **Keep the total-area model unchanged**

It is along the lines of $dA_T := -\kappa_T (A_T - A_T^\ast)dt + \sqrt{2D_T}dW_T$

And remember the actual $A_T$ is calculated fromt he indiviudal area bands summed, not directly from $dA_T$

2. **Transform observed $A_i$ increments into redistsribution**

Let $Q_n$ be the hlemert basis, satisfying

$Q_n^TQ_n = I$ and $Q_n^T \mathbf{1} = 0$ , this should already be calculated.

We are changing/fitting to $\mathbf{y_k}:= Q_n^T\left[\mathbf{A}_{k+1} - \mathbf{A}_{k} - \mathbf{w}_k(A_{T, k+1} - A_{T, k}) \right]$



3. **Keep the segment-specific constant transfer**

$\boldsymbol{\beta}_s \sim  \mathcal{N}(0, \sigma_b^2 I)$

4. **Replace the scalar colored noise OU with a 2d oscillatory OU process**

$\mathbf{z} := \begin{pmatrix} \mathbf{u} \\ \mathbf{r} \end{pmatrix}$

$d\mathbf{z} :=\begin{pmatrix} -\gamma I &  \omega I \\ -\omega I & -\gamma I  \end{pmatrix} \mathbf{z}dt + \sqrt{2D_u} d\mathbf{W}$

Thus 

$C_u(\tau) \sim e^{-\gamma\abs{\tau}} \cos(\omega \tau)$

5. **Use the exact transition for this case**

$F := \begin{pmatrix} -\gamma I &  \omega I \\ -\omega I & -\gamma I  \end{pmatrix}$

$\mathbf{z}_{k+1} = e^{F \Delta t}\mathbf{z}_k + \boldsymbol{\eta}_k$

\(\operatorname{Cov}(\boldsymbol\eta_k) = \frac{D_u}{\gamma} \left(1-e^{-2\gamma\Delta t_k}\right)I.\)

The analytic form of the matrix exponential which you should use, as it yields better performance in `JAX`, is:
\(e^{F\Delta t} = e^{-\gamma\Delta t} \begin{pmatrix} \cos(\omega\Delta t)&\sin(\omega\Delta t)\\ -\sin(\omega\Delta t)&\cos(\omega\Delta t) \end{pmatrix}.\)

So the transition is:

\[ \mathbf z_{k+1} = e^{-\gamma\Delta t} \begin{pmatrix} \cos(\omega\Delta t)&\sin(\omega\Delta t)\\ -\sin(\omega\Delta t)&\cos(\omega\Delta t) \end{pmatrix}\mathbf z_k + \sqrt{\frac{D_u}{\gamma} (1-e^{-2\gamma\Delta t_k})} \,\boldsymbol\xi_k  \]

6. **FIT**

You'll fit $\gamma$, $\omega$, $D_u$, $\sigma_b$ for the $\mathbf{y}_k$, all the other fits will remain the same



**Note about the events**: I don't care too much about the events fits for now, mostly just hte clean. If its easier you can just implement this for the clean.



If you need to intialize an empircally $\omega$ more robustly, do

Initialize

\[ \omega_0\approx\frac{\pi}{2t_{\mathrm{zero}}} \]'

where $t_\text{zero}$ is where the ACF of the observed MSD crosses the x-axis first. If this is too much work just do something on a similar order of magntiude like the current BFGS empiricial fits



## Discretization

As you seen before, we will not be using the regular Euler-Maruyama for these fits so use the exact version.

## Validation

Validation should focus primarily on pre-implementation and development-time tests rather than expensive runtime checks.

### Synthetic data validation

Before fitting real data, create synthetic trajectories from the known oscillatory OU model and verify that the fitting pipeline can recover the parameters.

Generate synthetic data using:

\[
\mathbf z_{k+1}
=
\Phi_k\mathbf z_k+\boldsymbol\eta_k
\]

with known:

\[
\gamma,\quad \omega,\quad D_u,\quad \sigma_b.
\]

Then:

1. Generate synthetic \(A_i\) trajectories using the same reconstruction pipeline as the real data:
\[
\Delta\mathbf A_k
=
\mathbf w_k\Delta A_{T,k}
+
Q_n\mathbf y_k .
\]

2. Run the full likelihood and optimizer on the synthetic data.

3. Verify that:
   - fitted \(\gamma,\omega,D_u,\sigma_b\) recover the known values within uncertainty;
   - the fitted model reproduces the input autocorrelation:
\[
C_u(\tau)=e^{-\gamma\tau}\cos(\omega\tau);
\]
   - the fitted model reproduces the synthetic MSD upto a baseline error.

Test regimes:
- weak oscillations (\(\omega\) small);
- strong oscillations;
- short and long persistence times;

### Regression tests

Add small deterministic tests to ensure:

1. The analytic transition matrix matches the generic matrix exponential:

\[
e^{F\Delta t}
\]

should agree with the analytic cosine/sine form for random values of \(\gamma,\omega,\Delta t\).

2. The simulated autocorrelation follows:

\[
\rho_u(\tau)\approx e^{-\gamma\tau}\cos(\omega\tau).
\]

4. Setting:

\[
\omega=0
\]

recovers the original non-oscillatory OU behavior.

### Runtime validation

Keep runtime validation minimal to avoid slowing inference.

Only validate quantities that would otherwise silently produce incorrect fits:

- fitted parameters remain in their valid 	 ranges;
- covariance matrices remain positive definite;
- JAX computations do not produce NaN/Inf values.

Do not add expensive trajectory-level checks or repeated synthetic validation during normal fitting.



## Code-specific

Use something like `z = jnp.concatenate([u, r])` with shape `(2, m)`, and compute everything vectorized from here. Recall that `r` does not apply directly to the $d\mathbf{A}$ but instead affects `u` which then affects the dA.



Also, for the transition matrix $F$ don't build a giant matrix (the notation i used above was mainly as a guide). Do something like this for `u` and `r`:

```python
u_new = a * (c*u + s*r)
r_new = a * (-s*u + c*r)
a = exp(-gamma*dt)
c = cos(omega*dt)
s = sin(omega*dt)
```

But be more explicit in the variable names of course



## Non-goals

1. Don't think about backwards compatibility at all. I.e don't try to make it backwards compatible, just be indifferent towards it. I am the only user and i can change my own commands

2. No input data verifciation. I.e assume the *.safetensors file is clean and has no issues. Just let things panic or not work in gneral if its not clean, don't error handle this.

## Q/A

### 1. `r` is latent — the likelihood needs a Kalman filter

**Problem.** Today `u_k` *is* the measured rate `Δ(Qᵀ A)/Δt`, so the AR(1) likelihood
factorizes into independent one-step Gaussian terms — that is what
`PersistentRateBlock` / `parameter_negative_log_likelihood` in
`fitting/model.py` exploit (all sequences of equal dimension are concatenated
into one batched block). In the 2D model only `u` is observed; `r` is latent, so
`u` alone is no longer Markov and the batched one-step form is wrong.

**Decision.** Marginalize `r` with an exact Gaussian Kalman filter: `u` observed
(exactly, or with a tiny jitter for numerical conditioning), `r` latent, state
propagated with the analytic `Φ_k` above. Implement as a `jax.lax.scan` per
sequence, initialized from the stationary covariance `(D_u/γ) I`.

**Consequence.** `PersistentRateBlock` and the batched rate likelihood are
replaced by per-sequence scans. Sequences must keep their per-sequence identity
(they can still be padded/stacked to a common length and `vmap`-ed, with a mask
zeroing padded steps), rather than being concatenated across sequences as they
are now in `build_persistent_rate_blocks`. `persistent_innovations` becomes the
filter's standardized innovation sequence, which is what
`_temporal_residual_acf` in `fitting/validation.py` should consume.

### 2. `w_k` is area-fraction weighted

`w_{i,k} = A_{i,k}/A_{T,k}`, **not** the uniform `1/n` currently used. This
matters: under uniform weights `Qᵀ w_k = 0` and the whole `w_k ΔA_T` term would
vanish from `y_k`, making step 2 a no-op.

**Consequence.** The uniform allocation must be replaced everywhere it appears:

- `transition()` in `fitting/model.py` — `allocation = ones/n` becomes
  `area/area.sum()`, both in the drift and in the `2 D_T dt · w wᵀ` total-mode
  covariance;
- `build_state_space_sequences` — subtracting the row mean is no longer the
  right projection of the total mode; use the observed `ΔA_T` and the observed
  `w_k` at the left endpoint of each step;
- the forward simulator in `validation.py::_paths`, which currently splits into
  `total/n` plus a mean-removed conservative part.

`w_k` is built from the *observed* areas at step `k`, so it stays exogenous and
the likelihood remains conditionally Gaussian — do not make it a function of the
latent state.

### 3. `σ_b` is fitted, and `β_s` is marginalized

Today `σ_b` is not a likelihood parameter at all: `conservative_segment_slope`
detrends each segment by its endpoint slope and `estimate_slope_variance`
computes `σ_b²` once outside the optimizer.

**Decision.** Drop the endpoint detrending. Treat `β_s ~ N(0, σ_b² I)` as a
per-segment random effect integrated out inside the Gaussian likelihood — it
folds naturally into the Kalman filter by augmenting the state with a constant
component (`β̇ = 0`, initial covariance `σ_b² I`), so `σ_b` is fit jointly with
`γ, ω, D_u`.

**Rationale.** Endpoint detrending and the oscillation compete for the same
low-frequency signal; detrending first will absorb part of the oscillation and
bias `ω` upward.

**Consequence.** `conservative_segment_slope`, `conservative_segment_slopes`,
`estimate_slope_variance`, and the `conservative_slopes` / `sigma_b_squared`
fields of `TrainingTransitions` all go away, as does the `sigma_b_squared`
argument threaded through `transition`, `transition_log_density`, and the
`_transition_rows` / `_density_rows` vmaps in `validation.py`.

### 4. Scope of the change

Parameters become `(gamma, omega, kappa_T, D_u, D_T, A_T_star, sigma_b)`;
`tau_p` is gone. Update in the same pass:

- **`fitting/model.py`** — `PARAMETER_NAMES`, `Scaling.parameter_factors` (`γ`
  and `ω` both scale as `1/time`; `σ_b` as `area/time`), the Kalman likelihood,
  and the empirical starts in `fitting/inference.py` (including the `ω₀ ≈
  π/(2 t_zero)` initialization and the `generic_values` list in `_starts`,
  which is hardcoded to five entries).
- **`fitting/bayesian.py`** — `coupled_area_model` priors: LogNormal for
  `γ, D_u, σ_b`, and a prior for `ω` centered on the empirical `ω₀`. The
  index-based `0.5 if index == 4` special case must be re-pointed at
  `A_T_star`.
- **`fitting/validation.py` + plots** — the forward simulator `_paths`
  propagates a scalar `rates` AR(1) today; it must carry `(u, r)` with the
  analytic rotation. `_posterior_matrix` hardcodes 5 columns. The
  transfer-rate ACF plot (`_transfer_rate_series`, `pipeline/plots.py`) is the
  diagnostic this whole change targets, so it must reflect the new model.
- **`tests/`** — synthetic recovery + the regression tests listed above
  (analytic `Φ` vs `expm`, simulated ACF vs `e^{-γτ}cos ωτ`, and `ω → 0`
  reducing to the old OU).

**Out of scope:** `fitting/masked_model.py` (the event path) keeps the old
non-oscillatory OU. Its parameter vector will therefore differ from the clean
path's — that is accepted, and the `TrainingData` union in `inference.py` /
`bayesian.py` should simply dispatch on type as it already does.
