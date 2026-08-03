# Workflow for fitting the coupled stable-band area model

## 0. I/O + analysis changes/design principles

You will now allow any number of `n` input directories in which you will use each as a sample point or an ensemble mean based on context for the fitting. You also support 1 seed as well. 

You will also remove any previous analysis in the `band_analysis` that does not directly relate to this, and remove all old plots as well. The only plots that remain will be the new ones detailed in this plan.

In terms of general design principles, yes this is a very complex plan, but keep it as simple as possible. Be concise in your code, unnecessary edge cases which are based on my error (i.e validating that the timesteps are all positive or monotonic) are redundant, the data is already very good. Also don't try for no backwards compatibility, its not needed at this exploratory stage.

Do not create overtly large files but neighter over modulate, like no files except the root `__init__.py` and such with only 10-15 lines. If there are any places here where we could use a python package to simplify some of this, mention it in the plannign stage and I'll confirm/deny whether to use this.

Add concise comments in the code about the general fitting or math, but not about the code. Create decent variable names such that excessive comments are unnecessary. 

## 1. Model definition

For a stable segment containing \(n\) continuously tracked bands, define

\[
\mathbf A=
\begin{pmatrix}
A_1,\ldots,A_n
\end{pmatrix}^{\mathsf T},
\qquad
A_T=\sum_{i=1}^n A_i,
\qquad
\overline A=\frac{A_T}{n}.
\]

Define

\[
P_n=I_n-\frac{1}{n}\mathbf 1\mathbf 1^{\mathsf T},
\]

which projects onto area differences between bands, and define the total-area allocation weights

\[
\mathbf w(\mathbf A)=\frac{\mathbf A}{A_T}.
\]

The model has five main parameters:

\[
\theta=
\left(
\kappa_c,\kappa_T,D_A,D_T,A_T^\ast
\right).
\]

Their meanings are:

- \(\kappa_c\): relaxation rate of differences between band areas. Should be similar or equal to -r_cons.
- \(\kappa_T\): mean-reversion rate of the total stable-band area.
- \(D_A\): stochastic fluctuations that exchange area between bands while preserving \(A_T\).
- \(D_T\): stochastic fluctuations of the total area.
- \(A_T^\ast\): stationary mean total area.

All four rate and diffusion parameters should be positive.

---

## 2. Continuous form

The total area follows an Ornstein–Uhlenbeck process:

\[
\boxed{
dA_T
=
-\kappa_T(A_T-A_T^\ast)\,d\tau
+
\sqrt{2D_T}\,dW_T
}
\]

The coupled individual-band dynamics are

\[
\boxed{
d\mathbf A
=
-\kappa_cP_n\mathbf A\,d\tau
+
\mathbf w(\mathbf A)\,dA_T
+
\sqrt{2D_A}\,P_n\,d\mathbf W
}
\]

Substituting the total-area equation gives

\[
\boxed{
\begin{aligned}
d\mathbf A={}&
\left[
-\kappa_cP_n\mathbf A
-\kappa_T(A_T-A_T^\ast)\mathbf w
\right]d\tau\\
&+
\sqrt{2D_A}\,P_n\,d\mathbf W
+
\sqrt{2D_T}\,\mathbf w\,dW_T .
\end{aligned}
}
\]

For one band,

\[
\begin{aligned}
dA_i={}&
-\kappa_c(A_i-\overline A)\,d\tau\\
&-\kappa_T\frac{A_i}{A_T}(A_T-A_T^\ast)\,d\tau\\
&+
\sqrt{2D_A}
\left(
dW_i-\frac1n\sum_jdW_j
\right)
+
\frac{A_i}{A_T}\sqrt{2D_T}\,dW_T .
\end{aligned}
\]

The projected \(D_A\) noise only redistributes area between bands. Summing the equations over \(i\) exactly recovers the \(A_T\) equation.

---

## 3. Discrete form used for fitting

For a transition from time \(\tau_k\) to \(\tau_{k+1}\), with

\[
\Delta\tau_k=\tau_{k+1}-\tau_k,
\]

Euler–Maruyama gives

\[
\begin{aligned}
\mathbf A_{k+1}
={}&
\mathbf A_k
+
\left[
-\kappa_cP_n\mathbf A_k
-\kappa_T(A_{T,k}-A_T^\ast)\mathbf w_k
\right]\Delta\tau_k\\
&+
\sqrt{2D_A\Delta\tau_k}\,P_n\boldsymbol\eta_k
+
\sqrt{2D_T\Delta\tau_k}\,\mathbf w_k\eta_{T,k},
\end{aligned}
\]

where

\[
\boldsymbol\eta_k\sim\mathcal N(0,I_n),
\qquad
\eta_{T,k}\sim\mathcal N(0,1).
\]

Therefore, each observed transition has the conditional distribution

\[
\boxed{
\mathbf A_{k+1}\mid\mathbf A_k
\sim
\mathcal N(\boldsymbol\mu_k,\Sigma_k)
}
\]

with mean

\[
\boxed{
\boldsymbol\mu_k
=
\mathbf A_k+
\left[
-\kappa_cP_n\mathbf A_k
-\kappa_T(A_{T,k}-A_T^\ast)\mathbf w_k
\right]\Delta\tau_k
}
\]

and covariance

\[
\boxed{
\Sigma_k
=
2\Delta\tau_k
\left[
D_AP_n+D_T\mathbf w_k\mathbf w_k^{\mathsf T}
\right].
}
\]

This multivariate Gaussian transition is the likelihood used for fitting.

Do not add a second likelihood for \(A_T\) when

\[
A_T=\sum_i A_i.
\]

The total-area dynamics are already included in the joint covariance and drift. Fitting both would double-count the same observations.

---

# 4. Prepare the data

Divide the trajectories into uninterrupted stable-band segments.

Each segment should satisfy the following:

- Band identities remain continuously tracked, using the cli flag for the stability checks. This means all properties including total area are calculated only for stable bands.
- The number of bands \(n\) is fixed.
- No transition crosses a birth, disappearance, merge, split, or tracking failure.
  - **NOTE**: The defintion of a split should also be changed if possible. Make sure that a split results in two new STABLE bands, i.e start the trackign when it splits, but if the bands split then remerges within the stable tracking peroid then don't count it as a split. Same with a merge, birth, dissapearence, etc.
- Every band is included in the same order at every frame.
- Exact physical times are retained, prefer simualation time scales \[\tau\]

For each segment, store:

\[
\mathbf A^{(s)}
=
\left[
\mathbf A_0,\mathbf A_1,\ldots,\mathbf A_{T_s-1}
\right],
\]

and

\[
\Delta\boldsymbol\tau^{(s)}
=
\left[
\Delta\tau_0,\ldots,\Delta\tau_{T_s-2}
\right].
\]

Different segments may contain different numbers of bands. The projection matrix \(P_n\) is constructed separately for each segment.

Do not include short-lived bands in these stable-segment fits. Birth and death events should be modeled separately if they are later included.

---

# 5. Implement the transition model in regular JAX

Write one central JAX function that takes

\[
\mathbf A_k,\quad \Delta\tau_k,\quad \theta
\]

and returns

\[
\boldsymbol\mu_k,\quad \Sigma_k.
\]

Everything else should call this same function:

- the L-BFGS likelihood;
- the NumPyro model;
- residual calculations;
- posterior predictive simulations.

The computational sequence for one transition is:

1. Determine \(n\) from the size of \(\mathbf A_k\).
2. Construct \(P_n\).
3. Calculate \(A_{T,k}\).
4. Calculate \(\mathbf w_k=\mathbf A_k/A_{T,k}\).
5. Calculate the drift.
6. Calculate \(\boldsymbol\mu_k\).
7. Calculate \(\Sigma_k\).
8. Add a very small diagonal numerical jitter.
9. Evaluate the multivariate-normal log likelihood.

The conceptual implementation is only:

```python
theta = transform(raw_parameters)
mean, covariance = transition(area, dt, theta)
logp = multivariate_normal_logpdf(next_area, mean, covariance)
loss = -sum(logp_over_all_transitions)
```

Use 64-bit JAX precision if possible because covariance matrices and relaxation rates can otherwise become numerically unstable.

Evaluate the Gaussian log density using a Cholesky decomposition rather than explicitly calculating \(\Sigma^{-1}\).

---

# 6. Parameter transformations

L-BFGS optimizes unconstrained real numbers, but the physical parameters must satisfy

\[
\kappa_c,\kappa_T,D_A,D_T,A_T^\ast>0.
\]

Represent each positive parameter using an unconstrained raw parameter, for example

\[
x=\operatorname{softplus}(x_{\rm raw})+\epsilon.
\]

An exponential transformation is also possible, but softplus usually behaves more gently for extreme initial values.

The optimizer should therefore work with

\[
\boldsymbol\phi\in\mathbb R^5,
\]

while the likelihood receives the transformed physical parameters

\[
\theta=T(\boldsymbol\phi).
\]

---

# 7. Construct empirical initial estimates

Use the empirical diagnostics only to initialize the optimizer. They are not the final estimates.

## Total-area parameters

Estimate

\[
A_T^\ast\approx \langle A_T\rangle.
\]

Regress

\[
\frac{\Delta A_T}{\Delta\tau}
\]

against

\[
A_T-A_T^\ast.
\]

The expected relation is

\[
\mathbb E\left[
\frac{\Delta A_T}{\Delta\tau}
\middle| A_T
\right]
=
-\kappa_T(A_T-A_T^\ast).
\]

Therefore, the fitted slope is approximately

\[
-\kappa_T.
\]

Estimate \(D_T\) from the conditional residual variance:

\[
D_T
\approx
\frac{
\operatorname{Var}
\left[
\Delta A_T+
\kappa_T(A_T-A_T^\ast)\Delta\tau
\right]
}{
2\Delta\tau
}.
\]

## Conservative parameters

Calculate the area deviations

\[
\delta A_i=A_i-\overline A.
\]

After subtracting the fitted total-area contribution, regress the conservative increment (the part of the change in band areas that only redistributes area between bands) against \(\delta A_i\). The expected slope is approximately

\[
-\kappa_c.
\]

Estimate \(D_A\) from the variance of the projected residual increments.

These estimates only need to have the correct order of magnitude, so you can sacrifice some accuracy if it leads to a drastic speedup or memory efficiency, but you don't need to worry about compute too much.

---

# 8. Obtain the initial point estimate using JAXopt L-BFGS

Define the objective as the negative total transition log likelihood:

\[
\boxed{
\mathcal L_{\rm opt}(\theta)
=
-\sum_s\sum_k
\log
p\left(
\mathbf A_{k+1}^{(s)}
\mid
\mathbf A_k^{(s)},\theta
\right).
}
\]

The procedure is:

1. Convert the empirical initial values into unconstrained raw parameters.
2. Pass the negative log likelihood to JAXopt L-BFGS.
3. Use automatic differentiation for the gradient.
4. Run several initializations rather than relying on one starting point.
5. Retain the result with the lowest final negative log likelihood.

A reasonable set of starts is:

- one empirical start;
- several perturbed versions of the empirical start;
- optionally one broad generic start.

The perturbations should be applied in unconstrained parameter space.

After optimization, check:

- the optimizer reports convergence;
- the gradient norm is close to zero;
- different starts converge to similar parameter values;
- the final likelihoods from the best starts are similar.

The output is the maximum-likelihood point estimate

\[
\widehat\theta_{\rm ML}.
\]

---

# 9. Check local identifiability after L-BFGS

Evaluate the Hessian of the negative log likelihood at the optimum:

\[
H
=
\nabla_\phi^2
\mathcal L_{\rm opt}
\left(
\widehat{\boldsymbol\phi}
\right).
\]

Inspect its eigenvalues.

A very small eigenvalue indicates that a combination of parameters is poorly constrained by the data.

Likely correlations include:

\[
(\kappa_c,D_A)
\]

and

\[
(\kappa_T,D_T).
\]

Also inspect a profile likelihood for any parameter that appears weakly identifiable. Fix that parameter at a sequence of values, reoptimize the remaining parameters, and examine how much the likelihood changes.

Do not rely only on the inverse Hessian for final uncertainties. It assumes a locally Gaussian likelihood and can fail when parameters are correlated or skewed.

You just report the hessian and the eigenvalues and id weakly idenfiiable , and i'll instruct you later if we need to modify anything or fix anything based on that.

---

# 10. Define the NumPyro model

Use the same JAX transition function inside NumPyro.

For each stable segment and each transition, the NumPyro model should construct

\[
\boldsymbol\mu_k(\theta)
\]

and

\[
\Sigma_k(\theta),
\]

then observe

\[
\mathbf A_{k+1}
\]

under the multivariate-normal transition distribution.

The NumPyro model should sample the five physical parameters:

\[
\kappa_c,\quad
\kappa_T,\quad
D_A,\quad
D_T,\quad
A_T^\ast.
\]

Use positive-support priors, normally lognormal or half-normal priors.

A suitable general form is

\[
\kappa_c\sim \operatorname{LogNormal}(\mu_c,\sigma_c),
\]

\[
\kappa_T\sim \operatorname{LogNormal}(\mu_T,\sigma_T),
\]

\[
D_A\sim \operatorname{LogNormal}(\mu_A,\sigma_A),
\]

\[
D_T\sim \operatorname{LogNormal}(\mu_{DT},\sigma_{DT}),
\]

\[
A_T^\ast\sim \operatorname{LogNormal}(\mu_{A_T},\sigma_{A_T}).
\]

Choose prior scales using physical orders of magnitude:

\[
\kappa\sim
\frac{1}{\text{typical relaxation time}},
\]

\[
D_A\sim
\frac{(\text{typical band area})^2}
{\text{typical time}},
\]

\[
D_T\sim
\frac{(\text{typical total area})^2}
{\text{typical time}},
\]

\[
A_T^\ast\sim
\text{typical observed total area}.
\]

The priors should be broad enough that the posterior is controlled mainly by the data, but not so broad that NUTS spends time in physically meaningless scales.

---

# 11. Initialize and run NUTS

Initialize the NumPyro sampler at the L-BFGS estimate:

\[
\theta_{\rm init}
=
\widehat\theta_{\rm ML}.
\]

This improves warmup because the chains begin near a region of high likelihood.

A suitable initial sampling configuration is:

- four chains;
- approximately 1,000–2,000 warmup iterations;
- approximately 1,000–2,000 retained samples per chain;
- target acceptance probability around \(0.9\);
- dense mass matrix if parameter correlations are strong.

Run the chains from slightly different initial positions near the L-BFGS result rather than using identical states.

The posterior is

\[
p(\theta\mid\text{data})
\propto
p(\text{data}\mid\theta)p(\theta).
\]

The posterior median or posterior mean can be used as the Bayesian point estimate, while credible intervals quantify uncertainty.

---

# 12. Transfer the result to ArviZ

Convert the NumPyro MCMC output into an ArviZ `InferenceData` object.

Use ArviZ to calculate, for every parameter:

- posterior mean;
- posterior median;
- posterior standard deviation;
- 95% highest-density interval;
- bulk effective sample size;
- tail effective sample size;
- rank-normalized \(\widehat R\).

Plot:

- chain traces;
- marginal posterior distributions;
- rank plots;
- pair plots for correlated parameters;
- energy or Bayesian fraction of missing information diagnostics if necessary.

A small ArviZ call may look conceptually like:

```python
idata = az.from_numpyro(mcmc)
az.summary(idata)
az.plot_trace(idata)
```

---

# 13. MCMC acceptance criteria

The fit should generally satisfy:

\[
\widehat R<1.01
\]

for every parameter.

Bulk and tail effective sample sizes should be at least several hundred and preferably much larger.

There should be no persistent separation between chains.

There should ideally be zero divergent transitions. A very small number should still be investigated rather than ignored.

If divergences occur:

1. Increase the target acceptance probability, for example from \(0.90\) to \(0.95\).
2. Run longer warmup.
3. Inspect pair plots for funnel-shaped or strongly curved parameter relationships.
4. Rescale areas and times if the numerical magnitudes differ greatly.
5. Reconsider overly broad priors.
6. Consider a more suitable parameterization.

Do not interpret posterior intervals until the MCMC diagnostics are satisfactory.

Report the diagnostics like such

---

# 14. One-step residual diagnostics

At each posterior estimate or posterior draw, calculate the conditional residual

\[
\mathbf r_k
=
\mathbf A_{k+1}-\boldsymbol\mu_k.
\]

Whiten it using the Cholesky decomposition

\[
\Sigma_k=L_kL_k^{\mathsf T},
\]

so that

\[
\boxed{
\mathbf z_k=L_k^{-1}\mathbf r_k.
}
\]

Under the fitted model, the pooled whitened residuals should approximately satisfy

\[
\mathbb E[\mathbf z]=0,
\qquad
\operatorname{Cov}(\mathbf z)=I.
\]

Check:

- residual mean;
- residual variance;
- cross-correlations;
- residual autocorrelation;
- residual histograms;
- normal quantile plots.

Also plot residuals against:

\[
A_T,
\qquad
A_i-\overline A,
\qquad
n,
\qquad
\Delta\tau.
\]

Systematic dependence indicates a missing state dependence in the drift or diffusion.

Examples include:

- diffusion increasing with band area;
- relaxation depending on band count;
- nonlinear mean reversion;
- memory remaining at the selected lag.

---

# 15. Posterior predictive validation

Generate posterior predictive transitions by:

1. Drawing a parameter vector from the posterior.
2. Taking an observed \(\mathbf A_k\).
3. Constructing \(\boldsymbol\mu_k\) and \(\Sigma_k\).
4. Drawing a simulated \(\mathbf A_{k+1}\).
5. Repeating across all transitions.

Compare observed and posterior predictive distributions for:

\[
\Delta A_T,
\]

\[
\Delta A_i,
\]

\[
A_i-\overline A,
\]

and projected conservative increments.

The predicted distributions should reproduce both their central values and their spread.

Posterior predictive intervals should contain approximately the expected fraction of observations. For example, roughly 95% of observations should lie inside 95% predictive intervals, subject to sampling uncertainty.

---

# 16. Long-trajectory validation

A good one-step likelihood fit does not guarantee correct long-time dynamics.

For each real stable segment:

1. Start from its observed initial area vector.
2. Draw one posterior parameter set.
3. Simulate the discrete model repeatedly using the same timestep sequence.
4. Repeat for many posterior draws.
5. Compare the resulting trajectories with the observed segment.

Compare:

## Total area

- stationary distribution of \(A_T\);
- mean of \(A_T\);
- variance of \(A_T\);
- autocorrelation of \(A_T\);
- relaxation time of \(A_T\).

## Individual bands

- distribution of \(A_i\);
- distribution of \(A_i-\overline A\);
- conservative-mode autocorrelation;
- area increment distributions;
- area MSD.

## Coupling

- covariance between band increments;
- covariance between conservative modes;
- covariance of each band with \(\Delta A_T\).

The model should reproduce these properties without refitting them separately.

---

# 17. Check positivity

The Gaussian SDE does not mathematically enforce

\[
A_i>0.
\]

Count how often posterior predictive simulations produce negative areas.

---

# 18. Validate the fitting lag

Repeat the entire fitting procedure for several candidate lags.

For every lag, compare the posterior estimates of

\[
\kappa_c,\quad
\kappa_T,\quad
D_A,\quad
D_T,\quad
A_T^\ast.
\]

Look for a lag interval where the inferred parameters are approximately stable within uncertainty.

The selected lag should satisfy both:

1. Short-time tracking and measurement correlations have decayed.
2. The lag remains short relative to the physical relaxation times.

Also inspect whitened residual autocorrelation. If residuals remain correlated, the state \(\mathbf A\) is not Markovian at that lag.

If \(\kappa_c\) continues changing strongly as the lag increases, do not conceal this by choosing one arbitrary lag. Report that the inferred conservative dynamics are scale dependent and that the simple Markov SDE is only an effective model over a specified lag range.

For our case, select the base lag is 2, but also run for lag 1, 3, 5, 10. 

---

# 19. Final fitting sequence

The complete sequence is:

1. Extract uninterrupted stable-band segments.
2. Choose several candidate fitting lags.
3. Implement one JAX transition function returning \(\boldsymbol\mu_k\) and \(\Sigma_k\).
4. Construct the joint multivariate Gaussian transition likelihood.
5. Obtain empirical starting estimates.
6. Transform constrained parameters to unconstrained optimizer variables.
7. Run multi-start JAXopt L-BFGS.
8. Check convergence, gradients, Hessian eigenvalues, and parameter correlations.
9. Use the L-BFGS estimate to initialize NumPyro NUTS.
10. Sample the posterior using several chains.
11. Convert the result to ArviZ.
12. Check \(\widehat R\), effective sample sizes, divergences, traces, and rank plots.
13. Calculate whitened one-step residuals.
14. Perform posterior predictive transition checks.
15. Simulate long trajectories.
16. Check negative-area frequency.
17. Repeat the fit over several lags.
18. Test the final model on held-out simulation seeds.
19. Report the posterior estimates and the lag range over which they are stable.

The final reported parameter table should contain

\[
\kappa_c,\quad
\kappa_T,\quad
D_A,\quad
D_T,\quad
A_T^\ast
\]

with posterior medians and 95% credible intervals, together with the fitting lag, number of stable segments, number of transitions, number of seeds, and MCMC diagnostics.


You should output an variety of new plots giving the validation essentially, so I can interpret them.  
