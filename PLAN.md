# One-mode colored-noise fit for dilute-band area

Location: `hexatic/band_analysis`

## Physical model

The total band area remains an OU process:

\[
dA_T = -\kappa_T(A_T-A_T^*)dt + \sqrt{2D_T}\,dW_T.
\]

For a segment with `n` bands, let `Q_n` be the Helmert basis and use the
observed left-endpoint area fractions

\[
w_{i,k}=\frac{A_{i,k}}{A_{T,k}}.
\]

The observed redistribution increment is

\[
y_k=Q_n^T\left[A_{k+1}-A_k-w_k(A_{T,k+1}-A_{T,k})\right].
\]

Each redistribution coordinate has one colored transfer rate:

\[
du=-\gamma u\,dt+\sqrt{2D_u}\,dW,
\qquad
\beta_s\sim\mathcal N(0,\sigma_b^2),
\]

and

\[
y_k=\int_{t_k}^{t_{k+1}}(u(t)+\beta_s)\,dt.
\]

Thus `tau_p = 1 / gamma`. There is no fitted frequency and no auxiliary
oscillatory state.

## Exact continuous-time transition

For an interval `dt`,

\[
u_{k+1}=a u_k+\eta_k,
\qquad
a=e^{-\gamma dt},
\qquad
\operatorname{Var}(\eta_k)=\frac{D_u}{\gamma}(1-a^2).
\]

The interval integral is handled jointly with the endpoint state using its
exact Gaussian moments. The coefficient of the left endpoint is

\[
G=\frac{1-a}{\gamma},
\]

and the stationary integral variance is

\[
\operatorname{Var}\left(\int_0^{dt}u(t)dt\right)
=\frac{2D_u}{\gamma^2}\left(dt-G\right).
\]

A Kalman scan marginalizes `u`; the segment-level `beta_s` is marginalized in
the likelihood. No Euler--Maruyama approximation is used.

## Fitted parameters

The clean fit parameters are

`(gamma, kappa_T, D_u, D_T, A_T_star, sigma_b)`.

The event fit remains separate and keeps its existing parameterization.

## Validation

Development tests cover the exact OU endpoint and interval moments, weighted
redistribution reconstruction, finite likelihood gradients, and the Kalman
likelihood. Runtime validation remains focused on posterior predictive,
residual, ACF, MSD, and trajectory diagnostics without fixed parameter-count
checks.
