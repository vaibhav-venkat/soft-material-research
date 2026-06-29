
So compute and analyze:

```text
η = -div ξ
```

You should report:

```text
rms(η)
std(η)
spatial power spectrum of η
autocorrelation of η
correlation time of η
```

This tells you what the unresolved density noise looks like.

---

## Build the simplest stochastic model

The simplest useful model is a **colored low-k noise model**.

Since your residual is low-wavenumber and temporally correlated, do not use white noise. Use an AR(1)/OU model in Fourier space.

For each low Fourier mode `k` of `ξ` or `η`, write:

```text
ξ_k(t + Δt) = α_k ξ_k(t) + σ_k ζ_k(t)
```

where:

```text
α_k = temporal memory of mode k
σ_k = innovation amplitude
ζ_k = random Gaussian noise
```

If `α_k = 0`, the noise is white in time.

If `α_k > 0`, the noise is colored and persistent.

Your data already suggest `α_k` is not zero.

---

## Step 4: estimate α and σ from data

For each retained low-k Fourier mode:

```text
ξ_k(t) = FFT[ξ(x,θ,t)]
```

Estimate:

```text
α_k = <ξ_k(t+1) ξ_k(t)*> / <|ξ_k(t)|²>
```

Then the innovation is:

```text
ε_k(t) = ξ_k(t+1) - α_k ξ_k(t)
```

and:

```text
σ_k² = var(ε_k)
```

So the stochastic model is:

```text
ξ_k(t+1) = α_k ξ_k(t) + σ_k ζ_k(t)
```

Then inverse FFT gives:

```text
ξ_model(x,θ,t)
```

and density evolves by:

```text
∂tρ =
-∇s·[J_base + J_sys]
+ S_cross
- ∇s·ξ_model
```

---

## Step 5: validate statistically, not pointwise

A stochastic model should not be judged by pointwise `R²` against the exact trajectory. It should reproduce statistics.

Check whether `ξ_model` reproduces:

```text
rms(-div ξ)
spatial spectrum of -div ξ
autocorrelation of -div ξ
histogram of -div ξ
density fluctuation variance
```

Then run stochastic rollouts and compare:

```text
mean density evolution
variance across stochastic realizations
spatial spectrum of density fluctuations
```

The ensemble mean should follow the deterministic part:

```text
∂tρ_mean ≈ -∇s·[J_base + J_sys] + S_cross
```

The ensemble spread should match the unresolved residual fluctuations.

---

Update the markdown reports and text reports with this new information.
