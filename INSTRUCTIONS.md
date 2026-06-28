You will be doing a change for model_fitting/. 

# Target model

Use this as the model form:

```other
partial_t ρ = -∇s · Jρ + S_cross_measured
partial_t P = FP
```

where:

```other
∇s = (∂x, ∂y),  y = Rθ
P = (P_x, P_y)
P_perp = (-P_y, P_x)
```

For each time frame, compute guassian smoothed fields on the same `(x, y = Rθ)` grid:

```other
ρ
P_x, P_y
chirality
D
hexatic_order
S_cross
```

Then compute derivatives using the same smoothed representation:

```other
partial_t ρ
partial_t P_x
partial_t P_y
∇ρ
∇D
∇|ψ6|
∇·P
∇·(χP_perp)
(P·∇)P
(P_perp·∇)P
laplacian P
laplacian P_perp
```
|ψ6| = hexatic_order

Everything must use the same fourier operator.

Use the same mask for all fits:

```other
valid density
valid P magnitude
```

## Do not mix finite-volume exact fields with FFT/smoothed fields in the same regression. For this workflow, use the **hydrodynamic smoothed fields**.

# Step 2 — Fit the density equation

Fit this target:

```other
Yρ = partial_t ρ - S_cross_measured
```

against a divergence library:

```other
Yρ ≈ c1[-div * P]
   + c2[-div (chirality * P_perp)]
   + c3[laplacian ρ]
   + c4[laplacian hexatic_order]
   + c5[laplacian D]
```

This corresponds to:

```other
Jρ =
    c1 P
  + c2 χP_perp
  - c3 ∇ρ
  - c4 ∇|ψ6|
  - c5 ∇D
```

So explicitly:

Do **not** include force density in this equation. 

# Step 3 — Fit the polarization equation

Fit `P_x` and `P_y` together, with shared scalar coefficients.

Targets:

```other
Y_Px = partial_t P_x
Y_Py = partial_t P_y
```

Stack them:

```other
[partial_t P_x]
[partial_t P_y]
```

```other
partial_t P =
    b1 P
  + b2 χP_perp
  + b3 ∇ρ
  + b4 ∇D
  + b5 ∇|ψ6|
  + b6 D P
  + b7 D χP_perp
  + b8 |P|² P
  + b9 (P·∇)P
  + b10 (P_perp·∇)P
  + b11 laplacian P
  + b12 laplacian P_perp
```


# Step 4 — Regression method

Use the same method for both fits:

```other
1. Flatten all valid space-time samples.
2. RMS-normalize every feature column.
3. Fit ridge first.
4. Then apply STLSQ/SR3 sparsification.
5. Convert coefficients back to physical units.
```

Do not use local coefficient maps for the main model, flatten the mask.

For density:

```other
Yρ.shape = (N_samples,)
fields.shape = (N_samples, 5)
c.shape  = (5,)
```

For polarization:

```other
YP.shape = (2N_samples,)
fields.shape = (2N_samples, 12)
b.shape  = (12,)
```

## The key is that `b1` multiplies both `P_x` and `P_y`, `b2` multiplies both components of `χP_perp`, etc. That keeps it a vector PDE.

# Step 5 — What to report after fitting

For the density fit, report:

```other
corr(Yρ_pred, Yρ)
R²(Yρ)
normalized MAE(Yρ)
term contribution maps for each density feature (coefficient * field) / Y_rho
```

For the polarization fit, report:

```other
corr(partial_t P_pred, partial_t P)
R² for partial_t P_x
R² for partial_t P_y
normalized MAE for partial_t P_x
normalized MAE for partial_t P_y
curl-related residual structure
```

Also plot:

```other
partial_t ρ true vs predicted
partial_t P_x true vs predicted
partial_t P_y true vs predicted
residual maps
```
# Refactor
- Cleanup a lot of the code within the model_fitting/fitting directory. Some of the files are very long
- Then, make sure adding/removing fields is easy. Doesn't mean to overmodulate and seaprate files, just make sure that you don't have to change things across 10 different files to compile successfully
- Reduce the complexity of the classes, config, types, etc. It doesn't need to be this much.
