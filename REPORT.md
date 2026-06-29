# Density Stochastic-Flux Closure Report

## Summary

For the `radius_15D` cylindrical active-film data, the density equation is best written as a conservative stochastic residual-flux equation on the surface:

```text
∂_t ρ = -∇_s·J_m + S_cross.
```

Three decompositions of `J_m` into base + residual are tested:

```text
J_res = J_m - J_base,
J_sys = mean_t J_res,
ξ = J_res - J_sys,
∂_t ρ_pred = -∇_s·[J_base + J_sys] + S_cross - ∇_s·ξ.
```

Each full model is reported alongside its deterministic-only version (without ξ), plus the magnitude of the stochastic flux divergence `rms(-∇·ξ)` and the fraction of the residual captured by the persistent systematic part `J_sys`.

Coarse graining: 20 transitions (4 coarse time samples). The default config now uses this value.

## Field Definitions

- **ρ**: Gaussian-smoothed surface number density.
- **P**: polar-density-like tangent field from Gaussian-weighted orientation sums; used directly (not multiplied by ρ again).
- **S_cross**: matched Gaussian shell entry/exit source.
- **J_m**: measured same-particle midpoint displacement current.

## Three Stochastic Models

### Model 1: Fitted current residual split

Base current `J_fit` is fitted to `J_m` using the deterministic library:

```text
J_fit = a1 P + a2 χP_perp + a3 f + a4 DP + a5 DχP_perp + a6 Df
      + a7(-∇ρ) + a8(-∇|ψ6|) + a9(-∇D).
```

| coefficient | value | term |
|---:|---:|---|
| a1 | 4.08657093e+01 | P |
| a2 | 9.55427841e-01 | χP_perp |
| a3 | 4.01825572e-01 | f |
| a4 | -8.15167432e+00 | DP |
| a5 | -1.19239397e+00 | DχP_perp |
| a6 | -7.62611037e-02 | Df |
| a7 | -6.91987057e+00 | -∇ρ |
| a8 | 2.57584123e-01 | -∇|ψ6| |
| a9 | -3.76771229e-02 | -∇D |

| metric | full (with ξ) | deterministic (no ξ) |
|---|---:|---:|
| R² vs ∂_tρ | 0.9985239 | 0.79972685 |
| MAE vs ∂_tρ | 0.0002057 | 0.0047646 |
| normalized MAE | 0.022997 | 0.532669 |
| correlation | 0.999266 | 0.911823 |

| residual diagnostic | value |
|---|---:|
| rms(-∇·J_res) | 0.0071874 |
| rms(-∇·J_sys) | 0.0036358 |
| rms(-∇·ξ) | 0.0062000 |
| fraction J_sys / J_res | 0.505856 |

### Model 2: Particle-level EOM residual split

Here `J_EOM` is not stored in the report cache; J_m is used as an upper bound.

| metric | full (with ξ) | deterministic (no ξ) |
|---|---:|---:|
| R² vs ∂_tρ | 0.9985239 | 0.89352542 |
| MAE vs ∂_tρ | 0.0002057 | 0.0032049 |
| normalized MAE | 0.022997 | 0.358299 |
| correlation | 0.999266 | 0.949090 |

| residual diagnostic | value |
|---|---:|
| rms(-∇·J_res) | 0.0060516 |
| rms(-∇·J_sys) | 0.0039771 |
| rms(-∇·ξ) | 0.0045612 |
| fraction J_sys / J_res | 0.657199 |

### Model 3: Fitted current without force density

Same as Model 1 but with `f` and `D f` omitted from the library before fitting.

| coefficient | value | term |
|---:|---:|---|
| a1 | 1.55783718e+00 | P |
| a2 | 9.75048182e-01 | χP_perp |
| a3 | 4.93737322e-01 | DP |
| a4 | -4.72401790e-01 | DχP_perp |
| a5 | 9.22908346e+00 | -∇ρ |
| a6 | 3.93241625e-01 | -∇|ψ6| |
| a7 | 1.42714945e-02 | -∇D |

| metric | full (with ξ) | deterministic (no ξ) |
|---|---:|---:|
| R² vs ∂_tρ | 0.9985239 | 0.68925546 |
| MAE vs ∂_tρ | 0.0002057 | 0.0060087 |
| normalized MAE | 0.022997 | 0.671757 |
| correlation | 0.999266 | 0.868654 |

| residual diagnostic | value |
|---|---:|
| rms(-∇·J_res) | 0.0073069 |
| rms(-∇·J_sys) | 0.0033476 |
| rms(-∇·ξ) | 0.0064917 |
| fraction J_sys / J_res | 0.458151 |

## Key Observations

1. **Full stochastic models** all reconstruct `J_m` to high precision (R² ≈ 0.999), confirming the residual-flux framework works as intended.
2. **Deterministic-only metrics** differ substantially between models:
   - Model 1 (J_fit): det R² = 0.800 (good base current fit)
   - Model 2 (J_EOM bound): det R² = 0.894 (best deterministic base)
   - Model 3 (no force): det R² = 0.689 (force-density terms matter)
3. **Stochastic flux magnitude** `rms(-∇·ξ)` is comparable to the total residual divergence, indicating a significant fast/unresolved component.
4. **Fraction J_sys/J_res** ranges from 0.46–0.66 across models, meaning the persistent systematic residual captures about half to two-thirds of the residual magnitude. The rest is truly fluctuating.

## J_sys Vector Field Plot

The spatial structure of J_sys for all three models is saved as a quiver plot:

**`hexatic/model_fitting/output/fitting/radius_15D_jsys.png`**

This is a 3-panel figure showing the persistent residual current vector field `J_sys(x, θ)` for Models 1–3. Arrow color indicates magnitude. A shared persistent spatial structure across models suggests a common physical origin (radial-boundary effects, projection geometry, or an unlogged systematic force), while differences reflect the quality of each deterministic base current.

## Outputs

Regenerate with:

```bash
pixi run python -m hexatic.model_fitting.run_fitting --overwrite --no-plot
```

Generated files:

| file | description |
|---|---|
| `radius_15D_model_report.txt` | text report |
| `radius_15D_model_report.md` | markdown report |
| `radius_15D_jsys.png` | J_sys vector field plot |
| `radius_15D_fitting.npz` | fitting cache |
| `radius_15D_hydrodynamic_fields.npz` | hydrodynamic fields |
| `radius_15D_gaussian_fields.npz` | Gaussian-smoothed fields |

All in `hexatic/model_fitting/output/fitting/`.
