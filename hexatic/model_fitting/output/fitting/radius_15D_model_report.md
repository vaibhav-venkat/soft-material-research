# Density Stochastic-Flux Report: `radius_15D`

## Governing Equation

```text
∂_t ρ_pred = -∇_s·[J_base + J_sys] + S_cross - ∇_s·ξ
J_res = J_m - J_base,  J_sys = mean_t J_res,  ξ = J_res - J_sys
```

Because `ξ` is defined from the residual, each full stochastic model reconstructs `J_m` by identity; the metrics below are against `∂_tρ`.

## Field Summaries

| field | mean | std | rms | min | max |
|---|---:|---:|---:|---:|---:|
| ∂_t ρ | `0.00742318` | `0.0136621` | `0.0155485` | `-0.0118795` | `0.0491443` |
| S_cross | `0.00742292` | `0.014668` | `0.0164393` | `-0.0118802` | `0.0809634` |
| Y_ρ | `2.64301e-07` | `0.00593726` | `0.00593726` | `-0.0514888` | `0.0214055` |

## Three Full Stochastic Density Models

| model | R² full | R² det (no ξ) | rms(-∇·ξ) | fraction J_sys/J_res |
|---|---:|---:|---:|---:|
| J_fit residual split | `0.9985239` | `0.79972685` | `0.0061999701` | `0.50585587` |
| J_EOM residual split | `0.9985239` | `0.89352542` | `0.0045611717` | `0.65719928` |
| J_fit without force_density residual split | `0.9985239` | `0.73861094` | `0.0070611948` | `0.68643877` |

### Model 1: J_fit residual split

Full equation:
```text
-∇_s·[J_fit + J_sys_fit] + S_cross - ∇_s·ξ_fit
```
Deterministic (no ξ):
```text
-∇_s·[J_fit + J_sys_fit] + S_cross
```

| metric | full (with ξ) | deterministic (no ξ) |
|---|---:|---:|
| R² vs `∂_tρ` | `0.9985239` | `0.79972685` |
| MAE vs `∂_tρ` | `0.00020570209` | `0.0047646304` |
| normalized MAE | `0.022996795` | `0.53266949` |
| correlation | `0.99926647` | `0.91182345` |

Residual diagnostics:

| quantity | value |
|---|---:|
| rms(-∇·J_res) | `0.0071873877` |
| rms(-∇·J_sys) | `0.0036357823` |
| rms(-∇·ξ) | `0.0061999701` |
| fraction J_sys / J_res | `0.50585587` |

Note: J_res_fit = J_m - J_fit; J_sys_fit = mean_t J_res_fit; ξ_fit = J_res_fit - J_sys_fit

| coefficient | value | term |
|---:|---:|---|
| `a1` | `4.08657093e+01` | P |
| `a2` | `9.55427841e-01` | chirality P_perp |
| `a3` | `4.01825572e-01` | force_density |
| `a4` | `-8.15167432e+00` | D P |
| `a5` | `-1.19239397e+00` | D chirality P_perp |
| `a6` | `-7.62611037e-02` | D force_density |
| `a7` | `-6.91987057e+00` | -grad rho |
| `a8` | `2.57584123e-01` | -grad hexatic_order |
| `a9` | `-3.76771229e-02` | -grad D |

### Model 2: J_EOM residual split

Full equation:
```text
-∇_s·[J_EOM + J_sys] + S_cross - ∇_s·ξ
```
Deterministic (no ξ):
```text
-∇_s·[J_EOM + J_sys] + S_cross
```

| metric | full (with ξ) | deterministic (no ξ) |
|---|---:|---:|
| R² vs `∂_tρ` | `0.9985239` | `0.89352542` |
| MAE vs `∂_tρ` | `0.00020570209` | `0.0032049176` |
| normalized MAE | `0.022996795` | `0.3582989` |
| correlation | `0.99926647` | `0.94908979` |

Residual diagnostics:

| quantity | value |
|---|---:|
| rms(-∇·J_res) | `0.0060515719` |
| rms(-∇·J_sys) | `0.0039770887` |
| rms(-∇·ξ) | `0.0045611717` |
| fraction J_sys / J_res | `0.65719928` |

Note: J_EOM = J_active + J_pair + J_wall; full residual identity reconstructs J_m

No fitted `a_i`; EOM terms use fixed microscopic coefficients.

### Model 3: J_fit without force_density residual split

Full equation:
```text
-∇_s·[J_fit_no_f + J_sys_no_f] + S_cross - ∇_s·ξ_no_f
```
Deterministic (no ξ):
```text
-∇_s·[J_fit_no_f + J_sys_no_f] + S_cross
```

| metric | full (with ξ) | deterministic (no ξ) |
|---|---:|---:|
| R² vs `∂_tρ` | `0.9985239` | `0.73861094` |
| MAE vs `∂_tρ` | `0.00020570209` | `0.0054857519` |
| normalized MAE | `0.022996795` | `0.61328843` |
| correlation | `0.99926647` | `0.88616915` |

Residual diagnostics:

| quantity | value |
|---|---:|
| rms(-∇·J_res) | `0.0097102793` |
| rms(-∇·J_sys) | `0.0066655122` |
| rms(-∇·ξ) | `0.0070611948` |
| fraction J_sys / J_res | `0.68643877` |

Note: force_density and D force_density omitted from J_fit before residual split

| coefficient | value | term |
|---:|---:|---|
| `a1` | `1.24444814e+00` | P |
| `a2` | `8.43386162e-01` | chirality P_perp |
| `a3` | `4.98806938e-01` | D P |
| `a4` | `-3.06525183e-01` | D chirality P_perp |
| `a5` | `-2.08889813e+00` | -grad rho |
| `a6` | `6.38386250e-01` | -grad hexatic_order |
| `a7` | `-1.06673006e-02` | -grad D |
| `a8` | `1.99801704e+00` | low-k(P) |

## Saved Outputs

- Text report: `/Users/vaibhavvenkat/structures_and_functions/hexatic/model_fitting/output/fitting/radius_15D_model_report.txt`
- Markdown report: `/Users/vaibhavvenkat/structures_and_functions/hexatic/model_fitting/output/fitting/radius_15D_model_report.md`