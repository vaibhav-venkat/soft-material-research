# Density Stochastic-Flux Report: `radius_15D`

## Headline Model 3 Result

```text
partial_t rho_pred = -div(J_fit_model3) + S_cross
J_fit_model3 = J_fit_no_force + J_sys_no_force + xi_AR1
```

| metric vs `partial_t rho` | value |
|---|---:|
| R² | `0.7959097` |
| MAE | `0.0046818884` |
| normalized MAE | `0.52341923` |
| correlation | `0.90964319` |

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
| S_cross_pred | `0.00742292` | `0.0125048` | `0.014542` | `-0.0205416` | `0.0533479` |
| Y_ρ | `2.64301e-07` | `0.00593726` | `0.00593726` | `-0.0514888` | `0.0214055` |

## Fitted S_cross Source Model

```text
S_cross_pred = c0 1 + c1 rho + c2 laplacian rho + c3 D + c4 |psi6| + c5 P_r + c6 h + c7 h rho + c8 h P_r + c9 D P_r + c10 D rho
```

| metric vs actual `S_cross` | value |
|---|---:|
| R² | `0.72679327` |
| MAE | `0.0059726377` |
| normalized MAE | `0.63419001` |
| correlation | `0.85252171` |

| coefficient | value | term |
|---:|---:|---|
| `c0` | `2.78233935e+02` | 1 |
| `c1` | `1.74951104e+03` | rho |
| `c2` | `-5.89207934e+00` | laplacian rho |
| `c3` | `3.07957076e-01` | D |
| `c4` | `1.68851926e-01` | |psi6| |
| `c5` | `-1.17617463e+04` | P_r |
| `c6` | `-5.55637177e+02` | h |
| `c7` | `-3.50606284e+03` | h rho |
| `c8` | `2.35239848e+04` | h P_r |
| `c9` | `2.07668460e+00` | D P_r |
| `c10` | `-2.72736597e+00` | D rho |

## Three Full Stochastic Density Models

| model | R² AR(1)/full | R² with S_cross_pred AR(1) | R² det (no ξ) | rms(-∇·ξ) | fraction J_sys/J_res |
|---|---:|---:|---:|---:|---:|
| J_fit residual split | `0.9985239` |  | `0.79972685` | `0.0061999701` | `0.50585587` |
| J_EOM residual split | `0.9985239` |  | `0.89352542` | `0.0045611717` | `0.65719928` |
| J_fit without force_density residual split + 85% η-power AR(1) | `0.7959097` | `0.53604607` | `0.73861094` | `0.0070611948` | `0.68643877` |

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

Current fit against `J_m`:

| quantity | R² |
|---|---:|
| combined components | `0.64615782` |
| x component | `0.66838023` |
| theta component | `0.46062846` |

Adaptive Fourier stochastic mechanism:

| quantity | value |
|---|---:|
| selected modes | empirical `eta` power ranking, keep 85% |
| retained Fourier current modes | `338` |
| mean abs(`alpha_k`) | `0.47998866` |
| mean `sigma_k` | `9.8577815` |
| mean modal correlation time | `2.7248224` |

Final density model with seeded AR(1) `xi`:

| metric | value |
|---|---:|
| R2 vs `partial_t rho` | `0.83722772` |
| MAE vs `partial_t rho` | `0.0042312293` |
| normalized MAE | `0.47303707` |
| correlation | `0.92637418` |

`eta = -div xi` statistics, empirical vs AR(1) mechanism:

| statistic | empirical | AR(1) mechanism |
|---|---:|---:|
| rms | `0.0061999701` | `0.0055126857` |
| std | `0.0061999701` | `0.0055126857` |
| retained eta power fraction | `0.85037549` | `0.99711752` |
| dominant mode `(x,theta)` | `(1, 0)` | `(1, 0)` |
| lag-1 autocorrelation | `-0.26408635` | `-0.42544221` |
| correlation time | `0` | `0` |

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

Adaptive Fourier stochastic mechanism:

| quantity | value |
|---|---:|
| selected modes | empirical `eta` power ranking, keep 85% |
| retained Fourier current modes | `278` |
| mean abs(`alpha_k`) | `0.36725562` |
| mean `sigma_k` | `13.1414` |
| mean modal correlation time | `1.9966114` |

Final density model with seeded AR(1) `xi`:

| metric | value |
|---|---:|
| R2 vs `partial_t rho` | `0.917926` |
| MAE vs `partial_t rho` | `0.0030311321` |
| normalized MAE | `0.33887028` |
| correlation | `0.9605412` |

`eta = -div xi` statistics, empirical vs AR(1) mechanism:

| statistic | empirical | AR(1) mechanism |
|---|---:|---:|
| rms | `0.0045611717` | `0.0043225049` |
| std | `0.0045611717` | `0.0043225049` |
| retained eta power fraction | `0.8511737` | `0.99920756` |
| dominant mode `(x,theta)` | `(1, 2)` | `(1, -3)` |
| lag-1 autocorrelation | `-0.19693555` | `-0.38616889` |
| correlation time | `0` | `0` |

Note: J_EOM = J_active + J_pair + J_wall; full residual identity reconstructs J_m

No fitted `a_i`; EOM terms use fixed microscopic coefficients.

### Model 3: J_fit without force_density residual split + 85% η-power AR(1)

Full equation:
```text
-∇_s·[J_fit_no_f + J_sys_no_f] + S_cross + η_AR1
```
Deterministic (no ξ):
```text
-∇_s·[J_fit_no_f + J_sys_no_f] + S_cross
```

| metric | full (with ξ) | deterministic (no ξ) |
|---|---:|---:|
| R² vs `∂_tρ` | `0.7959097` | `0.73861094` |
| MAE vs `∂_tρ` | `0.0046818884` | `0.0054857519` |
| normalized MAE | `0.52341923` | `0.61328843` |
| correlation | `0.90964319` | `0.88616915` |

Residual diagnostics:

| quantity | value |
|---|---:|
| rms(-∇·J_res) | `0.0097102793` |
| rms(-∇·J_sys) | `0.0066655122` |
| rms(-∇·ξ) | `0.0070611948` |
| fraction J_sys / J_res | `0.68643877` |

Current fit against `J_m`:

| quantity | R² |
|---|---:|
| combined components | `0.13457788` |
| x component | `0.17218961` |
| theta component | `-0.18720045` |

Actual `S_cross` baseline:

Adaptive Fourier stochastic mechanism:

| quantity | value |
|---|---:|
| selected modes | empirical `eta` power ranking, keep 85% |
| retained Fourier current modes | `394` |
| mean abs(`alpha_k`) | `0.42365618` |
| mean `sigma_k` | `10.790699` |
| mean modal correlation time | `2.3287413` |

Final density model with seeded AR(1) `xi`:

| metric | value |
|---|---:|
| R2 vs `partial_t rho` | `0.7959097` |
| MAE vs `partial_t rho` | `0.0046818884` |
| normalized MAE | `0.52341923` |
| correlation | `0.90964319` |

`eta = -div xi` statistics, empirical vs AR(1) mechanism:

| statistic | empirical | AR(1) mechanism |
|---|---:|---:|
| rms | `0.0070611948` | `0.0061714928` |
| std | `0.0070611948` | `0.0061714928` |
| retained eta power fraction | `0.85046645` | `0.99981847` |
| dominant mode `(x,theta)` | `(1, 2)` | `(1, 2)` |
| lag-1 autocorrelation | `-0.13607439` | `-0.35065818` |
| correlation time | `0` | `0` |

Fitted `S_cross_pred` substituted for actual `S_cross`:

Adaptive Fourier stochastic mechanism:

| quantity | value |
|---|---:|
| selected modes | empirical `eta` power ranking, keep 85% |
| retained Fourier current modes | `394` |
| mean abs(`alpha_k`) | `0.42365618` |
| mean `sigma_k` | `10.790699` |
| mean modal correlation time | `2.3287413` |

Final density model with seeded AR(1) `xi`:

| metric | value |
|---|---:|
| R2 vs `partial_t rho` | `0.53604607` |
| MAE vs `partial_t rho` | `0.0073621421` |
| normalized MAE | `0.82306249` |
| correlation | `0.762229` |

`eta = -div xi` statistics, empirical vs AR(1) mechanism:

| statistic | empirical | AR(1) mechanism |
|---|---:|---:|
| rms | `0.0070611948` | `0.0061714928` |
| std | `0.0070611948` | `0.0061714928` |
| retained eta power fraction | `0.85046645` | `0.99981847` |
| dominant mode `(x,theta)` | `(1, 2)` | `(1, 2)` |
| lag-1 autocorrelation | `-0.13607439` | `-0.35065818` |
| correlation time | `0` | `0` |

Note: force_density and D force_density omitted from J_fit before residual split; S_cross_pred substituted-source metrics are reported separately from the actual-S_cross baseline; final model uses seeded adaptive ξ_85 selected from empirical η power

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