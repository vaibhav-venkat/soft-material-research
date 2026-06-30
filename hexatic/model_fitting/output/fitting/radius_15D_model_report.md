# Density Stochastic-Flux Report: `radius_15D`

## Headline Model 3 Result

```text
partial_t rho_pred = -div(J_fit_no_force + J_sys_no_force) + S_cross + eta_AR1
```

| metric vs `partial_t rho` | value |
|---|---:|
| R² | `0.7914742` |
| seed ensemble mean R² | `0.78512208` |
| seed ensemble median R² | `0.78680359` |
| MAE | `0.0047906408` |
| normalized MAE | `0.53557737` |
| correlation | `0.90737905` |

With fitted `S_cross_pred` plus source AR(1):

```text
partial_t rho_pred = -div(J_fit_no_force + J_sys_no_force) + S_cross_pred + eta_AR1 + zeta_AR1
```

| metric vs `partial_t rho` | value |
|---|---:|
| R² | `0.56430574` |
| seed ensemble mean R² | `0.55540265` |
| seed ensemble median R² | `0.55582974` |
| normalized MAE | `0.79936431` |

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

| model | R² AR(1)/full | R² det (no ξ) | rms(-∇·ξ) | fraction J_sys/J_res |
|---|---:|---:|---:|---:|
| J_fit residual split | `0.9985239` | `0.79972685` | `0.0061999701` | `0.50585587` |
| J_EOM residual split | `0.9985239` | `0.89352542` | `0.0045611717` | `0.65719928` |
| J_fit without force_density residual split + 80% η-power AR(1) | `0.7914742` | `0.73861094` | `0.0070611948` | `0.68643877` |

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
| selected modes | empirical `eta` power ranking, keep 80% |
| retained Fourier current modes | `282` |
| mean abs(`alpha_k`) | `0.47371713` |
| mean `sigma_k` | `10.593786` |
| mean modal correlation time | `2.6768569` |

Final density model with seeded AR(1) `xi`:

| metric | value |
|---|---:|
| R2 vs `partial_t rho` | `0.83484141` |
| seed ensemble mean R2 | `0.83467419` |
| seed ensemble median R2 | `0.83536725` |
| MAE vs `partial_t rho` | `0.004278916` |
| normalized MAE | `0.47836828` |
| correlation | `0.92553074` |

`eta = -div xi` statistics, empirical vs AR(1) mechanism:

| statistic | empirical | AR(1) mechanism |
|---|---:|---:|
| rms | `0.0061999701` | `0.0052604857` |
| std | `0.0061999701` | `0.0052604857` |
| retained eta power fraction | `0.80018797` | `0.99889649` |
| dominant mode `(x,theta)` | `(1, 0)` | `(2, 1)` |
| lag-1 autocorrelation | `-0.26408635` | `-0.30975881` |
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
| selected modes | empirical `eta` power ranking, keep 80% |
| retained Fourier current modes | `222` |
| mean abs(`alpha_k`) | `0.36943095` |
| mean `sigma_k` | `15.251856` |
| mean modal correlation time | `2.0084527` |

Final density model with seeded AR(1) `xi`:

| metric | value |
|---|---:|
| R2 vs `partial_t rho` | `0.91214182` |
| seed ensemble mean R2 | `0.90735931` |
| seed ensemble median R2 | `0.90865531` |
| MAE vs `partial_t rho` | `0.0031173702` |
| normalized MAE | `0.3485114` |
| correlation | `0.95788267` |

`eta = -div xi` statistics, empirical vs AR(1) mechanism:

| statistic | empirical | AR(1) mechanism |
|---|---:|---:|
| rms | `0.0045611717` | `0.0043342142` |
| std | `0.0045611717` | `0.0043342142` |
| retained eta power fraction | `0.80080879` | `0.99923894` |
| dominant mode `(x,theta)` | `(1, 2)` | `(2, -4)` |
| lag-1 autocorrelation | `-0.19693555` | `-0.094956652` |
| correlation time | `0` | `0` |

Note: J_EOM = J_active + J_pair + J_wall; full residual identity reconstructs J_m

No fitted `a_i`; EOM terms use fixed microscopic coefficients.

### Model 3: J_fit without force_density residual split + 80% η-power AR(1)

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
| R² vs `∂_tρ` | `0.7914742` | `0.73861094` |
| MAE vs `∂_tρ` | `0.0047906408` | `0.0054857519` |
| normalized MAE | `0.53557737` | `0.61328843` |
| correlation | `0.90737905` | `0.88616915` |

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

Adaptive Fourier stochastic mechanism:

| quantity | value |
|---|---:|
| selected modes | empirical `eta` power ranking, keep 80% |
| retained Fourier current modes | `324` |
| mean abs(`alpha_k`) | `0.42120624` |
| mean `sigma_k` | `12.005221` |
| mean modal correlation time | `2.3131209` |

Final density model with seeded AR(1) `xi`:

| metric | value |
|---|---:|
| R2 vs `partial_t rho` | `0.7914742` |
| seed ensemble mean R2 | `0.78512208` |
| seed ensemble median R2 | `0.78680359` |
| MAE vs `partial_t rho` | `0.0047906408` |
| normalized MAE | `0.53557737` |
| correlation | `0.90737905` |

`eta = -div xi` statistics, empirical vs AR(1) mechanism:

| statistic | empirical | AR(1) mechanism |
|---|---:|---:|
| rms | `0.0070611948` | `0.0058863525` |
| std | `0.0070611948` | `0.0058863525` |
| retained eta power fraction | `0.80156952` | `1` |
| dominant mode `(x,theta)` | `(1, 2)` | `(1, 2)` |
| lag-1 autocorrelation | `-0.13607439` | `-0.084998415` |
| correlation time | `0` | `0` |

Adaptive Fourier source-residual stochastic mechanism:

| quantity | value |
|---|---:|
| selected modes | empirical `deltaS` power ranking, keep 80% |
| retained Fourier source modes | `106` |
| mean abs(`alpha_k`) | `0.47834576` |
| mean `sigma_k` | `2.7681675` |
| mean modal correlation time | `2.7121532` |
| empirical `deltaS` rms | `0.0076668474` |
| AR(1) `zeta` rms | `0.0056146217` |

Final density model with seeded AR(1) `xi` and `zeta`:

| metric | value |
|---|---:|
| R2 vs `partial_t rho` | `0.56430574` |
| seed ensemble mean R2 | `0.55540265` |
| seed ensemble median R2 | `0.55582974` |
| MAE vs `partial_t rho` | `0.0071501662` |
| normalized MAE | `0.79936431` |
| correlation | `0.77725763` |

Note: force_density and D force_density omitted from J_fit before residual split; final model uses seeded adaptive ξ_80 selected from empirical η power

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