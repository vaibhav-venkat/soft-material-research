# Density Stochastic-Flux Report: `radius_15D`

## Headline Model 3 Result

```text
partial_t rho_pred = -div(J_fit_no_force + J_sys_no_force) + S_cross + eta_AR1
```

| metric vs `partial_t rho` | value |
|---|---:|
| R² | `0.86747683` |
| seed ensemble mean R² | `0.85222159` |
| seed ensemble median R² | `0.85550309` |
| MAE | `0.0038284325` |
| normalized MAE | `0.42800575` |
| correlation | `0.93863231` |

With fitted `S_cross_pred` plus source AR(1):

```text
partial_t rho_pred = -div(J_fit_no_force + J_sys_no_force) + S_cross_pred + eta_AR1 + zeta_AR1
```

| metric vs `partial_t rho` | value |
|---|---:|
| R² | `0.65910508` |
| seed ensemble mean R² | `0.6302594` |
| seed ensemble median R² | `0.63288534` |
| normalized MAE | `0.71990168` |

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
| S_cross_pred | `0.00742292` | `0.0125658` | `0.0145945` | `-0.0215142` | `0.0602246` |
| Y_ρ | `2.64301e-07` | `0.00593726` | `0.00593726` | `-0.0514888` | `0.0214055` |

## Fitted S_cross Source Model

```text
S_cross_pred = c0 1 + c1 rho + c2 laplacian rho + c3 D + c4 |psi6| + c5 P_r + c6 h + c7 h rho + c8 h P_r + c9 D P_r + c10 D rho
```

| metric vs actual `S_cross` | value |
|---|---:|
| R² | `0.73390201` |
| MAE | `0.0058738559` |
| normalized MAE | `0.62370111` |
| correlation | `0.85668081` |

| coefficient | value | term |
|---:|---:|---|
| `c0` | `1.47543533e+03` | 1 |
| `c1` | `-8.25134640e+03` | rho |
| `c2` | `-5.70445467e+00` | laplacian rho |
| `c3` | `2.91144566e-01` | D |
| `c4` | `1.42742939e-01` | |psi6| |
| `c5` | `-7.08394856e+03` | P_r |
| `c6` | `-2.94898828e+03` | h |
| `c7` | `1.64876093e+04` | h rho |
| `c8` | `1.41704023e+04` | h P_r |
| `c9` | `3.36126279e+00` | D P_r |
| `c10` | `-2.83930637e+00` | D rho |

## Three Full Stochastic Density Models

| model | R² AR(1)/full | R² det (no ξ) | rms(-∇·ξ) | fraction J_sys/J_res |
|---|---:|---:|---:|---:|
| J_fit residual split | `0.9985239` | `0.67352204` | `0.0078738138` | `0.62365626` |
| J_EOM residual split | `0.9985239` | `0.89352542` | `0.0045611717` | `0.65719928` |
| J_fit without force_density residual split + 80% η-power AR(1) | `0.86747683` | `0.81862481` | `0.0059035387` | `0.7360491` |

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
| R² vs `∂_tρ` | `0.9985239` | `0.67352204` |
| MAE vs `∂_tρ` | `0.00020570209` | `0.0058919885` |
| normalized MAE | `0.022996795` | `0.6587043` |
| correlation | `0.99926647` | `0.86772467` |

Residual diagnostics:

| quantity | value |
|---|---:|
| rms(-∇·J_res) | `0.010072696` |
| rms(-∇·J_sys) | `0.0062819002` |
| rms(-∇·ξ) | `0.0078738138` |
| fraction J_sys / J_res | `0.62365626` |

Current fit against `J_m`:

| quantity | R² |
|---|---:|
| combined components | `0.62508864` |
| x component | `0.65164983` |
| theta component | `0.40473513` |

Adaptive Fourier stochastic mechanism:

| quantity | value |
|---|---:|
| selected modes | empirical `eta` power ranking, keep 80% |
| retained Fourier current modes | `430` |
| mean abs(`alpha_k`) | `0.47229498` |
| mean `sigma_k` | `8.0706962` |
| mean modal correlation time | `2.666128` |

Final density model with seeded AR(1) `xi`:

| metric | value |
|---|---:|
| R2 vs `partial_t rho` | `0.72661419` |
| seed ensemble mean R2 | `0.71772601` |
| seed ensemble median R2 | `0.71859449` |
| MAE vs `partial_t rho` | `0.0054101262` |
| normalized MAE | `0.60483374` |
| correlation | `0.88508382` |

`eta = -div xi` statistics, empirical vs AR(1) mechanism:

| statistic | empirical | AR(1) mechanism |
|---|---:|---:|
| rms | `0.0078738138` | `0.0063459633` |
| std | `0.0078738138` | `0.0063459633` |
| retained eta power fraction | `0.80041742` | `0.99927539` |
| dominant mode `(x,theta)` | `(3, -1)` | `(1, 0)` |
| lag-1 autocorrelation | `-0.17392068` | `-0.16597669` |
| correlation time | `0` | `0` |

Note: J_res_fit = J_m - J_fit; J_sys_fit = mean_t J_res_fit; ξ_fit = J_res_fit - J_sys_fit

| coefficient | value | term |
|---:|---:|---|
| `a1` | `2.93456187e+02` | P_density |
| `a2` | `7.61491162e+00` | chirality P_density_perp |
| `a3` | `3.94314386e-01` | force_density |
| `a4` | `-5.46130092e+01` | D P_density |
| `a5` | `-8.38933171e+00` | D chirality P_density_perp |
| `a6` | `-8.30635338e-02` | D force_density |

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
| R² vs `∂_tρ` | `0.86747683` | `0.81862481` |
| MAE vs `∂_tρ` | `0.0038284325` | `0.0045618074` |
| normalized MAE | `0.42800575` | `0.50999457` |
| correlation | `0.93863231` | `0.91715152` |

Residual diagnostics:

| quantity | value |
|---|---:|
| rms(-∇·J_res) | `0.0087210714` |
| rms(-∇·J_sys) | `0.0064191368` |
| rms(-∇·ξ) | `0.0059035387` |
| fraction J_sys / J_res | `0.7360491` |

Current fit against `J_m`:

| quantity | R² |
|---|---:|
| combined components | `0.12556041` |
| x component | `0.16027139` |
| theta component | `-0.17360909` |

Adaptive Fourier stochastic mechanism:

| quantity | value |
|---|---:|
| selected modes | empirical `eta` power ranking, keep 80% |
| retained Fourier current modes | `212` |
| mean abs(`alpha_k`) | `0.42828776` |
| mean `sigma_k` | `15.92304` |
| mean modal correlation time | `2.3586019` |

Final density model with seeded AR(1) `xi`:

| metric | value |
|---|---:|
| R2 vs `partial_t rho` | `0.86747683` |
| seed ensemble mean R2 | `0.85222159` |
| seed ensemble median R2 | `0.85550309` |
| MAE vs `partial_t rho` | `0.0038284325` |
| normalized MAE | `0.42800575` |
| correlation | `0.93863231` |

`eta = -div xi` statistics, empirical vs AR(1) mechanism:

| statistic | empirical | AR(1) mechanism |
|---|---:|---:|
| rms | `0.0059035387` | `0.0049379871` |
| std | `0.0059035387` | `0.0049379871` |
| retained eta power fraction | `0.80151045` | `1` |
| dominant mode `(x,theta)` | `(1, 1)` | `(1, 1)` |
| lag-1 autocorrelation | `-0.18454507` | `-0.29206168` |
| correlation time | `0` | `0` |

Adaptive Fourier source-residual stochastic mechanism:

| quantity | value |
|---|---:|
| selected modes | empirical `deltaS` power ranking, keep 80% |
| retained Fourier source modes | `106` |
| mean abs(`alpha_k`) | `0.45885141` |
| mean `sigma_k` | `2.703371` |
| mean modal correlation time | `2.567299` |
| empirical `deltaS` rms | `0.0075664457` |
| AR(1) `zeta` rms | `0.0054386346` |

Final density model with seeded AR(1) `xi` and `zeta`:

| metric | value |
|---|---:|
| R2 vs `partial_t rho` | `0.65910508` |
| seed ensemble mean R2 | `0.6302594` |
| seed ensemble median R2 | `0.63288534` |
| MAE vs `partial_t rho` | `0.0064393877` |
| normalized MAE | `0.71990168` |
| correlation | `0.81981833` |

Note: force_density and D force_density omitted from J_fit before residual split; final model uses seeded adaptive ξ_80 selected from empirical η power

| coefficient | value | term |
|---:|---:|---|
| `a1` | `7.80063856e+00` | P_density |
| `a2` | `7.57831772e+00` | chirality P_density_perp |
| `a3` | `3.47794302e+00` | D P_density |
| `a4` | `-5.28209709e-01` | D chirality P_density_perp |
| `a5` | `1.66496773e+01` | low-k(P_density) |

## Saved Outputs

- Text report: `/Users/vaibhavvenkat/structures_and_functions/hexatic/model_fitting/output/fitting/radius_15D_model_report.txt`
- Markdown report: `/Users/vaibhavvenkat/structures_and_functions/hexatic/model_fitting/output/fitting/radius_15D_model_report.md`