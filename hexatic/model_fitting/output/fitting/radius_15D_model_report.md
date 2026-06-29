# Density Stochastic-Flux Report: `radius_15D`

## Headline Model 3 Result

```text
partial_t rho_pred = -div(J_fit_no_force + J_sys_no_force) + S_cross + eta_AR1
```

| metric vs `partial_t rho` | value |
|---|---:|
| R² | `0.91881822` |
| seed ensemble mean R² | `0.91849559` |
| seed ensemble median R² | `0.91874161` |
| MAE | `0.0028007294` |
| normalized MAE | `0.26897127` |
| correlation | `0.96172108` |

With fitted `S_cross_pred` plus source AR(1):

```text
partial_t rho_pred = -div(J_fit_no_force + J_sys_no_force) + S_cross_pred + eta_AR1 + zeta_AR1
```

| metric vs `partial_t rho` | value |
|---|---:|
| R² | `0.1525717` |
| seed ensemble mean R² | `0.15948488` |
| seed ensemble median R² | `0.16374184` |
| normalized MAE | `0.86968105` |

## Governing Equation

```text
∂_t ρ_pred = -∇_s·[J_base + J_sys] + S_cross - ∇_s·ξ
J_res = J_m - J_base,  J_sys = mean_t J_res,  ξ = J_res - J_sys
```

Because `ξ` is defined from the residual, each full stochastic model reconstructs `J_m` by identity; the metrics below are against `∂_tρ`.

## Field Summaries

| field | mean | std | rms | min | max |
|---|---:|---:|---:|---:|---:|
| ∂_t ρ | `0.00143118` | `0.0128182` | `0.0128979` | `-0.039283` | `0.0371766` |
| S_cross | `0.00144285` | `0.0131774` | `0.0132561` | `-0.0419619` | `0.0466283` |
| S_cross_pred | `0.00144285` | `0.00416552` | `0.00440833` | `-0.0180684` | `0.0149101` |
| Y_ρ | `-1.16713e-05` | `0.0025509` | `0.00255093` | `-0.0157778` | `0.0136173` |

## Fitted S_cross Source Model

```text
S_cross_pred = c0 1 + c1 rho + c2 laplacian rho + c3 D + c4 |psi6| + c5 P_r + c6 h + c7 h rho + c8 h P_r + c9 D P_r + c10 D rho
```

| metric vs actual `S_cross` | value |
|---|---:|
| R² | `0.09992629` |
| MAE | `0.010073988` |
| normalized MAE | `0.94745702` |
| correlation | `0.3161112` |

| coefficient | value | term |
|---:|---:|---|
| `c0` | `3.33422410e+01` | 1 |
| `c1` | `1.29553560e+03` | rho |
| `c2` | `3.01511757e-01` | laplacian rho |
| `c3` | `3.33422410e+01` | D |
| `c4` | `-2.86962588e-02` | |psi6| |
| `c5` | `-6.19737243e+03` | P_r |
| `c6` | `-1.33279329e+02` | h |
| `c7` | `-5.18188774e+03` | h rho |
| `c8` | `2.47921318e+04` | h P_r |
| `c9` | `-6.19737243e+03` | D P_r |
| `c10` | `1.29553560e+03` | D rho |

## Three Full Stochastic Density Models

| model | R² AR(1)/full | R² det (no ξ) | rms(-∇·ξ) | fraction J_sys/J_res |
|---|---:|---:|---:|---:|
| J_fit residual split | `0.99966059` | `0.73205499` | `0.0066552997` | `2.6214722` |
| J_EOM residual split | `0.99966059` | `0.64058303` | `0.007702712` | `3.053387` |
| J_fit without force_density residual split + 85% η-power AR(1) | `0.91881822` | `0.66283464` | `0.0074620521` | `2.6146718` |

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
| R² vs `∂_tρ` | `0.99966059` | `0.73205499` |
| MAE vs `∂_tρ` | `8.9867475e-05` | `0.0044555008` |
| normalized MAE | `0.0086305262` | `0.42788914` |
| correlation | `0.99983093` | `0.88954564` |

Residual diagnostics:

| quantity | value |
|---|---:|
| rms(-∇·J_res) | `0.0026968366` |
| rms(-∇·J_sys) | `0.0070696823` |
| rms(-∇·ξ) | `0.0066552997` |
| fraction J_sys / J_res | `2.6214722` |

Current fit against `J_m`:

| quantity | R² |
|---|---:|
| combined components | `0.53153784` |
| x component | `0.55750644` |
| theta component | `0.4039943` |

Adaptive Fourier stochastic mechanism:

| quantity | value |
|---|---:|
| selected modes | empirical `eta` power ranking, keep 80% |
| retained Fourier current modes | `5464` |
| mean abs(`alpha_k`) | `0.93623483` |
| mean `sigma_k` | `0.21126384` |
| mean modal correlation time | `30.354104` |

Final density model with seeded AR(1) `xi`:

| metric | value |
|---|---:|
| R2 vs `partial_t rho` | `0.93127008` |
| seed ensemble mean R2 | `0.93104114` |
| seed ensemble median R2 | `0.93124053` |
| MAE vs `partial_t rho` | `0.0025768712` |
| normalized MAE | `0.24747278` |
| correlation | `0.96698654` |

`eta = -div xi` statistics, empirical vs AR(1) mechanism:

| statistic | empirical | AR(1) mechanism |
|---|---:|---:|
| rms | `0.0066552997` | `0.0062379844` |
| std | `0.006655291` | `0.0062379638` |
| retained eta power fraction | `0.80005834` | `0.90672117` |
| dominant mode `(x,theta)` | `(49, -27)` | `(49, -27)` |
| lag-1 autocorrelation | `0.6929453` | `0.74331568` |
| correlation time | `4.1601327` | `4.7000836` |

Note: J_res_fit = J_m - J_fit; J_sys_fit = mean_t J_res_fit; ξ_fit = J_res_fit - J_sys_fit

| coefficient | value | term |
|---:|---:|---|
| `a1` | `2.94470456e+00` | P |
| `a2` | `3.90656268e-02` | chirality P_perp |
| `a3` | `2.93627274e-02` | force_density |
| `a4` | `2.94470456e+00` | D P |
| `a5` | `3.90656268e-02` | D chirality P_perp |
| `a6` | `2.93627274e-02` | D force_density |
| `a7` | `-9.10395593e-02` | -grad rho |
| `a8` | `-2.84303880e-02` | -grad hexatic_order |
| `a9` | `-3.67848048e+11` | -grad D |

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
| R² vs `∂_tρ` | `0.99966059` | `0.64058303` |
| MAE vs `∂_tρ` | `8.9867475e-05` | `0.0051855135` |
| normalized MAE | `0.0086305262` | `0.49799675` |
| correlation | `0.99983093` | `0.85971964` |

Residual diagnostics:

| quantity | value |
|---|---:|
| rms(-∇·J_res) | `0.0026243663` |
| rms(-∇·J_sys) | `0.0080132061` |
| rms(-∇·ξ) | `0.007702712` |
| fraction J_sys / J_res | `3.053387` |

Adaptive Fourier stochastic mechanism:

| quantity | value |
|---|---:|
| selected modes | empirical `eta` power ranking, keep 80% |
| retained Fourier current modes | `5148` |
| mean abs(`alpha_k`) | `0.94220106` |
| mean `sigma_k` | `0.232215` |
| mean modal correlation time | `33.592787` |

Final density model with seeded AR(1) `xi`:

| metric | value |
|---|---:|
| R2 vs `partial_t rho` | `0.92061484` |
| seed ensemble mean R2 | `0.92071959` |
| seed ensemble median R2 | `0.920728` |
| MAE vs `partial_t rho` | `0.002771293` |
| normalized MAE | `0.2661443` |
| correlation | `0.96280712` |

`eta = -div xi` statistics, empirical vs AR(1) mechanism:

| statistic | empirical | AR(1) mechanism |
|---|---:|---:|
| rms | `0.007702712` | `0.0071821406` |
| std | `0.0077027059` | `0.0071821316` |
| retained eta power fraction | `0.80005307` | `0.92186356` |
| dominant mode `(x,theta)` | `(49, -25)` | `(49, -25)` |
| lag-1 autocorrelation | `0.80116975` | `0.83657165` |
| correlation time | `4.8069303` | `5.1900978` |

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
| R² vs `∂_tρ` | `0.91881822` | `0.66283464` |
| MAE vs `∂_tρ` | `0.0028007294` | `0.0051115513` |
| normalized MAE | `0.26897127` | `0.4908937` |
| correlation | `0.96172108` | `0.86763971` |

Residual diagnostics:

| quantity | value |
|---|---:|
| rms(-∇·J_res) | `0.0029250177` |
| rms(-∇·J_sys) | `0.0076479612` |
| rms(-∇·ξ) | `0.0074620521` |
| fraction J_sys / J_res | `2.6146718` |

Current fit against `J_m`:

| quantity | R² |
|---|---:|
| combined components | `0.10772019` |
| x component | `0.14777263` |
| theta component | `-0.089155403` |

Adaptive Fourier stochastic mechanism:

| quantity | value |
|---|---:|
| selected modes | empirical `eta` power ranking, keep 80% |
| retained Fourier current modes | `5048` |
| mean abs(`alpha_k`) | `0.93555486` |
| mean `sigma_k` | `0.25131419` |
| mean modal correlation time | `30.023047` |

Final density model with seeded AR(1) `xi`:

| metric | value |
|---|---:|
| R2 vs `partial_t rho` | `0.91881822` |
| seed ensemble mean R2 | `0.91849559` |
| seed ensemble median R2 | `0.91874161` |
| MAE vs `partial_t rho` | `0.0028007294` |
| normalized MAE | `0.26897127` |
| correlation | `0.96172108` |

`eta = -div xi` statistics, empirical vs AR(1) mechanism:

| statistic | empirical | AR(1) mechanism |
|---|---:|---:|
| rms | `0.0074620521` | `0.0069130801` |
| std | `0.0074620481` | `0.0069130335` |
| retained eta power fraction | `0.80002386` | `0.92605747` |
| dominant mode `(x,theta)` | `(49, -26)` | `(49, -26)` |
| lag-1 autocorrelation | `0.80337312` | `0.83649879` |
| correlation time | `4.7193573` | `5.1327042` |

Adaptive Fourier source-residual stochastic mechanism:

| quantity | value |
|---|---:|
| selected modes | empirical `deltaS` power ranking, keep 80% |
| retained Fourier source modes | `117` |
| mean abs(`alpha_k`) | `0.6619935` |
| mean `sigma_k` | `4.649516` |
| mean modal correlation time | `4.8484902` |
| empirical `deltaS` rms | `0.012501682` |
| AR(1) `zeta` rms | `0.0088575791` |

Final density model with seeded AR(1) `xi` and `zeta`:

| metric | value |
|---|---:|
| R2 vs `partial_t rho` | `0.1525717` |
| seed ensemble mean R2 | `0.15948488` |
| seed ensemble median R2 | `0.16374184` |
| MAE vs `partial_t rho` | `0.0090557677` |
| normalized MAE | `0.86968105` |
| correlation | `0.49996724` |

Note: force_density and D force_density omitted from J_fit before residual split; final model uses seeded adaptive ξ_85 selected from empirical η power

| coefficient | value | term |
|---:|---:|---|
| `a1` | `1.53731209e-01` | P |
| `a2` | `5.16140155e-02` | chirality P_perp |
| `a3` | `1.53731209e-01` | D P |
| `a4` | `5.16140155e-02` | D chirality P_perp |
| `a5` | `-3.91966422e-01` | -grad rho |
| `a6` | `3.70802118e-02` | -grad hexatic_order |
| `a7` | `-3.20138399e+11` | -grad D |
| `a8` | `3.33855038e-01` | low-k(P) |

## Saved Outputs

- Text report: `/Users/vaibhavvenkat/structures_and_functions/hexatic/model_fitting/output/fitting/radius_15D_model_report.txt`
- Markdown report: `/Users/vaibhavvenkat/structures_and_functions/hexatic/model_fitting/output/fitting/radius_15D_model_report.md`