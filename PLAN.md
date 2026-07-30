
Broad model:
## MATHEMATICAL MODEL

Kurzthaler et al. define the ISF in Eq. (6) as

\[
F(\mathbf k,\tau)
=
\left\langle
e^{-i\mathbf k\cdot\Delta\mathbf r(\tau)}
\right\rangle .
\]

It is the characteristic function of the displacement distribution. For isotropic 3D motion, directional averaging gives their Eq. (8),

\[
F(k,\tau)
=
\left\langle
\frac{\sin(k|\Delta\mathbf r|)}
{k|\Delta\mathbf r|}
\right\rangle ,
\]

and the small-\(k\) expansion, Eq. (9), is

\[
F(k,\tau)
=
1-\frac{k^2}{6}\langle\Delta r^2\rangle
+\frac{k^4}{120}\langle\Delta r^4\rangle+\cdots .
\]

Thus, the MSD is only the \(k^2\) part of the ISF. 

### COM version

Use the per-particle-equivalent displacement

\[
\mathbf X(t_0,\tau)
=
\sqrt N\left[
\mathbf R_{\rm COM}(t_0+\tau)
-\mathbf R_{\rm COM}(t_0)
\right].
\]

Then calculate

\[
F_{\rm COM}(\mathbf k,\tau)
=
\left\langle
e^{-i\mathbf k\cdot\mathbf X(t_0,\tau)}
\right\rangle_{t_0,\text{seeds}}.
\]

The \(\sqrt N\) factor is required because the raw COM displacement decreases as \(N^{-1/2}\).

For independent identical particles,

\[
F_{\rm COM}(k,\tau)
=
\left[
F_{\rm single}\left(\frac{k}{\sqrt N},\tau\right)
\right]^N,
\]

so as \(N\) grows,

\[
F_{\rm COM}(k,\tau)
\rightarrow
\exp\left[
-\frac{k^2M(\tau)}{2d}
\right],
\]

where

\[
M(\tau)=N\,\mathrm{MSD}_{\rm COM}(\tau).
\]

This is the Gaussian benchmark.

### Expected interpretation

The paper uses the persistence length

\[
\ell_p=\frac{v}{D_{\rm rot}}
\]

to scale the wavenumber. It finds:

- small \(k\ell_p\): large-scale effective diffusion,
  \[
  F\approx e^{-D_{\rm eff}k^2\tau};
  \]
- intermediate \(k\ell_p\): persistent swimming and possible damped oscillations;
- large \(k\ell_p\): short-scale translational diffusion.

For nearly deterministic persistent motion,

\[
F(k,\tau)\approx\frac{\sin(kv\tau)}{kv\tau}.
\]

These regimes are the interpretation used for Fig. 3 of the paper. 

---

## DISCRETIZATION / NUMERICAL METHOD

### Stored trajectories

```text
com_bulk      : float64[S, T, 3]
com_cylinder  : float64[S, T, 3]
surface_com   : float64[S, T, 2]    # x, R*theta_unwrapped
lags          : int64[L]
k_values      : float64[K]
```

All coordinates must be continuously unwrapped before calculating displacements.

For each lag index \(m\),

\[
\tau_m=m\Delta t,
\]

form all valid displacements:

\[
\mathbf X_{s,t_0,m}
=
\sqrt N
\left[
\mathbf R_{s,t_0+m}-\mathbf R_{s,t_0}
\right].
\]

### Bulk isotropic ISF

Use the direction-averaged 3D estimator:

\[
F_{\rm bulk}(k,\tau)
=
\left\langle
\operatorname{sinc}
\left(k|\mathbf X|\right)
\right\rangle,
\]

where here

\[
\operatorname{sinc}(z)=\frac{\sin z}{z}.
\]

This is usually cleaner than sampling many random \(\mathbf k\) directions.

### Cylinder component ISFs

Do not isotropically combine the cylinder first.

Calculate

\[
F_x(k,\tau)
=
\left\langle
e^{-ikX_x}
\right\rangle,
\]

\[
F_\theta(k,\tau)
=
\left\langle
e^{-ikX_\theta}
\right\rangle,
\qquad
X_\theta=\sqrt N\,R\Delta\theta_{\rm COM}.
\]

For symmetric displacement distributions, plot the real parts:

\[
\operatorname{Re}F_x
=
\langle\cos(kX_x)\rangle,
\]

\[
\operatorname{Re}F_\theta
=
\langle\cos(kX_\theta)\rangle.
\]

Keep the imaginary part as a diagnostic; it should be close to zero.

Choose \(k\) through the dimensionless variable

\[
q=k\ell_p.
\]

A practical initial set is

```text
q = [0.1, 0.3, 1, 3, 10]
k = q / persistence_length
```

---

## DATA + ALGORITHMS

### Main data flow

```text
[unwrapped particle positions]
            ↓
compute_COM()
            ↓
[COM trajectory : S×T×d]
            ↓
scale_displacements(sqrt(N))
            ↓
[displacements : origins×d]
            ↓
compute_ISF(k_values)
            ↓
[F : S×L×K]
            ↓
seed_average + confidence_interval
            ↓
plots
```

### Function interfaces

```text
compute_com(positions) → com

compute_isf_isotropic_3d(
    com: float64[S,T,3],
    lags: int64[L],
    k_values: float64[K],
    particle_count: int
) → F: float64[S,L,K]

compute_isf_component(
    coordinate: float64[S,T],
    lags: int64[L],
    k_values: float64[K],
    particle_count: int
) → F_complex: complex128[S,L,K]
```

### Bulk algorithm

```text
for seed s
    X ← sqrt(N) * com[s]

    for lag m
        ΔX ← X[m:] - X[:-m]
        radius ← norm(ΔX, axis=1)

        for k
            F[s,m,k] ← mean(sin(k*radius)/(k*radius))
```

Python detail:

```python
# np.sinc(z) means sin(pi*z)/(pi*z)
values = np.sinc(k * radius / np.pi)
```

### Cylinder component algorithm

```text
for seed s
    X ← sqrt(N) * coordinate[s]

    for lag m
        ΔX ← X[m:] - X[:-m]

        for k
            phase ← exp(-i*k*ΔX)
            F[s,m,k] ← mean(phase)
```

Equivalent vectorized core:

```python
delta = x[lag:] - x[:-lag]
F = np.mean(np.exp(-1j * delta[:, None] * k_values[None, :]), axis=0)
```

### Gaussian reference

From the same displacement samples,

\[
M(\tau)=\langle|\mathbf X|^2\rangle.
\]

For \(d\)-dimensional isotropic motion:

```text
M ──gaussian_isf(k,d)──▶ F_gaussian
```

\[
F_G(k,\tau)
=
\exp\left[-\frac{k^2M(\tau)}{2d}\right].
\]

For a single cylinder coordinate:

\[
F_{G,x}
=
\exp\left[-\frac{k^2\langle X_x^2\rangle}{2}\right].
\]

### Validation checks

```text
F(k, 0) ≈ 1
imaginary(F) ≈ 0
|F| ≤ 1
small-k ISF reproduces MSD
number of time origins decreases with lag
```

Check the small-\(k\) relation numerically:

\[
M_{\rm ISF}(\tau)
\approx
\frac{2d[1-F(k,\tau)]}{k^2}.
\]

It should agree with the directly calculated rescaled COM MSD for sufficiently small \(k\).

---

## IMPLEMENTATION + PERFORMANCE

For \(T\approx3000\), use the direct estimator. An FFT-based method is unnecessary initially.

```text
Time:
O(S × L × K × T)

Memory:
O(T × d + K × T)
```

Avoid storing all lag-displacement arrays simultaneously. Process one lag at a time.

```text
for lag
    construct temporary displacement array
    evaluate all k values
    discard temporary
```

### Plots to generate

#### 1. ISF versus lag time

```text
x-axis: tau / tau_r
y-axis: Re[F(k,tau)]
one curve per k*ell_p
```

Interpretation:

- monotonic exponential decay: diffusion-dominated;
- damped zero crossings: persistent directed motion;
- stronger oscillations at intermediate \(k\): expected ABP persistence;
- no oscillations in the COM but oscillations for single particles: COM Gaussianization.

#### 2. Measured versus Gaussian ISF

Plot

\[
F_{\rm measured}(k,\tau)
\]

and

\[
F_G(k,\tau)
\]

together, or plot

\[
\Delta F=F_{\rm measured}-F_G.
\]

Interpretation:

- \(\Delta F\approx0\): MSD nearly specifies the COM process;
- nonzero \(\Delta F\): higher-order displacement statistics matter.

#### 3. Single particle versus rescaled COM

Compare

\[
F_{\rm single}(k,\tau)
\]

with

\[
F_{\rm COM}(k,\tau).
\]

This is the main test of:

\[
\text{same MSD}\quad\text{but}\quad\text{different full dynamics}.
\]

#### 4. Cylinder comparison

Plot separately:

```text
bulk Fx
cylinder Fx
cylinder Ftheta
```

Interpretation:

- bulk \(F_x\approx\) cylinder \(F_x\): axial COM dynamics are bulk-like;
- \(F_\theta\neq F_x\): confinement-induced anisotropy;
- oscillations only in \(F_\theta\): angular or wall-related persistence;
- differences only at large \(k\): confinement affects short distances but not diffusion.

#### 5. ISF heatmap

```text
x-axis: tau/tau_r
y-axis: k*ell_p
value: Re[F]
```

Use a signed linear scale, not \(\log|F|\), because zero crossings and negative regions are physically informative.

### Recommended first calculation

```text
1. Compute bulk 3D isotropic F_COM(k,tau).
2. Compute cylinder Fx(k,tau).
3. Compute cylinder Ftheta(k,tau).
4. Overlay Gaussian predictions from each MSD.
5. Compare against single-particle ISFs.
6. Add seed-level confidence bands.
```

The main result to look for is:

\[
N\,\mathrm{MSD}_{\rm COM}
\approx
\mathrm{MSD}_{\rm single},
\]

but

\[
F_{\rm COM}(k,\tau)
\neq
F_{\rm single}(k,\tau)
\]

at intermediate \(k\), followed by convergence of the COM ISF toward the Gaussian prediction as \(N\) increases.


Do this for `k * l_p = [0.1, 0.3, 1, 3, 10]` and not that `l_p` is defined similar to the hoomd convention `l = U_0/D`.
so `k = [0.1, 0.3, 1, 3, 10]/l_p`


THE ISF should be separte from the other plots.
