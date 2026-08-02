

---

# A. Discrete characterization of one band

## Band area

\[
A_k(t)
=
\Delta x\,\Delta s
\sum_{i,j}B^{(k)}_{ij}(t).
\]

## Mean width

For a completely wrapped band,

\[
\bar w_k(t)
=
\frac{A_k(t)}{L_s}
=
\frac{1}{N_s}\sum_jw_{kj}(t).
\]

The area-based expression is generally more robust to individual noisy columns.

## Mean axial position

Use the average centerline:

\[
X_k(t)
=
\frac{1}{N_s}
\sum_{j=0}^{N_s-1}h_{kj}(t).
\]

This weights every circumferential position equally. The ordinary area centroid,

\[
X_k^{\rm area}
=
\frac{\sum_{ij}x_i B^{(k)}_{ij}}
{\sum_{ij}B^{(k)}_{ij}},
\]

may also be recorded, but it is biased toward locally wider portions of the band.

If \(x\) is periodic, unwrap \(X_k(t)\) in time before calculating displacements.

## Width nonuniformity

\[
\sigma_{w,k}^2(t)
=
\frac{1}{N_s}
\sum_j
\left[w_{kj}(t)-\bar w_k(t)\right]^2.
\]

This distinguishes a nearly uniform band from one that opens and closes locally.

## Centerline roughness

\[
\sigma_{h,k}^2(t)
=
\frac{1}{N_s}
\sum_j
\left[h_{kj}(t)-X_k(t)\right]^2.
\]

## Discrete interface length

Use periodic indexing \(j+1\equiv0\) at \(j=N_s-1\):

\[
P_k=
\sum_j
\sqrt{
\Delta s^2+
\left(x^-_{k,j+1}-x^-_{kj}\right)^2
}
+
\sum_j
\sqrt{
\Delta s^2+
\left(x^+_{k,j+1}-x^+_{kj}\right)^2
}.
\]

For a straight uniform band,

\[
P_k\approx2L_s.
\]

Obliqueness, bending, and roughness increase \(P_k\).

---

## Circumferential shape modes

Apply a discrete Fourier transform to the centerline:

\[
\widetilde h_{k,n}
=
\frac{1}{N_s}
\sum_{j=0}^{N_s-1}
\left[h_{kj}-X_k\right]
e^{-2\pi i n j/N_s}.
\]

Define the mode amplitude

\[
H_{k,n}=2\left|\widetilde h_{k,n}\right|.
\]

Similarly, width modes are

\[
\widetilde w_{k,n}
=
\frac{1}{N_s}
\sum_j
\left[w_{kj}-\bar w_k\right]
e^{-2\pi i n j/N_s}.
\]

Interpretation:

- \(H_{k,1}\): largest-scale displacement or oblique/trapezoidal shape;
- \(H_{k,2}\): two-lobed deformation;
- higher \(n\): smaller-scale interface roughness;
- \(|\widetilde w_{k,n}|\): local opening and closing around the circumference.

The allowed circumferential wave numbers are

\[
q_n=\frac{2\pi n}{L_s}=\frac{n}{R_s}.
\]

This gives a direct connection between the observed band modes and cylinder radius.

---

# B. Tracking bands through time

Let frames occur at

\[
t_m=m\Delta t.
\]

For band \(k\) at frame \(m\) and band \(\ell\) at frame \(m+1\), calculate their Jaccard overlap:

\[
O_{k\ell}
=
\frac{
\sum_{ij}
B^{(k)}_{ij}(t_m)
B^{(\ell)}_{ij}(t_{m+1})
}{
\sum_{ij}
\max\left[
B^{(k)}_{ij}(t_m),
B^{(\ell)}_{ij}(t_{m+1})
\right]
}.
\]

A possible matching cost is

\[
C_{k\ell}
=
1-O_{k\ell}
+
\alpha
\frac{|X_k-X_\ell|}{L_x}
+
\beta
\frac{|A_k-A_\ell|}{A_k+A_\ell}.
\]

Minimize the total cost using one-to-one assignment.

Classify exceptions separately:

- one old band overlaps two new bands: splitting;
- two old bands overlap one new band: merging;
- unmatched new band: birth;
- unmatched old band: disappearance.

For stable bands, require persistence for at least several frames before including them in the stochastic analysis.

---

# C. Discrete dynamics

For any measured band quantity

\[
z_k\in
\left\{
X_k,A_k,\bar w_k,P_k,H_{k,1},H_{k,2},\ldots
\right\},
\]

define the increment

\[
\Delta z_k(t_m)
=
z_k(t_{m+1})-z_k(t_m).
\]

Group observations into bins according to the current value \(z_k(t_m)\). The discrete conditional drift is

\[
F_z(z_b)
=
\frac{1}{\Delta t}
\left\langle
\Delta z_k
\mid z_k(t_m)\in z_b
\right\rangle.
\]

The conditional diffusion is

\[
D_z(z_b)
=
\frac{1}{2\Delta t}
\left\langle
\left[
\Delta z_k-F_z(z_b)\Delta t
\right]^2
\mid z_k(t_m)\in z_b
\right\rangle.
\]

Repeat this using several sampling intervals \(\Delta t\). The inferred drift and diffusion should remain reasonably stable over a range of \(\Delta t\).

The bubble study similarly obtained reduced stochastic dynamics by measuring conditional first and second moments of area changes. It found perimeter-dependent area fluctuations for compact bubbles; for your wrapped bands, that dependence should be tested rather than assumed. 

---

## Position dynamics

Test whether

\[
F_X(X)\approx v
\]

for systematic translation,

\[
F_X(X)\approx0
\]

for free diffusion, or

\[
F_X(X)\approx-\kappa_X(X-X_0)
\]

for fluctuations around a preferred position.

Also calculate

\[
\operatorname{MSD}_X(\tau)
=
\left\langle
[X_k(t+\tau)-X_k(t)]^2
\right\rangle.
\]

---

## Width or area dynamics

For a stable band, test

\[
F_A(A)\approx-\kappa_A(A-A_0)
\]

or

\[
F_{\bar w}(\bar w)
\approx
-\kappa_w(\bar w-w_0).
\]

Then examine whether the area diffusion is:

\[
D_A\approx\text{constant},
\]

or instead

\[
D_A(P)\propto P.
\]

Because a wrapped band has approximately two interfaces of length \(L_s\),

\[
P\approx2L_s
\]

over much of its evolution. Therefore, perimeter-dependent fluctuations may appear approximately as constant additive area noise.

The noise here is an **effective stochastic description** of unresolved active-particle dynamics, not thermal noise.

---

## Shape-mode dynamics

For each complex centerline mode, estimate

\[
\frac{
\widetilde h_{k,n}(t_{m+1})
-
\widetilde h_{k,n}(t_m)
}{\Delta t}.
\]

Test the linear relaxation model

\[
\frac{\Delta\widetilde h_{k,n}}{\Delta t}
=
-\lambda_n\widetilde h_{k,n}
+\eta_{k,n}.
\]

Then fit the relaxation rates against

\[
\lambda_n
=
\nu_hq_n^2+K_hq_n^4,
\qquad
q_n=\frac{n}{R_s}.
\]

This is the most direct test of how the cylindrical geometry controls band deformation and relaxation.

---

## Coupling between multiple bands

Sort bands by their axial positions and define their separations:

\[
\ell_k(t)=X_{k+1}(t)-X_k(t).
\]

To examine one band closing while another opens, calculate

\[
C_{k\ell}^{A}(\tau)
=
\left\langle
\delta A_k(t)
\delta A_\ell(t+\tau)
\right\rangle,
\]

where

\[
\delta A_k=A_k-\langle A_k\rangle.
\]

A negative correlation indicates that one band tends to gain dilute area while another loses it. Also track

\[
A_{\rm total}(t)=\sum_kA_k(t).
\]

Nearly constant \(A_{\rm total}\) together with anticorrelated \(A_k\) would support redistribution of dilute material between bands rather than independent creation and destruction.

---

## Minimum output table

For every detected band and every frame, store

\[
\boxed{
\{
t,\,
\text{band ID},\,
X,\,
A,\,
\bar w,\,
\sigma_w,\,
\sigma_h,\,
P,\,
H_1,\,
H_2,\ldots
\}
}
\]

plus event labels for birth, disappearance, merging, and splitting.

That table is sufficient for the initial geometric, positional, and stochastic characterization without modeling individual particles inside the bands.
