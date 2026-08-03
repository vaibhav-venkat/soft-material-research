Use everythign in terms of the simulation time based ont he timestep, trajectory_write_frame step (which does vary), and the total steps. Let this be tau. This will be used for plotting, however regular analyze on the discrete will be computed with the difference in frames `delta t` which can be mapped back to the float `tau`, and should be mapped as such.


For band \(k\) at frame \(m\) and band \(\ell\) at frame \(m+1\), calculate their Jaccard overlap using Scipy
\]


Minimize the total cost using one-to-one assignment.

Classify exceptions separately:

- one old band overlaps two new bands: splitting;
- two old bands overlap one new band: merging;
- unmatched new band: birth;
- unmatched old band: disappearance.

For stable bands, require persistence for 5 frames before including them in the stochastic analysis.

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
Plot these values for different `delta t` with starting with just 1 frame, up to 10 frames:

---

## Position dynamics

Plot `F_X(X)`

\[
\operatorname{MSD}_X(\tau)
=
\left\langle
[X_k(t+tau)-X_k(t)]^2
\right\rangle.
\]

and plot that on the same plot

---

## Width or area dynamics

For all stable band, plot:

\[
F_A(A)
\]

and `F_w_bar(w_bar)`

Then plot `D_A`

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

Nearly constant \(A_{\rm total}\) together with anticorrelated \(A_k\) would support redistribution of dilute material between bands rather than independent creation and destruction. In order to test this, plot A_total against the mean mangitude of velocity of A_k, so if that mean magntiude is not zero we know there is some distribution

---

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

With all the fields we mentioned here as well, i.e each F_X for each delta t. THis should be stored as a safetensor in the same safetensor you stored the other components. When something is passed with no `--overwrite` just plot using the safetensors.
