Once again, account for x being periodic so you will to unwrap it when storing axial componets or anything that depends on x, which is a lot. This isn't covered below. If it turns out a to be aproplem to distinguish whether one end is the max or min based on the periodic image, then handle it in the earlier analysis stage to identify the bands themselves. You can assume a band doesn't travel more than one box length in one frame.

The previous analysis delt with the x- and x+ storage, so you will need to make sure those are unwrapped. Either way a lot of the things we will be doing is based on relative motion of these bands later on, so it doesn't depend on the exact position of the each band.


# Discrete characterization of one band

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

Here are the outputs:
 - A safetensor containing each of these fields
 - A plot of each of the time based fields e.g field(t) netted over all bands, so the net (not the mean), so a plot of that versus time.
