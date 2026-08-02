You will implement everything in the hexatic/band_analysis folder, using the safetensor as an input. 
I.e i do an `--input-dir case1/safetensor_output/case1` which has a manifest output and everythgin, and is a `big-lx` case so read that folder.

# Discrete identification and analysis of dilute bands

First, Work only with particles in the cylindrical shell, i.e `R - D - eps < R < R + eps` or something which allows a buffer (i.e eps doesnt need to be like the acutal np.finfo(dtype).eps because that may be way to small). 
Before continuing assert in the code that there is no particle with R greater than the R + eps.

Then, use the unwrapped coordinates:

\[
x_p,\qquad s_p=R_s\theta_p\pmod{L_s},
\qquad L_s=2\pi R_s or C,
\]
keep in mind both x and s are periodic, the plan im about to give you may only assume one is so adapt it IF NECESSARY.

where \(R_s\) is one fixed reference radius, such as the midpoint of the shell. Do **not** use each particle’s individual \(r_p\theta_p\), because then particles at different radii would have different effective circumferences.

## 1. Construct the discrete shell-density field

Divide the unwrapped surface into
`N_x \times N_s` bins with dimensions

\[
\Delta x=\frac{L_x}{N_x},
\qquad
\Delta s=\frac{L_s}{N_s}.
\]

Let

`n_ij(t)`

be the number of shell particles in axial bin \(i\) and circumferential bin \(j\). The projected surface density is
`\rho_{ij}(t)=\frac{n_{ij}(t)}{\Delta x\,\Delta s}.`

The circumferential index is periodic:

\[
j+N_s\equiv j.
\]

And same for the axial `i + N_x \equiv i`
Smooth the density weakly, for example with a Gaussian or beta kernel (whichever is best) whose width is approximately one bin.

The kernel must also wrap periodically in \(j\) and `i`.

---

## 2. Convert density into a dilute mask

Choose one fixed dilute–dense threshold \(\rho_c\) from the steady-state density distribution, preferably from the valley or crossing between the dilute and dense populations. I will choose this, so for now choose an arbitarry rho_c like 0.5 while I get the actual. You will plot the density distribution however as well in 
this case, and mark the valley if possible, i'll just double check.

Define

\[
M_{ij}(t)=
\begin{cases}
1, & \bar\rho_{ij}(t)<\rho_c,\\
0, & \bar\rho_{ij}(t)\ge \rho_c.
\end{cases}
\]

Thus, \(M_{ij}=1\) represents a dilute cell.

Use the same \(\rho_c\) for every frame. A frame-dependent threshold can generate artificial changes in band area and position.

Very small isolated components may be removed using a minimum area cutoff, but avoid aggressive dilation or closing because that can artificially turn an almost-wrapped patch into a wrapped band.

## 3. Group dilute cells into connected components

Use 8-neighbor connectivity. Cell \((i,j)\) is connected to

\[
(i\pm1,j),\qquad
(i,j\pm1),\qquad
(i\pm1,j\pm1).
\]

The circumferential index is periodic. However, periodic connectivity alone does not determine whether a component actually winds around the cylinder.

### Discrete winding test

Create a three-copy mask:

\[
T_{iJ}=M_{i,\;J\bmod N_s},
\qquad
J=0,\ldots,3N_s-1.
\]

This produces a left copy, central copy, and right copy

Label connected components in \(T\) using ordinary, nonperiodic connectivity in the enlarged \(J\) direction.

For a dilute cell \((i,j)\) in the original mask, its copies are located at

\[
(i,j),\qquad
(i,j+N_s),\qquad
(i,j+2N_s).
\]

A component is classified as a circumferentially wrapped band when, for at least one cell,

\[
\operatorname{label}(i,j+N_s)
=
\operatorname{label}(i,j+2N_s).
\]

That equality means there is a continuous dilute path connecting the central copy of the component to its copy one complete circumference away.

A localized patch produces separate repeated components and fails this test.

**IMPORTANT**: Do the same thing for `x`, as the axial thing is also periodic. Like I warned in the beginniing. 


## 4. Extract the two band interfaces

For band \(k\), define

\[
B^{(k)}_{ij}(t)=
\begin{cases}
1,&\text{cell }(i,j)\text{ belongs to band }k,\\
0,&\text{otherwise}.
\end{cases}
\]

For each circumferential column \(j\), find

\[
i^-_{kj}=\min\{i:B^{(k)}_{ij}=1\},
\qquad
i^+_{kj}=\max\{i:B^{(k)}_{ij}=1\}.
\]

If \(x_i\) denotes the center of axial cell \(i\), define the interface locations using the cell edges:

\[
x^-_{kj}=x_{i^-_{kj}}-\frac{\Delta x}{2},
\]

\[
x^+_{kj}=x_{i^+_{kj}}+\frac{\Delta x}{2}.
\]

The discrete centerline and local width are then

\[
h_{kj}
=
\frac{x^+_{kj}+x^-_{kj}}{2},
\]

\[
w_{kj}
=
x^+_{kj}-x^-_{kj}
=
\left(i^+_{kj}-i^-_{kj}+1\right)\Delta x.
\]

If one circumferential column contains multiple widely separated axial intervals belonging to the same component, flag the frame as geometrically complex. This may indicate branching, merging, a hole, or an unsuitable density threshold.


For now, inputting a safetensor should yield:
- The density distribution with the peaks and valleys marked, in terms of the arial distribution. In fact, convert this to an area fraction if possible so we can standardize it, i.e divide it by the area of a single
 particle treated as as a circle with diameter D.
- A movie of B, with different bands colored different colors, while the rest of the cylinder is just black. This should be a movie across time and make sure it is unwrapped and only for shell particles, so we can visually see whats going on.

Use scipy if possible for a lot of the operations, and Jax on the GPU with CUDA instead of Numpy. This should also load the safetensor. THe current machine hosts no safetnesor nor does it have CUDA, i'll run this on a remote machine.
