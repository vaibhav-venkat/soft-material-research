
## Use a surface-excess density

\[
A_s = C L_x = 2\pi R L_x
\]

be the cylindrical surface area and

\[
V=L_xA_{\rm cross}
\]

the total accessible 3D volume.

Use the following dimensions:

\[
\rho_\gamma:\quad [N/L^3],
\]

\[
\rho_{\rm surface},\rho_\alpha,\rho_\beta:\quad [N/L^2],
\]

\[
x_\alpha,x_\beta:\quad \text{dimensionless area fractions}.
\]

Then the desired decomposition is

\[
\boxed{
N
=
V\rho_\gamma
+
A_s\left(
x_\alpha\rho_\alpha+x_\beta\rho_\beta
\right)
}
\]

with

\[
\boxed{x_\beta=1-x_\alpha.}
\]

The important point is that \(\rho_\alpha\) and \(\rho_\beta\) must be **surface-excess densities**, not 3D densities and not raw counts in an arbitrarily thick shell.

---

## 1. Calculate the 3D bulk density \(\rho_\gamma\)

First calculate the radial density profile away from the cylindrical wall:

\[
\rho(r)
=
\frac{\left\langle N(r,r+\Delta r)\right\rangle}
{V(r,r+\Delta r)}.
\]

Find the radial region where \(\rho(r)\) reaches a bulk plateau, sufficiently far from the wall-associated layer. Define

\[
\boxed{
\rho_\gamma
=
\frac{N_{\rm plateau}}{V_{\rm plateau}}
}
\]

using only that plateau region.

Do not calculate \(\rho_\gamma\) from all particles that are not assigned to the surface. That would make \(\rho_\gamma\) depend on the arbitrary shell cutoff and potentially include the depleted or accumulated boundary region.

If the bulk is not spatially uniform along \(x\), calculate at least

\[
\rho_\gamma(x,t)
\]

from the central bulk region at each axial position. A single global \(\rho_\gamma(t)\) is appropriate only after verifying that the bulk is well mixed.

---

## 2. Calculate the local surface density field

Divide the cylinder into equal-area bins indexed by \(i\):

\[
A_i=R\Delta\theta\Delta x.
\]

For each surface bin, include a radial layer extending far enough from the wall to capture the complete wall-associated film. Let its volume be \(V_i\) and its particle count be \(N_i\).

Define the local surface-excess density

\[
\boxed{
\rho_{s,i}
=
\frac{N_i-\rho_\gamma V_i}{A_i}.
}
\]

Interpretation:

- \(N_i\) is the number actually observed near the wall;
- \(\rho_\gamma V_i\) is the number that would be present there if the bulk density continued uniformly through the surface layer;
- the difference is the excess associated with the surface film.

Equivalently,

\[
\rho_{s,i}
=
\int_{\rm wall}^{\rm bulk}
\left[\rho_i(r)-\rho_\gamma\right]
\frac{dV}{A_i}.
\]

This definition is useful because increasing the radial cutoff beyond the adsorption layer should not appreciably change \(\rho_{s,i}\): the extra particles are canceled by the subtracted bulk background.

That cutoff-independence should be explicitly tested.

---

## 3. Calculate the mean surface density

The mean surface-excess density is

\[
\boxed{
\rho_{\rm surface}
=
\frac{1}{A_s}\sum_i A_i\rho_{s,i}.
}
\]

For equal-area bins,

\[
\rho_{\rm surface}
=
\frac{1}{N_{\rm bins}}\sum_i\rho_{s,i}.
\]

Globally, this is equivalent to

\[
\boxed{
\rho_{\rm surface}
=
\frac{N-\rho_\gamma V}{A_s}.
}
\]

Therefore,

\[
N=V\rho_\gamma+A_s\rho_{\rm surface}.
\]

This provides an immediate consistency check between the local bin calculation and the global particle count.

---

## 4. Determine \(\rho_\alpha\) and \(\rho_\beta\)

Construct the area-weighted distribution

\[
P(\rho_s)
=
\frac{1}{A_s}
\sum_i A_i\,
\delta(\rho_s-\rho_{s,i}).
\]

The paper follows this general logic using the distribution of local packing fractions: the two populations identify the dilute and dense coexistence densities, while changing the overall density primarily changes the amount of each phase along a tie line. 

Do not use Jenks to obtain the phase densities. Instead, fit the distribution using a probabilistic mixture:

\[
P(\rho_s)
=
x_\beta p_\beta(\rho_s)
+
x_\alpha p_\alpha(\rho_s).
\]

If interfacial bins produce a broad bridge between the peaks, use

\[
P(\rho_s)
=
x_\beta p_\beta
+
x_Ip_I
+
x_\alpha p_\alpha,
\]

where

\[
x_\alpha+x_\beta+x_I=1.
\]

### Peak versus phase density

For mass conservation, define \(\rho_\alpha\) and \(\rho_\beta\) as the **means of the fitted phase components**:

\[
\boxed{
\rho_\alpha
=
\mathbb{E}[\rho_s\mid\alpha],
\qquad
\rho_\beta
=
\mathbb{E}[\rho_s\mid\beta].
}
\]

The histogram mode can also be reported as the visual peak location, but the mode should not be used in the exact lever rule when the distribution is skewed.

Your current problem likely arises because the “dilute mean” is the mean of every bin below the Jenks boundary. That group includes dilute bins, interfacial bins, and potentially bulk-contaminated bins, so its mean is not the dilute coexistence density.

---

## 5. Calculate \(x_\alpha\) and \(x_\beta\)

For a two-phase surface,

\[
\rho_{\rm surface}
=
x_\alpha\rho_\alpha
+
x_\beta\rho_\beta,
\]

with

\[
x_\alpha+x_\beta=1.
\]

Therefore,

\[
\boxed{
x_\alpha
=
\frac{
\rho_{\rm surface}-\rho_\beta
}{
\rho_\alpha-\rho_\beta
}
}
\]

and

\[
\boxed{
x_\beta
=
\frac{
\rho_\alpha-\rho_{\rm surface}
}{
\rho_\alpha-\rho_\beta
}
=
1-x_\alpha.
}
\]

These are **surface-area fractions**:

\[
x_\alpha=\frac{A_\alpha}{A_s},
\qquad
x_\beta=\frac{A_\beta}{A_s}.
\]

They are not fractions of particles.

The full particle balance is then

\[
\boxed{
N
=
L_xA_{\rm cross}\rho_\gamma
+
2\pi RL_x
\left[
x_\alpha\rho_\alpha
+
(1-x_\alpha)\rho_\beta
\right].
}
\]

Solving directly for \(x_\alpha\),

\[
\boxed{
x_\alpha
=
\frac{
\dfrac{N-L_xA_{\rm cross}\rho_\gamma}{2\pi RL_x}
-\rho_\beta
}{
\rho_\alpha-\rho_\beta
}.
}
\]

---

## Area fractions versus particle fractions

Your phrase “fraction of alpha particles divided by surface area” describes a surface number density, not a fraction.

There are three different quantities:

### Area fraction

\[
x_\alpha^A=\frac{A_\alpha}{A_s}.
\]

This satisfies

\[
x_\beta^A=1-x_\alpha^A.
\]

### Alpha contribution per total surface area

\[
n_\alpha^s
=
\frac{N_\alpha}{A_s}
=
x_\alpha^A\rho_\alpha.
\]

Similarly,

\[
n_\beta^s=x_\beta^A\rho_\beta.
\]

These do not sum to one. Instead,

\[
n_\alpha^s+n_\beta^s=\rho_{\rm surface}.
\]

### Fraction of surface particles in alpha

\[
y_\alpha
=
\frac{N_\alpha}{N_\alpha+N_\beta}
=
\frac{x_\alpha^A\rho_\alpha}
{\rho_{\rm surface}}.
\]

Likewise,

\[
y_\beta
=
\frac{x_\beta^A\rho_\beta}
{\rho_{\rm surface}},
\]

and

\[
y_\alpha+y_\beta=1.
\]

Therefore, the \(x_\alpha\) that appears in your weighted-density equation must be the **area fraction**, not the particle fraction.

---

## Recommended practical estimator

After fitting the local-density distribution, assign each bin posterior probabilities

\[
P_{i\alpha},\qquad P_{i\beta}
\]

rather than a hard Jenks label. Then calculate

\[
x_\alpha
=
\frac{\sum_iA_iP_{i\alpha}}{A_s},
\]

\[
x_\beta
=
\frac{\sum_iA_iP_{i\beta}}{A_s}.
\]

The corresponding phase densities are

\[
\rho_\alpha
=
\frac{
\sum_iA_iP_{i\alpha}\rho_{s,i}
}{
\sum_iA_iP_{i\alpha}
},
\]

\[
\rho_\beta
=
\frac{
\sum_iA_iP_{i\beta}\rho_{s,i}
}{
\sum_iA_iP_{i\beta}
}.
\]

For a two-component model, these definitions automatically satisfy

\[
\rho_{\rm surface}
=
x_\alpha\rho_\alpha+x_\beta\rho_\beta
\]

up to numerical precision.

Finally report the mass-balance residual

\[
\boxed{
\epsilon_N
=
\frac{
N-
\left[
V\rho_\gamma+
A_s(x_\alpha\rho_\alpha+x_\beta\rho_\beta)
\right]
}{N}.
}
\]

A persistent nonzero residual indicates one of the following:

- the bulk density is not uniform;
- the radial surface layer is too thin;
- interfacial bins constitute a significant third component;
- the two-phase distribution model is inadequate;
- the bulk and surface densities are being defined using inconsistent reference volumes.

The core recommended definitions are therefore

\[
\boxed{
\rho_\gamma=\text{3D plateau density},
}
\]

\[
\boxed{
\rho_{s,i}
=
\frac{N_i-\rho_\gamma V_i}{A_i},
}
\]

\[
\boxed{
\rho_{\rm surface}
=
\frac{N-\rho_\gamma V}{A_s},
}
\]

\[
\boxed{
\rho_\alpha,\rho_\beta
=
\text{fitted component means of }P(\rho_s),
}
\]

\[
\boxed{
x_\alpha
=
\frac{\rho_{\rm surface}-\rho_\beta}
{\rho_\alpha-\rho_\beta},
\qquad
x_\beta=1-x_\alpha.
}
\]
