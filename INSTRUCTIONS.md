# Instructions

## Changing coarse-graining
Right now, we use boundary renormalization for all directions (x, r, theta) within the coarse graining. 

But instead, for x and theta which are periodic, do this instead, no renormalizing:
1. Create ghost particles to calculate and account for the guassian
That is, these are reflections like `x_i^{(m,n)} = x_i + (mL_x,nL_y),` for ` m,n\in\{-1,0,1\}`
2. Then use them in the coarse-graining kernel. For example, like
`\rho(t,x)=\sum_i\sum_{m,n} K\!\left[x-x_i(t)-(mL_x,nL_y)\right].`

Keep the direction `r` as closed and renormalized, but for `x` and `theta` use this method instead

Thats the gist for x, y, but here is how to do it with (x, r, theta)
Use a **mixed-boundary kernel**:

* periodic wrapping / ghost copies in (x) and (\theta)
* boundary renormalization only in the nonperiodic coordinate (r)

So the kernel should not be treated as “fully missing mass” in all coordinates. The periodic coordinates have no missing mass; only the finite (r)-interval truncates the kernel.

Suppose your state space is

[
x \in [0,L_x), \qquad \theta \in [0,2\pi), \qquad r \in [r_{\min},r_{\max}].
]

Let the base smoothing kernel factor as

[
K(\Delta x,\Delta \theta,\Delta r)
==================================

K_x(\Delta x)K_\theta(\Delta \theta)K_r(\Delta r),
]

with each 1D kernel normalized on the infinite line.

For a particle at ((x_i,\theta_i,r_i)), define the **periodic wrapped kernel** in (x) and (\theta):

[
K_x^{\mathrm{per}}(x-x_i)
=========================

\sum_{m\in\mathbb{Z}} K_x(x-x_i-mL_x),
]

[
K_\theta^{\mathrm{per}}(\theta-\theta_i)
========================================

\sum_{n\in\mathbb{Z}} K_\theta(\theta-\theta_i-2\pi n).
]

Then the only renormalization factor comes from the nonperiodic coordinate:

[
Z_r(r_i)
========

\int_{r_{\min}}^{r_{\max}} K_r(r-r_i),dr.
]

The mixed-boundary kernel is

[
K_{\mathrm{mixed}}
==================

K_x^{\mathrm{per}}(x-x_i)
K_\theta^{\mathrm{per}}(\theta-\theta_i)
\frac{K_r(r-r_i)}{Z_r(r_i)}.
]

Then

[
\int_0^{L_x}\int_0^{2\pi}\int_{r_{\min}}^{r_{\max}}
K_{\mathrm{mixed}},dr,d\theta,dx
================================

1.

]

That is the key point: **the full kernel still integrates to 1**, but the normalization correction is only for the truncated (r)-direction.

For a Gaussian (K_r),

[
K_r(r-r_i)
==========

\frac{1}{\sqrt{2\pi}\sigma_r}
\exp\left[-\frac{(r-r_i)^2}{2\sigma_r^2}\right],
]

the renormalization factor is

[
Z_r(r_i)
========

## \Phi\left(\frac{r_{\max}-r_i}{\sigma_r}\right)

\Phi\left(\frac{r_{\min}-r_i}{\sigma_r}\right),
]

where (\Phi) is the standard normal CDF.

So your coarse-grained density would be

[
\rho(x,\theta,r)
================

\sum_i
K_x^{\mathrm{per}}(x-x_i)
K_\theta^{\mathrm{per}}(\theta-\theta_i)
\frac{K_r(r-r_i)}{Z_r(r_i)}.
]

For a vector or orientation-weighted field (a),

[
A(x,\theta,r)
=============

\sum_i
K_x^{\mathrm{per}}(x-x_i)
K_\theta^{\mathrm{per}}(\theta-\theta_i)
\frac{K_r(r-r_i)}{Z_r(r_i)}
a_i.
]

### Implementation choice

You can implement this without explicitly making ghost particles by using wrapped distances:

[
\Delta x =
((x-x_i+L_x/2)\bmod L_x)-L_x/2,
]

[
\Delta \theta =
((\theta-\theta_i+\pi)\bmod 2\pi)-\pi.
]

Then use the ordinary Gaussian in those wrapped distances, and apply the (r)-renormalization.

This is equivalent to one layer of ghost particles when the kernel width is small compared with the periodic domain size.

### Important caveat

If (r) has a physical volume element, for example cylindrical or polar coordinates with measure

[
dV = r,dr,d\theta,dx,
]

then the normalization should respect that measure. In that case,

[
Z_r(r_i)
========

\int_{r_{\min}}^{r_{\max}} K_r(r-r_i), r,dr
]

or whatever the correct measure is for your coordinates.

Then use

[
K_{\mathrm{mixed}}
==================

K_x^{\mathrm{per}}K_\theta^{\mathrm{per}}
\frac{K_r}{Z_r}.
]

So the rule is:

**periodic directions: wrap or ghost-copy.
nonperiodic directions: truncate and renormalize.
full kernel normalization: enforce using only the missing-mass correction from the nonperiodic directions.**
