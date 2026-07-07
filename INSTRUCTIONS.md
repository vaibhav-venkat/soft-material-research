# Improve performance + Accuracy instructions

## Generic performance boosts
*Completely boot the non-GPU version of the coarse-graining, even as a fallback.*

First, do not use `burn-wgpu` with metal, instead use the new `burn-mlx` burn backend (will require maybe an installation) which has support for native metal, lowering the overhead. This will require changes to the device initialization.

Still keep the logic for the CUDA that checks if the system is CUDA or metal. If it isn't possible to have the same API with extremely minor changes because of the difference between `burn-cuda` and `burn-mlx` (i.e not much more than what is currently there), they just boot `burn-cuda` and stick to mlx.

Try to eliminate as much copying done within the script, so it is near zero-copy

## For the guassian stuff
We wills witch to 
particles → finite-volume cylindrical grid → TSC deposition → local normalization → density grid → optional conservative smoothing


1. Define the cylindrical grid

Use grid cells in:

[
(x,r,\theta)
]

Each cell is:

[
[x_i,x_{i+1}]
\times
[r_j,r_{j+1}]
\times
[\theta_k,\theta_{k+1}]
]

The physical volume of each cell is:

[
V_{ijk}
=======

\Delta x_i
\Delta \theta_k
\frac{r_{j+1}^2-r_j^2}{2}
]

This is important because cylindrical cells do **not** all have the same volume.

2. For each particle, find nearby TSC cells

For particle (p) at:

[
(x_p,r_p,\theta_p)
]

find the nearby TSC stencil:

```text
3 neighboring cells in x
3 neighboring cells in r
3 neighboring cells in theta
```

So each particle touches at most:

[
3 \times 3 \times 3 = 27
]

cells.

3. Compute raw TSC weights

Compute separate 1D TSC weights:

[
w_x(i)
]

[
w_r(j)
]

[
w_\theta(k)
]

Then combine them:

[
W_{p\to ijk}
============

w_x(i)w_r(j)w_\theta(k)
]

Only the local (3\times3\times3) cells get nonzero weight.

---

## 4. Remove invalid cells

Some cells may be outside the physical domain, for example outside the shell.

Only keep valid cells:

```text
valid_cells = cells inside the physical domain
```

So now you have:

[
W_{p\to ijk}
]

only for valid cells.

Near boundaries, the raw weights may no longer sum to one:

[
\sum_{\text{valid cells}} W_{p\to ijk} < 1
]

This is the same type of problem you had with the cutoff Gaussian.

5. Locally normalize the particle weights

Normalize the valid weights for each particle:

[
\tilde W_{p\to ijk}
===================

\frac{
W_{p\to ijk}
}{
\sum_{\text{valid cells}} W_{p\to ijk}
}
]

[
\sum_{\text{valid cells}}
\tilde W_{p\to ijk}
===================

1
]

6. Deposit particle mass

For density only, each particle contributes one unit of mass/number.

For every valid cell in the particle stencil:

[
m_{ijk}
\mathrel{+}=
\tilde W_{p\to ijk}
]

After this step:

[
\sum_{ijk} m_{ijk} = N
]

where (N) is the number of particles.

7. Convert mass to density

Density is mass per cell volume:

[
\rho_{ijk}
==========

\frac{m_{ijk}}{V_{ijk}}
]

---

# Optional conservative smoothing

TSC gives a conservative field, but it may be less smooth than your Gaussian field.

Instead of smoothing particle-by-particle, smooth the gridded mass field:

```text
mass → smoothed_mass → density
```

Do **not** smooth density directly unless the grid cells all have equal volume.

Because your grid is cylindrical, smooth:

[
m_{ijk}
]

not:

[
\rho_{ijk}
]

Then convert back:

[
\rho^{\text{smooth}}_{ijk}
==========================

\frac{
m^{\text{smooth}}*{ijk}
}{
V*{ijk}
}
]

Conservative smoothing method

For each source cell (a), redistribute its mass to nearby target cells (b) using a smoothing kernel:

[
m_b^{\text{smooth}}
\mathrel{+}=
m_a
\frac{
G_{a\to b}
}{
\sum_{\text{valid } b} G_{a\to b}
}
]

This is the same idea as per-particle normalization, but applied to grid cells.

The source cell’s mass is spread to nearby cells, but the spread weights are normalized so that:

[
\sum_{\text{valid } b}
\frac{
G_{a\to b}
}{
\sum_{\text{valid } b} G_{a\to b}
}
=

1
]

Therefore, each source cell conserves its mass during smoothing.

So:

[
\sum_{ijk} m^{\text{smooth}}_{ijk}
==================================

# \sum_{ijk} m_{ijk}

N
]

Then:

[
\rho^{\text{smooth}}_{ijk}
==========================

\frac{
m^{\text{smooth}}*{ijk}
}{
V*{ijk}
}
]

and:

[
\sum_{ijk}
\rho^{\text{smooth}}*{ijk}V*{ijk}
=================================

N
]

## Simple smoothing stencil example

A simple 1D smoothing kernel is:

[
[1,2,1]/4
]

In 3D, you can apply this separably in (x), (r), and (\theta). Use this for onw


For tensors, do **not** smooth or average tensor values alone.

For a particle tensor (T_p), deposit a density-weighted numerator:

[
M_{ijk}
\mathrel{+}=
\tilde W_{p\to ijk} T_p
]

while also depositing mass:

[
m_{ijk}
\mathrel{+}=
\tilde W_{p\to ijk}
]

Then the cell-average tensor is:

[
\langle T\rangle_{ijk}
======================

\frac{M_{ijk}}{m_{ijk}}
]

Smooth both (M) and (m) conservatively:

[
M \to M^{\text{smooth}}
]

[
m \to m^{\text{smooth}}
]

Then divide:

[
\langle T\rangle^{\text{smooth}}_{ijk}
======================================

\frac{
M^{\text{smooth}}*{ijk}
}{
m^{\text{smooth}}*{ijk}
}
]
```
