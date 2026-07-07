# Density-only process

Starting point:

```text
particles → non-renormalized Gaussian deposition → density grid
```

Suggested replacement:

```text
particles → finite-volume cylindrical grid → TSC deposition → local normalization → density grid → optional conservative smoothing
```

---

## 1. Define the cylindrical grid

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

You will store two arrays:

```text
mass[i,j,k]
density[i,j,k]
```

Initialize:

```text
mass[i,j,k] = 0
```

---

## 2. For each particle, find nearby TSC cells

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

This replaces the Gaussian loop over a potentially much larger support.

---

## 3. Compute raw TSC weights

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

---

## 5. Locally normalize the particle weights

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

Now each particle contributes exactly one particle’s worth of mass:

[
\sum_{\text{valid cells}}
\tilde W_{p\to ijk}
===================

1
]

This is the key conservation step.

---

## 6. Deposit particle mass

For density only, each particle contributes one unit of mass/number.

For every valid cell in the particle stencil:

[
m_{ijk}
\mathrel{+}=
\tilde W_{p\to ijk}
]

In pseudocode:

```python
for particle in particles:
    cells = get_3x3x3_TSC_stencil(particle)

    weights = []
    for cell in cells:
        if cell_is_valid(cell):
            w = wx[cell.i] * wr[cell.j] * wtheta[cell.k]
            weights.append((cell, w))

    norm = sum(w for cell, w in weights)

    for cell, w in weights:
        mass[cell] += w / norm
```

After this step:

[
\sum_{ijk} m_{ijk} = N
]

where (N) is the number of particles.

---

## 7. Convert mass to density

Density is mass per cell volume:

[
\rho_{ijk}
==========

\frac{m_{ijk}}{V_{ijk}}
]

In code form:

```python
density = mass / volume
```

Now the density should satisfy:

[
\sum_{ijk} \rho_{ijk}V_{ijk}
============================

N
]

This should be your main conservation check.

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

---

## Conservative smoothing method

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

---

## Simple smoothing stencil example

A simple 1D smoothing kernel is:

[
[1,2,1]/4
]

In 3D, you can apply this separably in (x), (r), and (\theta).

For stronger smoothing, use a wider stencil or repeat the smoothing step multiple times.

The important thing is that smoothing should be done conservatively:

```text
source cell mass → redistributed to valid neighboring cells → normalized weights
```

not:

```text
density value → averaged with neighbors
```

---

# Overall algorithm

```text
1. Build cylindrical grid.
2. Compute cell volumes V[i,j,k].
3. Initialize mass[i,j,k] = 0.

4. For each particle:
   a. Find the 3x3x3 TSC stencil.
   b. Compute raw TSC weights.
   c. Remove invalid cells outside the domain.
   d. Normalize the remaining weights so they sum to 1.
   e. Add normalized weights to mass[i,j,k].

5. Convert mass to density:
      density = mass / volume

6. Check conservation:
      sum(density * volume) should equal number of particles.

7. Optional:
   a. Smooth mass conservatively.
   b. Convert smoothed mass back to density:
          density_smooth = mass_smooth / volume
   c. Check conservation again.
```

---

# Main caveat for tensors

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

If smoothing is used, smooth both (M) and (m) conservatively:

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

The key caveat:

[
\boxed{
\frac{S[M]}{S[m]}
\neq
S\left[\frac{M}{m}\right]
}
]

Usually, you want:

[
\boxed{
\langle T\rangle_{\text{smooth}}
================================

\frac{
S[\rho T]
}{
S[\rho]
}
}
]

not direct smoothing of the already-averaged tensor field.


For a **uniform 1D grid**, the TSC weight is a compact quadratic function over the nearest **3 grid cells**.

Suppose a particle is at position (x_p), and the grid cell centers are at (x_i), with spacing (\Delta x).

Define the normalized distance:

[
s_i = \frac{x_p - x_i}{\Delta x}
]

Then the 1D TSC weight for cell (i) is:

[
w_i =
\begin{cases}
\frac{3}{4} - s_i^2, & |s_i| < \frac{1}{2} [6pt]
\frac{1}{2}\left(\frac{3}{2} - |s_i|\right)^2, & \frac{1}{2} \le |s_i| < \frac{3}{2} [6pt]
0, & |s_i| \ge \frac{3}{2}
\end{cases}
]

So each particle contributes only to cells whose centers satisfy:

[
|x_p - x_i| < \frac{3}{2}\Delta x
]

That means at most **3 cells in 1D**.

---

# In 3D

For cylindrical coordinates ((x,r,\theta)), compute the 1D weights separately:

[
w_x(i)
]

[
w_r(j)
]

[
w_\theta(k)
]

using the same TSC formula in each coordinate.

Then the full 3D weight is:

[
W_{p\to ijk}
============

w_x(i)w_r(j)w_\theta(k)
]

So the particle deposits to at most:

[
3 \times 3 \times 3 = 27
]

cells.

---

# Explicit form for your grid

For (x):

[
s_x(i) = \frac{x_p - x_i}{\Delta x}
]

[
w_x(i)=T(s_x(i))
]

For (r):

[
s_r(j) = \frac{r_p - r_j}{\Delta r}
]

[
w_r(j)=T(s_r(j))
]

For (\theta):

[
s_\theta(k) =
\frac{
\mathrm{wrap}(\theta_p-\theta_k)
}{
\Delta \theta
}
]

[
w_\theta(k)=T(s_\theta(k))
]

where

[
T(s)=
\begin{cases}
\frac{3}{4} - s^2, & |s| < \frac{1}{2} [6pt]
\frac{1}{2}\left(\frac{3}{2} - |s|\right)^2, & \frac{1}{2} \le |s| < \frac{3}{2} [6pt]
0, & |s| \ge \frac{3}{2}
\end{cases}
]

and (\mathrm{wrap}(\theta_p-\theta_k)) means use the periodic angular distance in ((-\pi,\pi]).

---

# Then normalize locally

After computing the raw 3D weights:

[
W_{p\to ijk}
============

w_x(i)w_r(j)w_\theta(k)
]

restrict to valid cells and normalize:

[
\tilde W_{p\to ijk}
===================

\frac{
W_{p\to ijk}
}{
\sum_{\text{valid cells}}W_{p\to ijk}
}
]

Then deposit:

[
m_{ijk}
\mathrel{+}=
\tilde W_{p\to ijk}
]

This guarantees that each particle contributes exactly one unit of mass.

---

# Caveat for nonuniform radial bins

The simple formula

[
s_r = \frac{r_p-r_j}{\Delta r}
]

assumes uniform (r)-spacing.

If you use nonuniform radial bins, for example equal-volume radial bins, TSC is less straightforward. In that case, it is often easier to define a uniform computational radial coordinate, such as:

[
u = r^2
]

or normalized volume coordinate,

[
u = \frac{r^2-r_{\min}^2}{r_{\max}^2-r_{\min}^2}
]

Then apply TSC in (u), not directly in (r).
