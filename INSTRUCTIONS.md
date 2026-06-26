# NEW PLOTS
Goal is:

```text
Understand whether the defect dynamics are controlled mainly by:

1. motion of persistent defects,
2. birth/death of defects,
3. cluster splitting/merging,
4. local particle rearrangement,
5. shell-core exchange,
6. lattice/hexatic disorder,
7. chirality,
8. stress/flow fields,
9. or some combination of these.
```

The current leading hypothesis is:

```text
Defect pattern changes are probably source/reaction dominated:
birth, death, annihilation, and cluster rearrangement matter more than smooth defect drift.
```

---

## 2.1 Cleanup
Delete all files within [disclination_order_fields](/Users/vaibhavvenkat/structures_and_functions/hexatic/multiple_sim_analysis/disclination_order_fields/), start from a clean slate

Before doing so, read the directory to understand context.

## 2.2 Persistent defect velocity

Only compute velocity for a defect particle that exists in two consecutive saved frames and is confidently matched:

```
v_i(t) = [X_i(t + Δt) - X_i(t)] / Δt
```

Do not compute velocity for:

```
newly born defects
dead defects
ambiguous identity swaps
```

Those become separate events.

---

## 2.3 Birth event

Define a birth as:

```
A defect appears at time t
and persists for at least k = 2 consecutive frames, including its origin frame.
```
depending on frame spacing.

For each birth event, save:

```text
birth position X_b
birth time t_b
birth charge/type (+1/-1)
nearest opposite defect
nearest same-charge defect
local fields at t_b - τ, t_b, t_b + τ
```
time is in term of the timestep, trajectory_write_period, and the frames. Not just frames

---

## 2.4 Death event

A death is:

```text
A defect existed for at least k = 2 frames and then disappears.
```
or a "birthed" defect that dissapeared.

For each death event, save:

```text
death position X_d
death time t_d
death charge/type
nearest opposite defect before death
whether it annihilated with opposite charge
local fields before death: t_b - tau
```
A death should be classified as annihilation if:

```text
+ and - defects approach within distance d_ann
and both disappear within a short time window.
d_ann ≈ 2.5a, where a = neighbor_count_radius
time window ≈ 2 saved frames
```

---

# 3. Clean local neighborhoods

If defects are clustered, then the annulus around one defect may contain other defects

## 3.1 Use defect-excluded annuli

For target defect (i), define:

```text
core:      d < a
annulus:   a < d < r_max
r_max = min(3a, 0.5 * nearest_defect_distance)
```

Exclude particles that are themselves defect particles with the annulus case

---

## 3.2 Compute local background velocity carefully

For each defect annulus, compute all three:

```text
u_mean  = |mean(v_j)|
u_rms   = mean(|v_j|)
u_fluct = sqrt(mean(|v_j - mean(v_j)|²))
```
# 4. Quantities to compute

For every (R), every frame, and every region ((x,R\theta)), compute the following.


```text
N_+(t)
N_-(t)
N_total(t) = N_+(t) + N_-(t)
defect density n_+(x,θ,t)
defect density n_-(x,θ,t)
```

```text
B_+(t), B_-(t) = birth counts per time
D_+(t), D_-(t) = death counts per time
A_+(t), A_-(t) = annihilation counts per time
```

Define total topological activity:

```text
A_top(t) = B_+(t) + B_-(t) + D_+(t) + D_-(t)
```

Normalize by defect number:

```
a_top(t) = A_top(t) / N_total(t)
```

## 4.3 Defect motion fields

```text
|v_defect|
v_defect,x
v_defect,θ
net displacement during lifetime
```

Compute:

```text
v_parallel = v_defect · u_local_hat

v_perp = |v_defect - v_parallel u_local_hat|

cos(v_defect, u_local)
```
## 4.4 Defect cluster fields

Connect defects if:

```text
distance(defect i, defect j) < ℓ_cluster
```

Choose:

```text
ℓ_cluster ≈ 4.5a
```

For each cluster:

```text
size = number of defects
charge = N_+ - N_-
total defects = N_+ + N_-
COM
velocity
```

This is essential if the relevant object is not a single disclination but a defect aggregate.

---

## 4.5 Local particle/order fields

Compute these in the shell and near defects:

```text
J_r                  radial exchange current
u_rms                local particle activity
u_fluct              local rearrangement activity
S                    tangent nematic order
Q                    tangent nematic tensor
|ψ6|                 hexatic order
χ                    chirality
F_density            force_density (F_ij + F_wall)
D²_min               nonaffine rearrangement
nearest_defect_distance
```
D^2_min is defined as:
Pick a particle i. Look at its neighbors j at time t, then compare their relative positions at a later time t+Δt.
`r_ij(t) = r_j(t) - r_i(t)`
`r_ij(t + Δt) = r_j(t + Δt) - r_i(t + Δt)`
Then find the best local linear deformation matrix `F_i`that maps the old neighbor configuration to the new one
`D²_min(i) = 1/neighbors_i min over F_i  Σ_j | r_ij(t + Δt) - F_i r_ij(t) |²`

# 5. Plots vs R

## 5.1 Defect number and turnover


```text
mean N_+(R)
mean N_-(R)
birth rate B(R)
death rate D(R)
annihilation rate A(R)
topological activity A_top(R)
track lifetime(R)
```

## 5.2 

Plot:

```text
A_top(R)
median |v_defect|(R)
median track lifetime(R)
median displacement before death(R)
```


define normalized motion activity:

```text
motion_activity(R) = median(|v_defect|) / a
```

Plot:

```text
a_top(R) / motion_activity(R)
```
where a_top is the normalized one

# 6. Event-centered plots: the most important section

For each birth event at time (t_b), plot fields from before to after:

```text
t - t_b = -τ, ..., 0, ..., +τ
```

For each death event at (t_d), do the same.

---

## 6.1 Birth-triggered averages

For each birth, compute local fields at the future birth position before it happens:

```text
ρ_birth(t_b - τ)
J_r_birth(t_b - τ)
u_rms_birth(t_b - τ)
u_fluct_birth(t_b - τ)
S_birth(t_b - τ)
|ψ6|_birth(t_b - τ)
χ_birth(t_b - τ)
D²_min_birth(t_b - τ)
strain_birth(t_b - τ)
```

Then average over birth events. DO this only for R = 20D across frames

Plot:

```text
field near future birth site vs time relative to birth
```

Example plots:

```text
|ψ6|(t - t_b)
χ(t - t_b)
D²_min(t - t_b)
u_rms(t - t_b)
J_r(t - t_b)
ρ(t - t_b)
```

---

## 6.2 Death-triggered averages

Same for death events:

```text
field near dying defect vs time relative to death
```

Plot:

```text
nearest opposite-charge distance(t - t_d)
|ψ6|(t - t_d)
χ(t - t_d)
D²_min(t - t_d)
u_rms(t - t_d)
density(t - t_d)
```

---


# 9. Radial exchange and shell-build-up plots

Because particles leave the shell and return, defect birth/death may be controlled by shell-core exchange.

Compute:

```text
J_r = ρ v_r
```

Separate:

```text
J_in  = inward shell-to-core flux
J_out = outward core-to-shell flux
J_abs = |J_r|
```

For births/deaths:

```text
J_r near future birth
J_r near future death
J_r near stable defects
J_r far from defects
```

Plots:

```text
birth rate vs |J_r|
birth rate vs J_in
birth rate vs J_out
death rate vs |J_r|
defect density vs shell density
defect birth rate vs shell thickness
```
for R = 20D


---

## 10.2 Quantitative plots

Plot:

```text
birth probability vs |ψ6|
birth probability vs 1 - |ψ6|
birth probability vs |∇ψ6|
death probability vs |ψ6|
annihilation probability vs |ψ6|
```

---

# 11. Chirality plots
Plot:

```text
birth probability vs χ
birth probability vs |χ|
birth probability vs ∂tχ
birth probability vs |∇χ|
death probability vs χ
annihilation probability vs χ
```

Event-centered plots:

```text
χ(t - t_birth)
χ(t - t_death)
χ_annulus(t - t_birth)
χ_annulus(t - t_death)
```


---


# 14. Maps you should make


For representative frames and radii, make panels:

```text
1. particles + defects
2. |ψ6| map + defects
3. chirality map + defects
4. u_rms or D²_min map + defects
5. J_r map + defects
6. density map + defects
7. birth/death events overlaid
```
