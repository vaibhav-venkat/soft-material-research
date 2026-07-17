1. Structural grains and boundaries
For each frame, build a geometry-aware neighbor graph and calculate
\[
\psi_{6,i}=\frac{1}{N_i}\sum_j e^{6i\theta_{ij}}, \qquad
\alpha_i=\frac{\arg\psi_{6,i}}{6}.
\]|ψ₆| measures local hexagonal order, while α is the local crystal orientation. This is the standard diagnostic for 2D crystalline/hexatic order. Example definition and application.
Classify a neighbor connection as belonging to the same grain when:
Both particles are sufficiently ordered, initially |ψ₆| > 0.7.
Their lattice misorientation is small, initially below approximately 5°.
They are actual Delaunay or first-shell neighbors.
Connected components of those compatible bonds are the grains.
Mark grain-boundary particles when they have one or more of:
Coordination other than six
A nonzero disclination charge
A nearby 5–7 dislocation pair
Low |ψ₆|
Strong orientation disagreement with neighbors
Using orientation-compatible edges is important: proximity alone can accidentally merge adjacent but differently oriented grains. This distinction is also emphasized in modern crystalline-cluster construction work. Journal of Chemical Physics discussion.
2. Coherently moving clusters
Choose a physical lag Δt and compute each particle’s minimum-image displacement:
\[
\mathbf u_i(t,\Delta t)
=
\mathbf r_i(t+\Delta t)-\mathbf r_i(t).
\]Then subtract global drift:
\[
\mathbf u_i'=\mathbf u_i-\langle\mathbf u\rangle.
\]Connect neighboring particles when their residual displacements are aligned, for example:
\[
\frac{\mathbf u_i'\cdot\mathbf u_j'}
{|\mathbf u_i'||\mathbf u_j'|}>0.8.
\]Optionally require similar displacement magnitudes so a nearly stationary particle is not joined to a fast-moving domain. Connected components of this graph are the moving clusters.
Do not use DBSCAN on positions alone: because these systems are dense, almost the entire crystal would become one spatial cluster.
Choose Δt by scanning several lags and looking for the peak in dynamic heterogeneity—using the non-Gaussian parameter, overlap susceptibility χ₄, or mean coherent-cluster size. Four-point correlations and χ₄ are established ways to quantify the spatial extent of slow or mobile domains. Physical Review E discussion.
