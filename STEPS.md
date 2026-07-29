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

# Package Recommendations

These recommendations come from a scan of all tracked Python, Zig, and Rust
source outside `hoomd-blue`. They prioritize logic that is valuable to this
project, has a credible reusable API, and is not already covered by a widely
used package in the same form. Names below are working names only.

## 1. `curvdeposit`: conservative curvilinear particle-to-grid deposition

This is the strongest package opportunity. The repository already contains a
GPU implementation that deposits 53 packed scalar, vector, tensor, and current
features in `crates/rho_fitting_gpu/src/coarse_grain_burn/`, NumPy Gaussian
deposition in `hexatic/active_matter_cylinder/math_utils.py`, and another
cell-list-backed implementation in
`packages/simulation_analysis/src/properties/polar.zig`.

The package should provide:

- Particle-to-grid deposition on Cartesian, cylindrical, and other orthogonal
  curvilinear grids.
- User-defined particle features rather than hardcoded `rho`, `P`, `Q`, and
  current tensors.
- Scalar, vector, symmetric-tensor, and rank-three moment deposition.
- Periodic axes, clipped axes, metric factors, and grid-cell volume forms.
- Discrete boundary renormalization so deposited mass and moments are
  conserved near finite radial boundaries.
- CPU, CUDA, and Metal backends with consistent results.
- Chunked execution, conservation diagnostics, masks, and nonfinite handling.
- A Rust core with NumPy-facing Python bindings.

A possible public API is:

```python
domain = CylindricalGrid(x=x, theta=theta, r=r, periods={"x": lx, "theta": 2 * pi})
plan = DepositPlan(
    kernel=Gaussian(sigma),
    moments={
        "rho": "1",
        "P": "direction",
        "Q": "symmetric_traceless(direction * direction)",
        "J": "velocity",
    },
)
fields = plan.deposit(positions, particle_data, domain)
```

Existing particle-mesh packages primarily target Cartesian cosmology,
electrostatics, or generic scatter operations. The differentiator here is the
combination of curvilinear volume elements, physical component frames,
boundary renormalization, arbitrary mechanical moments, and multiple GPU
backends. Potential users include active-matter, SPH, plasma, colloid,
granular, microscopy, and molecular-simulation researchers.

## 2. `conserveid`: conservation-form sparse system identification

This package would extract the divergence-primary fitting architecture from
`hexatic/rho_fitting/fit.py`, the regression-row assembly from
`crates/rho_fitting_numerics/src/fitting/`, the constrained solver from
`crates/rho_fitting_numerics/src/regression/`, and the rollout machinery from
`crates/rho_fitting_numerics/src/pde/`.

Its central workflow should be:

```text
measured flux + candidate fluxes
              -> differential operator
measured divergence + candidate divergences
              -> joint constrained sparse fit
              -> holdout and forward-rollout validation
```

The package should support:

- Joint fitting of flux values and their PDE-relevant divergences.
- Divergence-primary evaluation rather than treating a high flux R² as
  sufficient evidence for a usable closure.
- Scalar, vector, and tensor conservation laws.
- Sign, equality, symmetry, and shared-coefficient constraints.
- Independent finite filtering and weighting of field and divergence rows.
- Candidate normalization, regularization paths, coefficient importance, and
  feature-correlation diagnostics.
- Pluggable geometry and differential operators.
- Mandatory forward-rollout validation of learned closures.

PySINDy and PDE-discovery projects are established adjacent tools, but they do
not provide this complete combination of flux-level interpretability,
divergence-primary scoring, physical tensor closures, constrained regression,
and rollout validation. Active-matter-specific candidate names should remain
outside the reusable numerical core.

## 3. `tensortrail`: transactional safetensors time-series datasets

The immediate motivation is the nearly duplicated atomic safetensors writers,
frame-shard writers, manifests, overwrite/resume handling, and completion
markers in `hexatic/big_lx/storage.py` and
`hexatic/confinement_comparison/storage.py`. The Zig analyses also implement
their own mapped shard collections and cross-shard schema validation.

The package should provide:

- Immutable safetensors shards organized along a declared frame or record
  axis.
- Atomic shard and manifest publication.
- Crash recovery, resume support, and explicit dataset completion.
- Frame-range iteration and tensor projection without loading unrelated data.
- Cross-shard dtype, shape, axis, and metadata validation.
- Monotonic step/index validation and gap or overlap detection.
- Dataset fingerprints and versioned schema identifiers.
- Memory-mapped Rust and Zig readers plus a NumPy-facing Python API.
- Ragged tensors represented by values and offsets.

This should not become another safetensors parser. The official safetensors
implementations already fill that role. It should be the immutable,
transactional dataset layer above individual safetensors files. It is also not
intended to replace Zarr: the niche is independently transportable, safely
published shards with straightforward zero-copy readers. This is particularly
useful for HPC outputs, simulation trajectories, scientific acquisition, and
large ML activation datasets.

## 4. `quotientcell`: exact commensurate Bravais supercells

This package would generalize the exact twisted triangular-lattice construction
in `hexatic/unwrapped_analysis/cases.py` and
`simulate_case_perfect_hexatic.py`. The reusable mathematical object is a
finite quotient of a Bravais lattice under integer periodic identifications,
not a HOOMD-specific cylinder initializer.

The package should provide:

- Integer orthogonal complements under an arbitrary two-dimensional Gram
  metric.
- Primitive and multiplied axial or transverse supercell vectors.
- Exact enumeration of one fundamental parallelogram.
- Quotient, screw, and twisted periodic identifications.
- Verification that requested nearest-neighbor bonds survive every periodic
  identification.
- Embeddings onto planes, cylinders, tori, and helices.
- Export to NumPy, ASE, GSD, or plain coordinate and connectivity arrays.

A possible API is:

```python
cell = QuotientCell(
    gram=[[1.0, 0.5], [0.5, 1.0]],
    identification=(m, n),
    orthogonality="metric",
)
points, fractions = cell.enumerate()
cell.verify_neighbors(neighbor_vectors)
```

Crystallographic and nanotube packages cover many specific constructions, but
a small general package for metric-aware integer quotient cells and
graph-preserving periodic embeddings does not appear to be widespread.
Potential applications include nanotubes, wrapped crystals, moiré
approximants, screw-periodic simulations, topological lattice models, and mesh
generation.

## 5. `periodic-lift`: topology-aware components in periodic domains

The foundation already exists in `packages/cluster_analysis/src/clusters.zig`:
a reusable cell list, union-find components, adjacency construction, and
periodic component unwrapping. A public package should focus on the topology of
periodic components rather than provide only cluster labels.

The package should provide:

- Connected components built from periodic minimum-image edges.
- Consistent lifted or unwrapped coordinates for each component.
- Winding vectors and detection of components that cross or wrap the domain.
- Percolation classification.
- Periodic centers, covariance, gyration, and bounding geometry.
- Rectangular boxes and general lattice-period matrices.
- Optional node and edge compatibility predicates, including lattice
  misorientation or orientational order.
- Compact ragged outputs using values and offsets.

Freud already provides strong periodic clustering, so plain clustering would
not be a sufficiently differentiated package. The distinguishing features
should be lifted coordinates, winding topology, percolation, and custom edge
compatibility. The package could serve percolation studies, porous media,
polymer networks, crystalline domains, and periodic image segmentation.

## 6. `curvispec`: array-first cylindrical spectral calculus

This package would extract `CylindricalSpectralOperators` and radial transfer
logic from `crates/rho_fitting_numerics/src/spectral/` and
`hexatic/rho_fitting/spectral.py`.

The package should provide:

- Fourier-by-Chebyshev grids for periodic axial/angular directions and a
  bounded radial direction.
- Derivative, gradient, divergence, curl, and Laplacian operators.
- Correct metric and connection terms for physical-frame vectors and tensors.
- Arbitrary leading batch dimensions and trailing component dimensions.
- Stable barycentric transfers between uniform and spectral radial grids.
- Two-thirds and configurable spectral filtering.
- NumPy and Rust APIs with explicit axis and component conventions.

Dedalus and Shenfun are established spectral PDE frameworks, so this should
not attempt to become another complete solver. Its niche should be a
lightweight operator library for users who already have arrays and need
correct cylindrical vector and tensor calculus. Spherical and other orthogonal
coordinate charts could be added later.

## 7. `polemap`: damped-mode discovery from correlations

This lower-priority package would extract the complex Laplace analysis,
preferred-coordinate searches, and robust damped-cosine fitting from
`packages/laplacian_analysis/`.

It should provide:

- Fast complex Laplace evaluation over decay-rate and frequency grids.
- Uniform and irregular-time numerical integration.
- Stable recurrence-based grid evaluation.
- Preferred pole, peak, or ridge detection.
- Bootstrap uncertainty across trajectories or experimental replicates.
- Boundary-hit and identifiability diagnostics.
- Robust bounded multistart fitting of damped modes.

The useful product framing is decaying-mode discovery rather than another
generic Laplace-transform implementation. Applications include spectroscopy,
relaxation processes, autocorrelation analysis, active matter, and dynamical
systems.

## Packages not worth creating

Do not create the following as new general public packages:

- A generic molecular-dynamics trajectory analyzer; MDAnalysis and freud
  already occupy this space.
- A generic parameter-sweep or GPU subprocess scheduler; consolidate the
  duplicated repository code internally or use signac-flow and established
  workflow systems.
- Another safetensors parser; build `tensortrail` as a dataset layer instead.
- Another generic GSD or HDF5 wrapper; the local Zig wrappers are useful
  infrastructure but have a weak standalone differentiator.
- Another general dense-linear-algebra library.
- Another complete spectral PDE framework; keep `curvispec` focused on
  array-oriented physical-frame operators.
- A plotting or report-helper package; the repetition is real but does not
  define a compelling public product.

## Recommended implementation order

1. Extract `curvdeposit`; it has the highest project impact and the broadest
   external audience.
2. Build `tensortrail`; it directly removes repeated logic and supports the
   active safetensors-only `big_lx_analysis` workflow.
3. Extract `conserveid` after defining clean geometry, candidate-library, and
   differential-operator interfaces.
4. Publish `quotientcell` as the smaller and mathematically distinctive
   package.
5. Develop `periodic-lift` and `curvispec` after separating their public APIs
   from repository-specific schemas.
6. Consider `polemap` once replicate uncertainty and a clear standalone
   scientific workflow are included.

The best single public package is `curvdeposit`. The fastest useful extraction
is `tensortrail`. The most scientifically distinctive package is
`conserveid`, with `quotientcell` close behind.
