# Crystalline Cylinder Analysis in Rust

## Summary

Create:

- `crates/crystalline_cylinder_analysis`: reusable Rust analysis library, published internally as `crystalline-cylinder-analysis`.
- `packages/crystalline-cylinder-analysis`: CLI package named `crystalline-cylinder-analysis-cli`, exposing the `crystalline-cylinder-analysis` binary.

The library will stream current Big-Lx and confinement safetensor outputs, compute COM motion, lagged velocity Pearson correlation, complex Laplace transforms, preferred \(r/\omega\), and robust damped-cosine fits. The CLI will produce Kuva SVG plots plus machine-readable safetensor and JSON artifacts.

V1 is CPU-only on macOS and Linux, parallelized with Rayon and using Tenferro’s CPU-faer backend. It will not use Burn, PyO3, Maturin, or GPU dependencies yet.

## Workspace and Public Interfaces

- Raise the workspace `rust-version` from 1.83 to 1.87 for Kuva 0.4 compatibility and update `Cargo.lock`.
- Add shared dependencies for safetensors 0.8, memmap2, bytemuck, Tenferro runtime/CPU/linalg 0.2, Kuva 0.4 with embedded-font SVG support, Clap 4, Serde/JSON, `num-complex`, and Rayon. Use assertion-driven fail-fast validation instead of custom error enums.
- Do not add `integrate`: its Simpson API evaluates a function at new points, whereas this workflow integrates sampled correlation values. Implement sampled Simpson quadrature directly.
- Expose typed result structures including `CaseMetadata`, `ComSeries`, `CorrelationSeries`, `LaplaceGrid`, `PreferredEstimate`, and `DampedCosineFit`.
- Use focused enums for manifest schema, analysis kind, preferred axis, and output artifact type. Provide an `AnalysisBackend` trait with a `CpuAnalysisBackend`; keep I/O and plotting outside the backend abstraction.
- Keep the core library free of CLI and Kuva concerns so it can later be wrapped by PyO3 without reorganizing numerical code.

CLI shape:

```text
crystalline-cylinder-analysis [global options] <subcommand>

subcommands:
  com
  correlation
  laplace
  preferred --axis omega|r|both
  fit
  all
```

Global options include repeatable `--input-dir`, optional `--output-dir`, `--timestep` defaulting to `1e-6`, optional `--threads`, and explicit `--overwrite`. Correlation-derived commands expose `--max-lag`; every selected lag must retain at least two paired samples. Transform commands preserve the Python range/point defaults.

## Implementation Changes

### Streaming input and validation

- Scan each resolved `--input-dir/safetensors_output/*/manifest.json`; accept complete `hexatic.big_lx.analysis.v1` and `hexatic.confinement_comparison.analysis.v1` manifests.
- Identify cases from manifest metadata—not shard filenames, which currently only encode frame ranges. Group replicates by `(schema, case_id)` and require matching geometry, particle count, \(L_x\), multiplier, and confinement metadata. Do not inspect or verify seed values.
- Validate contained paths, contiguous zero-based shards, declared frame counts, required tensors, dtypes, shapes, finite coordinates, and globally increasing steps before using results.
- Encapsulate a read-only `memmap2::Mmap` and direct `safetensors::SafeTensors` views in the shard reader. Borrow only `coords`/`position_cartesian` and `step`, expose typed zero-copy slices through bytemuck, decode one frame at a time, and drop each mapping before opening the next shard. Never materialize every tensor or every shard. Document the mmap safety requirement that inputs remain unchanged while mapped.
- Refuse to overwrite existing analysis artifacts unless `--overwrite` is supplied; write JSON and safetensors atomically.

### COM, correlation, and replicate handling

- Maintain only previous wrapped coordinates and current unwrapped coordinates, both \(O(N_\text{particles})\). Apply the per-particle minimum-image update
  \(dx \leftarrow dx-L_x\operatorname{round}(dx/L_x)\), then average in `f64`.
- Compute elapsed time from steps and `--timestep`. Reproduce NumPy-style first/second-order endpoint gradients and centered interior gradients, including nonuniform COM sampling; require uniform sampling before correlation.
- Implement Pearson correlation manually for every lag using centered paired windows, compensated `f64` accumulation, explicit zero-variance assertions, lag-zero normalization to one, and clipping to \([-1,1]\).
- Parallelize independent cases, lags, and transform grid rows with Rayon while keeping only one shard per active replicate mapped.
- For duplicate cases, require a compatible common elapsed/lag prefix, compute each replicate’s COM velocity and Pearson series independently, then average those series pointwise. Store sample-standard-deviation bands and summed time-origin counts. The averaged Pearson series is the primary input to all downstream transforms and fitting.

### Laplace analysis and fitting

- Implement sampled composite Simpson integration, including trapezoidal handling for two samples and SciPy’s final-interval Cartwright correction when the number of samples is even.
- Evaluate
  \[
  \widehat C_v(r+i\omega)=\int e^{(r+i\omega)\tau}C_v(\tau)\,d\tau
  \]
  directly in `f64`/`Complex64`, parallelized over grid coordinates without allocating an `(r, omega, time)` temporary.
- Preserve Python defaults:
  - \(r_{\min}=-10/T\), \(r_{\max}=0\);
  - \(\omega_{\max}=\min(\omega_\mathrm{Nyquist},20\pi/T)\);
  - 161×241 full heatmap grid;
  - 241-point preferred-coordinate grids.
- Preferred \(\omega\) searches strictly positive frequencies at \(r=0\); preferred \(r\) searches strictly negative values at \(\omega=0\). Maximize `log10(max(abs(transform), f64::MIN_POSITIVE))` and report boundary warnings.
- Reproduce the current robust model
  \[
  C_v(\tau)=A e^{-r\tau}\cos(\omega\tau+\phi)+B,\qquad
  B=1-A\cos\phi.
  \]
  Preserve the Python bounds, origin-count weights, soft-L1 scale `0.05`, nine rate/frequency starts, 20,000-evaluation ceiling, boundary diagnostics, and unweighted \(R^2\).
- Implement analytic Jacobians and a bounded Levenberg–Marquardt/IRLS loop. Each damped linearized system will be converted explicitly to Tenferro’s column-major layout and solved through `tenferro-linalg` SVD/pseudoinverse with a documented rank tolerance. Accept/reject steps using the exact soft-L1 objective and stop on gradient, step, or objective tolerances of `1e-8`.
- Keep external safetensor data as borrowed row-major frame views; only the small optimizer matrices are compacted into Tenferro’s column-major representation. [Tenferro memory-order guidance](https://tensor4all.org/tenferro-rs/guides/memory-order.html)

### Plotting and artifacts

- Default output directory: `<first-input-dir>/crystalline_cylinder_analysis_output`.
- Write an output manifest using schema `crystalline_cylinder_analysis.output.v1`, a JSON scalar summary, and per-case safetensors containing numerical arrays and replicate metadata.
- Use Kuva 0.4 to generate self-contained SVGs:
  - two-panel unwrapped COM and COM-velocity plot with replicate bands;
  - velocity Pearson correlation versus lag with replicate bands;
  - schema-grouped `log10|Ĉv|` Laplace heatmap panels for Big-Lx and confinement cases;
  - preferred \(r\) and \(\omega\) versus \(L_x\) multiplier, grouped by circumference with confinement cases at multiplier one;
  - measured correlation and damped-cosine fit overlays with fitted parameters and \(R^2\).
- `all` performs discovery and streaming once, reuses shared COM/correlation intermediates, and emits every artifact. [Kuva plotting pipeline](https://docs.rs/kuva/latest/kuva/)

## Validation

Repository policy forbids adding or running a test suite. Verification will therefore be limited to:

- `pixi run cargo fmt --all --check`
- targeted `cargo check` for the new core crate and CLI package
- CLI `--help`/argument parsing checks
- review that no Burn, PyO3, Maturin, GPU, or generated analysis outputs were introduced
- runtime fail-fast checks for malformed manifests, escaping shard paths, inconsistent replicas, invalid tensors, nonuniform correlation samples, zero variance, transform overflow, invalid grids, and singular/nonconvergent fits

No production safetensors or generated plots will be overwritten during implementation.

## Assumptions

- CPU-first means no Metal/CUDA feature flags in v1; Rayon and Tenferro CPU-faer are supported on both target platforms.
- Big-Lx and confinement replicas with the same case ID represent the same physical case once validated, regardless of seed metadata.
- The existing `1e-6` timestep remains the default, but callers may override it.
- Only axial COM velocity Pearson correlation is included; the Python hexatic-order autocorrelation is out of scope.
- SVG is the canonical plot format; numeric safetensor and JSON artifacts provide future Python/PyO3 interoperability.
