# Coupled Stable-Band Area Inference

## Summary

Replace the existing `hexatic.band_analysis` stochastic/morphology workflow with a cached multi-seed pipeline for the five-parameter coupled area SDE. Retain only density-based detection, overlap tracking, stable-segment construction, inference, and the validations specified in `PLAN.md`.

Training seeds contribute pooled transitions; holdout seeds never enter inference. Run the complete workflow sequentially at frame lags 1, 2, 3, 5, and 10, with lag 2 as the reference. Add no multi-GPU or cross-lag parallelization machinery.

## Implementation changes

- Reduce detected-band data to physical area, periodic axial center, mask, identity, event state, and physical time. Remove old morphology modes, neighbor statistics, conditional-bin fits, MSD analysis, and their plotting code.
- Debounce topology events using `persistence_frames`:
  - Confirm persistent topology changes retroactively from their first frame.
  - Confirmed splits/merges retire all participating identities and assign new identities to the resulting stable bands.
  - Treat a topology that reverses before confirmation as an excluded transient gap; restore uniquely matched identities afterward and never construct transitions across the gap.
  - Treat uncertain restoration as a hard tracking boundary.
- Construct maximal contiguous joint segments with a fixed, track-ID-ordered set of stable bands. Include their full clean lifetime after stability is confirmed. Retain \(n=1\) segments for the total-area mode only.
- Use exact \(\tau=\text{simulation step}\times\text{timestep}\). Require identical geometry, physical parameters, preprocessing settings, run duration, and trajectory write period across seeds; only seed and storage metadata may differ.
- Normalize area by the training median total area and time by the training median frame interval. Infer dimensionless parameters and report physical values through
  \[
  \kappa=\kappa'/\tau_s,\qquad
  D=D'A_s^2/\tau_s,\qquad
  A_T^\ast=A_sA_T^{\ast\prime}.
  \]
- Implement one float64 JAX transition function shared by optimization, NumPyro, residuals, and simulations. Use Cholesky log densities and \(10^{-10}I\) normalized covariance jitter.
- At lag \(h\), use every transition contained in a clean segment and weight each log-likelihood contribution by \(1/h\), retaining all phase offsets without treating all overlapping intervals as independent information.
- Calculate training-only empirical starts from weighted total-area OU regression and projected conservative increments.
- Use eight Optimistix BFGS starts: one empirical, six reproducibly perturbed raw-space starts, and one generic normalized start. Limit each to 2,000 iterations and retain the lowest finite objective.
- Add Optimistix, NumPyro, ArviZ, and `h5netcdf` as direct Pixi dependencies. Do not add JAXopt.
- Compute the raw-parameter Hessian at every optimum. Report its matrix, eigenvalues/eigenvectors, condition number, and flag nonpositive eigenvalues or eigenvalues below \(10^{-6}\lambda_{\max}\). Do not calculate profile likelihoods.
- Use empirical-centered lognormal priors in normalized units, with log standard deviation 1.0 for rates/diffusions and 0.5 for \(A_T^\ast\).
- Run four vectorized NUTS chains per lag with 1,500 warmup and 1,500 retained draws, target acceptance 0.90, and a dense mass matrix.
- Require \(\widehat R<1.01\), bulk/tail ESS of at least 400, and zero divergences. Retry a failed lag once with target acceptance 0.95 and 3,000 warmup steps. If it fails again, retain diagnostics but exclude it from accepted predictive and lag-stability conclusions.
- Process lags strictly in the configured order. Complete and atomically save one lag before starting the next so interrupted runs can resume from successful caches.

## Interfaces and artifacts

- Expose one command with repeatable training `--input-dir`, optional repeatable `--holdout-dir`, and required `--output-dir`.
- Do not add `--gpu-ids`, lag-worker counts, subprocess schedulers, or device-assignment logic. Use the active JAX device selected by the execution environment.
- Retain extraction controls including `--persistence-frames` and overlap threshold. Add configurable lags, base lag, optimizer starts, MCMC budget, predictive draws, and random seed.
- Without `--overwrite`, reuse compatible per-seed and completed per-lag caches and regenerate reports/plots. Reject incompatible caches. With `--overwrite`, rebuild generated artifacts.
- Store:
  - Per-seed stable segments in SafeTensor.
  - BFGS, Hessian, residual, and predictive arrays in per-lag SafeTensor files.
  - Physical-unit ArviZ `InferenceData` in per-lag NetCDF files.
  - Configuration and metrics in JSON plus a consolidated Markdown report.
  - Static plots as PNG.
- Produce per-lag posterior/trace, rank, pair, energy/BFMI, Hessian, whitened-residual, predictive, trajectory, long-time, covariance, and negative-area diagnostics. Also produce cross-lag parameter/HDI and physical-\(\Delta\tau\) plots plus aggregate and per-seed holdout panels.
- Use 200 posterior draws for one-step summaries and 200 paths per stable segment. Do not clip negative areas; count them and terminate only numerically singular/nonfinite paths.
- Label a lag stable relative to lag 2 only when every parameter median lies inside lag 2's 95% HDI and lag 2's median lies inside that lag's HDI. Report no stable range if lag 2 fails MCMC acceptance.
- Validate holdouts using the frozen training posterior and scaling; report predictive log score, coverage, residual behavior, and long-time summaries without refitting.

## Test plan

- Verify transition sums, covariance decomposition, conservative total-area preservation, positive definiteness, normalized-unit round trips, and \(n=1\) behavior.
- Recover known parameters from synthetic multi-\(n\), irregular-\(\Delta\tau\) trajectories with BFGS; include a reduced-budget NUTS smoke test.
- Test persistent/transient births, disappearances, splits, merges, uncertain restoration, retroactive stability, and exclusion of gap-crossing transitions.
- Test all-start lag construction, \(1/h\) likelihood weighting, train/holdout isolation, and manifest compatibility.
- Test deterministic sequential lag ordering, atomic per-lag writes, interrupted-run resumption, and reuse of successful caches.
- Round-trip SafeTensor and NetCDF artifacts, test cache fingerprints and automatic NUTS retry, and render all plots from synthetic results.
- Run `compileall`, `ty check`, and focused standard-library unit tests only; do not run production extraction or full five-lag NUTS during routine verification.

## Assumptions

- At least one training directory is required; holdouts are optional.
- All seeds represent the same physical case and use identical analysis thresholds and output cadence.
- GPU acceleration may still be used naturally by existing JAX density and inference operations, but the workflow contains no explicit multi-GPU or concurrent-lag parallelization.
- Existing generated user artifacts are not deleted during implementation and are replaced only by a later explicitly authorized `--overwrite` run.
- The Euler-Maruyama transition, Gaussian area model, empirical-prior construction, and absence of measurement noise remain the modeling assumptions from `PLAN.md`.
