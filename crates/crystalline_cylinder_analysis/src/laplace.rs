//! Complex Laplace transform and preferred-coordinate searches.

use crate::backend::AnalysisBackend;
use crate::error::AnalysisResult;
use crate::model::{CorrelationSeries, LaplaceGrid, PreferredAxis, PreferredEstimate};

/// Transform-grid controls.
#[derive(Clone, Copy, Debug)]
pub struct LaplaceConfig {
    pub r_min: Option<f64>,
    pub r_max: f64,
    pub r_points: usize,
    pub omega_min: Option<f64>,
    pub omega_max: Option<f64>,
    pub omega_points: usize,
}

/// Construct validated real and imaginary transform axes.
pub fn transform_axes(
    _correlations: &[CorrelationSeries],
    _config: LaplaceConfig,
) -> AnalysisResult<(Vec<f64>, Vec<f64>)> {
    todo!("derive duration- and Nyquist-limited shared axes")
}

/// Evaluate the complete complex transform grid.
pub fn analyze_laplace<B: AnalysisBackend>(
    _backend: &B,
    _correlation: &CorrelationSeries,
    _r: &[f64],
    _omega: &[f64],
) -> AnalysisResult<LaplaceGrid> {
    todo!("dispatch a memory-bounded transform-grid evaluation")
}

/// Locate the maximum log magnitude on the selected transform axis.
pub fn preferred_coordinate(
    _correlation: &CorrelationSeries,
    _axis: PreferredAxis,
    _coordinates: &[f64],
) -> AnalysisResult<PreferredEstimate> {
    todo!("evaluate the r=0 or omega=0 transform and diagnose boundary maxima")
}
