//! Lagged axial-velocity Pearson correlation.

use crate::backend::AnalysisBackend;
use crate::error::AnalysisResult;
use crate::model::{ComSeries, CorrelationSeries};

/// Controls the number of valid lag origins.
#[derive(Clone, Copy, Debug)]
pub struct CorrelationConfig {
    pub min_origins: usize,
    pub max_lag: Option<usize>,
}

/// Compute one pairwise Pearson coefficient using stable accumulation.
pub fn pearson(_left: &[f64], _right: &[f64]) -> AnalysisResult<f64> {
    todo!("compute centered covariance and variances with compensated sums")
}

/// Compute the lagged velocity correlation for one COM series.
pub fn analyze_correlation<B: AnalysisBackend>(
    _backend: &B,
    _com: &ComSeries,
    _config: CorrelationConfig,
) -> AnalysisResult<CorrelationSeries> {
    todo!("validate uniform samples and dispatch all requested lags")
}
