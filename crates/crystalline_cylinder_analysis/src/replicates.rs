//! Pointwise aggregation of compatible seed replicas.

use crate::error::AnalysisResult;
use crate::model::{ComSeries, CorrelationSeries};

/// Average compatible COM series on their common elapsed-time prefix.
pub fn average_com_series(_replicas: &[ComSeries]) -> AnalysisResult<ComSeries> {
    todo!("compute pointwise means and sample-standard-deviation bands")
}

/// Average compatible correlations and sum their time-origin counts.
pub fn average_correlations(_replicas: &[CorrelationSeries]) -> AnalysisResult<CorrelationSeries> {
    todo!("validate lag grids and aggregate seed correlations")
}
