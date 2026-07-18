//! Rayon and Tenferro CPU backend declaration.

use crate::backend::AnalysisBackend;
use crate::error::AnalysisResult;
use crate::model::{CorrelationSeries, LaplaceGrid};

/// CPU backend backed by Rayon and Tenferro's faer provider.
#[derive(Clone, Debug)]
pub struct CpuAnalysisBackend {
    pub thread_count: Option<usize>,
}

impl CpuAnalysisBackend {
    /// Construct a CPU backend with an optional thread limit.
    pub fn new(_thread_count: Option<usize>) -> AnalysisResult<Self> {
        todo!("initialize the Rayon and Tenferro CPU runtimes")
    }
}

impl AnalysisBackend for CpuAnalysisBackend {
    fn lagged_pearson(&self, _values: &[f64], _max_lag: usize) -> AnalysisResult<Vec<f64>> {
        todo!("compute stable lagged Pearson coefficients in parallel")
    }

    fn laplace_grid(
        &self,
        _correlation: &CorrelationSeries,
        _r: &[f64],
        _omega: &[f64],
    ) -> AnalysisResult<LaplaceGrid> {
        todo!("evaluate independent Laplace grid rows in parallel")
    }

    fn linear_least_squares(
        &self,
        _matrix_row_major: &[f64],
        _rows: usize,
        _columns: usize,
        _rhs: &[f64],
        _rank_tolerance: f64,
    ) -> AnalysisResult<Vec<f64>> {
        todo!("solve through tenferro-linalg SVD or pseudoinverse")
    }
}
