//! Rayon and Tenferro CPU backend declaration.

use crate::backend::AnalysisBackend;
use crate::model::{CorrelationSeries, LaplaceGrid};
use tenferro_cpu::CpuContext;

/// CPU backend backed by Rayon and Tenferro's faer provider.
#[derive(Clone, Debug)]
pub struct CpuAnalysisBackend {
    pub thread_count: usize,
    context: CpuContext,
}

impl CpuAnalysisBackend {
    /// Construct a CPU backend with an optional thread limit.
    pub fn new(thread_count: Option<usize>) -> Self {
        let context = match thread_count {
            Some(0) => panic!("bad threads"),
            Some(count) => CpuContext::with_threads(count),
            None => CpuContext::try_from_env(),
        }
        .expect("bad CPU");

        Self {
            thread_count: context.num_threads(),
            context,
        }
    }

    /// Execute work inside the backend's configured Rayon pool.
    pub fn install<R: Send>(&self, operation: impl FnOnce() -> R + Send) -> R {
        self.context.install(operation)
    }
}

impl AnalysisBackend for CpuAnalysisBackend {
    fn lagged_pearson(&self, _values: &[f64], _max_lag: usize) -> Vec<f64> {
        todo!("compute stable lagged Pearson coefficients in parallel")
    }

    fn laplace_grid(
        &self,
        _correlation: &CorrelationSeries,
        _r: &[f64],
        _omega: &[f64],
    ) -> LaplaceGrid {
        todo!("evaluate independent Laplace grid rows in parallel")
    }

    fn linear_least_squares(
        &self,
        _matrix_row_major: &[f64],
        _rows: usize,
        _columns: usize,
        _rhs: &[f64],
        _rank_tolerance: f64,
    ) -> Vec<f64> {
        todo!("solve through tenferro-linalg SVD or pseudoinverse")
    }
}
