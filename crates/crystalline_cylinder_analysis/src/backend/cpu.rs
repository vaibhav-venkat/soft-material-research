//! Rayon and Tenferro CPU backend declaration.

use crate::backend::AnalysisBackend;
use crate::correlation::pearson;
use crate::integration::simpson_weights;
use crate::model::{CorrelationSeries, LaplaceGrid};
use rayon::prelude::*;
use tenferro_cpu::{CpuBackend, CpuContext};
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};

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
            Some(0) => panic!("thread count must be positive"),
            Some(count) => CpuContext::with_threads(count),
            None => CpuContext::try_from_env(),
        }
        .expect("CPU backend initialization failed");

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
    fn lagged_pearson(&self, values: &[f64], max_lag: usize) -> Vec<f64> {
        assert!(values.len() >= 2, "Pearson needs two samples");
        assert!(
            max_lag <= values.len() - 2,
            "lag leaves fewer than two pairs"
        );
        self.install(|| {
            (0..=max_lag)
                .into_par_iter()
                .map(|lag| {
                    if lag == 0 {
                        let _ = pearson(values, values);
                        1.0
                    } else {
                        pearson(&values[..values.len() - lag], &values[lag..])
                    }
                })
                .collect()
        })
    }

    fn laplace_grid(
        &self,
        correlation: &CorrelationSeries,
        r: &[f64],
        omega: &[f64],
    ) -> LaplaceGrid {
        let sample_count = correlation.lag_times.len();
        assert_eq!(
            correlation.pearson_mean.len(),
            sample_count,
            "correlation lengths differ"
        );
        assert!(sample_count >= 2, "Laplace transform needs two lags");
        assert!(!r.is_empty(), "r grid is empty");
        assert!(!omega.is_empty(), "omega grid is empty");
        let spacing = correlation.lag_times[1] - correlation.lag_times[0];
        let weights = simpson_weights(sample_count, spacing);
        let column_count = r.len();
        let values = self.install(|| {
            (0..omega.len() * column_count)
                .into_par_iter()
                .map(|flat_index| {
                    let omega_value = omega[flat_index / column_count];
                    let r_value = r[flat_index % column_count];
                    let mut real = 0.0;
                    let mut real_compensation = 0.0;
                    let mut imaginary = 0.0;
                    let mut imaginary_compensation = 0.0;
                    for ((&time, &correlation_value), &weight) in correlation
                        .lag_times
                        .iter()
                        .zip(&correlation.pearson_mean)
                        .zip(&weights)
                    {
                        let envelope = (r_value * time).exp();
                        assert!(envelope.is_finite(), "Laplace exponential overflow");
                        let (sine, cosine) = (omega_value * time).sin_cos();
                        compensated_add(
                            weight * correlation_value * envelope * cosine,
                            &mut real,
                            &mut real_compensation,
                        );
                        compensated_add(
                            weight * correlation_value * envelope * sine,
                            &mut imaginary,
                            &mut imaginary_compensation,
                        );
                    }
                    let value = num_complex::Complex64::new(real, imaginary);
                    assert!(
                        value.re.is_finite() && value.im.is_finite(),
                        "Laplace result is non-finite"
                    );
                    value
                })
                .collect()
        });
        LaplaceGrid {
            r: r.to_vec(),
            omega: omega.to_vec(),
            values,
            shape: [omega.len(), r.len()],
        }
    }

    fn linear_least_squares(
        &self,
        matrix_row_major: &[f64],
        rows: usize,
        columns: usize,
        rhs: &[f64],
        rank_tolerance: f64,
    ) -> Vec<f64> {
        assert!(rows >= columns, "least-squares matrix is too short");
        assert!(columns > 0, "least-squares matrix has no columns");
        assert_eq!(
            matrix_row_major.len(),
            rows * columns,
            "matrix shape differs"
        );
        assert_eq!(rhs.len(), rows, "right-hand side shape differs");
        assert!(
            matrix_row_major.iter().all(|value| value.is_finite()),
            "least-squares matrix is non-finite"
        );
        assert!(
            rhs.iter().all(|value| value.is_finite()),
            "least-squares right-hand side is non-finite"
        );
        assert!(
            rank_tolerance.is_finite() && rank_tolerance >= 0.0,
            "rank tolerance is invalid"
        );

        // Tenferro/faer is column-major. Compact only this small optimizer
        // matrix; trajectory and correlation storage remains in its native order.
        let mut matrix_column_major = vec![0.0; matrix_row_major.len()];
        for row in 0..rows {
            for column in 0..columns {
                matrix_column_major[column * rows + row] = matrix_row_major[row * columns + column];
            }
        }
        let matrix = TracedTensor::from_vec_col_major(vec![rows, columns], matrix_column_major)
            .expect("create least-squares matrix");
        let right_hand_side = TracedTensor::from_vec_col_major(vec![rows, 1], rhs.to_vec())
            .expect("create least-squares right-hand side");
        let solution = matrix
            .pinv_with_rtol(rank_tolerance)
            .expect("build least-squares pseudoinverse")
            .matmul(&right_hand_side)
            .expect("build least-squares product");
        let mut compiler = GraphCompiler::new();
        let program = compiler
            .compile(&solution)
            .expect("compile least-squares graph");
        let mut executor = GraphExecutor::new(CpuBackend::new());
        executor
            .register_extension(tenferro_linalg::register_runtime)
            .expect("register Tenferro linear algebra");
        let output = self.install(|| executor.run(&program).expect("solve least-squares system"));
        let (_, values) = output
            .into_vec_col_major::<f64>()
            .expect("read least-squares solution");
        assert_eq!(values.len(), columns, "solution shape differs");
        assert!(
            values.iter().all(|value| value.is_finite()),
            "least-squares solution is non-finite"
        );
        values
    }
}

fn compensated_add(value: f64, sum: &mut f64, compensation: &mut f64) {
    let corrected = value - *compensation;
    let updated = *sum + corrected;
    *compensation = (updated - *sum) - corrected;
    *sum = updated;
}
