//! Device-abstracted CUDA analysis backend with Rayon host orchestration.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::backend::AnalysisBackend;
use crate::integration::simpson_weights;
use crate::model::{CorrelationSeries, LaplaceGrid};
use tenferro_gpu::{download_tensor, upload_tensor, CudaBackend};
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, GraphProgram, Tensor, TracedTensor};

/// Compute placement selected by the CLI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputeDevice {
    Cuda { ordinal: usize },
}

impl ComputeDevice {
    pub const fn cuda(ordinal: usize) -> Self {
        Self::Cuda { ordinal }
    }

    pub fn label(self) -> String {
        match self {
            Self::Cuda { ordinal } => format!("cuda:{ordinal}"),
        }
    }

    const fn ordinal(self) -> usize {
        match self {
            Self::Cuda { ordinal } => ordinal,
        }
    }
}

impl std::fmt::Display for ComputeDevice {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.label())
    }
}

impl std::str::FromStr for ComputeDevice {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (provider, ordinal) = value.split_once(':').unwrap_or((value, "0"));
        if provider != "cuda" {
            return Err(format!(
                "unsupported compute provider {provider:?}; use cuda:N"
            ));
        }
        let ordinal = ordinal
            .parse::<usize>()
            .map_err(|error| format!("invalid CUDA device ordinal {ordinal:?}: {error}"))?;
        Ok(Self::cuda(ordinal))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ShapeKey {
    rows: usize,
    columns: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct LeastSquaresKey {
    rows: usize,
    columns: usize,
    rank_tolerance_bits: u64,
}

struct PearsonGraph {
    left: TracedTensor,
    right: TracedTensor,
    mask: TracedTensor,
    counts: TracedTensor,
    program: GraphProgram,
}

struct LaplaceGraph {
    time: TracedTensor,
    r: TracedTensor,
    omega: TracedTensor,
    weighted_correlation: TracedTensor,
    program: GraphProgram,
}

struct LeastSquaresGraph {
    matrix: TracedTensor,
    right_hand_side: TracedTensor,
    program: GraphProgram,
}

/// CUDA numerical execution behind one device-neutral application boundary.
pub struct DeviceAnalysisBackend {
    pub thread_count: usize,
    pub device: ComputeDevice,
    host_pool: rayon::ThreadPool,
    executor: Mutex<GraphExecutor<CudaBackend>>,
    pearson_graphs: Mutex<HashMap<ShapeKey, Arc<PearsonGraph>>>,
    laplace_graphs: Mutex<HashMap<ShapeKey, Arc<LaplaceGraph>>>,
    least_squares_graphs: Mutex<HashMap<LeastSquaresKey, Arc<LeastSquaresGraph>>>,
}

impl std::fmt::Debug for DeviceAnalysisBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeviceAnalysisBackend")
            .field("thread_count", &self.thread_count)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl DeviceAnalysisBackend {
    /// Initialize one CUDA executor and an independent host Rayon pool.
    pub fn new(thread_count: Option<usize>, device: ComputeDevice) -> Self {
        assert!(thread_count != Some(0), "thread count must be positive");
        let mut builder = rayon::ThreadPoolBuilder::new();
        if let Some(count) = thread_count {
            builder = builder.num_threads(count);
        }
        let host_pool = builder.build().expect("Rayon pool initialization failed");
        let thread_count = host_pool.current_num_threads();
        let cuda = CudaBackend::new(device.ordinal()).expect("CUDA backend initialization failed");
        let mut executor = GraphExecutor::new(cuda);
        executor
            .register_extension(tenferro_linalg::register_runtime)
            .expect("register Tenferro CUDA linear algebra");
        Self {
            thread_count,
            device,
            host_pool,
            executor: Mutex::new(executor),
            pearson_graphs: Mutex::new(HashMap::new()),
            laplace_graphs: Mutex::new(HashMap::new()),
            least_squares_graphs: Mutex::new(HashMap::new()),
        }
    }

    /// Execute host-side orchestration in the configured Rayon pool.
    pub fn install<R: Send>(&self, operation: impl FnOnce() -> R + Send) -> R {
        self.host_pool.install(operation)
    }

    pub fn device_label(&self) -> String {
        self.device.label()
    }

    fn upload(executor: &GraphExecutor<CudaBackend>, tensor: &Tensor) -> Tensor {
        upload_tensor(executor.backend().runtime(), tensor).expect("upload tensor to CUDA")
    }

    fn download(executor: &GraphExecutor<CudaBackend>, tensor: &Tensor) -> Tensor {
        download_tensor(executor.backend().runtime(), tensor).expect("download tensor from CUDA")
    }

    fn pearson_graph(&self, lag_count: usize, sample_count: usize) -> Arc<PearsonGraph> {
        let key = ShapeKey {
            rows: lag_count,
            columns: sample_count,
        };
        let mut cache = self
            .pearson_graphs
            .lock()
            .expect("Pearson graph cache poisoned");
        if let Some(graph) = cache.get(&key) {
            return Arc::clone(graph);
        }
        let left = TracedTensor::input_concrete_shape(DType::F64, &[lag_count, sample_count])
            .expect("create Pearson left input");
        let right = TracedTensor::input_concrete_shape(DType::F64, &[lag_count, sample_count])
            .expect("create Pearson right input");
        let mask = TracedTensor::input_concrete_shape(DType::F64, &[lag_count, sample_count])
            .expect("create Pearson mask input");
        let counts = TracedTensor::input_concrete_shape(DType::F64, &[lag_count, 1])
            .expect("create Pearson count input");
        let left_masked = (&left * &mask).expect("build masked Pearson left input");
        let right_masked = (&right * &mask).expect("build masked Pearson right input");
        let left_sum = left_masked
            .reduce_sum(&[1])
            .expect("build Pearson left sum")
            .reshape(&[lag_count, 1]);
        let right_sum = right_masked
            .reduce_sum(&[1])
            .expect("build Pearson right sum")
            .reshape(&[lag_count, 1]);
        let left_mean = (&left_sum / &counts).expect("build Pearson left mean");
        let right_mean = (&right_sum / &counts).expect("build Pearson right mean");
        let left_centered = (&(&left - &left_mean).expect("center Pearson left") * &mask)
            .expect("mask centered Pearson left");
        let right_centered = (&(&right - &right_mean).expect("center Pearson right") * &mask)
            .expect("mask centered Pearson right");
        let covariance = (&left_centered * &right_centered)
            .expect("build Pearson covariance")
            .reduce_sum(&[1])
            .expect("reduce Pearson covariance");
        let left_variance = (&left_centered * &left_centered)
            .expect("build Pearson left variance")
            .reduce_sum(&[1])
            .expect("reduce Pearson left variance");
        let right_variance = (&right_centered * &right_centered)
            .expect("build Pearson right variance")
            .reduce_sum(&[1])
            .expect("reduce Pearson right variance");
        let denominator = (&left_variance * &right_variance)
            .expect("build Pearson denominator")
            .sqrt();
        let output = (&covariance / &denominator).expect("build Pearson coefficient");
        let mut compiler = GraphCompiler::new();
        let program = compiler
            .compile_with_input_specs(
                &output,
                &[
                    (&left, DType::F64, &[lag_count, sample_count]),
                    (&right, DType::F64, &[lag_count, sample_count]),
                    (&mask, DType::F64, &[lag_count, sample_count]),
                    (&counts, DType::F64, &[lag_count, 1]),
                ],
            )
            .expect("compile CUDA Pearson graph");
        let graph = Arc::new(PearsonGraph {
            left,
            right,
            mask,
            counts,
            program,
        });
        cache.insert(key, Arc::clone(&graph));
        graph
    }

    fn laplace_graph(&self, grid_count: usize, sample_count: usize) -> Arc<LaplaceGraph> {
        let key = ShapeKey {
            rows: grid_count,
            columns: sample_count,
        };
        let mut cache = self
            .laplace_graphs
            .lock()
            .expect("Laplace graph cache poisoned");
        if let Some(graph) = cache.get(&key) {
            return Arc::clone(graph);
        }
        let time = TracedTensor::input_concrete_shape(DType::F64, &[1, sample_count])
            .expect("create Laplace time input");
        let r = TracedTensor::input_concrete_shape(DType::F64, &[grid_count, 1])
            .expect("create Laplace r input");
        let omega = TracedTensor::input_concrete_shape(DType::F64, &[grid_count, 1])
            .expect("create Laplace omega input");
        let weighted_correlation =
            TracedTensor::input_concrete_shape(DType::F64, &[1, sample_count])
                .expect("create weighted correlation input");
        let envelope = (&r * &time).expect("build Laplace envelope exponent").exp();
        let phase = (&omega * &time).expect("build Laplace phase");
        let weighted_envelope =
            (&envelope * &weighted_correlation).expect("build weighted Laplace envelope");
        let real = (&weighted_envelope * &phase.cos())
            .expect("build real Laplace integrand")
            .reduce_sum(&[1])
            .expect("reduce real Laplace grid");
        let imaginary = (&weighted_envelope * &phase.sin())
            .expect("build imaginary Laplace integrand")
            .reduce_sum(&[1])
            .expect("reduce imaginary Laplace grid");
        let mut compiler = GraphCompiler::new();
        let program = compiler
            .compile_many(&[&real, &imaginary])
            .expect("compile CUDA Laplace graph");
        let graph = Arc::new(LaplaceGraph {
            time,
            r,
            omega,
            weighted_correlation,
            program,
        });
        cache.insert(key, Arc::clone(&graph));
        graph
    }

    fn least_squares_graph(
        &self,
        rows: usize,
        columns: usize,
        rank_tolerance: f64,
    ) -> Arc<LeastSquaresGraph> {
        let key = LeastSquaresKey {
            rows,
            columns,
            rank_tolerance_bits: rank_tolerance.to_bits(),
        };
        let mut cache = self
            .least_squares_graphs
            .lock()
            .expect("least-squares graph cache poisoned");
        if let Some(graph) = cache.get(&key) {
            return Arc::clone(graph);
        }
        let matrix = TracedTensor::input_concrete_shape(DType::F64, &[rows, columns])
            .expect("create least-squares matrix input");
        let right_hand_side = TracedTensor::input_concrete_shape(DType::F64, &[rows, 1])
            .expect("create least-squares right-hand side input");
        let solution = matrix
            .pinv_with_rtol(rank_tolerance)
            .expect("build least-squares pseudoinverse")
            .matmul(&right_hand_side)
            .expect("build least-squares product");
        let mut compiler = GraphCompiler::new();
        let program = compiler
            .compile_with_input_specs(
                &solution,
                &[
                    (&matrix, DType::F64, &[rows, columns]),
                    (&right_hand_side, DType::F64, &[rows, 1]),
                ],
            )
            .expect("compile CUDA least-squares graph");
        let graph = Arc::new(LeastSquaresGraph {
            matrix,
            right_hand_side,
            program,
        });
        cache.insert(key, Arc::clone(&graph));
        graph
    }
}

impl AnalysisBackend for DeviceAnalysisBackend {
    fn lagged_pearson(&self, values: &[f64], max_lag: usize) -> Vec<f64> {
        assert!(values.len() >= 2, "Pearson needs two samples");
        assert!(
            max_lag <= values.len() - 2,
            "lag leaves fewer than two pairs"
        );
        assert!(
            values.iter().all(|value| value.is_finite()),
            "bad Pearson input"
        );
        let sample_count = values.len();
        let lag_count = max_lag + 1;
        let mut left = vec![0.0; lag_count * sample_count];
        let mut right = vec![0.0; lag_count * sample_count];
        let mut mask = vec![0.0; lag_count * sample_count];
        let counts = (0..lag_count)
            .map(|lag| (sample_count - lag) as f64)
            .collect::<Vec<_>>();
        for sample in 0..sample_count {
            for lag in 0..lag_count {
                if sample + lag < sample_count {
                    let index = lag + sample * lag_count;
                    left[index] = values[sample];
                    right[index] = values[sample + lag];
                    mask[index] = 1.0;
                }
            }
        }
        let left = Tensor::from_vec_col_major(vec![lag_count, sample_count], left)
            .expect("create Pearson left tensor");
        let right = Tensor::from_vec_col_major(vec![lag_count, sample_count], right)
            .expect("create Pearson right tensor");
        let mask = Tensor::from_vec_col_major(vec![lag_count, sample_count], mask)
            .expect("create Pearson mask tensor");
        let counts = Tensor::from_vec_col_major(vec![lag_count, 1], counts)
            .expect("create Pearson count tensor");
        let graph = self.pearson_graph(lag_count, sample_count);
        let mut executor = self.executor.lock().expect("CUDA executor poisoned");
        let left = Self::upload(&executor, &left);
        let right = Self::upload(&executor, &right);
        let mask = Self::upload(&executor, &mask);
        let counts = Self::upload(&executor, &counts);
        let output = executor
            .run_with_inputs(
                &graph.program,
                &[
                    (&graph.left, &left),
                    (&graph.right, &right),
                    (&graph.mask, &mask),
                    (&graph.counts, &counts),
                ],
            )
            .expect("execute CUDA Pearson graph");
        let output = Self::download(&executor, &output);
        let (_, mut coefficients) = output
            .into_vec_col_major::<f64>()
            .expect("read CUDA Pearson result");
        assert_eq!(coefficients.len(), lag_count, "bad Pearson result shape");
        assert!(
            coefficients.iter().all(|value| value.is_finite()),
            "constant Pearson input"
        );
        coefficients
            .iter_mut()
            .for_each(|value| *value = value.clamp(-1.0, 1.0));
        coefficients[0] = 1.0;
        coefficients
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
        assert!(!r.is_empty() && !omega.is_empty(), "Laplace grid is empty");
        let spacing = correlation.lag_times[1] - correlation.lag_times[0];
        let weights = simpson_weights(sample_count, spacing);
        let weighted = weights
            .iter()
            .zip(&correlation.pearson_mean)
            .map(|(&weight, &value)| weight * value)
            .collect::<Vec<_>>();
        let grid_count = r.len() * omega.len();
        let mut r_values = Vec::with_capacity(grid_count);
        let mut omega_values = Vec::with_capacity(grid_count);
        for &omega_value in omega {
            for &r_value in r {
                r_values.push(r_value);
                omega_values.push(omega_value);
            }
        }
        let time = Tensor::from_vec_col_major(vec![1, sample_count], correlation.lag_times.clone())
            .expect("create Laplace time tensor");
        let r_tensor = Tensor::from_vec_col_major(vec![grid_count, 1], r_values)
            .expect("create Laplace r tensor");
        let omega_tensor = Tensor::from_vec_col_major(vec![grid_count, 1], omega_values)
            .expect("create Laplace omega tensor");
        let weighted = Tensor::from_vec_col_major(vec![1, sample_count], weighted)
            .expect("create weighted correlation tensor");
        let graph = self.laplace_graph(grid_count, sample_count);
        let mut executor = self.executor.lock().expect("CUDA executor poisoned");
        let time = Self::upload(&executor, &time);
        let r_tensor = Self::upload(&executor, &r_tensor);
        let omega_tensor = Self::upload(&executor, &omega_tensor);
        let weighted = Self::upload(&executor, &weighted);
        let outputs = executor
            .run_many_with_inputs(
                &graph.program,
                &[
                    (&graph.time, &time),
                    (&graph.r, &r_tensor),
                    (&graph.omega, &omega_tensor),
                    (&graph.weighted_correlation, &weighted),
                ],
            )
            .expect("execute CUDA Laplace graph");
        assert_eq!(outputs.len(), 2, "bad CUDA Laplace output count");
        let real = Self::download(&executor, &outputs[0]);
        let imaginary = Self::download(&executor, &outputs[1]);
        let (_, real) = real
            .into_vec_col_major::<f64>()
            .expect("read real CUDA Laplace result");
        let (_, imaginary) = imaginary
            .into_vec_col_major::<f64>()
            .expect("read imaginary CUDA Laplace result");
        assert_eq!(real.len(), grid_count, "bad real Laplace shape");
        assert_eq!(imaginary.len(), grid_count, "bad imaginary Laplace shape");
        let values = real
            .into_iter()
            .zip(imaginary)
            .map(|(real, imaginary)| {
                let value = num_complex::Complex64::new(real, imaginary);
                assert!(
                    value.re.is_finite() && value.im.is_finite(),
                    "bad Laplace result"
                );
                value
            })
            .collect();
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
        assert!(rows >= columns && columns > 0, "bad least-squares shape");
        assert_eq!(matrix_row_major.len(), rows * columns, "bad matrix length");
        assert_eq!(rhs.len(), rows, "bad right-hand side length");
        assert!(
            matrix_row_major.iter().all(|value| value.is_finite()),
            "bad matrix"
        );
        assert!(
            rhs.iter().all(|value| value.is_finite()),
            "bad right-hand side"
        );
        assert!(
            rank_tolerance.is_finite() && rank_tolerance >= 0.0,
            "bad rank tolerance"
        );
        let mut matrix_column_major = vec![0.0; matrix_row_major.len()];
        for row in 0..rows {
            for column in 0..columns {
                matrix_column_major[column * rows + row] = matrix_row_major[row * columns + column];
            }
        }
        let matrix = Tensor::from_vec_col_major(vec![rows, columns], matrix_column_major)
            .expect("create least-squares matrix");
        let right_hand_side = Tensor::from_vec_col_major(vec![rows, 1], rhs.to_vec())
            .expect("create least-squares right-hand side");
        let graph = self.least_squares_graph(rows, columns, rank_tolerance);
        let mut executor = self.executor.lock().expect("CUDA executor poisoned");
        let matrix = Self::upload(&executor, &matrix);
        let right_hand_side = Self::upload(&executor, &right_hand_side);
        let output = executor
            .run_with_inputs(
                &graph.program,
                &[
                    (&graph.matrix, &matrix),
                    (&graph.right_hand_side, &right_hand_side),
                ],
            )
            .expect("solve CUDA least-squares system");
        let output = Self::download(&executor, &output);
        let (_, values) = output
            .into_vec_col_major::<f64>()
            .expect("read CUDA least-squares solution");
        assert_eq!(values.len(), columns, "bad least-squares solution shape");
        assert!(
            values.iter().all(|value| value.is_finite()),
            "bad least-squares solution"
        );
        values
    }
}
