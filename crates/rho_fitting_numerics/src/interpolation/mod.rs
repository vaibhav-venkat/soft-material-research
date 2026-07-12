use ndarray::{Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD, IxDyn};
use rayon::prelude::*;

use rho_fitting_types::{CoreError, CoreResult};

#[derive(Clone)]
pub struct RadialTransfer {
    matrix: Array2<f64>,
    source_len: usize,
    target_len: usize,
}

impl RadialTransfer {
    pub fn new(source: ArrayView1<'_, f64>, target: ArrayView1<'_, f64>) -> CoreResult<Self> {
        let matrix = barycentric_matrix(source, target)?;
        Ok(Self {
            source_len: source.len(),
            target_len: target.len(),
            matrix,
        })
    }

    pub fn from_matrix(matrix: ArrayView2<'_, f64>) -> CoreResult<Self> {
        if matrix.nrows() == 0
            || matrix.ncols() < 2
            || !matrix.iter().all(|value| value.is_finite())
        {
            return Err(CoreError::InvalidInput(
                "radial transfer matrix must be finite with shape (target, source>=2)".to_string(),
            ));
        }
        Ok(Self {
            source_len: matrix.ncols(),
            target_len: matrix.nrows(),
            matrix: matrix.to_owned(),
        })
    }

    pub fn matrix(&self) -> &Array2<f64> {
        &self.matrix
    }

    pub fn apply(&self, values: ArrayViewD<'_, f64>, axis: usize) -> CoreResult<ArrayD<f64>> {
        if axis >= values.ndim() {
            return Err(CoreError::Shape(format!(
                "radial transfer axis {axis} is out of range for rank {}",
                values.ndim()
            )));
        }
        if values.shape()[axis] != self.source_len {
            return Err(CoreError::Shape(format!(
                "radial axis has length {}; expected {}",
                values.shape()[axis],
                self.source_len
            )));
        }
        let input = values.to_owned();
        let input_slice = input.as_slice().ok_or_else(|| {
            CoreError::InvalidInput("failed to create contiguous radial input".to_string())
        })?;
        let before = values.shape()[..axis].iter().product::<usize>().max(1);
        let after = values.shape()[axis + 1..].iter().product::<usize>().max(1);
        let mut output = vec![0.0; before * self.target_len * after];
        output
            .par_chunks_mut(self.target_len * after)
            .enumerate()
            .for_each(|(outer, block)| {
                let input_base = outer * self.source_len * after;
                for target in 0..self.target_len {
                    for trailing in 0..after {
                        block[target * after + trailing] = (0..self.source_len)
                            .map(|source| {
                                self.matrix[[target, source]]
                                    * input_slice[input_base + source * after + trailing]
                            })
                            .sum();
                    }
                }
            });
        let mut shape = values.shape().to_vec();
        shape[axis] = self.target_len;
        ArrayD::from_shape_vec(IxDyn(&shape), output).map_err(|error| {
            CoreError::InvalidInput(format!("radial transfer assembly failed: {error}"))
        })
    }
}

pub fn barycentric_matrix(
    source: ArrayView1<'_, f64>,
    target: ArrayView1<'_, f64>,
) -> CoreResult<Array2<f64>> {
    if source.len() < 2 || target.is_empty() {
        return Err(CoreError::InvalidInput(
            "barycentric nodes require at least two source points and one target point".to_string(),
        ));
    }
    if !source
        .iter()
        .chain(target.iter())
        .all(|value| value.is_finite())
    {
        return Err(CoreError::InvalidInput(
            "barycentric nodes must be finite".to_string(),
        ));
    }
    let center = 0.5 * (source[0] + source[source.len() - 1]);
    let scale = source
        .iter()
        .map(|value| (value - center).abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err(CoreError::InvalidInput(
            "barycentric source nodes must be distinct".to_string(),
        ));
    }
    let normalized = source.mapv(|value| (value - center) / scale);
    let mut signs = vec![1.0; source.len()];
    let mut log_magnitudes = vec![0.0; source.len()];
    for index in 0..source.len() {
        for other in 0..source.len() {
            if index == other {
                continue;
            }
            let difference = normalized[index] - normalized[other];
            if difference == 0.0 {
                return Err(CoreError::InvalidInput(
                    "barycentric source nodes must be distinct".to_string(),
                ));
            }
            signs[index] *= difference.signum();
            log_magnitudes[index] -= difference.abs().ln();
        }
    }
    let max_log = log_magnitudes
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let weights = signs
        .iter()
        .zip(log_magnitudes.iter())
        .map(|(sign, log_value)| sign * (log_value - max_log).exp())
        .collect::<Vec<_>>();
    let mut matrix = Array2::zeros((target.len(), source.len()));
    for (row, point) in target.iter().enumerate() {
        if let Some(exact) = source
            .iter()
            .position(|value| (*value - *point).abs() <= 1.0e-13)
        {
            matrix[[row, exact]] = 1.0;
            continue;
        }
        let scaled_point = (*point - center) / scale;
        let denominator = weights
            .iter()
            .enumerate()
            .map(|(index, weight)| weight / (scaled_point - normalized[index]))
            .sum::<f64>();
        if !denominator.is_finite() || denominator == 0.0 {
            return Err(CoreError::InvalidInput(
                "barycentric interpolation produced a singular row".to_string(),
            ));
        }
        for index in 0..source.len() {
            matrix[[row, index]] =
                (weights[index] / (scaled_point - normalized[index])) / denominator;
        }
        let row_sum = matrix.row(row).sum();
        matrix.row_mut(row).mapv_inplace(|value| value / row_sum);
    }
    Ok(matrix)
}
