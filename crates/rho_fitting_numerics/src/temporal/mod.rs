use faer::linalg::solvers::SolveLstsq;
use faer::Mat;
use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayViewD, IxDyn};
use rayon::prelude::*;

use rho_fitting_types::{CoreError, CoreResult};

pub fn power_spectrum(coefficients: &[ArrayViewD<'_, f64>]) -> CoreResult<Array1<f64>> {
    let Some(first) = coefficients.first() else {
        return Err(CoreError::InvalidInput(
            "at least one coefficient field is required".to_string(),
        ));
    };
    if first.ndim() == 0 {
        return Err(CoreError::Shape(
            "coefficient fields must have a leading mode axis".to_string(),
        ));
    }
    let modes = first.shape()[0];
    if coefficients
        .iter()
        .any(|values| values.ndim() == 0 || values.shape()[0] != modes)
    {
        return Err(CoreError::Shape(
            "coefficient fields must share the leading mode axis".to_string(),
        ));
    }
    let power = (0..modes)
        .into_par_iter()
        .map(|mode| {
            coefficients
                .iter()
                .map(|values| {
                    values
                        .index_axis(ndarray::Axis(0), mode)
                        .iter()
                        .map(|value| value * value)
                        .sum::<f64>()
                })
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    Ok(Array1::from_vec(power))
}

/// Time-grid operators shared by every spatial field in one fitting run.
pub struct TemporalOperators {
    pub times: Array1<f64>,
    pub scaled_times: Array1<f64>,
    pub diagnostic_nodes: Array1<f64>,
    pub fit_operator: Array2<f64>,
    pub diagnostic_fit_operator: Array2<f64>,
    pub diagnostic_operator: Array2<f64>,
    pub evaluation_operator: Array2<f64>,
    pub derivative_operator: Array2<f64>,
    pub cutoff: usize,
}

pub struct TemporalFieldResult {
    pub cleaned: ArrayD<f64>,
    pub filtered: ArrayD<f64>,
    pub derivative: ArrayD<f64>,
    pub coefficients: ArrayD<f64>,
}

impl TemporalOperators {
    pub fn new(steps: ArrayView1<'_, i64>, timestep: f64, cutoff: usize) -> CoreResult<Self> {
        if steps.is_empty() {
            return Err(CoreError::InvalidInput(
                "steps must be a non-empty 1D array".to_string(),
            ));
        }
        if !(timestep.is_finite() && timestep > 0.0) {
            return Err(CoreError::InvalidInput(
                "timestep must be finite and positive".to_string(),
            ));
        }
        if cutoff == 0 {
            return Err(CoreError::InvalidInput(
                "cheb_cutoff must be positive".to_string(),
            ));
        }
        let frame_count = steps.len();
        let cutoff = cutoff.min(frame_count);
        let times = Array1::from_iter(
            steps
                .iter()
                .map(|step| (*step - steps[0]) as f64 * timestep),
        );
        for pair in times.windows(2) {
            if !(pair[1] > pair[0]) {
                return Err(CoreError::InvalidInput(
                    "steps must increase strictly over time".to_string(),
                ));
            }
        }
        let (scaled_times, half_span) = if frame_count == 1 {
            (Array1::zeros(1), 1.0)
        } else {
            let span = times[frame_count - 1] - times[0];
            let center = 0.5 * (times[0] + times[frame_count - 1]);
            (
                times.mapv(|time| (time - center) / (0.5 * span)),
                0.5 * span,
            )
        };
        let fit_operator = coefficient_operator(&scaled_times, cutoff - 1)?;
        let diagnostic_nodes = if frame_count == 1 {
            Array1::zeros(1)
        } else {
            Array1::from_iter(
                (0..frame_count)
                    .map(|index| {
                        (std::f64::consts::PI * index as f64 / (frame_count - 1) as f64).cos()
                    })
                    .rev(),
            )
        };
        let diagnostic_fit_operator = coefficient_operator(&diagnostic_nodes, frame_count - 1)?;
        let diagnostic_resample_operator =
            cubic_resample_operator(&scaled_times, &diagnostic_nodes)?;
        let diagnostic_operator = diagnostic_fit_operator.dot(&diagnostic_resample_operator);
        let evaluation_operator = evaluation_matrix(&scaled_times, cutoff - 1, false, 1.0);
        let derivative_basis = evaluation_matrix(&scaled_times, cutoff - 1, true, half_span);
        let derivative_operator = derivative_basis.dot(&fit_operator);
        Ok(Self {
            times,
            scaled_times,
            diagnostic_nodes,
            fit_operator,
            diagnostic_fit_operator,
            diagnostic_operator,
            evaluation_operator,
            derivative_operator,
            cutoff,
        })
    }

    pub fn apply(&self, values: ArrayViewD<'_, f64>) -> CoreResult<TemporalFieldResult> {
        if values.ndim() == 0 || values.shape()[0] != self.times.len() {
            return Err(CoreError::Shape(format!(
                "values must have frame axis of length {}; got {:?}",
                self.times.len(),
                values.shape()
            )));
        }
        let shape = values.shape().to_vec();
        let columns = shape[1..].iter().product::<usize>().max(1);
        let frames = self.times.len();
        let cleaned = cleaned_columns(values, &self.times);
        let mut fit_coefficients = vec![0.0; self.cutoff * columns];
        let mut filtered = vec![0.0; frames * columns];
        let mut derivative = vec![0.0; frames * columns];

        fit_coefficients
            .par_chunks_mut(self.cutoff)
            .enumerate()
            .for_each(|(column, output)| {
                for mode in 0..self.cutoff {
                    output[mode] = (0..frames)
                        .map(|frame| {
                            self.fit_operator[[mode, frame]] * cleaned[column * frames + frame]
                        })
                        .sum();
                }
            });
        filtered
            .par_chunks_mut(columns)
            .enumerate()
            .for_each(|(frame, output)| {
                for column in 0..columns {
                    output[column] = (0..self.cutoff)
                        .map(|mode| {
                            self.evaluation_operator[[frame, mode]]
                                * fit_coefficients[column * self.cutoff + mode]
                        })
                        .sum();
                }
            });
        derivative
            .par_chunks_mut(columns)
            .enumerate()
            .for_each(|(frame, output)| {
                for column in 0..columns {
                    output[column] = (0..frames)
                        .map(|source| {
                            self.derivative_operator[[frame, source]]
                                * cleaned[column * frames + source]
                        })
                        .sum();
                }
            });

        let mut diagnostic_coefficients = vec![0.0; frames * columns];
        diagnostic_coefficients
            .par_chunks_mut(columns)
            .enumerate()
            .for_each(|(mode, output)| {
                for column in 0..columns {
                    output[column] = (0..frames)
                        .map(|frame| {
                            self.diagnostic_operator[[mode, frame]]
                                * cleaned[column * frames + frame]
                        })
                        .sum();
                }
            });

        let mut coefficient_shape = vec![frames];
        coefficient_shape.extend_from_slice(&shape[1..]);
        let mut cleaned_output = vec![0.0; frames * columns];
        for column in 0..columns {
            for frame in 0..frames {
                cleaned_output[frame * columns + column] = cleaned[column * frames + frame];
            }
        }
        let filtered = ArrayD::from_shape_vec(IxDyn(&shape), filtered).map_err(|error| {
            CoreError::InvalidInput(format!("filtered field assembly failed: {error}"))
        })?;
        let derivative = ArrayD::from_shape_vec(IxDyn(&shape), derivative).map_err(|error| {
            CoreError::InvalidInput(format!("derivative field assembly failed: {error}"))
        })?;
        let coefficients =
            ArrayD::from_shape_vec(IxDyn(&coefficient_shape), diagnostic_coefficients).map_err(
                |error| CoreError::InvalidInput(format!("coefficient assembly failed: {error}")),
            )?;
        Ok(TemporalFieldResult {
            cleaned: ArrayD::from_shape_vec(IxDyn(&shape), cleaned_output).map_err(|error| {
                CoreError::InvalidInput(format!("cleaned field assembly failed: {error}"))
            })?,
            filtered,
            derivative,
            coefficients,
        })
    }

    pub fn diagnostic_coefficients(
        &self,
        values_at_nodes: ArrayViewD<'_, f64>,
    ) -> CoreResult<ArrayD<f64>> {
        if values_at_nodes.ndim() == 0 || values_at_nodes.shape()[0] != self.times.len() {
            return Err(CoreError::Shape(
                "diagnostic values must match the temporal frame count".to_string(),
            ));
        }
        let shape = values_at_nodes.shape().to_vec();
        let columns = shape[1..].iter().product::<usize>().max(1);
        let frames = self.times.len();
        let mut columns_data = vec![0.0; columns * frames];
        for frame in 0..frames {
            for column in 0..columns {
                let mut index = vec![frame];
                index.extend(unravel(column, &shape[1..]));
                columns_data[column * frames + frame] = values_at_nodes[IxDyn(&index)];
            }
        }
        let mut output = vec![0.0; columns * frames];
        output
            .par_chunks_mut(frames)
            .enumerate()
            .for_each(|(column, coefficients)| {
                for mode in 0..frames {
                    coefficients[mode] = (0..frames)
                        .map(|frame| {
                            self.diagnostic_fit_operator[[mode, frame]]
                                * columns_data[column * frames + frame]
                        })
                        .sum();
                }
            });
        let mut coefficient_output = vec![0.0; columns * frames];
        for column in 0..columns {
            for mode in 0..frames {
                coefficient_output[mode * columns + column] = output[column * frames + mode];
            }
        }
        let mut coefficient_shape = vec![frames];
        coefficient_shape.extend_from_slice(&shape[1..]);
        ArrayD::from_shape_vec(IxDyn(&coefficient_shape), coefficient_output).map_err(|error| {
            CoreError::InvalidInput(format!("diagnostic coefficient assembly failed: {error}"))
        })
    }
}

fn cubic_resample_operator(source: &Array1<f64>, target: &Array1<f64>) -> CoreResult<Array2<f64>> {
    let n = source.len();
    if n == 1 {
        return Ok(Array2::ones((target.len(), 1)));
    }
    if n == 2 {
        let span = source[1] - source[0];
        return Ok(Array2::from_shape_fn((target.len(), 2), |(row, col)| {
            let weight = (target[row] - source[0]) / span;
            if col == 0 {
                1.0 - weight
            } else {
                weight
            }
        }));
    }
    if n == 3 {
        return Ok(Array2::from_shape_fn((target.len(), 3), |(row, col)| {
            let x = target[row];
            (0..3)
                .filter(|other| *other != col)
                .map(|other| (x - source[other]) / (source[col] - source[other]))
                .product()
        }));
    }

    // Solve the not-a-knot cubic-spline system once for every unit sample vector.
    let mut system = Mat::<f64>::zeros(n, n);
    let mut rhs = Mat::<f64>::zeros(n, n);
    let h = (0..n - 1)
        .map(|index| source[index + 1] - source[index])
        .collect::<Vec<_>>();
    system[(0, 0)] = -h[1];
    system[(0, 1)] = h[0] + h[1];
    system[(0, 2)] = -h[0];
    for row in 1..n - 1 {
        system[(row, row - 1)] = h[row - 1];
        system[(row, row)] = 2.0 * (h[row - 1] + h[row]);
        system[(row, row + 1)] = h[row];
        rhs[(row, row - 1)] = 6.0 / h[row - 1];
        rhs[(row, row)] = -6.0 * (1.0 / h[row - 1] + 1.0 / h[row]);
        rhs[(row, row + 1)] = 6.0 / h[row];
    }
    system[(n - 1, n - 3)] = -h[n - 2];
    system[(n - 1, n - 2)] = h[n - 3] + h[n - 2];
    system[(n - 1, n - 1)] = -h[n - 3];
    let second = system.as_ref().qr().solve_lstsq(rhs);

    Ok(Array2::from_shape_fn((target.len(), n), |(row, column)| {
        let x = target[row];
        let interval = source
            .iter()
            .position(|value| *value > x)
            .map(|index| index.saturating_sub(1))
            .unwrap_or(n - 2)
            .min(n - 2);
        let width = h[interval];
        let a = (source[interval + 1] - x) / width;
        let b = (x - source[interval]) / width;
        let direct = if column == interval {
            a
        } else if column == interval + 1 {
            b
        } else {
            0.0
        };
        direct
            + ((a * a * a - a) * second[(interval, column)]
                + (b * b * b - b) * second[(interval + 1, column)])
                * width
                * width
                / 6.0
    }))
}

fn cleaned_columns(values: ArrayViewD<'_, f64>, times: &Array1<f64>) -> Vec<f64> {
    let shape = values.shape();
    let frames = shape[0];
    let columns = shape[1..].iter().product::<usize>().max(1);
    let mut output = vec![0.0; columns * frames];
    for frame in 0..frames {
        for column in 0..columns {
            let mut index = vec![frame];
            index.extend(unravel(column, &shape[1..]));
            output[column * frames + frame] = values[IxDyn(&index)];
        }
    }
    output
        .par_chunks_mut(frames)
        .for_each(|series| fill_nonfinite(series, times));
    output
}

fn fill_nonfinite(series: &mut [f64], times: &Array1<f64>) {
    let finite = series
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.is_finite().then_some(index))
        .collect::<Vec<_>>();
    match finite.as_slice() {
        [] => series.fill(0.0),
        [index] => {
            let value = series[*index];
            series.fill(value);
        }
        _ => {
            for index in 0..series.len() {
                if series[index].is_finite() {
                    continue;
                }
                let upper = finite.partition_point(|known| *known < index);
                if upper == 0 {
                    series[index] = series[finite[0]];
                } else if upper == finite.len() {
                    series[index] = series[*finite.last().unwrap()];
                } else {
                    let left = finite[upper - 1];
                    let right = finite[upper];
                    let fraction = (times[index] - times[left]) / (times[right] - times[left]);
                    series[index] = series[left] + fraction * (series[right] - series[left]);
                }
            }
        }
    }
}

fn coefficient_operator(sampled_times: &Array1<f64>, degree: usize) -> CoreResult<Array2<f64>> {
    let rows = sampled_times.len();
    let columns = degree + 1;
    let vandermonde = Mat::from_fn(rows, columns, |row, column| {
        chebyshev_value(column, sampled_times[row])
    });
    let identity = Mat::<f64>::identity(rows, rows);
    let solution = vandermonde.as_ref().qr().solve_lstsq(identity);
    Ok(Array2::from_shape_fn((columns, rows), |(row, column)| {
        solution[(row, column)]
    }))
}

fn evaluation_matrix(
    times: &Array1<f64>,
    degree: usize,
    derivative: bool,
    scale: f64,
) -> Array2<f64> {
    Array2::from_shape_fn((times.len(), degree + 1), |(row, column)| {
        if derivative {
            chebyshev_derivative_value(column, times[row]) / scale
        } else {
            chebyshev_value(column, times[row])
        }
    })
}

fn chebyshev_value(mode: usize, x: f64) -> f64 {
    match mode {
        0 => 1.0,
        1 => x,
        _ => {
            let (mut previous, mut current) = (1.0, x);
            for _ in 2..=mode {
                let next = 2.0 * x * current - previous;
                previous = current;
                current = next;
            }
            current
        }
    }
}

fn chebyshev_derivative_value(mode: usize, x: f64) -> f64 {
    if mode == 0 {
        return 0.0;
    }
    let sine_squared = (1.0 - x * x).max(0.0);
    if sine_squared > 1.0e-14 {
        let theta = x.acos();
        mode as f64 * (mode as f64 * theta).sin() / sine_squared.sqrt()
    } else {
        let endpoint: f64 = if x >= 0.0 { 1.0 } else { -1.0 };
        endpoint.powi((mode - 1) as i32) * (mode * mode) as f64
    }
}

fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        indices[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
    indices
}
