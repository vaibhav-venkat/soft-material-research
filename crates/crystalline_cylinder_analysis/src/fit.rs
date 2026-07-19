//! Robust constrained fitting of the damped-cosine correlation model.

use crate::backend::AnalysisBackend;
use crate::laplace::preferred_coordinate;
use crate::model::{CorrelationSeries, DampedCosineFit};
use crate::PreferredAxis;

const PARAMETER_COUNT: usize = 4;
const PHASE_MARGIN: f64 = 1.0e-6;

/// Numerical controls for bounded soft-L1 fitting.
#[derive(Clone, Copy, Debug)]
pub struct FitConfig {
    pub soft_l1_scale: f64,
    pub tolerance: f64,
    pub maximum_evaluations: usize,
    pub rank_tolerance: f64,
}

/// Evaluate the constrained damped-cosine model.
pub fn damped_cosine(time: &[f64], parameters: &[f64; PARAMETER_COUNT]) -> Vec<f64> {
    let [amplitude, rate, omega, phase] = *parameters;
    let offset = 1.0 - amplitude * phase.cos();
    time.iter()
        .map(|&sample_time| {
            amplitude * (-rate * sample_time).exp() * (omega * sample_time + phase).cos() + offset
        })
        .collect()
}

/// Evaluate the analytic model Jacobian in row-major sample order.
pub fn damped_cosine_jacobian(time: &[f64], parameters: &[f64; PARAMETER_COUNT]) -> Vec<f64> {
    let [amplitude, rate, omega, phase] = *parameters;
    let mut jacobian = Vec::with_capacity(time.len() * PARAMETER_COUNT);
    for &sample_time in time {
        let envelope = (-rate * sample_time).exp();
        let argument = omega * sample_time + phase;
        let (sine, cosine) = argument.sin_cos();
        jacobian.extend_from_slice(&[
            envelope * cosine - phase.cos(),
            -amplitude * sample_time * envelope * cosine,
            -amplitude * sample_time * envelope * sine,
            -amplitude * envelope * sine + amplitude * phase.sin(),
        ]);
    }
    jacobian
}

/// Fit the averaged correlation with multistart bounded least squares.
pub fn fit_damped_cosine<B: AnalysisBackend>(
    backend: &B,
    correlation: &CorrelationSeries,
    omega_grid: &[f64],
    config: FitConfig,
) -> DampedCosineFit {
    validate_inputs(correlation, omega_grid, config);
    let time = &correlation.lag_times;
    let observed = &correlation.pearson_mean;
    let dt = time[1] - time[0];
    let duration = *time.last().expect("correlation is empty");
    let nyquist = std::f64::consts::PI / dt;
    let preferred_omega =
        preferred_coordinate(correlation, PreferredAxis::Omega, omega_grid).coordinate;
    let tail_count = 3.max(observed.len() / 10);
    let mut tail = observed[observed.len() - tail_count..].to_vec();
    tail.sort_by(|left, right| left.total_cmp(right));
    let initial_offset = median(&tail).clamp(-0.5, 0.5);
    let initial_amplitude = (1.0 - initial_offset).clamp(0.05, 2.0);
    let lower = [0.0, 0.0, 0.0, -0.5 * std::f64::consts::PI + PHASE_MARGIN];
    let upper = [
        2.0,
        1.0 / dt,
        nyquist,
        0.5 * std::f64::consts::PI - PHASE_MARGIN,
    ];
    let origin_zero = correlation.origin_counts[0] as f64;
    let weights = correlation
        .origin_counts
        .iter()
        .map(|&count| (count as f64 / origin_zero).sqrt())
        .collect::<Vec<_>>();
    let rate_starts = unique_clamped(
        &[1.0 / duration, 3.0 / duration, 10.0 / duration],
        f64::EPSILON,
        upper[1] * (1.0 - 1.0e-8),
    );
    let omega_starts = unique_clamped(
        &[
            0.5 * preferred_omega,
            preferred_omega,
            1.5 * preferred_omega,
        ],
        f64::EPSILON,
        upper[2] * (1.0 - 1.0e-8),
    );

    let mut evaluations = 0;
    let mut best: Option<FitState> = None;
    for &initial_rate in &rate_starts {
        for &initial_omega in &omega_starts {
            let initial = [initial_amplitude, initial_rate, initial_omega, 0.0];
            let state = optimize_start(
                backend,
                time,
                observed,
                &weights,
                initial,
                lower,
                upper,
                config,
                &mut evaluations,
            );
            if state.converged
                && state.parameters.iter().all(|value| value.is_finite())
                && best
                    .as_ref()
                    .is_none_or(|current| state.objective < current.objective)
            {
                best = Some(state);
            }
        }
    }
    let best = best.expect("damped-cosine fit found no finite solution");
    let prediction = damped_cosine(time, &best.parameters);
    let mean = observed.iter().sum::<f64>() / observed.len() as f64;
    let residual_sum = observed
        .iter()
        .zip(&prediction)
        .map(|(&actual, &fitted)| (actual - fitted).powi(2))
        .sum::<f64>();
    let total_sum = observed
        .iter()
        .map(|&actual| (actual - mean).powi(2))
        .sum::<f64>();
    assert!(total_sum > 0.0, "correlation has no variation");
    let [amplitude, rate, omega, phase] = best.parameters;
    DampedCosineFit {
        amplitude,
        rate,
        omega,
        phase,
        offset: 1.0 - amplitude * phase.cos(),
        r_squared: 1.0 - residual_sum / total_sum,
        evaluations,
        converged: best.converged,
        rate_at_lower_boundary: rate <= (1.0 / duration) * 1.0e-6,
        rate_at_upper_boundary: rate >= upper[1] * (1.0 - 1.0e-4),
        amplitude_at_upper_boundary: amplitude >= upper[0] * (1.0 - 1.0e-4),
        prediction,
    }
}

#[derive(Clone, Copy)]
struct FitState {
    parameters: [f64; PARAMETER_COUNT],
    objective: f64,
    converged: bool,
}

#[allow(clippy::too_many_arguments)]
fn optimize_start<B: AnalysisBackend>(
    backend: &B,
    time: &[f64],
    observed: &[f64],
    weights: &[f64],
    mut parameters: [f64; PARAMETER_COUNT],
    lower: [f64; PARAMETER_COUNT],
    upper: [f64; PARAMETER_COUNT],
    config: FitConfig,
    evaluations: &mut usize,
) -> FitState {
    let mut objective_value = objective(time, observed, weights, &parameters, config.soft_l1_scale);
    *evaluations += 1;
    let mut start_evaluations = 1;
    let mut damping: f64 = 1.0e-3;
    let mut converged = false;
    while start_evaluations < config.maximum_evaluations {
        let prediction = damped_cosine(time, &parameters);
        let jacobian = damped_cosine_jacobian(time, &parameters);
        let rows = time.len();
        let mut normal_matrix = [0.0; PARAMETER_COUNT * PARAMETER_COUNT];
        let mut gradient = [0.0; PARAMETER_COUNT];
        let mut column_norm_squared = [0.0; PARAMETER_COUNT];
        for row in 0..rows {
            let weighted_residual = weights[row] * (prediction[row] - observed[row]);
            let robust_weight =
                (1.0 + (weighted_residual / config.soft_l1_scale).powi(2)).powf(-0.25);
            let residual = robust_weight * weighted_residual;
            let mut jacobian_row = [0.0; PARAMETER_COUNT];
            for column in 0..PARAMETER_COUNT {
                let value = robust_weight * weights[row] * jacobian[row * PARAMETER_COUNT + column];
                jacobian_row[column] = value;
                column_norm_squared[column] += value * value;
                gradient[column] += value * residual;
            }
            for left in 0..PARAMETER_COUNT {
                for right in 0..PARAMETER_COUNT {
                    normal_matrix[left * PARAMETER_COUNT + right] +=
                        jacobian_row[left] * jacobian_row[right];
                }
            }
        }
        let gradient_maximum = projected_gradient_maximum(&parameters, &gradient, &lower, &upper);
        if gradient_maximum <= config.tolerance {
            converged = true;
            break;
        }
        for column in 0..PARAMETER_COUNT {
            normal_matrix[column * PARAMETER_COUNT + column] +=
                damping * column_norm_squared[column].max(1.0e-24);
        }
        let rhs = gradient.map(|value| -value);
        let step = backend.linear_least_squares(
            &normal_matrix,
            PARAMETER_COUNT,
            PARAMETER_COUNT,
            &rhs,
            config.rank_tolerance,
        );
        let mut candidate = parameters;
        for index in 0..PARAMETER_COUNT {
            candidate[index] = (parameters[index] + step[index]).clamp(lower[index], upper[index]);
        }
        let scaled_step = candidate
            .iter()
            .zip(parameters)
            .map(|(&next, current)| (next - current).abs() / (1.0 + current.abs()))
            .fold(0.0, f64::max);
        let candidate_objective =
            objective(time, observed, weights, &candidate, config.soft_l1_scale);
        *evaluations += 1;
        start_evaluations += 1;
        if candidate_objective < objective_value {
            let relative_change =
                (objective_value - candidate_objective) / (1.0 + objective_value.abs());
            parameters = candidate;
            objective_value = candidate_objective;
            damping = (damping * 0.3).max(1.0e-12);
            let secondary_gradient_tolerance = config.tolerance.sqrt();
            if (relative_change <= config.tolerance || scaled_step <= config.tolerance)
                && gradient_maximum <= secondary_gradient_tolerance
            {
                converged = true;
                break;
            }
        } else {
            damping = (damping * 10.0).min(1.0e12);
        }
    }
    FitState {
        parameters,
        objective: objective_value,
        converged,
    }
}

fn projected_gradient_maximum(
    parameters: &[f64; PARAMETER_COUNT],
    gradient: &[f64; PARAMETER_COUNT],
    lower: &[f64; PARAMETER_COUNT],
    upper: &[f64; PARAMETER_COUNT],
) -> f64 {
    (0..PARAMETER_COUNT)
        .map(|index| {
            let at_lower = parameters[index] <= lower[index];
            let at_upper = parameters[index] >= upper[index];
            if (at_lower && gradient[index] > 0.0) || (at_upper && gradient[index] < 0.0) {
                0.0
            } else {
                gradient[index].abs()
            }
        })
        .fold(0.0, f64::max)
}

fn objective(
    time: &[f64],
    observed: &[f64],
    weights: &[f64],
    parameters: &[f64; PARAMETER_COUNT],
    scale: f64,
) -> f64 {
    damped_cosine(time, parameters)
        .into_iter()
        .zip(observed)
        .zip(weights)
        .map(|((prediction, &actual), &weight)| {
            let scaled = weight * (prediction - actual) / scale;
            2.0 * scale * scale * ((1.0 + scaled * scaled).sqrt() - 1.0)
        })
        .sum()
}

fn unique_clamped(values: &[f64], lower: f64, upper: f64) -> Vec<f64> {
    let mut values = values
        .iter()
        .map(|value| value.clamp(lower, upper))
        .collect::<Vec<_>>();
    values.sort_by(|left, right| left.total_cmp(right));
    values.dedup_by(|left, right| left.to_bits() == right.to_bits());
    values
}

fn median(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[middle - 1] + sorted[middle])
    } else {
        sorted[middle]
    }
}

fn validate_inputs(correlation: &CorrelationSeries, omega_grid: &[f64], config: FitConfig) {
    let count = correlation.lag_times.len();
    assert!(count >= 4, "fit needs four lag samples");
    assert_eq!(
        correlation.pearson_mean.len(),
        count,
        "correlation lengths differ"
    );
    assert_eq!(
        correlation.origin_counts.len(),
        count,
        "origin counts differ"
    );
    assert_eq!(correlation.lag_times[0], 0.0, "fit must start at zero lag");
    assert!(
        correlation
            .lag_times
            .windows(2)
            .all(|pair| pair[1] > pair[0]),
        "lag times are not increasing"
    );
    let dt = correlation.lag_times[1] - correlation.lag_times[0];
    assert!(
        correlation
            .lag_times
            .windows(2)
            .all(|pair| ((pair[1] - pair[0]) - dt).abs() <= dt.abs().max(1.0) * 1.0e-9),
        "lag spacing is not uniform"
    );
    assert!(omega_grid.len() >= 2, "omega grid needs two points");
    assert!(
        omega_grid
            .iter()
            .all(|value| value.is_finite() && *value > 0.0),
        "fit omega grid must be positive"
    );
    assert!(
        omega_grid.windows(2).all(|pair| pair[1] > pair[0]),
        "omega grid is not increasing"
    );
    assert!(
        config.soft_l1_scale.is_finite() && config.soft_l1_scale > 0.0,
        "soft-L1 scale is invalid"
    );
    assert!(
        config.tolerance.is_finite() && config.tolerance > 0.0,
        "fit tolerance is invalid"
    );
    assert!(
        config.maximum_evaluations >= 2,
        "fit evaluation limit is too small"
    );
    assert!(
        config.rank_tolerance.is_finite() && config.rank_tolerance >= 0.0,
        "rank tolerance is invalid"
    );
}
