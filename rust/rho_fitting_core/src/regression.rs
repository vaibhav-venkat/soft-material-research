use faer::prelude::SolveLstsq;
use faer::Mat;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::{CoreError, CoreResult};

const LEGACY_SELECTED_TAU_FRACTION: f64 = 30.0 / 39.0;

pub struct StabilitySelection {
    pub coefficients: Array1<f64>,
    pub importance: Array1<f64>,
    pub raw_correlations: Array1<f64>,
    pub importance_path: Array2<f64>,
    pub tau_values: Array1<f64>,
    pub active: Array1<bool>,
    pub tau_index: i64,
    pub y_pred: Array1<f64>,
    pub rmse: f64,
    pub r2: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn stability_selection(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    seed: u64,
    tau_eps: f64,
    subsamples: usize,
    importance_threshold: f64,
    alpha: f64,
    max_iter: usize,
) -> CoreResult<StabilitySelection> {
    validate_inputs(x, y, tau_eps, subsamples, importance_threshold, alpha)?;
    let raw_correlations = raw_feature_correlations(x, y);
    let (xn, yn) = normalize_for_path(x, y);
    let coef0 = stlsq(&xn.view(), &yn.view(), 0.0, alpha, max_iter, false)?;
    let tau_max = coef0.iter().fold(0.0_f64, |acc, value| acc.max(value.abs())) * 1.01;
    let tau = selected_tau(tau_max, tau_eps);
    let tau_values = Array1::from_vec(vec![tau]);

    if tau_max == 0.0 {
        let active = Array1::<bool>::from_elem(x.dim().1, false);
        let coefficients = Array1::<f64>::zeros(x.dim().1);
        let y_pred = Array1::<f64>::zeros(y.len());
        return finish(
            coefficients,
            Array1::<f64>::zeros(x.dim().1),
            raw_correlations,
            Array2::<f64>::zeros((1, x.dim().1)),
            tau_values,
            active,
            -1,
            y,
            y_pred,
        );
    }

    let tau_index = 0;
    let mut importance_path = Array2::<f64>::zeros((1, x.dim().1));
    let mut kept = Array1::<f64>::zeros(x.dim().1);
    let n_sub = usize::max(1, x.dim().0 / 2);
    let mut rows: Vec<usize> = (0..x.dim().0).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    for _ in 0..subsamples {
        rows.shuffle(&mut rng);
        let sample = &rows[..n_sub];
        let x_sub = select_rows(&xn.view(), sample);
        let y_sub = select_values(&yn.view(), sample);
        let coefficients = stlsq(&x_sub.view(), &y_sub.view(), tau, alpha, max_iter, false)?;
        for feature in 0..coefficients.len() {
            if coefficients[feature].abs() > 0.0 {
                kept[feature] += 1.0;
            }
        }
    }

    for feature in 0..kept.len() {
        importance_path[[tau_index, feature]] = kept[feature] / subsamples as f64;
    }
    let importance = importance_path.row(tau_index).to_owned();
    let active = importance.mapv(|value| value >= importance_threshold);
    let coefficients = final_refit(x, y, active.view())?;
    let y_pred = predict(x, coefficients.view());
    finish(
        coefficients,
        importance,
        raw_correlations,
        importance_path,
        tau_values,
        active,
        tau_index as i64,
        y,
        y_pred,
    )
}

fn validate_inputs(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    tau_eps: f64,
    subsamples: usize,
    importance_threshold: f64,
    alpha: f64,
) -> CoreResult<()> {
    if x.dim().0 == 0 || x.dim().1 == 0 || y.len() != x.dim().0 {
        return Err(CoreError::Shape(
            "X must be (N, terms), y must be (N,), and both must be non-empty".to_string(),
        ));
    }
    if !(tau_eps.is_finite() && tau_eps > 0.0) || subsamples == 0 {
        return Err(CoreError::InvalidInput(
            "tau_eps and subsamples must be positive".to_string(),
        ));
    }
    if !(importance_threshold.is_finite()
        && alpha.is_finite()
        && alpha >= 0.0
        && (0.0..=1.0).contains(&importance_threshold))
    {
        return Err(CoreError::InvalidInput(
            "importance_threshold must be in [0,1] and alpha must be nonnegative".to_string(),
        ));
    }
    if !x.iter().all(|value| value.is_finite()) || !y.iter().all(|value| value.is_finite()) {
        return Err(CoreError::InvalidInput(
            "regression inputs must be finite".to_string(),
        ));
    }
    Ok(())
}

fn raw_feature_correlations(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Array1<f64> {
    let y_mean = mean(y);
    let y_centered: Vec<f64> = y.iter().map(|value| value - y_mean).collect();
    let y_norm = y_centered.iter().map(|value| value * value).sum::<f64>().sqrt();
    let mut out = Array1::<f64>::from_elem(x.dim().1, f64::NAN);
    for feature in 0..x.dim().1 {
        let column = x.column(feature);
        let x_mean = mean(column);
        let mut numerator = 0.0;
        let mut x_norm2 = 0.0;
        for row in 0..x.dim().0 {
            let centered = x[[row, feature]] - x_mean;
            numerator += centered * y_centered[row];
            x_norm2 += centered * centered;
        }
        let denominator = x_norm2.sqrt() * y_norm;
        if denominator > 0.0 {
            out[feature] = numerator / denominator;
        }
    }
    out
}

fn normalize_for_path(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> (Array2<f64>, Array1<f64>) {
    let mut xn = Array2::<f64>::zeros(x.dim());
    for feature in 0..x.dim().1 {
        let column = x.column(feature);
        let column_mean = mean(column);
        let mut variance = 0.0;
        for row in 0..x.dim().0 {
            let centered = x[[row, feature]] - column_mean;
            xn[[row, feature]] = centered;
            variance += centered * centered;
        }
        let scale = (variance / x.dim().0 as f64).sqrt();
        if scale > 0.0 {
            for row in 0..x.dim().0 {
                xn[[row, feature]] /= scale;
            }
        }
    }

    let y_mean = mean(y);
    let mut yn = Array1::<f64>::zeros(y.len());
    let mut variance = 0.0;
    for row in 0..y.len() {
        yn[row] = y[row] - y_mean;
        variance += yn[row] * yn[row];
    }
    let scale = (variance / y.len() as f64).sqrt();
    if scale > 0.0 {
        yn.mapv_inplace(|value| value / scale);
    }
    (xn, yn)
}

fn selected_tau(tau_max: f64, eps: f64) -> f64 {
    if tau_max == 0.0 {
        return 0.0;
    }
    tau_max * 10.0_f64.powf(eps.log10() * LEGACY_SELECTED_TAU_FRACTION)
}

fn stlsq(
    x: &ArrayView2<'_, f64>,
    y: &ArrayView1<'_, f64>,
    threshold: f64,
    alpha: f64,
    max_iter: usize,
    unbias: bool,
) -> CoreResult<Array1<f64>> {
    let mut coefficients = solve_least_squares(x, y, alpha)?;
    let mut support = vec![true; x.dim().1];
    for _ in 0..max_iter {
        let mut next_support = vec![false; x.dim().1];
        for feature in 0..x.dim().1 {
            next_support[feature] = coefficients[feature].abs() >= threshold;
            if !next_support[feature] {
                coefficients[feature] = 0.0;
            }
        }
        if next_support == support {
            support = next_support;
            break;
        }
        support = next_support;
        if !support.iter().any(|keep| *keep) {
            coefficients.fill(0.0);
            break;
        }
        coefficients = refit_support(x, y, &support, alpha)?;
    }
    if unbias && support.iter().any(|keep| *keep) {
        refit_support(x, y, &support, 0.0)
    } else {
        Ok(coefficients)
    }
}

fn final_refit(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    active: ArrayView1<'_, bool>,
) -> CoreResult<Array1<f64>> {
    if !active.iter().any(|value| *value) {
        return Ok(Array1::<f64>::zeros(x.dim().1));
    }
    let support: Vec<bool> = active.iter().copied().collect();
    refit_support(&x, &y, &support, 0.0)
}

fn refit_support(
    x: &ArrayView2<'_, f64>,
    y: &ArrayView1<'_, f64>,
    support: &[bool],
    alpha: f64,
) -> CoreResult<Array1<f64>> {
    let active: Vec<usize> = support
        .iter()
        .enumerate()
        .filter_map(|(index, keep)| if *keep { Some(index) } else { None })
        .collect();
    let mut coefficients = Array1::<f64>::zeros(x.dim().1);
    if active.is_empty() {
        return Ok(coefficients);
    }
    let mut x_active = Array2::<f64>::zeros((x.dim().0, active.len()));
    for (local, feature) in active.iter().enumerate() {
        x_active.column_mut(local).assign(&x.column(*feature));
    }
    let active_coefficients = solve_least_squares(&x_active.view(), y, alpha)?;
    for (local, feature) in active.iter().enumerate() {
        coefficients[*feature] = active_coefficients[local];
    }
    Ok(coefficients)
}

fn solve_least_squares(
    x: &ArrayView2<'_, f64>,
    y: &ArrayView1<'_, f64>,
    alpha: f64,
) -> CoreResult<Array1<f64>> {
    let rows = x.dim().0 + if alpha > 0.0 { x.dim().1 } else { 0 };
    let cols = x.dim().1;
    let ridge = alpha.sqrt();
    let lhs = Mat::from_fn(rows, cols, |row, col| {
        if row < x.dim().0 {
            x[[row, col]]
        } else if row - x.dim().0 == col {
            ridge
        } else {
            0.0
        }
    });
    let rhs = Mat::from_fn(rows, 1, |row, _| if row < y.len() { y[row] } else { 0.0 });
    let solution = lhs.qr().solve_lstsq(&rhs);
    let mut out = Array1::<f64>::zeros(cols);
    for row in 0..cols {
        out[row] = solution[(row, 0)];
    }
    if out.iter().all(|value| value.is_finite()) {
        Ok(out)
    } else {
        Err(CoreError::InvalidInput(
            "least-squares solve produced non-finite coefficients".to_string(),
        ))
    }
}

fn select_rows(x: &ArrayView2<'_, f64>, rows: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((rows.len(), x.dim().1));
    for (out_row, source_row) in rows.iter().enumerate() {
        out.row_mut(out_row).assign(&x.row(*source_row));
    }
    out
}

fn select_values(y: &ArrayView1<'_, f64>, rows: &[usize]) -> Array1<f64> {
    Array1::from_iter(rows.iter().map(|row| y[*row]))
}

fn predict(x: ArrayView2<'_, f64>, coefficients: ArrayView1<'_, f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(x.dim().0);
    for row in 0..x.dim().0 {
        let mut value = 0.0;
        for feature in 0..x.dim().1 {
            value += x[[row, feature]] * coefficients[feature];
        }
        out[row] = value;
    }
    out
}

fn finish(
    coefficients: Array1<f64>,
    importance: Array1<f64>,
    raw_correlations: Array1<f64>,
    importance_path: Array2<f64>,
    tau_values: Array1<f64>,
    active: Array1<bool>,
    tau_index: i64,
    y: ArrayView1<'_, f64>,
    y_pred: Array1<f64>,
) -> CoreResult<StabilitySelection> {
    let residual_sum = y
        .iter()
        .zip(y_pred.iter())
        .map(|(actual, predicted)| (actual - predicted).powi(2))
        .sum::<f64>();
    let rmse = (residual_sum / y.len() as f64).sqrt();
    let y_mean = mean(y);
    let denom = y.iter().map(|value| (value - y_mean).powi(2)).sum::<f64>();
    let r2 = if denom > 0.0 {
        1.0 - residual_sum / denom
    } else {
        f64::NAN
    };
    Ok(StabilitySelection {
        coefficients,
        importance,
        raw_correlations,
        importance_path,
        tau_values,
        active,
        tau_index,
        y_pred,
        rmse,
        r2,
    })
}

fn mean(values: ArrayView1<'_, f64>) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn stlsq_recovers_sparse_coefficients() {
        let x = array![[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = stability_selection(x.view(), y.view(), 7, 1.0e-2, 8, 0.5, 1.0e-9, 20)
            .expect("regression should succeed");
        assert!((result.coefficients[0] - 2.0).abs() < 1.0e-8);
        assert!(result.coefficients[1].abs() < 1.0e-8);
        assert_eq!(result.tau_index, 0);
        assert_eq!(result.tau_values.len(), 1);
        assert_eq!(result.importance_path.dim(), (1, 2));
    }

    #[test]
    fn stlsq_final_coefficients_are_unbiased() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let coefficients = stlsq(&x.view(), &y.view(), 0.1, 10.0, 20, true)
            .expect("regression should succeed");
        assert!((coefficients[0] - 2.0).abs() < 1.0e-10);
    }

    #[test]
    fn stlsq_can_threshold_everything_out() {
        let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let y = array![1.0, 1.0, 2.0];
        let coefficients = stlsq(&x.view(), &y.view(), 10.0, 0.0, 20, true)
            .expect("regression should succeed");
        assert_eq!(coefficients.to_vec(), vec![0.0, 0.0]);
    }
}
