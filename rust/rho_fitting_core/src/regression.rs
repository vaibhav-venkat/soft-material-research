use ndarray::Array1;

pub fn threshold_active(coefficients: &Array1<f64>, tau: f64) -> Vec<bool> {
    coefficients
        .iter()
        .map(|value| value.abs() >= tau)
        .collect()
}
