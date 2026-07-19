//! Quadrature for already-sampled real and complex series.

use num_complex::Complex64;

/// Integrate uniformly sampled real values with composite Simpson quadrature.
pub fn simpson_samples(values: &[f64], spacing: f64) -> f64 {
    assert!(values.len() >= 2, "Simpson needs two samples");
    assert!(
        spacing.is_finite() && spacing > 0.0,
        "Simpson spacing must be positive"
    );
    assert!(
        values.iter().all(|value| value.is_finite()),
        "Simpson input is non-finite"
    );
    values
        .iter()
        .zip(simpson_weights(values.len(), spacing))
        .map(|(&value, weight)| value * weight)
        .sum()
}

/// Integrate uniformly sampled complex values with composite Simpson quadrature.
pub fn simpson_samples_complex(values: &[Complex64], spacing: f64) -> Complex64 {
    assert!(values.len() >= 2, "Simpson needs two samples");
    assert!(
        spacing.is_finite() && spacing > 0.0,
        "Simpson spacing must be positive"
    );
    assert!(
        values
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite()),
        "Simpson input is non-finite"
    );
    values
        .iter()
        .zip(simpson_weights(values.len(), spacing))
        .map(|(&value, weight)| value * weight)
        .sum()
}

/// Return uniform sampled-Simpson weights, including SciPy's even-count correction.
pub fn simpson_weights(sample_count: usize, spacing: f64) -> Vec<f64> {
    assert!(sample_count >= 2, "Simpson needs two samples");
    assert!(
        spacing.is_finite() && spacing > 0.0,
        "Simpson spacing must be positive"
    );
    if sample_count == 2 {
        return vec![spacing / 2.0; 2];
    }

    let simpson_count = if sample_count.is_multiple_of(2) {
        sample_count - 1
    } else {
        sample_count
    };
    let mut weights = vec![0.0; sample_count];
    for (index, weight) in weights[..simpson_count].iter_mut().enumerate() {
        let multiplier = if index == 0 || index + 1 == simpson_count {
            1.0
        } else if index % 2 == 1 {
            4.0
        } else {
            2.0
        };
        *weight = multiplier * spacing / 3.0;
    }

    if sample_count.is_multiple_of(2) {
        // Cartwright correction for the final interval on a uniform grid.
        weights[sample_count - 1] += 5.0 * spacing / 12.0;
        weights[sample_count - 2] += 2.0 * spacing / 3.0;
        weights[sample_count - 3] -= spacing / 12.0;
    }
    weights
}
