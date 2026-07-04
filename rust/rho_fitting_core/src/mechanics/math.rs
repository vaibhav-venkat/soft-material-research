use ndarray::{Array3, ArrayView3};

pub(super) fn delta(a: usize, b: usize) -> f64 {
    if a == b {
        1.0
    } else {
        0.0
    }
}

pub(super) fn vector_component(
    vector: ndarray::ArrayView4<'_, f64>,
    t: usize,
    ix: usize,
    iy: usize,
    component: usize,
    components: usize,
) -> f64 {
    if component < components {
        vector[[t, ix, iy, component]]
    } else {
        0.0
    }
}

pub(super) fn centered_scalar(values: ArrayView3<'_, f64>) -> Array3<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values.iter().copied() {
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }
    let mean = if count > 0 { sum / count as f64 } else { 0.0 };
    values.mapv(|value| value - mean)
}
