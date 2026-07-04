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
