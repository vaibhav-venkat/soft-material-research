use ndarray::{Array4, Array5, Array6, ArrayD};

pub const COMPONENT_COUNT: usize = 3;

/// Dynamic ndarray with canonical shape `(T, Nx, Ntheta, Nr, 3, 3, 3)`.
pub type CurrentQField = ArrayD<f64>;

/// All particle arrays passed to mechanical deposition
#[derive(Clone)]
pub struct ParticleFieldSet<'a> {
    pub coords: ndarray::ArrayView3<'a, f64>,
    pub directions: ndarray::ArrayView3<'a, f64>,
    pub velocities: ndarray::ArrayView3<'a, f64>,
    pub hexatic_order: ndarray::ArrayView2<'a, f64>,
    pub mask: ndarray::ArrayView2<'a, bool>,
}

/// Canonical raw mechanical fields. Grid axes `(T, Nx, Ntheta, Nr)` always come first.
pub struct MechanicalFieldSet {
    pub rho: Array4<f64>,
    pub p: Array5<f64>,
    pub q: Array6<f64>,
    pub a: Array6<f64>,
    pub hexatic_order: Array4<f64>,
    pub j_rho: Array5<f64>,
    pub j_p: Array6<f64>,
    pub j_q: CurrentQField,
}
