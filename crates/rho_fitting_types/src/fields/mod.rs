use ndarray::{Array4, Array5, Array6};

pub const COMPONENT_COUNT: usize = 3;

/// A rank-three tensor in the canonical `(x, e_theta, e_r)` frame.
pub type Rank3Tensor = [[[f64; COMPONENT_COUNT]; COMPONENT_COUNT]; COMPONENT_COUNT];
pub type CurrentQField = Array4<Rank3Tensor>;

/// All particle arrays passed to mechanical deposition, with names replacing positional tuples.
#[derive(Clone)]
pub struct ParticleFieldSet<'a> {
    pub coords: ndarray::ArrayView3<'a, f64>,
    pub directions: ndarray::ArrayView3<'a, f64>,
    pub velocities: ndarray::ArrayView3<'a, f64>,
    pub psi6_abs: ndarray::ArrayView2<'a, f64>,
    pub mask: ndarray::ArrayView2<'a, bool>,
}

/// Canonical raw mechanical fields. Grid axes `(T, Nx, Ntheta, Nr)` always come first.
pub struct MechanicalFieldSet {
    pub rho: Array4<f64>,
    pub p: Array5<f64>,
    pub q: Array6<f64>,
    pub a: Array6<f64>,
    pub psi6_sq: Array4<f64>,
    pub j_rho: Array5<f64>,
    pub j_p: Array6<f64>,
    pub j_q: CurrentQField,
}
