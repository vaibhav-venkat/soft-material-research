use ndarray::{Array4, Array5, Array6};

pub(crate) const COMPONENT_COUNT: usize = 3;

/// A rank-three tensor in the canonical `(x, e_theta, e_r)` frame.
pub(crate) type Rank3Tensor = [[[f64; COMPONENT_COUNT]; COMPONENT_COUNT]; COMPONENT_COUNT];
pub(crate) type CurrentQField = Array4<Rank3Tensor>;

/// All particle arrays passed to mechanical deposition, with names replacing positional tuples.
#[derive(Clone)]
pub(crate) struct ParticleFieldSet<'a> {
    pub(crate) coords: ndarray::ArrayView3<'a, f64>,
    pub(crate) directions: ndarray::ArrayView3<'a, f64>,
    pub(crate) velocities: ndarray::ArrayView3<'a, f64>,
    pub(crate) psi6_abs: ndarray::ArrayView2<'a, f64>,
    pub(crate) mask: ndarray::ArrayView2<'a, bool>,
}

/// Canonical raw mechanical fields. Grid axes `(T, Nx, Ntheta, Nr)` always come first.
pub(crate) struct MechanicalFieldSet {
    pub(crate) rho: Array4<f64>,
    pub(crate) p: Array5<f64>,
    pub(crate) q: Array6<f64>,
    pub(crate) a: Array6<f64>,
    pub(crate) psi6_sq: Array4<f64>,
    pub(crate) j_rho: Array5<f64>,
    pub(crate) j_p: Array6<f64>,
    pub(crate) j_q: CurrentQField,
}
