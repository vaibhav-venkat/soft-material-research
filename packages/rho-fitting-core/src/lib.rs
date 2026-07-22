//! PyO3 adapter for the reusable rho-fitting Rust crates.
//!
//! The adapter intentionally contains no numerical implementation. It converts
//! NumPy inputs, dispatches to the pure Rust crates, and translates domain
//! errors into Python exceptions.

pub use rho_fitting_gpu::coarse_grain_burn;
pub use rho_fitting_numerics::{
    fitting, interpolation, particles, pde, regression, spectral, temporal,
};
pub use rho_fitting_types::{constants, fields, mechanics, CoreError, CoreResult};

mod python;
