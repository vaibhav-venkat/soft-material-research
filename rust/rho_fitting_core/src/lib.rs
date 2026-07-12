pub mod arrays;
pub mod coarse_grain_burn;
pub mod errors;
pub mod fields;
pub mod fitting;
pub mod geometry;
pub mod interpolation;
pub mod mechanics;
pub mod particles;
pub mod pde;
pub mod python;
pub mod regression;
pub mod spectral;
pub mod temporal;

pub use errors::{CoreError, CoreResult};
