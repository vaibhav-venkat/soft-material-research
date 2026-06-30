pub mod arrays;
pub mod coarse_grain;
#[cfg(feature = "gpu-metal")]
pub mod coarse_grain_cubecl;
pub mod errors;
pub mod fft_ops;
pub mod geometry;
pub mod library;
pub mod python;
pub mod regression;
pub mod sampling;

pub use errors::{CoreError, CoreResult};
