//! Kuva plot declarations for every analysis result.

mod center_of_mass;
mod correlation;
mod fit;
mod laplace;
mod preferred;

pub use center_of_mass::write_com_plot;
pub use correlation::write_correlation_plot;
pub use fit::write_fit_plot;
pub use laplace::write_laplace_plots;
pub use preferred::write_preferred_plot;
