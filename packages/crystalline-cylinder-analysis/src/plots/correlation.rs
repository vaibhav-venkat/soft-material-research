//! Kuva velocity-correlation figure interface.

use std::path::{Path, PathBuf};

use crystalline_cylinder_analysis::pipeline::CaseAnalysis;

/// Write lagged Pearson curves and replicate-standard-deviation bands.
pub fn write_correlation_plot(_analyses: &[CaseAnalysis], _output: &Path) -> PathBuf {
    todo!("build a Kuva line-and-band Pearson correlation plot")
}
