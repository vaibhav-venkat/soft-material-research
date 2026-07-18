//! Kuva velocity-correlation figure interface.

use std::path::{Path, PathBuf};

use crystalline_cylinder_analysis::pipeline::CaseAnalysis;

use crate::error::AppResult;

/// Write lagged Pearson curves and replicate-standard-deviation bands.
pub fn write_correlation_plot(_analyses: &[CaseAnalysis], _output: &Path) -> AppResult<PathBuf> {
    todo!("build a Kuva line-and-band Pearson correlation plot")
}
