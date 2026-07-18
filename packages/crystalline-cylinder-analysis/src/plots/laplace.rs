//! Kuva complex-Laplace heatmap interface.

use std::path::{Path, PathBuf};

use crystalline_cylinder_analysis::pipeline::CaseAnalysis;

use crate::error::AppResult;

/// Write paginated or per-case log-magnitude transform heatmaps.
pub fn write_laplace_plots(
    _analyses: &[CaseAnalysis],
    _output_dir: &Path,
) -> AppResult<Vec<PathBuf>> {
    todo!("build Kuva heatmaps of log10 transform magnitude")
}
