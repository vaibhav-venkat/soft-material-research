//! Kuva complex-Laplace heatmap interface.

use std::path::{Path, PathBuf};

use crystalline_cylinder_analysis::pipeline::CaseAnalysis;

/// Write paginated or per-case log-magnitude transform heatmaps.
pub fn write_laplace_plots(_analyses: &[CaseAnalysis], _output_dir: &Path) -> Vec<PathBuf> {
    todo!("build Kuva heatmaps of log10 transform magnitude")
}
