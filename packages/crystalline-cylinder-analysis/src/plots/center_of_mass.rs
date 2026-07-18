//! Kuva COM and velocity figure interface.

use std::path::{Path, PathBuf};

use crystalline_cylinder_analysis::pipeline::CaseAnalysis;

use crate::error::AppResult;

/// Write the two-panel COM and velocity SVG with replicate bands.
pub fn write_com_plot(_analyses: &[CaseAnalysis], _output: &Path) -> AppResult<PathBuf> {
    todo!("build a Kuva Figure with COM and velocity panels")
}
