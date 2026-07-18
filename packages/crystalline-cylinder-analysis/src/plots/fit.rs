//! Kuva damped-cosine fit figure interface.

use std::path::{Path, PathBuf};

use crystalline_cylinder_analysis::pipeline::CaseAnalysis;

use crate::error::AppResult;

/// Write measured-correlation and fitted-model overlays.
pub fn write_fit_plot(_analyses: &[CaseAnalysis], _output: &Path) -> AppResult<PathBuf> {
    todo!("build paginated Kuva fit panels with parameter annotations")
}
