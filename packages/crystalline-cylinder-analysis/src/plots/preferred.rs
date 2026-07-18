//! Kuva preferred-coordinate summary interface.

use std::path::{Path, PathBuf};

use crystalline_cylinder_analysis::model::PreferredAxis;
use crystalline_cylinder_analysis::pipeline::CaseAnalysis;

use crate::error::AppResult;

/// Write preferred-coordinate summaries versus axial-length multiplier.
pub fn write_preferred_plot(
    _analyses: &[CaseAnalysis],
    _axis: PreferredAxis,
    _output: &Path,
) -> AppResult<PathBuf> {
    todo!("build Big-Lx families and confinement markers with Kuva")
}
