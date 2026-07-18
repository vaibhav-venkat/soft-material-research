//! Strict manifest parsing and shard-index validation.

use std::path::Path;

use crate::error::AnalysisResult;
use crate::model::{AnalysisManifest, CaseSchema};

/// Parse a manifest and classify its supported producer schema.
pub fn load_manifest(_path: &Path) -> AnalysisResult<(CaseSchema, AnalysisManifest)> {
    todo!("deserialize JSON and classify Big-Lx or confinement input")
}

/// Validate completeness, case fields, contained paths, and contiguous shards.
pub fn validate_manifest(
    _path: &Path,
    _schema: CaseSchema,
    _manifest: &AnalysisManifest,
) -> AnalysisResult<()> {
    todo!("enforce all fail-fast manifest and shard invariants")
}
