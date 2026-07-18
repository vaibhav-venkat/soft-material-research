//! Input-root scanning and physical-case grouping.

use std::path::PathBuf;

use crate::error::AnalysisResult;
use crate::model::{DiscoveredDataset, ReplicateGroup};

/// Discover supported complete manifests beneath all input roots.
pub fn discover_datasets(_input_roots: &[PathBuf]) -> AnalysisResult<Vec<DiscoveredDataset>> {
    todo!("scan each safetensors_output directory and validate every manifest")
}

/// Group compatible datasets by schema and case ID without checking seeds.
pub fn group_replicates(_datasets: Vec<DiscoveredDataset>) -> AnalysisResult<Vec<ReplicateGroup>> {
    todo!("validate physical metadata and group independent replicas")
}
