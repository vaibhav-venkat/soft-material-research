//! Streaming axial COM unwrapping and differentiation.

use crate::error::AnalysisResult;
use crate::model::{ComSeries, DiscoveredDataset};

/// COM analysis controls shared by individual replicas.
#[derive(Clone, Copy, Debug)]
pub struct ComConfig {
    pub timestep: f64,
}

/// Stream one dataset and compute its unwrapped axial COM and velocity.
pub fn analyze_replica_com(
    _dataset: &DiscoveredDataset,
    _config: ComConfig,
) -> AnalysisResult<ComSeries> {
    todo!("unwrap particle positions shard by shard and differentiate the COM")
}

/// Differentiate samples with NumPy-compatible endpoint behavior.
pub fn finite_gradient(_values: &[f64], _coordinates: &[f64]) -> AnalysisResult<Vec<f64>> {
    todo!("apply first- or second-order nonuniform finite differences")
}
