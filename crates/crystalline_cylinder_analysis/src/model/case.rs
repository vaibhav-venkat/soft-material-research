//! Case identity and replicate grouping.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::AnalysisManifest;

/// Supported producer schemas.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
pub enum CaseSchema {
    BigLx,
    Confinement,
}

/// Geometry fields required by the Rust analysis.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CaseMetadata {
    pub case_id: String,
    pub label: Option<String>,
    pub lx: f64,
    pub n_particles: usize,
    pub lx_multiplier: i64,
    pub circumference_diameters: Option<f64>,
    pub geometry_kind: Option<String>,
    pub seed: Option<u64>,
    #[serde(default)]
    pub particle_diameter: Option<f64>,
    #[serde(default)]
    pub radius: Option<f64>,
    #[serde(default)]
    pub circumference: Option<f64>,
    #[serde(default)]
    pub transverse_span: Option<f64>,
}

/// One validated manifest discovered below an input directory.
#[derive(Clone, Debug)]
pub struct DiscoveredDataset {
    pub input_root: PathBuf,
    pub manifest_path: PathBuf,
    pub schema: CaseSchema,
    pub manifest: AnalysisManifest,
}

/// Datasets that represent independent replicas of one physical case.
#[derive(Clone, Debug)]
pub struct ReplicateGroup {
    pub schema: CaseSchema,
    pub case: CaseMetadata,
    pub datasets: Vec<DiscoveredDataset>,
}
