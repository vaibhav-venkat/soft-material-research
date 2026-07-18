//! Atomic JSON and safetensor output declarations.

use std::path::{Path, PathBuf};

use crystalline_cylinder_analysis::pipeline::CaseAnalysis;
use serde::Serialize;

use crate::error::AppResult;

/// Versioned provenance record written beside every result set.
#[derive(Clone, Debug, Serialize)]
pub struct OutputManifest {
    pub schema: String,
    pub command: String,
    pub input_dirs: Vec<PathBuf>,
    pub timestep: f64,
    pub cases: Vec<OutputCaseRecord>,
}

/// Input and replicate provenance for one physical case.
#[derive(Clone, Debug, Serialize)]
pub struct OutputCaseRecord {
    pub case_id: String,
    pub label: String,
    pub replicate_count: usize,
    pub input_manifests: Vec<PathBuf>,
}

/// Prepare an empty output root or replace it with explicit authorization.
pub fn prepare_output_dir(_path: &Path, _overwrite: bool) -> AppResult<()> {
    todo!("fail on existing output unless overwrite is explicitly enabled")
}

/// Write a serializable value through a temporary file and atomic rename.
pub fn write_json_atomic<T: Serialize>(_path: &Path, _value: &T) -> AppResult<PathBuf> {
    todo!("serialize JSON and atomically publish it")
}

/// Write per-case numerical arrays as a versioned safetensor artifact.
pub fn write_case_safetensors(_output_dir: &Path, _analysis: &CaseAnalysis) -> AppResult<PathBuf> {
    todo!("encode result vectors without Python or ndarray dependencies")
}

/// Build provenance records from completed case analyses.
pub fn output_manifest(
    _command: &str,
    _input_dirs: &[PathBuf],
    _timestep: f64,
    _analyses: &[CaseAnalysis],
) -> OutputManifest {
    todo!("collect case metadata and all contributing manifest paths")
}
