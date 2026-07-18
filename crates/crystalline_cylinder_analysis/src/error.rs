//! Error types shared by input, numerical, and aggregation stages.

use std::path::PathBuf;

pub type AnalysisResult<T> = Result<T, AnalysisError>;

/// Fail-fast errors produced by malformed inputs or invalid numerical states.
#[derive(Debug, thiserror::Error)]
pub enum AnalysisError {
    #[error("invalid analysis configuration: {0}")]
    InvalidConfiguration(String),
    #[error("invalid manifest {path}: {message}")]
    InvalidManifest { path: PathBuf, message: String },
    #[error("invalid tensor {name} in {path}: {message}")]
    InvalidTensor {
        path: PathBuf,
        name: String,
        message: String,
    },
    #[error("incompatible replicates for {case_id}: {message}")]
    IncompatibleReplicates { case_id: String, message: String },
    #[error("numerical analysis failed: {0}")]
    Numerical(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("invalid zero-copy tensor byte layout: {0}")]
    ByteLayout(String),
}
