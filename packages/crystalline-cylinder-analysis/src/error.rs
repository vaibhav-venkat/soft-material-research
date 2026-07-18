//! Errors produced by CLI orchestration and artifact rendering.

/// Result type returned by the command-line package.
pub type AppResult<T> = Result<T, AppError>;

/// Top-level application error.
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error(transparent)]
    Analysis(#[from] crystalline_cylinder_analysis::AnalysisError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("invalid command-line configuration: {0}")]
    InvalidConfiguration(String),
    #[error("plot rendering failed: {0}")]
    Plot(String),
}
