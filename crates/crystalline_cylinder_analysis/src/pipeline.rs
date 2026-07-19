//! Reusable orchestration that keeps loaded intermediates out of the CLI.

use std::path::PathBuf;

use crate::backend::AnalysisBackend;
use crate::center_of_mass::ComConfig;
use crate::correlation::CorrelationConfig;
use crate::fit::FitConfig;
use crate::laplace::LaplaceConfig;
use crate::model::{
    ComSeries, CorrelationSeries, DampedCosineFit, LaplaceGrid, PreferredEstimate, ReplicateGroup,
};

/// End-to-end analysis controls independent of command-line parsing.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub input_dirs: Vec<PathBuf>,
    pub com: ComConfig,
    pub correlation: CorrelationConfig,
    pub laplace: LaplaceConfig,
    pub fit: FitConfig,
}

/// Cached results for one physical case.
#[derive(Clone, Debug)]
pub struct CaseAnalysis {
    pub group: ReplicateGroup,
    pub com: Option<ComSeries>,
    pub correlation: Option<CorrelationSeries>,
    pub laplace: Option<LaplaceGrid>,
    pub preferred: Vec<PreferredEstimate>,
    pub fit: Option<DampedCosineFit>,
}

/// Execute only the stages requested by an analysis command.
pub fn run_pipeline<B: AnalysisBackend>(
    _backend: &B,
    _config: &PipelineConfig,
) -> Vec<CaseAnalysis> {
    todo!("discover once, stream once, and reuse all requested intermediates")
}
