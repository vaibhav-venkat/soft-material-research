//! Translation from CLI declarations to the reusable analysis pipeline.

use crate::cli::{AnalysisCommand, Cli};
use crate::error::AppResult;

/// Validate a command, run the requested stages, and write their artifacts.
pub fn run(_cli: Cli) -> AppResult<()> {
    todo!("construct the CPU backend, dispatch the pipeline, and render outputs")
}

/// Return a stable name for each public subcommand.
pub fn command_name(_command: &AnalysisCommand) -> &'static str {
    todo!("map command variants to artifact directory names")
}
