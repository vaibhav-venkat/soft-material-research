//! Command-line entrypoint for crystalline-cylinder analysis.

use clap::Parser;
use crystalline_cylinder_analysis_cli::cli::Cli;
use crystalline_cylinder_analysis_cli::error::AppResult;
use crystalline_cylinder_analysis_cli::run;

fn main() -> AppResult<()> {
    run::run(Cli::parse())
}
