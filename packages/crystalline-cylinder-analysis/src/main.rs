//! Command-line entrypoint for crystalline-cylinder analysis.

use clap::Parser;
use crystalline_cylinder_analysis_cli::cli::Cli;
use crystalline_cylinder_analysis_cli::run;

fn main() {
    run::run(Cli::parse());
}
