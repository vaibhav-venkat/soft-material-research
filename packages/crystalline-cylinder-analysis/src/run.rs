//! Translation from CLI declarations to the reusable analysis pipeline.

use crate::cli::{AnalysisCommand, Cli};
use crate::error::{AppError, AppResult};
use crystalline_cylinder_analysis::input::{
    discover_datasets, group_replicates, inspect_dataset, DatasetShape,
};
use crystalline_cylinder_analysis::{CaseSchema, CpuAnalysisBackend};

/// Validate a command, run the requested stages, and write their artifacts.
pub fn run(cli: Cli) -> AppResult<()> {
    let backend = CpuAnalysisBackend::new(cli.common.threads)?;
    eprintln!(
        "[debug:init] backend=tenferro-cpu threads={}",
        backend.thread_count,
    );

    match cli.command {
        AnalysisCommand::Inspect => inspect_inputs(&cli.common.input_dir),
        command => Err(AppError::InvalidConfiguration(format!(
            "the {} analysis stage is scaffolded but not implemented; use `inspect` to validate inputs",
            command_name(&command)
        ))),
    }
}

/// Return a stable name for each public subcommand.
pub fn command_name(command: &AnalysisCommand) -> &'static str {
    match command {
        AnalysisCommand::Inspect => "inspect",
        AnalysisCommand::Com => "com",
        AnalysisCommand::Correlation(_) => "correlation",
        AnalysisCommand::Laplace(_) => "laplace",
        AnalysisCommand::Preferred(_) => "preferred",
        AnalysisCommand::Fit(_) => "fit",
        AnalysisCommand::All(_) => "all",
    }
}

fn inspect_inputs(input_dirs: &[std::path::PathBuf]) -> AppResult<()> {
    let datasets = discover_datasets(input_dirs)?;
    let groups = group_replicates(datasets)?;
    for group in groups {
        let schema = match group.schema {
            CaseSchema::BigLx => "big-lx",
            CaseSchema::Confinement => "confinement",
        };
        eprintln!(
            "[debug:case] case={} schema={} replicates={} expected=[frames, particles, components]",
            group.case.case_id,
            schema,
            group.datasets.len()
        );
        for (replicate_index, dataset) in group.datasets.iter().enumerate() {
            let shape = inspect_dataset(dataset)?;
            print_dataset_shape(replicate_index, &shape);
        }
    }
    Ok(())
}

fn print_dataset_shape(replicate_index: usize, shape: &DatasetShape) {
    for (shard_index, shard) in shape.shards.iter().enumerate() {
        eprintln!(
            "[debug:shard] case={} replicate={} shard={} file={} {}={:?} dtype={:?} step={:?}",
            shape.case_id,
            replicate_index + 1,
            shard_index + 1,
            shard.path.display(),
            shard.coordinate_name,
            shard.coordinate_shape,
            shard.coordinate_dtype,
            shard.step_shape
        );
    }
    eprintln!(
        "[debug:dataset] case={} replicate={} {}={:?} dtype={:?} step={:?}",
        shape.case_id,
        replicate_index + 1,
        shape.coordinate_name,
        shape.coordinate_shape,
        shape.coordinate_dtype,
        shape.step_shape
    );
}
