//! Streaming validation and shape summaries for trajectory shards.

use std::path::PathBuf;

use crate::error::{AnalysisError, AnalysisResult};
use crate::model::{CaseSchema, DiscoveredDataset};

use super::{resolve_shard_path, MappedShard, SafetensorView, TensorDtype};

/// Shape and dtype information for one mapped frame shard.
#[derive(Clone, Debug)]
pub struct ShardShape {
    pub path: PathBuf,
    pub coordinate_name: &'static str,
    pub coordinate_dtype: TensorDtype,
    pub coordinate_shape: Vec<usize>,
    pub step_shape: Vec<usize>,
}

/// Aggregate tensor shapes for one manifest
#[derive(Clone, Debug)]
pub struct DatasetShape {
    pub case_id: String,
    pub coordinate_name: &'static str,
    pub coordinate_dtype: TensorDtype,
    pub coordinate_shape: Vec<usize>,
    pub step_shape: Vec<usize>,
    pub shards: Vec<ShardShape>,
}

/// Map and validate each shard in sequence
pub fn inspect_dataset(dataset: &DiscoveredDataset) -> AnalysisResult<DatasetShape> {
    let mut summaries = Vec::with_capacity(dataset.manifest.shards.len());
    let mut aggregate_dtype = None;
    let mut aggregate_components = None;
    let mut previous_step = None;
    let mut coordinate_name = None;

    for shard_manifest in &dataset.manifest.shards {
        let path = resolve_shard_path(&dataset.manifest_path, &shard_manifest.file)?;
        let mapped = MappedShard::open(&path)?;
        let selected_name = select_coordinate_name(dataset.schema, &mapped, &path)?;
        let coordinates = mapped.tensor(selected_name)?;
        let steps = mapped.tensor("step")?;
        let expected_frames = shard_manifest.frame_stop - shard_manifest.frame_start;
        validate_coordinate_tensor(
            &path,
            selected_name,
            &coordinates,
            expected_frames,
            dataset.manifest.case.n_particles,
        )?;
        validate_step_tensor(&path, &steps, expected_frames, &mut previous_step)?;

        let components = coordinates.shape[2];
        if aggregate_dtype.is_some_and(|dtype| dtype != coordinates.dtype) {
            return Err(invalid_tensor(
                &path,
                selected_name,
                "coordinate dtype changes between shards",
            ));
        }
        if aggregate_components.is_some_and(|count| count != components) {
            return Err(invalid_tensor(
                &path,
                selected_name,
                "coordinate component count changes between shards",
            ));
        }
        if coordinate_name.is_some_and(|name| name != selected_name) {
            return Err(invalid_tensor(
                &path,
                selected_name,
                "coordinate tensor name changes between shards",
            ));
        }
        aggregate_dtype = Some(coordinates.dtype);
        aggregate_components = Some(components);
        coordinate_name = Some(selected_name);
        summaries.push(ShardShape {
            path,
            coordinate_name: selected_name,
            coordinate_dtype: coordinates.dtype,
            coordinate_shape: coordinates.shape,
            step_shape: steps.shape,
        });
        // `mapped` is dropped here before the next shard is opened.
    }

    Ok(DatasetShape {
        case_id: dataset.manifest.case.case_id.clone(),
        coordinate_name: coordinate_name.expect("a validated manifest has at least one shard"),
        coordinate_dtype: aggregate_dtype.expect("a validated manifest has at least one shard"),
        coordinate_shape: vec![
            dataset.manifest.frame_count,
            dataset.manifest.case.n_particles,
            aggregate_components.expect("a validated manifest has at least one shard"),
        ],
        step_shape: vec![dataset.manifest.frame_count],
        shards: summaries,
    })
}

fn select_coordinate_name(
    schema: CaseSchema,
    mapped: &MappedShard,
    path: &std::path::Path,
) -> AnalysisResult<&'static str> {
    match schema {
        CaseSchema::BigLx if mapped.contains("coords") => Ok("coords"),
        CaseSchema::Confinement if mapped.contains("coords") => Ok("coords"),
        CaseSchema::Confinement if mapped.contains("position_cartesian") => {
            Ok("position_cartesian")
        }
        _ => Err(invalid_tensor(
            path,
            "coords",
            "required coordinate tensor is missing",
        )),
    }
}

fn validate_coordinate_tensor(
    path: &std::path::Path,
    name: &str,
    tensor: &SafetensorView<'_>,
    frames: usize,
    particles: usize,
) -> AnalysisResult<()> {
    if !matches!(tensor.dtype, TensorDtype::F32 | TensorDtype::F64) {
        return Err(invalid_tensor(path, name, "coordinates must be F32 or F64"));
    }
    if tensor.shape.len() != 3
        || tensor.shape[0] != frames
        || tensor.shape[1] != particles
        || tensor.shape[2] == 0
    {
        return Err(invalid_tensor(
            path,
            name,
            &format!(
                "expected [{frames}, {particles}, components], found {:?}",
                tensor.shape
            ),
        ));
    }
    let finite = match tensor.dtype {
        TensorDtype::F32 => tensor.as_f32()?.iter().all(|value| value.is_finite()),
        TensorDtype::F64 => tensor.as_f64()?.iter().all(|value| value.is_finite()),
        TensorDtype::I32 | TensorDtype::I64 => false,
    };
    if !finite {
        return Err(invalid_tensor(
            path,
            name,
            "coordinates contain non-finite values",
        ));
    }
    Ok(())
}

fn validate_step_tensor(
    path: &std::path::Path,
    tensor: &SafetensorView<'_>,
    frames: usize,
    previous_step: &mut Option<i64>,
) -> AnalysisResult<()> {
    if !matches!(tensor.dtype, TensorDtype::I32 | TensorDtype::I64) {
        return Err(invalid_tensor(path, "step", "steps must be I32 or I64"));
    }
    if tensor.shape != [frames] {
        return Err(invalid_tensor(
            path,
            "step",
            &format!("expected [{frames}], found {:?}", tensor.shape),
        ));
    }
    match tensor.dtype {
        TensorDtype::I32 => validate_steps(
            path,
            tensor.as_i32()?.iter().map(|&step| i64::from(step)),
            previous_step,
        ),
        TensorDtype::I64 => validate_steps(path, tensor.as_i64()?.iter().copied(), previous_step),
        TensorDtype::F32 | TensorDtype::F64 => unreachable!("step dtype checked above"),
    }
}

fn validate_steps(
    path: &std::path::Path,
    steps: impl Iterator<Item = i64>,
    previous_step: &mut Option<i64>,
) -> AnalysisResult<()> {
    for step in steps {
        if previous_step.is_some_and(|previous| step <= previous) {
            return Err(invalid_tensor(
                path,
                "step",
                "steps must be globally strictly increasing",
            ));
        }
        *previous_step = Some(step);
    }
    Ok(())
}

fn invalid_tensor(path: &std::path::Path, name: &str, message: &str) -> AnalysisError {
    AnalysisError::InvalidTensor {
        path: path.to_path_buf(),
        name: name.to_owned(),
        message: message.to_owned(),
    }
}
