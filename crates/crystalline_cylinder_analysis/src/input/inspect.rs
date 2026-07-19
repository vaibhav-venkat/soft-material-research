//! Streaming validation and shape summaries for trajectory shards.

use std::path::PathBuf;

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
pub fn inspect_dataset(dataset: &DiscoveredDataset) -> DatasetShape {
    let mut summaries = Vec::with_capacity(dataset.manifest.shards.len());
    let mut aggregate_dtype = None;
    let mut aggregate_components = None;
    let mut previous_step = None;
    let mut coordinate_name = None;

    for shard_manifest in &dataset.manifest.shards {
        let path = resolve_shard_path(&dataset.manifest_path, &shard_manifest.file);
        let mapped = MappedShard::open(&path);
        let selected_name = select_coordinate_name(dataset.schema, &mapped);
        let coordinates = mapped.tensor(selected_name);
        let steps = mapped.tensor("step");
        let expected_frames = shard_manifest.frame_stop - shard_manifest.frame_start;
        validate_coordinate_tensor(
            &path,
            selected_name,
            &coordinates,
            expected_frames,
            dataset.manifest.case.n_particles,
        );
        validate_step_tensor(&steps, expected_frames, &mut previous_step);

        let components = coordinates.shape[2];
        assert!(
            aggregate_dtype.is_none_or(|dtype| dtype == coordinates.dtype),
            "bad dtype"
        );
        assert!(
            aggregate_components.is_none_or(|count| count == components),
            "bad shape"
        );
        assert!(
            coordinate_name.is_none_or(|name| name == selected_name),
            "bad tensor"
        );
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

    DatasetShape {
        case_id: dataset.manifest.case.case_id.clone(),
        coordinate_name: coordinate_name.expect("no shard"),
        coordinate_dtype: aggregate_dtype.expect("no shard"),
        coordinate_shape: vec![
            dataset.manifest.frame_count,
            dataset.manifest.case.n_particles,
            aggregate_components.expect("no shard"),
        ],
        step_shape: vec![dataset.manifest.frame_count],
        shards: summaries,
    }
}

fn select_coordinate_name(schema: CaseSchema, mapped: &MappedShard) -> &'static str {
    match schema {
        CaseSchema::BigLx if mapped.contains("coords") => "coords",
        CaseSchema::Confinement if mapped.contains("coords") => "coords",
        CaseSchema::Confinement if mapped.contains("position_cartesian") => "position_cartesian",
        _ => panic!("missing coords"),
    }
}

fn validate_coordinate_tensor(
    path: &std::path::Path,
    name: &str,
    tensor: &SafetensorView<'_>,
    frames: usize,
    particles: usize,
) {
    let _ = (path, name);
    assert!(
        matches!(tensor.dtype, TensorDtype::F32 | TensorDtype::F64),
        "bad dtype"
    );
    assert_eq!(tensor.shape.len(), 3, "bad shape");
    assert_eq!(tensor.shape[0], frames, "bad shape");
    assert_eq!(tensor.shape[1], particles, "bad shape");
    assert_ne!(tensor.shape[2], 0, "bad shape");
    let finite = match tensor.dtype {
        TensorDtype::F32 => tensor.as_f32().iter().all(|value| value.is_finite()),
        TensorDtype::F64 => tensor.as_f64().iter().all(|value| value.is_finite()),
        TensorDtype::Bool | TensorDtype::I8 | TensorDtype::I32 | TensorDtype::I64 => false,
    };
    assert!(finite, "bad coords");
}

fn validate_step_tensor(
    tensor: &SafetensorView<'_>,
    frames: usize,
    previous_step: &mut Option<i64>,
) {
    assert!(
        matches!(tensor.dtype, TensorDtype::I32 | TensorDtype::I64),
        "bad dtype"
    );
    assert_eq!(tensor.shape, [frames], "bad shape");
    match tensor.dtype {
        TensorDtype::I32 => validate_steps(
            tensor.as_i32().iter().map(|&step| i64::from(step)),
            previous_step,
        ),
        TensorDtype::I64 => validate_steps(tensor.as_i64().iter().copied(), previous_step),
        TensorDtype::Bool | TensorDtype::F32 | TensorDtype::F64 | TensorDtype::I8 => {
            unreachable!("step dtype checked above")
        }
    }
}

fn validate_steps(steps: impl Iterator<Item = i64>, previous_step: &mut Option<i64>) {
    for step in steps {
        assert!(
            previous_step.is_none_or(|previous| step > previous),
            "bad steps"
        );
        *previous_step = Some(step);
    }
}
