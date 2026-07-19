//! Streaming axial COM unwrapping and differentiation.

use crate::input::{resolve_shard_path, MappedShard, SafetensorView, TensorDtype};
use crate::model::{CaseSchema, ComSeries, DiscoveredDataset};
use ndarray::{Array1, ArrayView1, ArrayView3, Axis};

/// COM analysis controls shared by individual replicas.
#[derive(Clone, Copy, Debug)]
pub struct ComConfig {
    pub timestep: f64,
}

/// Stream one dataset and compute its unwrapped axial COM and velocity.
pub fn analyze_replica_com(dataset: &DiscoveredDataset, config: ComConfig) -> ComSeries {
    assert!(
        config.timestep.is_finite() && config.timestep > 0.0,
        "bad timestep"
    );
    let case = &dataset.manifest.case;
    let mut previous_wrapped = Array1::<f64>::zeros(case.n_particles);
    let mut unwrapped = Array1::<f64>::zeros(case.n_particles);
    let mut initialized = false;
    let mut steps = Vec::with_capacity(dataset.manifest.frame_count);
    let mut centers = Vec::with_capacity(dataset.manifest.frame_count);

    for shard in &dataset.manifest.shards {
        let path = resolve_shard_path(&dataset.manifest_path, &shard.file);
        let mapped = MappedShard::open(&path);
        let coordinate_name = coordinate_name(dataset.schema, &mapped);
        let coordinates = mapped.tensor(coordinate_name);
        let shard_steps = mapped.tensor("step");
        let frame_count = shard.frame_stop - shard.frame_start;
        validate_shapes(
            &path,
            coordinate_name,
            &coordinates,
            &shard_steps,
            frame_count,
            case.n_particles,
        );
        let shape = (frame_count, case.n_particles, coordinates.shape[2]);
        match coordinates.dtype {
            TensorDtype::F32 => process_shard(
                ArrayView3::from_shape(shape, coordinates.as_f32()).expect("bad shape"),
                &shard_steps,
                case.lx,
                &mut previous_wrapped,
                &mut unwrapped,
                &mut initialized,
                &mut steps,
                &mut centers,
            ),
            TensorDtype::F64 => process_shard(
                ArrayView3::from_shape(shape, coordinates.as_f64()).expect("bad shape"),
                &shard_steps,
                case.lx,
                &mut previous_wrapped,
                &mut unwrapped,
                &mut initialized,
                &mut steps,
                &mut centers,
            ),
            TensorDtype::Bool | TensorDtype::I8 | TensorDtype::I32 | TensorDtype::I64 => {
                panic!("bad dtype")
            }
        }
        // The mapping and all borrowed tensor views are dropped here.
    }

    assert_eq!(
        centers.len(),
        dataset.manifest.frame_count,
        "bad frame count"
    );
    assert!(centers.len() >= 2, "too few frames");
    let initial_step = steps[0] as f64;
    let elapsed_time: Vec<f64> = steps
        .iter()
        .map(|&step| (step as f64 - initial_step) * config.timestep)
        .collect();
    let velocity = finite_gradient(&centers, &elapsed_time);
    let zeros = vec![0.0; centers.len()];
    ComSeries {
        elapsed_time,
        x_center_mean: centers,
        x_center_std: zeros.clone(),
        x_velocity_mean: velocity,
        x_velocity_std: zeros,
        replicate_count: 1,
    }
}

/// Differentiate samples with NumPy-compatible endpoint behavior.
pub fn finite_gradient(values: &[f64], coordinates: &[f64]) -> Vec<f64> {
    assert_eq!(values.len(), coordinates.len(), "bad gradient");
    assert!(values.len() >= 2, "bad gradient");
    assert!(values.iter().all(|value| value.is_finite()), "bad values");
    assert!(
        coordinates.iter().all(|value| value.is_finite()),
        "bad time"
    );
    assert!(
        coordinates.windows(2).all(|pair| pair[1] > pair[0]),
        "bad time"
    );
    if values.len() == 2 {
        let slope = (values[1] - values[0]) / (coordinates[1] - coordinates[0]);
        return vec![slope, slope];
    }

    let count = values.len();
    let mut gradient = vec![0.0; count];
    for index in 1..count - 1 {
        let before = coordinates[index] - coordinates[index - 1];
        let after = coordinates[index + 1] - coordinates[index];
        let a = -after / (before * (before + after));
        let b = (after - before) / (before * after);
        let c = before / (after * (before + after));
        gradient[index] = a * values[index - 1] + b * values[index] + c * values[index + 1];
    }

    let first = coordinates[1] - coordinates[0];
    let second = coordinates[2] - coordinates[1];
    gradient[0] = -(2.0 * first + second) / (first * (first + second)) * values[0]
        + (first + second) / (first * second) * values[1]
        - first / (second * (first + second)) * values[2];

    let before = coordinates[count - 2] - coordinates[count - 3];
    let last = coordinates[count - 1] - coordinates[count - 2];
    gradient[count - 1] = last / (before * (before + last)) * values[count - 3]
        - (last + before) / (before * last) * values[count - 2]
        + (2.0 * last + before) / (last * (before + last)) * values[count - 1];
    assert!(
        gradient.iter().all(|value| value.is_finite()),
        "bad velocity"
    );
    gradient
}

fn coordinate_name(schema: CaseSchema, mapped: &MappedShard) -> &'static str {
    match schema {
        CaseSchema::BigLx if mapped.contains("coords") => "coords",
        CaseSchema::Confinement if mapped.contains("coords") => "coords",
        CaseSchema::Confinement if mapped.contains("position_cartesian") => "position_cartesian",
        _ => panic!("missing coords"),
    }
}

fn validate_shapes(
    path: &std::path::Path,
    coordinate_name: &str,
    coordinates: &SafetensorView<'_>,
    steps: &SafetensorView<'_>,
    frames: usize,
    particles: usize,
) {
    let _ = (path, coordinate_name);
    assert_eq!(coordinates.shape.len(), 3, "bad shape");
    assert_eq!(coordinates.shape[0], frames, "bad shape");
    assert_eq!(coordinates.shape[1], particles, "bad shape");
    assert_ne!(coordinates.shape[2], 0, "bad shape");
    assert_eq!(steps.shape, [frames], "bad shape");
    assert!(
        matches!(steps.dtype, TensorDtype::I32 | TensorDtype::I64),
        "bad dtype"
    );
}

fn step_at(steps: &SafetensorView<'_>, index: usize) -> i64 {
    match steps.dtype {
        TensorDtype::I32 => i64::from(steps.as_i32()[index]),
        TensorDtype::I64 => steps.as_i64()[index],
        TensorDtype::Bool | TensorDtype::F32 | TensorDtype::F64 | TensorDtype::I8 => {
            unreachable!("step dtype validated")
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn process_shard<T: Copy + Into<f64>>(
    coordinates: ArrayView3<'_, T>,
    shard_steps: &SafetensorView<'_>,
    lx: f64,
    previous_wrapped: &mut Array1<f64>,
    unwrapped: &mut Array1<f64>,
    initialized: &mut bool,
    steps: &mut Vec<i64>,
    centers: &mut Vec<f64>,
) {
    for local_frame in 0..coordinates.len_of(Axis(0)) {
        let step = step_at(shard_steps, local_frame);
        assert!(
            steps.last().is_none_or(|previous| step > *previous),
            "bad steps"
        );
        let frame = coordinates.index_axis(Axis(0), local_frame);
        let axial = frame.index_axis(Axis(1), 0);
        centers.push(update_unwrapped_center(
            axial,
            lx,
            previous_wrapped,
            unwrapped,
            initialized,
        ));
        steps.push(step);
    }
}

fn update_unwrapped_center<T: Copy + Into<f64>>(
    axial: ArrayView1<'_, T>,
    lx: f64,
    previous_wrapped: &mut Array1<f64>,
    unwrapped: &mut Array1<f64>,
    initialized: &mut bool,
) -> f64 {
    assert_eq!(axial.len(), previous_wrapped.len(), "bad shape");
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for ((&value, previous), accumulated) in axial
        .iter()
        .zip(previous_wrapped.iter_mut())
        .zip(unwrapped.iter_mut())
    {
        let wrapped = value.into();
        assert!(wrapped.is_finite(), "bad coords");
        if *initialized {
            let displacement = wrapped - *previous;
            *accumulated += displacement - lx * (displacement / lx).round();
        } else {
            *accumulated = wrapped;
        }
        *previous = wrapped;
        let corrected = *accumulated - compensation;
        let updated = sum + corrected;
        compensation = (updated - sum) - corrected;
        sum = updated;
    }
    *initialized = true;
    sum / previous_wrapped.len() as f64
}
