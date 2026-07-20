//! Geometry-aware structural and coherent-motion clusters on surface domains.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::f64::consts::{PI, TAU};

use kiddo::{KdTree, SquaredEuclidean};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::input::{resolve_shard_path, MappedShard, SafetensorView, TensorDtype};
use crate::model::{CaseSchema, DiscoveredDataset};

/// Numerical controls shared by structural and motion clustering.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct ClusterConfig {
    pub frame_start: usize,
    pub frame_stop: Option<usize>,
    pub lag_frames: usize,
    pub psi6_minimum: f64,
    pub misorientation_degrees: f64,
    pub neighbor_radius_diameters: f64,
    pub motion_cosine_minimum: f64,
    pub motion_rms_fraction: f64,
    pub motion_magnitude_ratio: f64,
    pub minimum_particles: usize,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            frame_start: 0,
            frame_stop: None,
            lag_frames: 1,
            psi6_minimum: 0.7,
            misorientation_degrees: 5.0,
            neighbor_radius_diameters: 1.7272,
            motion_cosine_minimum: 0.8,
            motion_rms_fraction: 0.1,
            motion_magnitude_ratio: 0.5,
            minimum_particles: 2,
        }
    }
}

impl ClusterConfig {
    /// Reject non-physical or numerically ambiguous controls.
    pub fn validate(self) {
        if let Some(frame_stop) = self.frame_stop {
            assert!(
                frame_stop > self.frame_start,
                "cluster frame stop must exceed frame start"
            );
        }
        assert!(self.lag_frames > 0, "cluster lag must be positive");
        assert!(
            self.psi6_minimum.is_finite() && (0.0..=1.0).contains(&self.psi6_minimum),
            "bad psi6 threshold"
        );
        assert!(
            self.misorientation_degrees.is_finite()
                && self.misorientation_degrees >= 0.0
                && self.misorientation_degrees <= 30.0,
            "bad misorientation threshold"
        );
        assert!(
            self.neighbor_radius_diameters.is_finite() && self.neighbor_radius_diameters > 0.0,
            "bad neighbor radius"
        );
        assert!(
            self.motion_cosine_minimum.is_finite()
                && (-1.0..=1.0).contains(&self.motion_cosine_minimum),
            "bad motion cosine"
        );
        assert!(
            self.motion_rms_fraction.is_finite() && self.motion_rms_fraction >= 0.0,
            "bad motion RMS fraction"
        );
        assert!(
            self.motion_magnitude_ratio.is_finite()
                && (0.0..=1.0).contains(&self.motion_magnitude_ratio),
            "bad motion magnitude ratio"
        );
        assert!(self.minimum_particles >= 2, "clusters need two particles");
    }
}

/// Which compatibility graph produced a cluster record.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ClusterKind {
    Structural,
    Motion,
}

/// One frame-local connected component and its compact membership.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ClusterRecord {
    pub kind: ClusterKind,
    pub frame_index: usize,
    pub step: i64,
    pub destination_frame_index: Option<usize>,
    pub destination_step: Option<i64>,
    pub local_id: usize,
    pub domain_id: i16,
    pub centroid: [f64; 2],
    pub particle_count: usize,
    pub equivalent_circumference: f64,
    pub surface_equivalent_circumference: f64,
    pub normalized_circumference: f64,
    pub members: Vec<u64>,
}

/// Cluster records computed for one input manifest/replicate.
#[derive(Clone, Debug)]
pub struct DatasetClusterAnalysis {
    pub case_id: String,
    pub frame_count: usize,
    pub particle_count: usize,
    pub structural: Vec<ClusterRecord>,
    pub motion: Vec<ClusterRecord>,
    pub snapshots: Vec<ClusterSnapshot>,
}

/// Particle positions and frame-local cluster assignments for static rendering.
#[derive(Clone, Debug)]
pub struct ClusterSnapshot {
    pub frame_index: usize,
    pub step: i64,
    pub motion_destination_frame_index: Option<usize>,
    pub motion_destination_step: Option<i64>,
    pub positions: Vec<[f64; 3]>,
    pub eligible: Vec<bool>,
    pub structural_cluster_ids: Vec<Option<usize>>,
    pub motion_cluster_ids: Vec<Option<usize>>,
}

/// A normalized fixed-bin histogram for one case and cluster kind.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ClusterHistogram {
    pub bin_edges: Vec<f64>,
    pub probabilities: Vec<f64>,
    pub sample_count: usize,
}

/// Pairwise membership agreement between one structural and one motion cluster.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct ClusterOverlapSample {
    pub jaccard: f64,
    pub shared_particle_count: usize,
}

/// Bin samples on a caller-supplied common range and normalize by cluster count.
pub fn cluster_probability_histogram(
    samples: &[f64],
    bins: usize,
    range: (f64, f64),
) -> ClusterHistogram {
    let width = validate_histogram_inputs(samples, bins, range) / bins as f64;
    let edges = (0..=bins)
        .map(|index| range.0 + index as f64 * width)
        .collect::<Vec<_>>();
    cluster_histogram_from_edges(samples, None, edges)
}

/// Bin cluster counts on logarithmically spaced circumference-ratio edges.
pub fn cluster_log_probability_histogram(
    samples: &[f64],
    bins: usize,
    range: (f64, f64),
) -> ClusterHistogram {
    validate_histogram_inputs(samples, bins, range);
    assert!(range.0 > 0.0, "log histogram range must be positive");
    assert!(
        samples.iter().all(|value| *value > 0.0),
        "log histogram samples must be positive"
    );
    let log_minimum = range.0.ln();
    let log_width = (range.1.ln() - log_minimum) / bins as f64;
    let mut edges = (0..=bins)
        .map(|index| (log_minimum + index as f64 * log_width).exp())
        .collect::<Vec<_>>();
    edges[0] = range.0;
    edges[bins] = range.1;
    cluster_histogram_from_edges(samples, None, edges)
}

/// Bin by equivalent-circumference ratio and weight by that linear size.
///
/// The surface equivalent circumference is constant within one physical case,
/// so the normalized ratio is proportional to cluster equivalent circumference.
pub fn cluster_circumference_weighted_probability_histogram(
    samples: &[f64],
    bins: usize,
    range: (f64, f64),
) -> ClusterHistogram {
    assert!(
        samples.iter().all(|value| *value > 0.0),
        "circumference weights must be positive"
    );
    cluster_weighted_probability_histogram(samples, samples, bins, range)
}

/// Bin samples linearly with caller-supplied non-negative weights.
pub fn cluster_weighted_probability_histogram(
    samples: &[f64],
    weights: &[f64],
    bins: usize,
    range: (f64, f64),
) -> ClusterHistogram {
    let width = validate_histogram_inputs(samples, bins, range) / bins as f64;
    let edges = (0..=bins)
        .map(|index| range.0 + index as f64 * width)
        .collect::<Vec<_>>();
    cluster_histogram_from_edges(samples, Some(weights), edges)
}

/// Compute same-frame structural–motion pair overlaps using particle membership.
pub fn structural_motion_overlap_samples(
    analysis: &DatasetClusterAnalysis,
) -> Vec<ClusterOverlapSample> {
    let mut structural_by_frame = BTreeMap::<usize, Vec<&ClusterRecord>>::new();
    for record in &analysis.structural {
        structural_by_frame
            .entry(record.frame_index)
            .or_default()
            .push(record);
    }
    let mut motion_by_frame = BTreeMap::<usize, Vec<&ClusterRecord>>::new();
    for record in &analysis.motion {
        motion_by_frame
            .entry(record.frame_index)
            .or_default()
            .push(record);
    }

    let mut overlaps = Vec::new();
    for (frame_index, motion_records) in motion_by_frame {
        let Some(structural_records) = structural_by_frame.get(&frame_index) else {
            continue;
        };
        let structural_assignments = cluster_assignments_for_records(
            analysis.particle_count,
            structural_records.iter().copied(),
        );
        let motion_assignments = cluster_assignments_for_records(
            analysis.particle_count,
            motion_records.iter().copied(),
        );
        let structural_sizes = structural_records
            .iter()
            .map(|record| (record.local_id, record.particle_count))
            .collect::<HashMap<_, _>>();
        let motion_sizes = motion_records
            .iter()
            .map(|record| (record.local_id, record.particle_count))
            .collect::<HashMap<_, _>>();
        let mut shared_counts = HashMap::<(usize, usize), usize>::new();
        for (structural_id, motion_id) in structural_assignments.into_iter().zip(motion_assignments)
        {
            if let (Some(structural_id), Some(motion_id)) = (structural_id, motion_id) {
                *shared_counts.entry((structural_id, motion_id)).or_default() += 1;
            }
        }
        for ((structural_id, motion_id), shared_particle_count) in shared_counts {
            let union_count =
                structural_sizes[&structural_id] + motion_sizes[&motion_id] - shared_particle_count;
            let jaccard = shared_particle_count as f64 / union_count as f64;
            assert!(
                jaccard.is_finite() && jaccard > 0.0 && jaccard <= 1.0,
                "bad structural-motion cluster overlap"
            );
            overlaps.push(ClusterOverlapSample {
                jaccard,
                shared_particle_count,
            });
        }
    }
    overlaps
}

fn validate_histogram_inputs(samples: &[f64], bins: usize, range: (f64, f64)) -> f64 {
    assert!(bins > 0, "cluster histogram needs bins");
    assert!(
        range.0.is_finite() && range.1.is_finite() && range.1 > range.0,
        "bad cluster histogram range"
    );
    assert!(
        samples.iter().all(|value| value.is_finite()),
        "bad cluster sample"
    );
    range.1 - range.0
}

fn cluster_histogram_from_edges(
    samples: &[f64],
    weights: Option<&[f64]>,
    bin_edges: Vec<f64>,
) -> ClusterHistogram {
    let bins = bin_edges.len() - 1;
    if let Some(weights) = weights {
        assert_eq!(
            weights.len(),
            samples.len(),
            "cluster weights are misaligned"
        );
        assert!(
            weights
                .iter()
                .all(|weight| weight.is_finite() && *weight >= 0.0),
            "bad cluster weight"
        );
    }
    let minimum = bin_edges[0];
    let maximum = bin_edges[bins];
    let mut probabilities = vec![0.0; bins];
    let mut total_weight = 0.0;
    for (sample_index, &sample) in samples.iter().enumerate() {
        assert!(
            sample >= minimum && sample <= maximum,
            "cluster sample outside range"
        );
        let index = if sample == maximum {
            bins - 1
        } else {
            bin_edges.partition_point(|edge| *edge <= sample) - 1
        };
        let weight = weights.map_or(1.0, |values| values[sample_index]);
        probabilities[index] += weight;
        total_weight += weight;
    }
    if total_weight > 0.0 {
        for probability in &mut probabilities {
            *probability /= total_weight;
        }
        let total: f64 = probabilities.iter().sum();
        assert!(
            (total - 1.0).abs() < 1.0e-12,
            "cluster probabilities do not sum to one"
        );
    }
    ClusterHistogram {
        bin_edges,
        probabilities,
        sample_count: samples.len(),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Geometry {
    Cylinder,
    Wall { periodic_tangent: bool },
    TwoDimensional,
}

#[derive(Clone, Debug)]
struct SurfaceFrame {
    frame_index: usize,
    step: i64,
    points: Vec<[f64; 2]>,
    positions: Vec<[f64; 3]>,
    domains: Vec<i16>,
    eligible: Vec<bool>,
    psi6: Vec<[f64; 2]>,
    neighbors: Vec<Vec<usize>>,
    periods: [Option<f64>; 2],
}

/// Stream one dataset, retaining only the configured lag window in memory.
pub fn analyze_dataset_clusters(
    dataset: &DiscoveredDataset,
    config: ClusterConfig,
) -> DatasetClusterAnalysis {
    analyze_dataset_clusters_with_snapshots(dataset, config, &[])
}

/// Stream one dataset and retain renderable state only for requested cylinder frames.
pub fn analyze_dataset_clusters_with_snapshots(
    dataset: &DiscoveredDataset,
    config: ClusterConfig,
    snapshot_frames: &[usize],
) -> DatasetClusterAnalysis {
    config.validate();
    assert_eq!(
        geometry(dataset),
        Geometry::Cylinder,
        "cluster analysis currently supports cylinders only"
    );
    let frame_start = config.frame_start;
    let frame_stop = config.frame_stop.unwrap_or(dataset.manifest.frame_count);
    assert!(
        frame_stop <= dataset.manifest.frame_count,
        "cluster frame stop exceeds trajectory"
    );
    assert!(
        frame_start < frame_stop,
        "cluster frame start must be inside the trajectory"
    );
    let analysis_frame_count = frame_stop - frame_start;
    assert!(
        analysis_frame_count > config.lag_frames,
        "cluster lag exceeds selected frame range"
    );
    let diameter = dataset
        .manifest
        .case
        .particle_diameter
        .expect("case metadata has no particle diameter");
    assert!(
        diameter.is_finite() && diameter > 0.0,
        "bad particle diameter"
    );
    let cutoff = diameter * config.neighbor_radius_diameters;
    let mut structural = Vec::new();
    let mut motion = Vec::new();
    let requested_snapshots = snapshot_frames.iter().copied().collect::<HashSet<_>>();
    let collect_snapshots = true;
    let mut snapshots = BTreeMap::new();
    let mut lag_window = VecDeque::with_capacity(config.lag_frames + 1);

    for shard in &dataset.manifest.shards {
        let selected_start = shard.frame_start.max(frame_start);
        let selected_stop = shard.frame_stop.min(frame_stop);
        if selected_start >= selected_stop {
            continue;
        }
        let path = resolve_shard_path(&dataset.manifest_path, &shard.file);
        let mapped = MappedShard::open(&path);
        for frame_index in selected_start..selected_stop {
            let local_frame = frame_index - shard.frame_start;
            let capture_snapshot = collect_snapshots && requested_snapshots.contains(&frame_index);
            let mut frame =
                load_surface_frame(dataset, &mapped, local_frame, frame_index, capture_snapshot);
            frame.neighbors = neighbor_graph(&frame, cutoff);
            if dataset.schema == CaseSchema::Confinement
                && dataset.manifest.case.geometry_kind.as_deref() == Some("two_dimension")
            {
                frame.psi6 = psi6_from_neighbors(&frame);
            }
            let frame_structural = structural_clusters(&frame, diameter, config);
            if capture_snapshot {
                snapshots.insert(
                    frame_index,
                    ClusterSnapshot {
                        frame_index,
                        step: frame.step,
                        motion_destination_frame_index: None,
                        motion_destination_step: None,
                        positions: frame.positions.clone(),
                        eligible: frame.eligible.clone(),
                        structural_cluster_ids: cluster_assignments(
                            frame.points.len(),
                            &frame_structural,
                        ),
                        motion_cluster_ids: vec![None; frame.points.len()],
                    },
                );
            }
            structural.extend(frame_structural);
            lag_window.push_back(frame);
            if lag_window.len() > config.lag_frames {
                let origin = lag_window.pop_front().expect("missing lag origin");
                let destination = lag_window.back().expect("missing lag destination");
                let frame_motion = motion_clusters(&origin, destination, diameter, config);
                if let Some(snapshot) = snapshots.get_mut(&origin.frame_index) {
                    snapshot.motion_destination_frame_index = Some(destination.frame_index);
                    snapshot.motion_destination_step = Some(destination.step);
                    snapshot.motion_cluster_ids =
                        cluster_assignments(origin.points.len(), &frame_motion);
                }
                motion.extend(frame_motion);
            }
        }
    }

    assert_eq!(
        lag_window.len(),
        config.lag_frames,
        "bad terminal cluster lag window"
    );
    DatasetClusterAnalysis {
        case_id: dataset.manifest.case.case_id.clone(),
        frame_count: dataset.manifest.frame_count,
        particle_count: dataset.manifest.case.n_particles,
        structural,
        motion,
        snapshots: snapshots.into_values().collect(),
    }
}

fn cluster_assignments(particle_count: usize, records: &[ClusterRecord]) -> Vec<Option<usize>> {
    cluster_assignments_for_records(particle_count, records)
}

fn cluster_assignments_for_records<'a>(
    particle_count: usize,
    records: impl IntoIterator<Item = &'a ClusterRecord>,
) -> Vec<Option<usize>> {
    let mut selected = vec![None; particle_count];
    for record in records {
        for &member in &record.members {
            let particle = usize::try_from(member).expect("bad cluster member");
            assert!(particle < particle_count, "cluster member is out of range");
            let replace = selected[particle].is_none_or(|(linear_size, local_id): (f64, usize)| {
                record.equivalent_circumference > linear_size
                    || (record.equivalent_circumference == linear_size
                        && record.local_id < local_id)
            });
            if replace {
                selected[particle] = Some((record.equivalent_circumference, record.local_id));
            }
        }
    }
    selected
        .into_iter()
        .map(|choice| choice.map(|(_, local_id)| local_id))
        .collect()
}

fn load_surface_frame(
    dataset: &DiscoveredDataset,
    mapped: &MappedShard,
    local_frame: usize,
    frame_index: usize,
    capture_positions: bool,
) -> SurfaceFrame {
    let case = &dataset.manifest.case;
    let particle_count = case.n_particles;
    let step = tensor_integer_at(&mapped.tensor("step"), local_frame);
    let geometry = geometry(dataset);
    let mut points = vec![[0.0; 2]; particle_count];
    let mut positions = capture_positions.then(|| vec![[0.0; 3]; particle_count]);
    let mut domains = vec![0_i16; particle_count];
    let mut eligible = vec![true; particle_count];
    let mut psi6 = vec![[0.0; 2]; particle_count];
    let periods;

    match geometry {
        Geometry::Cylinder => {
            let coordinates = mapped.tensor("coords");
            validate_frame_particle_tensor(&coordinates, local_frame, particle_count, 3);
            let radius = case.radius.expect("cylinder metadata has no radius");
            let circumference = case.circumference.unwrap_or(TAU * radius);
            assert!(radius > 0.0 && circumference > 0.0, "bad cylinder geometry");
            for particle in 0..particle_count {
                let x = tensor_float_3(&coordinates, local_frame, particle, 0);
                let theta = tensor_float_3(&coordinates, local_frame, particle, 1);
                points[particle] = [x, radius * theta];
                if let Some(positions) = &mut positions {
                    let radial = tensor_float_3(&coordinates, local_frame, particle, 2);
                    positions[particle] = [x, radial * theta.cos(), radial * theta.sin()];
                }
            }
            let mask_name = match dataset.schema {
                CaseSchema::BigLx => "hexatic_shell_mask",
                CaseSchema::Confinement => "surface_mask",
            };
            eligible = tensor_bool_frame(&mapped.tensor(mask_name), local_frame, particle_count);
            psi6 = psi6_frame(mapped, local_frame, particle_count);
            periods = [Some(case.lx), Some(circumference)];
        }
        Geometry::TwoDimensional => {
            let coordinates = mapped.tensor("coords");
            validate_frame_particle_tensor(&coordinates, local_frame, particle_count, 2);
            for particle in 0..particle_count {
                let x = tensor_float_3(&coordinates, local_frame, particle, 0);
                let y = tensor_float_3(&coordinates, local_frame, particle, 1);
                points[particle] = [x, y];
                if let Some(positions) = &mut positions {
                    positions[particle] = [x, y, 0.0];
                }
            }
            periods = [Some(case.lx), None];
        }
        Geometry::Wall { periodic_tangent } => {
            let position_tensor = mapped.tensor("position_cartesian");
            let tangents = mapped.tensor("face_tangent");
            validate_frame_particle_tensor(&position_tensor, local_frame, particle_count, 3);
            validate_frame_particle_tensor(&tangents, local_frame, particle_count, 3);
            domains = tensor_i8_frame(&mapped.tensor("face_id"), local_frame, particle_count)
                .into_iter()
                .map(i16::from)
                .collect();
            eligible = tensor_bool_frame(
                &mapped.tensor("face_hexatic_valid"),
                local_frame,
                particle_count,
            );
            for particle in 0..particle_count {
                let position = [
                    tensor_float_3(&position_tensor, local_frame, particle, 0),
                    tensor_float_3(&position_tensor, local_frame, particle, 1),
                    tensor_float_3(&position_tensor, local_frame, particle, 2),
                ];
                if let Some(positions) = &mut positions {
                    positions[particle] = position;
                }
                let tangent = [
                    tensor_float_3(&tangents, local_frame, particle, 0),
                    tensor_float_3(&tangents, local_frame, particle, 1),
                    tensor_float_3(&tangents, local_frame, particle, 2),
                ];
                points[particle] = [
                    position[0],
                    position
                        .iter()
                        .zip(tangent)
                        .map(|(&value, direction)| value * direction)
                        .sum(),
                ];
            }
            psi6 = psi6_frame(mapped, local_frame, particle_count);
            periods = [
                Some(case.lx),
                periodic_tangent.then(|| {
                    case.transverse_span
                        .expect("sandwich metadata has no transverse span")
                }),
            ];
        }
    }
    assert!(
        points.iter().flatten().all(|value| value.is_finite()),
        "bad coordinates"
    );
    assert!(
        positions
            .iter()
            .flatten()
            .flatten()
            .all(|value| value.is_finite()),
        "bad Cartesian positions"
    );
    SurfaceFrame {
        frame_index,
        step,
        points,
        positions: positions.unwrap_or_default(),
        domains,
        eligible,
        psi6,
        neighbors: Vec::new(),
        periods,
    }
}

fn geometry(dataset: &DiscoveredDataset) -> Geometry {
    if dataset.schema == CaseSchema::BigLx {
        return Geometry::Cylinder;
    }
    match dataset.manifest.case.geometry_kind.as_deref() {
        Some("cylinder_rattle" | "cylinder_rattle_tangent") => Geometry::Cylinder,
        Some("two_dimension") => Geometry::TwoDimensional,
        Some("sandwich_volume" | "sandwich_surface_area") => Geometry::Wall {
            periodic_tangent: true,
        },
        Some("prism_volume" | "prism_surface_area") => Geometry::Wall {
            periodic_tangent: false,
        },
        _ => panic!("unsupported confinement geometry"),
    }
}

fn neighbor_graph(frame: &SurfaceFrame, cutoff: f64) -> Vec<Vec<usize>> {
    let particle_count = frame.points.len();
    let mut graph = vec![Vec::new(); particle_count];
    let mut by_domain: BTreeMap<i16, Vec<usize>> = BTreeMap::new();
    for particle in 0..particle_count {
        if frame.eligible[particle] {
            by_domain
                .entry(frame.domains[particle])
                .or_default()
                .push(particle);
        }
    }
    for particles in by_domain.values() {
        let mut tree: KdTree<f64, 2> = KdTree::new();
        for &particle in particles {
            for shift_x in periodic_shifts(frame.periods[0]) {
                for shift_y in periodic_shifts(frame.periods[1]) {
                    let point = frame.points[particle];
                    tree.add(
                        &[point[0] + shift_x, point[1] + shift_y],
                        u64::try_from(particle).expect("particle ID is too large"),
                    );
                }
            }
        }
        let domain_neighbors = particles
            .par_iter()
            .map(|&particle| {
                let mut neighbors = tree
                    .within_unsorted::<SquaredEuclidean>(&frame.points[particle], cutoff * cutoff)
                    .into_iter()
                    .map(|neighbor| usize::try_from(neighbor.item).expect("bad particle ID"))
                    .filter(|&neighbor| neighbor != particle)
                    .collect::<Vec<_>>();
                neighbors.sort_unstable();
                neighbors.dedup();
                (particle, neighbors)
            })
            .collect::<Vec<_>>();
        for (particle, neighbors) in domain_neighbors {
            graph[particle] = neighbors;
        }
    }
    graph
}

fn periodic_shifts(period: Option<f64>) -> Vec<f64> {
    match period {
        Some(period) => {
            assert!(period.is_finite() && period > 0.0, "bad period");
            vec![-period, 0.0, period]
        }
        None => vec![0.0],
    }
}

fn psi6_from_neighbors(frame: &SurfaceFrame) -> Vec<[f64; 2]> {
    frame
        .neighbors
        .par_iter()
        .enumerate()
        .map(|(particle, neighbors)| {
            if neighbors.is_empty() {
                return [0.0, 0.0];
            }
            let mut real = 0.0;
            let mut imaginary = 0.0;
            for &neighbor in neighbors {
                let delta = displacement(
                    frame.points[particle],
                    frame.points[neighbor],
                    frame.periods,
                );
                let angle = delta[1].atan2(delta[0]);
                real += (6.0 * angle).cos();
                imaginary += (6.0 * angle).sin();
            }
            let count = neighbors.len() as f64;
            [real / count, imaginary / count]
        })
        .collect()
}

fn structural_clusters(
    frame: &SurfaceFrame,
    particle_diameter: f64,
    config: ClusterConfig,
) -> Vec<ClusterRecord> {
    let ordered = frame
        .psi6
        .iter()
        .zip(&frame.eligible)
        .map(|(psi, &eligible)| eligible && psi[0].hypot(psi[1]) > config.psi6_minimum)
        .collect::<Vec<_>>();
    let orientations = frame
        .psi6
        .iter()
        .map(|psi| psi[1].atan2(psi[0]) / 6.0)
        .collect::<Vec<_>>();
    let maximum_misorientation = config.misorientation_degrees.to_radians();
    components_from_edges(
        frame,
        |left, right| {
            ordered[left]
                && ordered[right]
                && lattice_misorientation(orientations[left], orientations[right])
                    < maximum_misorientation
        },
        ClusterKind::Structural,
        None,
        particle_diameter,
        config.minimum_particles,
    )
}

fn motion_clusters(
    origin: &SurfaceFrame,
    destination: &SurfaceFrame,
    particle_diameter: f64,
    config: ClusterConfig,
) -> Vec<ClusterRecord> {
    assert_eq!(
        origin.points.len(),
        destination.points.len(),
        "particle count changed"
    );
    let count = origin.points.len();
    let paired = (0..count)
        .map(|particle| {
            origin.eligible[particle]
                && destination.eligible[particle]
                && origin.domains[particle] == destination.domains[particle]
        })
        .collect::<Vec<_>>();
    let raw = (0..count)
        .map(|particle| {
            displacement(
                origin.points[particle],
                destination.points[particle],
                origin.periods,
            )
        })
        .collect::<Vec<_>>();
    let mut sums: HashMap<i16, ([f64; 2], usize)> = HashMap::new();
    for particle in 0..count {
        if paired[particle] {
            let entry = sums
                .entry(origin.domains[particle])
                .or_insert(([0.0; 2], 0));
            entry.0[0] += raw[particle][0];
            entry.0[1] += raw[particle][1];
            entry.1 += 1;
        }
    }
    let means = sums
        .iter()
        .map(|(&domain, &(sum, count))| {
            assert!(count > 0, "empty motion domain");
            (domain, [sum[0] / count as f64, sum[1] / count as f64])
        })
        .collect::<HashMap<_, _>>();
    let residual = (0..count)
        .map(|particle| {
            if !paired[particle] {
                return [0.0, 0.0];
            }
            let mean = means[&origin.domains[particle]];
            [raw[particle][0] - mean[0], raw[particle][1] - mean[1]]
        })
        .collect::<Vec<_>>();
    let mut square_sums: HashMap<i16, (f64, usize)> = HashMap::new();
    for particle in 0..count {
        if paired[particle] {
            let entry = square_sums
                .entry(origin.domains[particle])
                .or_insert((0.0, 0));
            entry.0 += residual[particle][0].mul_add(
                residual[particle][0],
                residual[particle][1] * residual[particle][1],
            );
            entry.1 += 1;
        }
    }
    let rms = square_sums
        .into_iter()
        .map(|(domain, (sum, count))| (domain, (sum / count as f64).sqrt()))
        .collect::<HashMap<_, _>>();
    let magnitudes = residual
        .iter()
        .map(|value| value[0].hypot(value[1]))
        .collect::<Vec<_>>();
    let active = (0..count)
        .map(|particle| {
            paired[particle]
                && rms[&origin.domains[particle]] > 0.0
                && magnitudes[particle]
                    >= config.motion_rms_fraction * rms[&origin.domains[particle]]
        })
        .collect::<Vec<_>>();

    components_from_edges(
        origin,
        |left, right| {
            if !active[left] || !active[right] {
                return false;
            }
            let denominator = magnitudes[left] * magnitudes[right];
            let cosine = residual[left][0]
                .mul_add(residual[right][0], residual[left][1] * residual[right][1])
                / denominator;
            let ratio =
                magnitudes[left].min(magnitudes[right]) / magnitudes[left].max(magnitudes[right]);
            cosine > config.motion_cosine_minimum && ratio >= config.motion_magnitude_ratio
        },
        ClusterKind::Motion,
        Some(destination),
        particle_diameter,
        config.minimum_particles,
    )
}

#[allow(clippy::too_many_arguments)]
fn components_from_edges(
    frame: &SurfaceFrame,
    compatible: impl Fn(usize, usize) -> bool,
    kind: ClusterKind,
    destination: Option<&SurfaceFrame>,
    particle_diameter: f64,
    minimum_particles: usize,
) -> Vec<ClusterRecord> {
    let mut components = DisjointSet::new(frame.points.len());
    let mut present = vec![false; frame.points.len()];
    for left in 0..frame.points.len() {
        for &right in &frame.neighbors[left] {
            if left < right && compatible(left, right) {
                components.union(left, right);
                present[left] = true;
                present[right] = true;
            }
        }
    }
    let mut grouped: HashMap<usize, Vec<usize>> = HashMap::new();
    for particle in 0..frame.points.len() {
        if present[particle] {
            grouped
                .entry(components.find(particle))
                .or_default()
                .push(particle);
        }
    }
    let mut groups = grouped
        .into_values()
        .filter(|members| members.len() >= minimum_particles)
        .collect::<Vec<_>>();
    for members in &mut groups {
        members.sort_unstable();
    }
    groups.sort_by_key(|members| members[0]);
    groups
        .into_iter()
        .enumerate()
        .map(|(local_id, members)| {
            let domain_id = frame.domains[members[0]];
            assert!(
                members
                    .iter()
                    .all(|&particle| frame.domains[particle] == domain_id),
                "cluster crosses surface domains"
            );
            let particle_count = members.len();
            let axial_period = frame.periods[0].expect("axial period missing");
            let cylinder_circumference = frame.periods[1].expect("circumference missing");
            let equivalent_circumference = PI * particle_diameter * (particle_count as f64).sqrt();
            let surface_equivalent_circumference =
                2.0 * (PI * axial_period * cylinder_circumference).sqrt();
            assert!(
                equivalent_circumference.is_finite()
                    && equivalent_circumference > 0.0
                    && surface_equivalent_circumference.is_finite()
                    && surface_equivalent_circumference > 0.0,
                "bad equivalent circumference"
            );
            ClusterRecord {
                kind,
                frame_index: frame.frame_index,
                step: frame.step,
                destination_frame_index: destination.map(|value| value.frame_index),
                destination_step: destination.map(|value| value.step),
                local_id,
                domain_id,
                centroid: component_centroid(frame, &members),
                particle_count,
                equivalent_circumference,
                surface_equivalent_circumference,
                normalized_circumference: equivalent_circumference
                    / surface_equivalent_circumference,
                members: members
                    .into_iter()
                    .map(|particle| u64::try_from(particle).expect("particle ID is too large"))
                    .collect(),
            }
        })
        .collect()
}

fn component_centroid(frame: &SurfaceFrame, members: &[usize]) -> [f64; 2] {
    let included = members.iter().copied().collect::<HashSet<_>>();
    let anchor = members[0];
    let mut unwrapped = HashMap::from([(anchor, frame.points[anchor])]);
    let mut queue = VecDeque::from([anchor]);
    while let Some(particle) = queue.pop_front() {
        let base = unwrapped[&particle];
        for &neighbor in &frame.neighbors[particle] {
            if included.contains(&neighbor) && !unwrapped.contains_key(&neighbor) {
                let delta = displacement(
                    frame.points[particle],
                    frame.points[neighbor],
                    frame.periods,
                );
                unwrapped.insert(neighbor, [base[0] + delta[0], base[1] + delta[1]]);
                queue.push_back(neighbor);
            }
        }
    }
    assert_eq!(
        unwrapped.len(),
        members.len(),
        "cluster graph is disconnected"
    );
    let mut centroid = [0.0; 2];
    for particle in members {
        centroid[0] += unwrapped[particle][0];
        centroid[1] += unwrapped[particle][1];
    }
    centroid[0] /= members.len() as f64;
    centroid[1] /= members.len() as f64;
    for axis in 0..2 {
        if let Some(period) = frame.periods[axis] {
            centroid[axis] = centroid[axis].rem_euclid(period);
        }
    }
    assert!(
        centroid.iter().all(|value| value.is_finite()),
        "bad centroid"
    );
    centroid
}

fn displacement(from: [f64; 2], to: [f64; 2], periods: [Option<f64>; 2]) -> [f64; 2] {
    let mut delta = [to[0] - from[0], to[1] - from[1]];
    for axis in 0..2 {
        if let Some(period) = periods[axis] {
            delta[axis] -= period * (delta[axis] / period).round();
        }
    }
    delta
}

fn lattice_misorientation(left: f64, right: f64) -> f64 {
    let period = PI / 3.0;
    let difference = (left - right).rem_euclid(period);
    difference.min(period - difference)
}

fn psi6_frame(mapped: &MappedShard, frame: usize, particles: usize) -> Vec<[f64; 2]> {
    let real = mapped.tensor("psi_real");
    let imaginary = mapped.tensor("psi_imag");
    validate_frame_particle_tensor(&real, frame, particles, 1);
    validate_frame_particle_tensor(&imaginary, frame, particles, 1);
    (0..particles)
        .map(|particle| {
            [
                tensor_float_2(&real, frame, particle),
                tensor_float_2(&imaginary, frame, particle),
            ]
        })
        .collect()
}

fn validate_frame_particle_tensor(
    tensor: &SafetensorView<'_>,
    frame: usize,
    particles: usize,
    components: usize,
) {
    let expected_rank = if components == 1 { 2 } else { 3 };
    assert_eq!(
        tensor.shape.len(),
        expected_rank,
        "bad cluster tensor shape"
    );
    assert!(frame < tensor.shape[0], "bad cluster frame index");
    assert_eq!(tensor.shape[1], particles, "bad cluster particle count");
    if components > 1 {
        assert_eq!(tensor.shape[2], components, "bad cluster components");
    }
}

fn tensor_float_2(tensor: &SafetensorView<'_>, frame: usize, particle: usize) -> f64 {
    let index = frame * tensor.shape[1] + particle;
    tensor_float_at(tensor, index)
}

fn tensor_float_3(
    tensor: &SafetensorView<'_>,
    frame: usize,
    particle: usize,
    component: usize,
) -> f64 {
    let index = (frame * tensor.shape[1] + particle) * tensor.shape[2] + component;
    tensor_float_at(tensor, index)
}

fn tensor_float_at(tensor: &SafetensorView<'_>, index: usize) -> f64 {
    match tensor.dtype {
        TensorDtype::F32 => f64::from(tensor.as_f32()[index]),
        TensorDtype::F64 => tensor.as_f64()[index],
        _ => panic!("cluster field is not floating point"),
    }
}

fn tensor_integer_at(tensor: &SafetensorView<'_>, index: usize) -> i64 {
    match tensor.dtype {
        TensorDtype::I32 => i64::from(tensor.as_i32()[index]),
        TensorDtype::I64 => tensor.as_i64()[index],
        _ => panic!("cluster step is not an integer"),
    }
}

fn tensor_bool_frame(tensor: &SafetensorView<'_>, frame: usize, particles: usize) -> Vec<bool> {
    validate_frame_particle_tensor(tensor, frame, particles, 1);
    let start = frame * particles;
    match tensor.dtype {
        TensorDtype::Bool => tensor.as_bool_bytes()[start..start + particles]
            .iter()
            .map(|&value| value != 0)
            .collect(),
        _ => panic!("cluster mask is not boolean"),
    }
}

fn tensor_i8_frame(tensor: &SafetensorView<'_>, frame: usize, particles: usize) -> Vec<i8> {
    validate_frame_particle_tensor(tensor, frame, particles, 1);
    assert_eq!(tensor.dtype, TensorDtype::I8, "face ID is not I8");
    let start = frame * particles;
    tensor.as_i8()[start..start + particles].to_vec()
}

struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(count: usize) -> Self {
        Self {
            parent: (0..count).collect(),
            rank: vec![0; count],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        if self.parent[value] != value {
            self.parent[value] = self.find(self.parent[value]);
        }
        self.parent[value]
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        if self.rank[left_root] < self.rank[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = left_root;
        if self.rank[left_root] == self.rank[right_root] {
            self.rank[left_root] += 1;
        }
    }
}
