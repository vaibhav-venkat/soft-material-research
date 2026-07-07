use ndarray::{Array5, Array6, ArrayD, ArrayView2, ArrayView3, IxDyn};

use super::grid::{flat_index, Grid3};
use super::EPS;

pub(super) struct FrameFields {
    pub(super) mass: Vec<f32>,
    pub(super) psi6_num: Vec<f32>,
    pub(super) p_num: [Vec<f32>; 3],
    pub(super) q_num: [Vec<f32>; 9],
    pub(super) j_rho_num: [Vec<f32>; 3],
    pub(super) j_p_num: [Vec<f32>; 9],
    pub(super) j_q_num: [Vec<f32>; 27],
}

impl FrameFields {
    pub(super) fn new(grid: usize) -> Self {
        Self {
            mass: vec![0.0; grid],
            psi6_num: vec![0.0; grid],
            p_num: std::array::from_fn(|_| vec![0.0; grid]),
            q_num: std::array::from_fn(|_| vec![0.0; grid]),
            j_rho_num: std::array::from_fn(|_| vec![0.0; grid]),
            j_p_num: std::array::from_fn(|_| vec![0.0; grid]),
            j_q_num: std::array::from_fn(|_| vec![0.0; grid]),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_mechanical_frame(
    t: usize,
    grid: &Grid3,
    fields: &FrameFields,
    rho: &mut ndarray::Array4<f64>,
    psi6_sq: &mut ndarray::Array4<f64>,
    p: &mut Array5<f64>,
    q: &mut Array6<f64>,
    a: &mut Array6<f64>,
    j_rho: &mut Array5<f64>,
    j_p: &mut Array6<f64>,
    j_q: &mut ArrayD<f64>,
) {
    for ix in 0..grid.nx {
        for itheta in 0..grid.ntheta {
            for ir in 0..grid.nr {
                let flat = flat_index(ix, itheta, ir, grid.ntheta, grid.nr);
                let mass = fields.mass[flat];
                let volume = grid.volumes[flat];
                let density = mass / volume;
                rho[[t, ix, itheta, ir]] = density as f64;
                if mass > EPS {
                    let psi = fields.psi6_num[flat] / mass;
                    psi6_sq[[t, ix, itheta, ir]] = (psi * psi) as f64;
                    for component in 0..3 {
                        p[[t, ix, itheta, ir, component]] =
                            (fields.p_num[component][flat] / volume) as f64;
                        j_rho[[t, ix, itheta, ir, component]] =
                            (fields.j_rho_num[component][flat] / volume) as f64;
                    }
                    for row in 0..3 {
                        for col in 0..3 {
                            let q_index = row * 3 + col;
                            q[[t, ix, itheta, ir, row, col]] =
                                (fields.q_num[q_index][flat] / volume) as f64;
                            a[[t, ix, itheta, ir, row, col]] = q[[t, ix, itheta, ir, row, col]]
                                + if row == col {
                                    density as f64 / 3.0
                                } else {
                                    0.0
                                };
                            for flux in 0..3 {
                                j_p[[t, ix, itheta, ir, flux, row]] =
                                    (fields.j_p_num[flux * 3 + row][flat] / volume) as f64;
                                j_q[IxDyn(&[t, ix, itheta, ir, flux, row, col])] =
                                    (fields.j_q_num[flux * 9 + q_index][flat] / volume) as f64;
                            }
                        }
                    }
                }
            }
        }
    }
}

pub(super) fn print_conservation(label: &str, mass: &[f32], expected: usize) {
    let total: f64 = mass.iter().map(|value| *value as f64).sum();
    let rel_error = if expected > 0 {
        (total - expected as f64).abs() / expected as f64
    } else {
        0.0
    };
    println!(
        "[rho_fitting] GPU {label} mass conservation: total={:.6} expected={} rel_error={:.3e}",
        total, expected, rel_error
    );
}

pub(super) fn combine_particle_components(
    left: &[f32],
    right: &[f32],
    diagonal_shift: f32,
) -> Vec<f32> {
    left.iter()
        .zip(right.iter())
        .map(|(a_value, b_value)| a_value * b_value + diagonal_shift)
        .collect()
}

pub(super) fn frame_component(
    values: ArrayView3<'_, f64>,
    frame: usize,
    particles: usize,
    component: usize,
    scale: f64,
) -> Vec<f32> {
    (0..particles)
        .map(|particle| {
            let value = scale * values[[frame, particle, component]];
            if value.is_finite() {
                value as f32
            } else {
                0.0
            }
        })
        .collect()
}

pub(super) fn sanitized_frame_component(
    values: ArrayView3<'_, f64>,
    frame: usize,
    particles: usize,
    component: usize,
) -> Vec<f32> {
    frame_component(values, frame, particles, component, 1.0)
}

pub(super) fn sanitized_frame_component2(
    values: ArrayView3<'_, f64>,
    frame: usize,
    particles: usize,
    component: usize,
) -> Vec<f32> {
    (0..particles)
        .map(|particle| {
            let value = values[[frame, particle, component]];
            if value.is_finite() {
                value as f32
            } else {
                0.0
            }
        })
        .collect()
}

pub(super) fn sanitized_frame_scalar(
    values: ArrayView2<'_, f64>,
    frame: usize,
    particles: usize,
) -> Vec<f32> {
    (0..particles)
        .map(|particle| {
            let value = values[[frame, particle]];
            if value.is_finite() {
                value as f32
            } else {
                0.0
            }
        })
        .collect()
}

pub(super) fn surface_frame_mask(
    coords: ArrayView3<'_, f64>,
    p_particles: ArrayView3<'_, f64>,
    mask: ArrayView2<'_, bool>,
    frame: usize,
    particles: usize,
) -> Vec<f32> {
    (0..particles)
        .map(|particle| {
            let valid = mask[[frame, particle]]
                && (0..3).all(|component| coords[[frame, particle, component]].is_finite())
                && (0..2).all(|component| p_particles[[frame, particle, component]].is_finite());
            if valid {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

pub(super) fn mechanical_frame_mask(
    coords: ArrayView3<'_, f64>,
    directions: ArrayView3<'_, f64>,
    velocities: ArrayView3<'_, f64>,
    psi6_abs: ArrayView2<'_, f64>,
    mask: ArrayView2<'_, bool>,
    frame: usize,
    particles: usize,
) -> Vec<f32> {
    (0..particles)
        .map(|particle| {
            let valid = mask[[frame, particle]]
                && (0..3).all(|component| coords[[frame, particle, component]].is_finite())
                && (0..3).all(|component| directions[[frame, particle, component]].is_finite())
                && (0..3).all(|component| velocities[[frame, particle, component]].is_finite())
                && psi6_abs[[frame, particle]].is_finite();
            if valid {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}
