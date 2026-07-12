use ndarray::{ArrayView2, ArrayView3};

use super::grid::Grid3;
use crate::mechanics::{relative_mass_error, MechanicalFieldSet, MechanicalFrame};

pub(super) type FrameFields = MechanicalFrame;

pub(super) fn write_mechanical_frame(
    t: usize,
    grid: &Grid3,
    fields: &FrameFields,
    output: &mut MechanicalFieldSet,
) {
    fields.write_into(t, &grid.domain, output);
}

pub(super) fn print_conservation(label: &str, mass: &[f64], expected: usize) {
    let total: f64 = mass.iter().sum();
    let rel_error = relative_mass_error(mass, expected);
    println!(
        "[rho_fitting] Burn {label} mass conservation: total={:.6} expected={} rel_error={:.3e}",
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
