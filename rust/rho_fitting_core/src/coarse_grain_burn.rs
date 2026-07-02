use burn::prelude::*;
use burn::tensor::TensorData;
use burn_wgpu::{graphics, init_setup, Metal, WgpuDevice};
use ndarray::{Array3, Array4, Array5, Array6, ArrayView1, ArrayView2, ArrayView3};
use std::panic::{catch_unwind, set_hook, take_hook, AssertUnwindSafe};

use crate::coarse_grain;
use crate::mechanics::MechanicalFields;
use crate::{CoreError, CoreResult};

type BurnBackend = Metal<f32, i32, u8>;

const GRID_CHUNK: usize = 512;

#[allow(clippy::too_many_arguments)]
pub fn coarse_grain_fields(
    coords: ArrayView3<'_, f64>,
    p_particles: ArrayView3<'_, f64>,
    shell_mask: ArrayView2<'_, bool>,
    x_centers: ArrayView1<'_, f64>,
    y_centers: ArrayView1<'_, f64>,
    lx: f64,
    ly: f64,
    radius: f64,
    sigma: f64,
) -> CoreResult<(Array3<f64>, Array4<f64>)> {
    coarse_grain::validate_inputs(
        coords,
        p_particles,
        shell_mask,
        x_centers,
        y_centers,
        lx,
        ly,
        radius,
        sigma,
    )?;

    catch_burn_panic(|| {
        let device = WgpuDevice::DefaultDevice;
        init_setup::<graphics::Metal>(&device, Default::default());
        coarse_grain_burn_device(
            coords,
            p_particles,
            shell_mask,
            x_centers,
            y_centers,
            lx,
            ly,
            radius,
            sigma,
            &device,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn build_mechanical_fields(
    coords: ArrayView3<'_, f64>,
    directions: ArrayView3<'_, f64>,
    velocities: ArrayView3<'_, f64>,
    mask: ArrayView2<'_, bool>,
    x_centers: ArrayView1<'_, f64>,
    y_centers: ArrayView1<'_, f64>,
    lx: f64,
    ly: f64,
    radius: f64,
    sigma: f64,
    gamma: f64,
    u0: f64,
) -> CoreResult<MechanicalFields> {
    coarse_grain::validate_inputs(
        coords, directions, mask, x_centers, y_centers, lx, ly, radius, sigma,
    )?;
    let (frames, particles, _) = coords.dim();
    if velocities.dim() != (frames, particles, 2) {
        return Err(CoreError::Shape(
            "velocities must have shape (T,N,2)".to_string(),
        ));
    }
    if !(gamma.is_finite() && u0.is_finite() && u0 != 0.0) {
        return Err(CoreError::InvalidInput(
            "gamma must be finite and u0 must be nonzero".to_string(),
        ));
    }

    catch_burn_panic(|| {
        let device = WgpuDevice::DefaultDevice;
        init_setup::<graphics::Metal>(&device, Default::default());
        mechanical_burn_device(
            coords, directions, velocities, mask, x_centers, y_centers, lx, ly, radius, sigma,
            &device,
        )
    })
}

fn catch_burn_panic<T>(func: impl FnOnce() -> CoreResult<T>) -> CoreResult<T> {
    let hook = take_hook();
    set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(func));
    set_hook(hook);
    result.map_err(|_| {
        CoreError::InvalidInput("Burn Metal coarse-grain initialization panicked".to_string())
    })?
}

#[allow(clippy::too_many_arguments)]
fn coarse_grain_burn_device(
    coords: ArrayView3<'_, f64>,
    p_particles: ArrayView3<'_, f64>,
    shell_mask: ArrayView2<'_, bool>,
    x_centers: ArrayView1<'_, f64>,
    y_centers: ArrayView1<'_, f64>,
    lx: f64,
    ly: f64,
    radius: f64,
    sigma: f64,
    device: &WgpuDevice,
) -> CoreResult<(Array3<f64>, Array4<f64>)> {
    let (frames, particles, _) = coords.dim();
    let nx = x_centers.len();
    let ny = y_centers.len();
    let grid = nx * ny;
    let mut rho = Array3::<f64>::zeros((frames, nx, ny));
    let mut p_density = Array4::<f64>::zeros((frames, nx, ny, 2));
    let x_grid = repeated_grid_values(x_centers, ny);
    let y_grid = tiled_grid_values(y_centers, nx);
    let norm = (1.0 / (2.0 * std::f64::consts::PI * sigma * sigma)) as f32;
    let sigma2 = (sigma * sigma) as f32;
    let cutoff2 = (16.0 * sigma * sigma) as f32;

    for t in 0..frames {
        println!("[rho_fitting] burn coarse-grain frame {}/{}", t + 1, frames);
        let particle_x = frame_component(coords, t, particles, 0, 1.0);
        let particle_y = frame_component(coords, t, particles, 1, radius);
        let px = frame_component(p_particles, t, particles, 0, 1.0);
        let py = frame_component(p_particles, t, particles, 1, 1.0);
        let mask = frame_mask(shell_mask, t, particles);

        let x_particles = tensor2(particle_x, [1, particles], device);
        let y_particles = tensor2(particle_y, [1, particles], device);
        let px_tensor = tensor2(px, [1, particles], device);
        let py_tensor = tensor2(py, [1, particles], device);
        let mask_tensor = tensor2(mask, [1, particles], device);

        for start in (0..grid).step_by(GRID_CHUNK) {
            let end = (start + GRID_CHUNK).min(grid);
            let chunk = end - start;
            let x_chunk = tensor2(x_grid[start..end].to_vec(), [chunk, 1], device);
            let y_chunk = tensor2(y_grid[start..end].to_vec(), [chunk, 1], device);
            let dx = minimum_image_tensor(x_chunk - x_particles.clone(), lx as f32);
            let dy = minimum_image_tensor(y_chunk - y_particles.clone(), ly as f32);
            let dist2 = dx.clone() * dx + dy.clone() * dy;
            let weights = (dist2.clone() * (-0.5 / sigma2)).exp() * norm;
            let weights = weights.mask_fill(dist2.greater_elem(cutoff2), 0.0);
            let weights = weights * mask_tensor.clone();
            let rho_chunk = weights.clone().sum_dim(1).reshape([chunk]);
            let px_chunk = (weights.clone() * px_tensor.clone())
                .sum_dim(1)
                .reshape([chunk]);
            let py_chunk = (weights * py_tensor.clone()).sum_dim(1).reshape([chunk]);

            let rho_values = tensor_vec(rho_chunk)?;
            let px_values = tensor_vec(px_chunk)?;
            let py_values = tensor_vec(py_chunk)?;
            for local in 0..chunk {
                let flat = start + local;
                let ix = flat / ny;
                let iy = flat % ny;
                rho[[t, ix, iy]] = rho_values[local] as f64;
                p_density[[t, ix, iy, 0]] = px_values[local] as f64;
                p_density[[t, ix, iy, 1]] = py_values[local] as f64;
            }
        }
    }

    Ok((rho, p_density))
}

fn minimum_image_tensor(tensor: Tensor<BurnBackend, 2>, period: f32) -> Tensor<BurnBackend, 2> {
    tensor.clone() - (tensor / period).round() * period
}

fn tensor2(values: Vec<f32>, shape: [usize; 2], device: &WgpuDevice) -> Tensor<BurnBackend, 2> {
    Tensor::<BurnBackend, 2>::from_data(TensorData::new(values, shape), device)
}

fn tensor_vec(tensor: Tensor<BurnBackend, 1>) -> CoreResult<Vec<f32>> {
    tensor
        .try_into_data()
        .map_err(|error| CoreError::InvalidInput(format!("Burn readback failed: {error}")))?
        .to_vec::<f32>()
        .map_err(|error| CoreError::InvalidInput(format!("Burn data conversion failed: {error}")))
}

#[allow(clippy::too_many_arguments)]
fn mechanical_burn_device(
    coords: ArrayView3<'_, f64>,
    directions: ArrayView3<'_, f64>,
    velocities: ArrayView3<'_, f64>,
    mask: ArrayView2<'_, bool>,
    x_centers: ArrayView1<'_, f64>,
    y_centers: ArrayView1<'_, f64>,
    lx: f64,
    ly: f64,
    radius: f64,
    sigma: f64,
    device: &WgpuDevice,
) -> CoreResult<MechanicalFields> {
    let (frames, particles, _) = coords.dim();
    let nx = x_centers.len();
    let ny = y_centers.len();
    let grid = nx * ny;
    let mut rho = Array3::<f64>::zeros((frames, nx, ny));
    let mut p = Array4::<f64>::zeros((frames, nx, ny, 2));
    let mut q = Array5::<f64>::zeros((frames, nx, ny, 2, 2));
    let mut a = Array5::<f64>::zeros((frames, nx, ny, 2, 2));
    let mut j_rho = Array4::<f64>::zeros((frames, nx, ny, 2));
    let mut j_p = Array5::<f64>::zeros((frames, nx, ny, 2, 2));
    let mut j_q = Array6::<f64>::zeros((frames, nx, ny, 2, 2, 2));
    let x_grid = repeated_grid_values(x_centers, ny);
    let y_grid = tiled_grid_values(y_centers, nx);
    let norm = (1.0 / (2.0 * std::f64::consts::PI * sigma * sigma)) as f32;
    let sigma2 = (sigma * sigma) as f32;
    let cutoff2 = (16.0 * sigma * sigma) as f32;

    for t in 0..frames {
        println!(
            "[rho_fitting] burn mechanical coarse-grain frame {}/{}",
            t + 1,
            frames
        );
        let particle_x = frame_component(coords, t, particles, 0, 1.0);
        let particle_y = frame_component(coords, t, particles, 1, radius);
        let dir = [
            frame_component(directions, t, particles, 0, 1.0),
            frame_component(directions, t, particles, 1, 1.0),
        ];
        let vel = [
            frame_component(velocities, t, particles, 0, 1.0),
            frame_component(velocities, t, particles, 1, 1.0),
        ];
        let q_particle = [
            combine_particle_components(&dir[0], &dir[0], -0.5),
            combine_particle_components(&dir[0], &dir[1], 0.0),
            combine_particle_components(&dir[1], &dir[0], 0.0),
            combine_particle_components(&dir[1], &dir[1], -0.5),
        ];
        let mask_tensor = tensor2(frame_mask(mask, t, particles), [1, particles], device);
        let x_particles = tensor2(particle_x, [1, particles], device);
        let y_particles = tensor2(particle_y, [1, particles], device);
        let dir_tensors = [
            tensor2(dir[0].clone(), [1, particles], device),
            tensor2(dir[1].clone(), [1, particles], device),
        ];
        let vel_tensors = [
            tensor2(vel[0].clone(), [1, particles], device),
            tensor2(vel[1].clone(), [1, particles], device),
        ];
        let q_tensors = [
            tensor2(q_particle[0].clone(), [1, particles], device),
            tensor2(q_particle[1].clone(), [1, particles], device),
            tensor2(q_particle[2].clone(), [1, particles], device),
            tensor2(q_particle[3].clone(), [1, particles], device),
        ];

        for start in (0..grid).step_by(GRID_CHUNK) {
            let end = (start + GRID_CHUNK).min(grid);
            let chunk = end - start;
            let x_chunk = tensor2(x_grid[start..end].to_vec(), [chunk, 1], device);
            let y_chunk = tensor2(y_grid[start..end].to_vec(), [chunk, 1], device);
            let dx = minimum_image_tensor(x_chunk - x_particles.clone(), lx as f32);
            let dy = minimum_image_tensor(y_chunk - y_particles.clone(), ly as f32);
            let dist2 = dx.clone() * dx + dy.clone() * dy;
            let weights = (dist2.clone() * (-0.5 / sigma2)).exp() * norm;
            let weights = weights.mask_fill(dist2.greater_elem(cutoff2), 0.0) * mask_tensor.clone();

            write_chunk3(
                &mut rho,
                t,
                ny,
                start,
                &tensor_vec(weights.clone().sum_dim(1).reshape([chunk]))?,
            );
            for component in 0..2 {
                write_chunk4(
                    &mut p,
                    t,
                    ny,
                    start,
                    component,
                    &weighted_sum(&weights, &dir_tensors[component], chunk)?,
                );
                write_chunk4(
                    &mut j_rho,
                    t,
                    ny,
                    start,
                    component,
                    &weighted_sum(&weights, &vel_tensors[component], chunk)?,
                );
            }
            for i in 0..2 {
                for j in 0..2 {
                    let q_index = i * 2 + j;
                    let q_values = weighted_sum(&weights, &q_tensors[q_index], chunk)?;
                    write_chunk5(&mut q, t, ny, start, i, j, &q_values);
                    for k in 0..2 {
                        let jp_values = weighted_sum_product(
                            &weights,
                            &vel_tensors[k],
                            &dir_tensors[i],
                            chunk,
                        )?;
                        write_chunk5(&mut j_p, t, ny, start, k, i, &jp_values);
                        let jq_values = weighted_sum_product(
                            &weights,
                            &vel_tensors[k],
                            &q_tensors[q_index],
                            chunk,
                        )?;
                        write_chunk6(&mut j_q, t, ny, start, k, i, j, &jq_values);
                    }
                }
            }
        }
    }

    a.assign(&q);
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for component in 0..2 {
                    a[[t, ix, iy, component, component]] += 0.5 * rho[[t, ix, iy]];
                }
            }
        }
    }
    Ok(MechanicalFields {
        rho,
        p,
        q,
        a,
        j_rho,
        j_p,
        j_q,
    })
}

fn weighted_sum(
    weights: &Tensor<BurnBackend, 2>,
    values: &Tensor<BurnBackend, 2>,
    chunk: usize,
) -> CoreResult<Vec<f32>> {
    tensor_vec(
        (weights.clone() * values.clone())
            .sum_dim(1)
            .reshape([chunk]),
    )
}

fn weighted_sum_product(
    weights: &Tensor<BurnBackend, 2>,
    left: &Tensor<BurnBackend, 2>,
    right: &Tensor<BurnBackend, 2>,
    chunk: usize,
) -> CoreResult<Vec<f32>> {
    tensor_vec(
        (weights.clone() * left.clone() * right.clone())
            .sum_dim(1)
            .reshape([chunk]),
    )
}

fn combine_particle_components(left: &[f32], right: &[f32], diagonal_shift: f32) -> Vec<f32> {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| a * b + diagonal_shift)
        .collect()
}

fn write_chunk3(out: &mut Array3<f64>, t: usize, ny: usize, start: usize, values: &[f32]) {
    for (local, value) in values.iter().enumerate() {
        let flat = start + local;
        out[[t, flat / ny, flat % ny]] = *value as f64;
    }
}

fn write_chunk4(
    out: &mut Array4<f64>,
    t: usize,
    ny: usize,
    start: usize,
    component: usize,
    values: &[f32],
) {
    for (local, value) in values.iter().enumerate() {
        let flat = start + local;
        out[[t, flat / ny, flat % ny, component]] = *value as f64;
    }
}

fn write_chunk5(
    out: &mut Array5<f64>,
    t: usize,
    ny: usize,
    start: usize,
    i: usize,
    j: usize,
    values: &[f32],
) {
    for (local, value) in values.iter().enumerate() {
        let flat = start + local;
        out[[t, flat / ny, flat % ny, i, j]] = *value as f64;
    }
}

fn write_chunk6(
    out: &mut Array6<f64>,
    t: usize,
    ny: usize,
    start: usize,
    k: usize,
    i: usize,
    j: usize,
    values: &[f32],
) {
    for (local, value) in values.iter().enumerate() {
        let flat = start + local;
        out[[t, flat / ny, flat % ny, k, i, j]] = *value as f64;
    }
}

fn repeated_grid_values(values: ArrayView1<'_, f64>, repeat: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len() * repeat);
    for value in values {
        for _ in 0..repeat {
            out.push(*value as f32);
        }
    }
    out
}

fn tiled_grid_values(values: ArrayView1<'_, f64>, tiles: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len() * tiles);
    for _ in 0..tiles {
        out.extend(values.iter().map(|value| *value as f32));
    }
    out
}

fn frame_component(
    values: ArrayView3<'_, f64>,
    frame: usize,
    particles: usize,
    component: usize,
    scale: f64,
) -> Vec<f32> {
    (0..particles)
        .map(|particle| (scale * values[[frame, particle, component]]) as f32)
        .collect()
}

fn frame_mask(mask: ArrayView2<'_, bool>, frame: usize, particles: usize) -> Vec<f32> {
    (0..particles)
        .map(|particle| if mask[[frame, particle]] { 1.0 } else { 0.0 })
        .collect()
}
