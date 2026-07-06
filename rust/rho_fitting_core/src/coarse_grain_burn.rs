use burn::prelude::*;
use burn::tensor::TensorData;
use burn_wgpu::{graphics, init_setup, Metal, WgpuDevice};
use ndarray::{Array3, Array4, Array5, Array6, ArrayD, ArrayView1, ArrayView2, ArrayView3, IxDyn};
use std::panic::{catch_unwind, set_hook, take_hook, AssertUnwindSafe};

use crate::coarse_grain;
use crate::mechanics::MechanicalFields;
use crate::{CoreError, CoreResult};

type BurnBackend = Metal<f32, i32, u8>;

const GRID_CHUNK: usize = 512;

struct CoarseGrainInputs<'a> {
    coords: ArrayView3<'a, f64>,
    p_particles: ArrayView3<'a, f64>,
    shell_mask: ArrayView2<'a, bool>,
    x_centers: ArrayView1<'a, f64>,
    y_centers: ArrayView1<'a, f64>,
    lx: f64,
    ly: f64,
    radius: f64,
    sigma: f64,
}

struct MechanicalInputs<'a> {
    coords: ArrayView3<'a, f64>,
    directions: ArrayView3<'a, f64>,
    velocities: ArrayView3<'a, f64>,
    psi6_abs: ArrayView2<'a, f64>,
    mask: ArrayView2<'a, bool>,
    x_centers: ArrayView1<'a, f64>,
    theta_centers: ArrayView1<'a, f64>,
    r_centers: ArrayView1<'a, f64>,
    lx: f64,
    theta_period: f64,
    sigma: f64,
}

/// Coarse-grain density and two-component polarization density on the Metal backend.
///
/// `coords` is `(T,N,3)`, `p_particles` is `(T,N,2)`, and `shell_mask` is
/// `(T,N)`. Returns `rho` shaped `(T,Nx,Ny)` and `P_density` shaped
/// `(T,Nx,Ny,2)`.
///
/// Example: `coarse_grain_fields(coords, p_particles, mask, x_centers, y_centers, lx, ly, radius, sigma)`.
///
/// Edge cases: computation runs in `f32` on the GPU, and Burn initialization
/// panics are converted to `CoreError` so the Python wrapper can fall back.
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
        let inputs = CoarseGrainInputs {
            coords,
            p_particles,
            shell_mask,
            x_centers,
            y_centers,
            lx,
            ly,
            radius,
            sigma,
        };
        coarse_grain_burn_device(inputs, &device)
    })
}

/// Build mechanical moment fields and current tensors on the Metal backend.
///
/// Particle arrays use `(T,N,component)` axes, output moments use 3D
/// orientation components, and fluxes use 3D cylindrical directions.
///
/// Example: `build_mechanical_fields(coords, directions, velocities, psi6, mask, xs, thetas, rs, lx, 2*pi, sigma, gamma, u0)`.
///
/// Edge cases: this path validates shapes locally, computes in `f32`, and leaves
/// `psi6_sq` as `NaN` where coarse-grained density is zero.
#[allow(clippy::too_many_arguments)]
pub fn build_mechanical_fields(
    coords: ArrayView3<'_, f64>,
    directions: ArrayView3<'_, f64>,
    velocities: ArrayView3<'_, f64>,
    psi6_abs: ArrayView2<'_, f64>,
    mask: ArrayView2<'_, bool>,
    x_centers: ArrayView1<'_, f64>,
    theta_centers: ArrayView1<'_, f64>,
    r_centers: ArrayView1<'_, f64>,
    lx: f64,
    theta_period: f64,
    sigma: f64,
    gamma: f64,
    u0: f64,
) -> CoreResult<MechanicalFields> {
    let (frames, particles, _) = coords.dim();
    if coords.dim().2 != 3 {
        return Err(CoreError::Shape(
            "coords must have shape (T,N,3)".to_string(),
        ));
    }
    if directions.dim() != (frames, particles, 3) || velocities.dim() != (frames, particles, 3) {
        return Err(CoreError::Shape(
            "directions and velocities must have shape (T,N,3)".to_string(),
        ));
    }
    if psi6_abs.dim() != (frames, particles) {
        return Err(CoreError::Shape(
            "psi6_abs must have shape (T,N)".to_string(),
        ));
    }
    if mask.dim() != (frames, particles) {
        return Err(CoreError::Shape("mask must have shape (T,N)".to_string()));
    }
    if x_centers.is_empty() || theta_centers.is_empty() || r_centers.is_empty() {
        return Err(CoreError::InvalidInput(
            "grid centers must be non-empty".to_string(),
        ));
    }
    if !(lx.is_finite()
        && lx > 0.0
        && theta_period.is_finite()
        && theta_period > 0.0
        && sigma.is_finite()
        && sigma > 0.0)
    {
        return Err(CoreError::InvalidInput(
            "geometry values must be positive".to_string(),
        ));
    }
    if !(gamma.is_finite() && u0.is_finite() && u0 != 0.0) {
        return Err(CoreError::InvalidInput(
            "gamma must be finite and u0 must be nonzero".to_string(),
        ));
    }
    if !r_centers.iter().all(|value| value.is_finite() && *value > 0.0) {
        return Err(CoreError::InvalidInput(
            "radial centers must be finite and positive".to_string(),
        ));
    }

    catch_burn_panic(|| {
        let device = WgpuDevice::DefaultDevice;
        init_setup::<graphics::Metal>(&device, Default::default());
        let inputs = MechanicalInputs {
            coords,
            directions,
            velocities,
            psi6_abs,
            mask,
            x_centers,
            theta_centers,
            r_centers,
            lx,
            theta_period,
            sigma,
        };
        mechanical_burn_device(inputs, &device)
    })
}

fn catch_burn_panic<T>(func: impl FnOnce() -> CoreResult<T>) -> CoreResult<T> {
    // Suppress Burn's panic hook and report initialization panics as recoverable errors.
    let hook = take_hook();
    set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(func));
    set_hook(hook);
    result.map_err(|_| {
        CoreError::InvalidInput("Burn Metal coarse-grain initialization panicked".to_string())
    })?
}

fn coarse_grain_burn_device(
    inputs: CoarseGrainInputs<'_>,
    device: &WgpuDevice,
) -> CoreResult<(Array3<f64>, Array4<f64>)> {
    // Execute legacy coarse-graining on an already-initialized Burn device.
    let CoarseGrainInputs {
        coords,
        p_particles,
        shell_mask,
        x_centers,
        y_centers,
        lx,
        ly,
        radius,
        sigma,
    } = inputs;
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
    // Wrap tensor displacements into the centered periodic minimum-image interval.
    tensor.clone() - (tensor / period).round() * period
}

fn tensor2(values: Vec<f32>, shape: [usize; 2], device: &WgpuDevice) -> Tensor<BurnBackend, 2> {
    // Move a row/column matrix of f32 values onto the Burn device.
    Tensor::<BurnBackend, 2>::from_data(TensorData::new(values, shape), device)
}

fn tensor_vec(tensor: Tensor<BurnBackend, 1>) -> CoreResult<Vec<f32>> {
    // Read a one-dimensional Burn tensor back to host memory as f32 values.
    tensor
        .try_into_data()
        .map_err(|error| CoreError::InvalidInput(format!("Burn readback failed: {error}")))?
        .to_vec::<f32>()
        .map_err(|error| CoreError::InvalidInput(format!("Burn data conversion failed: {error}")))
}

fn mechanical_burn_device(
    inputs: MechanicalInputs<'_>,
    device: &WgpuDevice,
) -> CoreResult<MechanicalFields> {
    let MechanicalInputs {
        coords,
        directions,
        velocities,
        psi6_abs,
        mask,
        x_centers,
        theta_centers,
        r_centers,
        lx,
        theta_period,
        sigma,
    } = inputs;
    let (frames, particles, _) = coords.dim();
    let nx = x_centers.len();
    let ntheta = theta_centers.len();
    let nr = r_centers.len();
    let grid = nx * ntheta * nr;
    let mut rho = ndarray::Array4::<f64>::zeros((frames, nx, ntheta, nr));
    let mut psi6 = ndarray::Array4::<f64>::zeros((frames, nx, ntheta, nr));
    let mut p = Array5::<f64>::zeros((frames, nx, ntheta, nr, 3));
    let mut q = Array6::<f64>::zeros((frames, nx, ntheta, nr, 3, 3));
    let mut j_rho = Array5::<f64>::zeros((frames, nx, ntheta, nr, 3));
    let mut j_p = Array6::<f64>::zeros((frames, nx, ntheta, nr, 3, 3));
    let mut j_q = ArrayD::<f64>::zeros(IxDyn(&[frames, nx, ntheta, nr, 3, 3, 3]));
    let x_grid = repeated_3d_x_values(x_centers, ntheta, nr);
    let theta_grid = repeated_3d_theta_values(theta_centers, nx, nr);
    let r_grid = tiled_3d_r_values(r_centers, nx, ntheta);
    let norm = (1.0 / ((2.0 * std::f64::consts::PI).powf(1.5) * sigma * sigma * sigma)) as f32;
    let sigma2 = (sigma * sigma) as f32;
    let cutoff2 = (16.0 * sigma * sigma) as f32;

    for t in 0..frames {
        println!(
            "[rho_fitting] burn mechanical coarse-grain frame {}/{}",
            t + 1,
            frames
        );
        let valid = mechanical_frame_mask(coords, directions, velocities, psi6_abs, mask, t, particles);
        let particle_x = sanitized_frame_component(coords, t, particles, 0);
        let particle_theta = sanitized_frame_component(coords, t, particles, 1);
        let particle_r = sanitized_frame_component(coords, t, particles, 2);
        let dir = [
            sanitized_frame_component(directions, t, particles, 0),
            sanitized_frame_component(directions, t, particles, 1),
            sanitized_frame_component(directions, t, particles, 2),
        ];
        let vel = [
            sanitized_frame_component(velocities, t, particles, 0),
            sanitized_frame_component(velocities, t, particles, 1),
            sanitized_frame_component(velocities, t, particles, 2),
        ];
        let psi6_particle = sanitized_frame_scalar(psi6_abs, t, particles);
        let q_particle = [
            combine_particle_components(&dir[0], &dir[0], -1.0 / 3.0),
            combine_particle_components(&dir[0], &dir[1], 0.0),
            combine_particle_components(&dir[0], &dir[2], 0.0),
            combine_particle_components(&dir[1], &dir[0], 0.0),
            combine_particle_components(&dir[1], &dir[1], -1.0 / 3.0),
            combine_particle_components(&dir[1], &dir[2], 0.0),
            combine_particle_components(&dir[2], &dir[0], 0.0),
            combine_particle_components(&dir[2], &dir[1], 0.0),
            combine_particle_components(&dir[2], &dir[2], -1.0 / 3.0),
        ];
        let mask_tensor = tensor2(valid, [1, particles], device);
        let x_particles = tensor2(particle_x, [1, particles], device);
        let theta_particles = tensor2(particle_theta, [1, particles], device);
        let r_particles = tensor2(particle_r, [1, particles], device);
        let psi6_tensor = tensor2(psi6_particle, [1, particles], device);
        let dir_tensors = [
            tensor2(dir[0].clone(), [1, particles], device),
            tensor2(dir[1].clone(), [1, particles], device),
            tensor2(dir[2].clone(), [1, particles], device),
        ];
        let vel_tensors = [
            tensor2(vel[0].clone(), [1, particles], device),
            tensor2(vel[1].clone(), [1, particles], device),
            tensor2(vel[2].clone(), [1, particles], device),
        ];
        let q_tensors = [
            tensor2(q_particle[0].clone(), [1, particles], device),
            tensor2(q_particle[1].clone(), [1, particles], device),
            tensor2(q_particle[2].clone(), [1, particles], device),
            tensor2(q_particle[3].clone(), [1, particles], device),
            tensor2(q_particle[4].clone(), [1, particles], device),
            tensor2(q_particle[5].clone(), [1, particles], device),
            tensor2(q_particle[6].clone(), [1, particles], device),
            tensor2(q_particle[7].clone(), [1, particles], device),
            tensor2(q_particle[8].clone(), [1, particles], device),
        ];

        for start in (0..grid).step_by(GRID_CHUNK) {
            let end = (start + GRID_CHUNK).min(grid);
            let chunk = end - start;
            let x_chunk = tensor2(x_grid[start..end].to_vec(), [chunk, 1], device);
            let theta_chunk = tensor2(theta_grid[start..end].to_vec(), [chunk, 1], device);
            let r_chunk = tensor2(r_grid[start..end].to_vec(), [chunk, 1], device);
            let dx = minimum_image_tensor(x_chunk - x_particles.clone(), lx as f32);
            let dtheta = minimum_image_tensor(theta_chunk - theta_particles.clone(), theta_period as f32);
            let dy = r_chunk.clone() * dtheta;
            let dr = r_chunk - r_particles.clone();
            let dist2: Tensor<BurnBackend, 2> =
                dx.clone() * dx + dy.clone() * dy + dr.clone() * dr;
            let weights = (dist2.clone() * (-0.5 / sigma2)).exp() * norm;
            let weights = weights.mask_fill(dist2.greater_elem(cutoff2), 0.0) * mask_tensor.clone();

            write_chunk4_grid(
                &mut rho,
                t,
                ntheta,
                nr,
                start,
                &tensor_vec(weights.clone().sum_dim(1).reshape([chunk]))?,
            );
            write_chunk4_grid(
                &mut psi6,
                t,
                ntheta,
                nr,
                start,
                &weighted_sum(&weights, &psi6_tensor, chunk)?,
            );
            for component in 0..3 {
                write_chunk5_grid(
                    &mut p,
                    t,
                    ntheta,
                    nr,
                    start,
                    component,
                    &weighted_sum(&weights, &dir_tensors[component], chunk)?,
                );
                write_chunk5_grid(
                    &mut j_rho,
                    t,
                    ntheta,
                    nr,
                    start,
                    component,
                    &weighted_sum(&weights, &vel_tensors[component], chunk)?,
                );
            }
            for i in 0..3 {
                for j in 0..3 {
                    let q_index = i * 3 + j;
                    let q_values = weighted_sum(&weights, &q_tensors[q_index], chunk)?;
                    write_chunk6_grid(&mut q, t, ntheta, nr, start, i, j, &q_values);
                    for k in 0..3 {
                        let jp_values = weighted_sum_product(
                            &weights,
                            &vel_tensors[k],
                            &dir_tensors[i],
                            chunk,
                        )?;
                        write_chunk6_grid(&mut j_p, t, ntheta, nr, start, k, i, &jp_values);
                        let jq_values = weighted_sum_product(
                            &weights,
                            &vel_tensors[k],
                            &q_tensors[q_index],
                            chunk,
                        )?;
                        write_chunk7_grid(&mut j_q, t, ntheta, nr, start, k, i, j, &jq_values);
                    }
                }
            }
        }
    }

    let mut a = q.clone();
    let mut psi6_sq = ndarray::Array4::<f64>::from_elem((frames, nx, ntheta, nr), f64::NAN);
    for t in 0..frames {
        for ix in 0..nx {
            for itheta in 0..ntheta {
                for ir in 0..nr {
                    if rho[[t, ix, itheta, ir]].is_finite() && rho[[t, ix, itheta, ir]] > 0.0 {
                        let value = psi6[[t, ix, itheta, ir]] / rho[[t, ix, itheta, ir]];
                        psi6_sq[[t, ix, itheta, ir]] = value * value;
                    }
                    for component in 0..3 {
                        a[[t, ix, itheta, ir, component, component]] +=
                            rho[[t, ix, itheta, ir]] / 3.0;
                    }
                }
            }
        }
    }

    Ok(MechanicalFields {
        rho,
        p,
        q,
        a,
        psi6_sq,
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

fn repeated_grid_values(values: ArrayView1<'_, f64>, repeat: usize) -> Vec<f32> {
    // Expand x centers so flat grid order maps `flat / ny` to x index.
    let mut out = Vec::with_capacity(values.len() * repeat);
    for value in values {
        for _ in 0..repeat {
            out.push(*value as f32);
        }
    }
    out
}

fn tiled_grid_values(values: ArrayView1<'_, f64>, tiles: usize) -> Vec<f32> {
    // Tile y centers so flat grid order maps `flat % ny` to y index.
    let mut out = Vec::with_capacity(values.len() * tiles);
    for _ in 0..tiles {
        out.extend(values.iter().map(|value| *value as f32));
    }
    out
}

fn repeated_3d_x_values(values: ArrayView1<'_, f64>, ntheta: usize, nr: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len() * ntheta * nr);
    for value in values {
        for _ in 0..ntheta * nr {
            out.push(*value as f32);
        }
    }
    out
}

fn repeated_3d_theta_values(values: ArrayView1<'_, f64>, nx: usize, nr: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(nx * values.len() * nr);
    for _ in 0..nx {
        for value in values {
            for _ in 0..nr {
                out.push(*value as f32);
            }
        }
    }
    out
}

fn tiled_3d_r_values(values: ArrayView1<'_, f64>, nx: usize, ntheta: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(nx * ntheta * values.len());
    for _ in 0..nx * ntheta {
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
    // Extract one frame/component from a particle array and apply an optional scale.
    (0..particles)
        .map(|particle| (scale * values[[frame, particle, component]]) as f32)
        .collect()
}

fn sanitized_frame_component(
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

fn sanitized_frame_scalar(values: ArrayView2<'_, f64>, frame: usize, particles: usize) -> Vec<f32> {
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

fn frame_mask(mask: ArrayView2<'_, bool>, frame: usize, particles: usize) -> Vec<f32> {
    // Convert one boolean mask frame into multiplicative f32 weights.
    (0..particles)
        .map(|particle| if mask[[frame, particle]] { 1.0 } else { 0.0 })
        .collect()
}

fn mechanical_frame_mask(
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

fn grid_indices(flat: usize, ntheta: usize, nr: usize) -> (usize, usize, usize) {
    let ix = flat / (ntheta * nr);
    let rem = flat % (ntheta * nr);
    (ix, rem / nr, rem % nr)
}

fn write_chunk4_grid(
    out: &mut ndarray::Array4<f64>,
    t: usize,
    ntheta: usize,
    nr: usize,
    start: usize,
    values: &[f32],
) {
    for (local, value) in values.iter().enumerate() {
        let (ix, itheta, ir) = grid_indices(start + local, ntheta, nr);
        out[[t, ix, itheta, ir]] = *value as f64;
    }
}

fn write_chunk5_grid(
    out: &mut Array5<f64>,
    t: usize,
    ntheta: usize,
    nr: usize,
    start: usize,
    component: usize,
    values: &[f32],
) {
    for (local, value) in values.iter().enumerate() {
        let (ix, itheta, ir) = grid_indices(start + local, ntheta, nr);
        out[[t, ix, itheta, ir, component]] = *value as f64;
    }
}

fn write_chunk6_grid(
    out: &mut Array6<f64>,
    t: usize,
    ntheta: usize,
    nr: usize,
    start: usize,
    i: usize,
    j: usize,
    values: &[f32],
) {
    for (local, value) in values.iter().enumerate() {
        let (ix, itheta, ir) = grid_indices(start + local, ntheta, nr);
        out[[t, ix, itheta, ir, i, j]] = *value as f64;
    }
}

fn write_chunk7_grid(
    out: &mut ArrayD<f64>,
    t: usize,
    ntheta: usize,
    nr: usize,
    start: usize,
    k: usize,
    i: usize,
    j: usize,
    values: &[f32],
) {
    for (local, value) in values.iter().enumerate() {
        let (ix, itheta, ir) = grid_indices(start + local, ntheta, nr);
        out[IxDyn(&[t, ix, itheta, ir, k, i, j])] = *value as f64;
    }
}
