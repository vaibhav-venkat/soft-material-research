mod frame;
mod gaussian;
mod grid;

use burn::prelude::*;
use burn::tensor::TensorData;
#[cfg(feature = "gpu-cuda")]
use burn_cuda::{Cuda, CudaDevice};
#[cfg(feature = "gpu-metal")]
use burn_wgpu::{graphics, init_setup, Metal, WgpuDevice};
use ndarray::{Array5, Array6, ArrayD, ArrayView1, ArrayView2, ArrayView3, IxDyn};
use std::panic::{catch_unwind, set_hook, take_hook, AssertUnwindSafe};

use frame::{
    combine_particle_components, mechanical_frame_mask, print_conservation,
    sanitized_frame_component, sanitized_frame_scalar, write_mechanical_frame, FrameFields,
};
use gaussian::zr_integral;
use grid::{radial_spacing, Grid3};

use crate::mechanics::MechanicalFields;
use crate::{CoreError, CoreResult};

#[cfg(feature = "gpu-cuda")]
type CudaBackend = Cuda<f32, i32>;
#[cfg(feature = "gpu-metal")]
type MetalBackend = Metal<f32, i32, u8>;

const EPS: f32 = 1.0e-12;
const GAUSSIAN_GRID_CHUNK: usize = 256;

pub(super) struct MechanicalInputs<'a> {
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

/// Build mechanical moment fields and current tensors with renormalized GPU Gaussian deposition.
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
    validate_mechanical_inputs(
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
        gamma,
        u0,
    )?;
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
    run_mechanical(inputs)
}

fn catch_burn_panic<T>(backend: &str, func: impl FnOnce() -> CoreResult<T>) -> CoreResult<T> {
    if std::env::var("RHO_FITTING_DEBUG_PANIC").is_ok_and(|value| value == "1") {
        return func();
    }
    let hook = take_hook();
    set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(func));
    set_hook(hook);
    result.map_err(|_| {
        CoreError::InvalidInput(format!(
            "Burn {backend} coarse-grain initialization panicked"
        ))
    })?
}

fn run_mechanical(inputs: MechanicalInputs<'_>) -> CoreResult<MechanicalFields> {
    #[cfg(feature = "gpu-cuda")]
    if <CudaBackend as Backend>::device_count(0) > 0 {
        let device = CudaDevice::new(0);
        println!(
            "[rho_fitting] using Burn CUDA backend for renormalized mechanical Gaussian fields"
        );
        return catch_burn_panic("CUDA", || {
            mechanical_gaussian_device::<CudaBackend>(inputs, &device)
        });
    }

    #[cfg(feature = "gpu-metal")]
    {
        let device = WgpuDevice::DefaultDevice;
        println!("[rho_fitting] using Burn Metal/WGPU backend for renormalized mechanical Gaussian fields");
        return catch_burn_panic("Metal/WGPU", || {
            init_setup::<graphics::Metal>(&device, Default::default());
            mechanical_gaussian_device::<MetalBackend>(inputs, &device)
        });
    }

    #[allow(unreachable_code)]
    Err(CoreError::InvalidInput(
        "extension was built without a supported Burn GPU backend".to_string(),
    ))
}

fn mechanical_gaussian_device<B: Backend>(
    inputs: MechanicalInputs<'_>,
    device: &B::Device,
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
    let grid = Grid3::new(x_centers, theta_centers, r_centers, lx, theta_period)?;
    let mut rho = ndarray::Array4::<f64>::zeros((frames, grid.nx, grid.ntheta, grid.nr));
    let mut psi6_sq =
        ndarray::Array4::<f64>::from_elem((frames, grid.nx, grid.ntheta, grid.nr), f64::NAN);
    let mut p = Array5::<f64>::zeros((frames, grid.nx, grid.ntheta, grid.nr, 3));
    let mut q = Array6::<f64>::zeros((frames, grid.nx, grid.ntheta, grid.nr, 3, 3));
    let mut a = Array6::<f64>::zeros((frames, grid.nx, grid.ntheta, grid.nr, 3, 3));
    let mut j_rho = Array5::<f64>::zeros((frames, grid.nx, grid.ntheta, grid.nr, 3));
    let mut j_p = Array6::<f64>::zeros((frames, grid.nx, grid.ntheta, grid.nr, 3, 3));
    let mut j_q = ArrayD::<f64>::zeros(IxDyn(&[frames, grid.nx, grid.ntheta, grid.nr, 3, 3, 3]));

    for t in 0..frames {
        println!(
            "[rho_fitting] GPU Gaussian mechanical coarse-grain frame {}/{}",
            t + 1,
            frames
        );
        let valid =
            mechanical_frame_mask(coords, directions, velocities, psi6_abs, mask, t, particles);
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
        let psi6 = sanitized_frame_scalar(psi6_abs, t, particles);
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
        let mut fields = FrameFields::new(grid.len());
        let deposited = valid.iter().filter(|value| **value > 0.0).count();
        deposit_gaussian_mechanical_frame::<B>(
            GaussianFrameInputs {
                particle_x: &particle_x,
                particle_theta: &particle_theta,
                particle_r: &particle_r,
                valid: &valid,
                dir: &dir,
                vel: &vel,
                psi6: &psi6,
                q_particle: &q_particle,
                sigma: sigma as f32,
            },
            &grid,
            &mut fields,
            device,
        )?;
        write_mechanical_frame(
            t,
            &grid,
            &fields,
            &mut rho,
            &mut psi6_sq,
            &mut p,
            &mut q,
            &mut a,
            &mut j_rho,
            &mut j_p,
            &mut j_q,
        );
        print_conservation("mechanical", &fields.mass, deposited);
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

#[allow(clippy::too_many_arguments)]
fn validate_mechanical_inputs(
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
) -> CoreResult<()> {
    let (frames, particles, components) = coords.dim();
    if components != 3 {
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
        && sigma > 0.0
        && gamma.is_finite()
        && u0.is_finite()
        && u0 != 0.0)
    {
        return Err(CoreError::InvalidInput(
            "geometry and dynamics values must be finite and positive where required".to_string(),
        ));
    }
    if !r_centers
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
    {
        return Err(CoreError::InvalidInput(
            "radial centers must be finite and positive".to_string(),
        ));
    }
    radial_spacing(r_centers)?;
    Ok(())
}

struct GaussianFrameInputs<'a> {
    particle_x: &'a [f32],
    particle_theta: &'a [f32],
    particle_r: &'a [f32],
    valid: &'a [f32],
    dir: &'a [Vec<f32>; 3],
    vel: &'a [Vec<f32>; 3],
    psi6: &'a [f32],
    q_particle: &'a [Vec<f32>; 9],
    sigma: f32,
}

fn deposit_gaussian_mechanical_frame<B: Backend>(
    inputs: GaussianFrameInputs<'_>,
    grid: &Grid3,
    fields: &mut FrameFields,
    device: &B::Device,
) -> CoreResult<()> {
    let particles = inputs.particle_x.len();
    let valid = tensor2::<B>(inputs.valid.to_vec(), [1, particles], device);
    let particle_x = tensor2::<B>(inputs.particle_x.to_vec(), [1, particles], device);
    let particle_theta = tensor2::<B>(inputs.particle_theta.to_vec(), [1, particles], device);
    let particle_r = tensor2::<B>(inputs.particle_r.to_vec(), [1, particles], device);
    let dir_tensors = [
        tensor2::<B>(inputs.dir[0].clone(), [1, particles], device),
        tensor2::<B>(inputs.dir[1].clone(), [1, particles], device),
        tensor2::<B>(inputs.dir[2].clone(), [1, particles], device),
    ];
    let vel_tensors = [
        tensor2::<B>(inputs.vel[0].clone(), [1, particles], device),
        tensor2::<B>(inputs.vel[1].clone(), [1, particles], device),
        tensor2::<B>(inputs.vel[2].clone(), [1, particles], device),
    ];
    let psi6 = tensor2::<B>(inputs.psi6.to_vec(), [1, particles], device);
    let q_tensors: Vec<Tensor<B, 2>> = inputs
        .q_particle
        .iter()
        .map(|values| tensor2::<B>(values.clone(), [1, particles], device))
        .collect();
    let zr: Vec<f32> = (0..particles)
        .map(|p| {
            let ri = inputs.particle_r[p];
            zr_integral(ri, grid.r_min, grid.r_max, inputs.sigma)
        })
        .collect();
    let norm = tensor2::<B>(zr, [1, particles], device).reshape([particles]) + EPS;
    for start in (0..grid.len()).step_by(GAUSSIAN_GRID_CHUNK) {
        let chunk = (grid.len() - start).min(GAUSSIAN_GRID_CHUNK);
        let (_, _, _, volumes) = grid_chunk(grid, start, chunk);
        let weights = gaussian_weights::<B>(
            grid,
            start,
            chunk,
            &particle_x,
            &particle_theta,
            &particle_r,
            &valid,
            inputs.sigma,
            device,
        );
        let volume_tensor = tensor2::<B>(volumes, [chunk, 1], device);
        let mass_weights = weights * volume_tensor / norm.clone().reshape([1, particles]);
        write_chunk(
            &mut fields.mass,
            start,
            tensor_vec(mass_weights.clone().sum_dim(1).reshape([chunk]))?,
        );
        write_chunk(
            &mut fields.psi6_num,
            start,
            weighted_sum(&mass_weights, &psi6, chunk)?,
        );
        for component in 0..3 {
            write_chunk(
                &mut fields.p_num[component],
                start,
                weighted_sum(&mass_weights, &dir_tensors[component], chunk)?,
            );
            write_chunk(
                &mut fields.j_rho_num[component],
                start,
                weighted_sum(&mass_weights, &vel_tensors[component], chunk)?,
            );
        }
        for flux in 0..3 {
            for component in 0..3 {
                write_chunk(
                    &mut fields.j_p_num[flux * 3 + component],
                    start,
                    weighted_sum_product(
                        &mass_weights,
                        &vel_tensors[flux],
                        &dir_tensors[component],
                        chunk,
                    )?,
                );
            }
            for q_index in 0..9 {
                write_chunk(
                    &mut fields.j_q_num[flux * 9 + q_index],
                    start,
                    weighted_sum_product(
                        &mass_weights,
                        &vel_tensors[flux],
                        &q_tensors[q_index],
                        chunk,
                    )?,
                );
            }
        }
        for q_index in 0..9 {
            write_chunk(
                &mut fields.q_num[q_index],
                start,
                weighted_sum(&mass_weights, &q_tensors[q_index], chunk)?,
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gaussian_weights<B: Backend>(
    grid: &Grid3,
    start: usize,
    chunk: usize,
    particle_x: &Tensor<B, 2>,
    particle_theta: &Tensor<B, 2>,
    particle_r: &Tensor<B, 2>,
    valid: &Tensor<B, 2>,
    sigma: f32,
    device: &B::Device,
) -> Tensor<B, 2> {
    let (xs, thetas, rs, _) = grid_chunk(grid, start, chunk);
    let grid_x = tensor2::<B>(xs, [chunk, 1], device);
    let grid_theta = tensor2::<B>(thetas, [chunk, 1], device);
    let grid_r = tensor2::<B>(rs, [chunk, 1], device);
    let dx_raw = grid_x - particle_x.clone();
    let dx = dx_raw.clone() - (dx_raw / grid.lx).round() * grid.lx;
    let dtheta_raw = grid_theta - particle_theta.clone();
    let dtheta = dtheta_raw.clone() - (dtheta_raw / grid.theta_period).round() * grid.theta_period;
    let dy = grid_r.clone() * dtheta;
    let dr = grid_r - particle_r.clone();
    let dist2 = dx.clone() * dx + dy.clone() * dy + dr.clone() * dr;
    let cutoff2 = (4.0 * sigma) * (4.0 * sigma);
    let weights = (dist2.clone() * (-0.5 / (sigma * sigma))).exp();
    weights.mask_fill(dist2.greater_elem(cutoff2), 0.0) * valid.clone()
}

fn weighted_sum<B: Backend>(
    weights: &Tensor<B, 2>,
    values: &Tensor<B, 2>,
    chunk: usize,
) -> CoreResult<Vec<f32>> {
    tensor_vec(
        (weights.clone() * values.clone())
            .sum_dim(1)
            .reshape([chunk]),
    )
}

fn weighted_sum_product<B: Backend>(
    weights: &Tensor<B, 2>,
    left: &Tensor<B, 2>,
    right: &Tensor<B, 2>,
    chunk: usize,
) -> CoreResult<Vec<f32>> {
    tensor_vec(
        (weights.clone() * left.clone() * right.clone())
            .sum_dim(1)
            .reshape([chunk]),
    )
}

fn write_chunk(target: &mut [f32], start: usize, values: Vec<f32>) {
    for (offset, value) in values.into_iter().enumerate() {
        target[start + offset] = value;
    }
}

fn tensor2<B: Backend>(values: Vec<f32>, shape: [usize; 2], device: &B::Device) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data(TensorData::new(values, shape), device)
}

fn tensor_vec<B: Backend>(tensor: Tensor<B, 1>) -> CoreResult<Vec<f32>> {
    tensor
        .try_into_data()
        .map_err(|error| CoreError::InvalidInput(format!("Burn readback failed: {error}")))?
        .to_vec::<f32>()
        .map_err(|error| CoreError::InvalidInput(format!("Burn data conversion failed: {error}")))
}

fn grid_chunk(
    grid: &Grid3,
    start: usize,
    chunk: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut xs = Vec::with_capacity(chunk);
    let mut thetas = Vec::with_capacity(chunk);
    let mut rs = Vec::with_capacity(chunk);
    let mut volumes = Vec::with_capacity(chunk);
    for flat in start..start + chunk {
        let ix = flat / (grid.ntheta * grid.nr);
        let rem = flat % (grid.ntheta * grid.nr);
        let itheta = rem / grid.nr;
        let ir = rem % grid.nr;
        xs.push(grid.x_centers[ix]);
        thetas.push(grid.theta_centers[itheta]);
        rs.push(grid.r_centers[ir]);
        volumes.push(grid.volumes[flat]);
    }
    (xs, thetas, rs, volumes)
}
