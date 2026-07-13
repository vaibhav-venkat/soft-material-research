mod frame;
mod grid;

use burn::backend::{flex::FlexDevice, Flex};
use burn::prelude::*;
use burn::tensor::TensorData;
#[cfg(feature = "gpu-cuda")]
use burn_cuda::{Cuda, CudaDevice};
#[cfg(feature = "gpu-metal")]
use burn_wgpu::{graphics, init_setup, Metal, WgpuDevice};
use ndarray::{ArrayView1, ArrayView2, ArrayView3};
use std::panic::{catch_unwind, set_hook, take_hook, AssertUnwindSafe};

use frame::{
    combine_particle_components, mechanical_frame_mask, print_conservation,
    sanitized_frame_component, sanitized_frame_scalar, write_mechanical_frame,
};
use grid::Grid3;

use rho_fitting_types::mechanics::{
    MechanicalFieldSet, MechanicalFrame, MechanicalInputViews, TENSOR_COMPONENTS,
};
use rho_fitting_types::{CoreError, CoreResult};

#[cfg(feature = "gpu-cuda")]
type CudaBackend = Cuda<f32, i32>;
#[cfg(feature = "gpu-metal")]
type MetalBackend = Metal<f32, i32, u8>;

type FlexBackend = Flex<f32, i32>;

const EPS: f32 = 1.0e-12;
const GAUSSIAN_GRID_CHUNK: usize = 256;
const PACKED_FEATURES: usize = 53;

/// Build mechanical moment fields and current tensors with renormalized Burn Gaussian deposition.
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
) -> CoreResult<MechanicalFieldSet> {
    let inputs = MechanicalInputViews::new(
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

fn run_mechanical(inputs: MechanicalInputViews<'_>) -> CoreResult<MechanicalFieldSet> {
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
        println!("[rho_fitting] trying Burn Metal/WGPU backend for renormalized mechanical Gaussian fields");
        if let Ok(fields) = catch_burn_panic("Metal/WGPU", || {
            init_setup::<graphics::Metal>(&device, Default::default());
            mechanical_gaussian_device::<MetalBackend>(inputs.clone(), &device)
        }) {
            return Ok(fields);
        }
        println!("[rho_fitting] Metal/WGPU unavailable; falling back to Burn Flex");
    }

    println!(
        "[rho_fitting] using Burn Flex CPU backend for renormalized mechanical Gaussian fields"
    );
    catch_burn_panic("Flex", || {
        mechanical_gaussian_device::<FlexBackend>(inputs, &FlexDevice)
    })
}

fn mechanical_gaussian_device<B: Backend>(
    inputs: MechanicalInputViews<'_>,
    device: &B::Device,
) -> CoreResult<MechanicalFieldSet> {
    let MechanicalInputViews {
        particles,
        grid: domain,
        sigma,
        ..
    } = inputs;
    let rho_fitting_types::fields::ParticleFieldSet {
        coords,
        directions,
        velocities,
        hexatic_order: psi6_abs,
        mask,
    } = particles;
    let (frames, particles, _) = coords.dim();
    let grid = Grid3::from_domain(domain);
    let device_grid = DeviceGrid::<B>::new(&grid, device);
    let mut output = MechanicalFieldSet::zeros(frames, &grid.domain);

    for t in 0..frames {
        println!(
            "[rho_fitting] Burn Gaussian mechanical coarse-grain frame {}/{}",
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
        let q_particle = std::array::from_fn(|index| {
            let (row, col) = TENSOR_COMPONENTS[index];
            combine_particle_components(
                &dir[row.index()],
                &dir[col.index()],
                if row == col { -1.0 / 3.0 } else { 0.0 },
            )
        });
        let mut fields = MechanicalFrame::new(grid.len());
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
            &device_grid,
            &mut fields,
            device,
        )?;
        write_mechanical_frame(t, &grid, &fields, &mut output);
        print_conservation("mechanical", &fields.mass, deposited);
    }

    Ok(output)
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
    device_grid: &DeviceGrid<B>,
    fields: &mut MechanicalFrame,
    device: &B::Device,
) -> CoreResult<()> {
    let particles = inputs.particle_x.len();
    let valid = tensor2::<B>(inputs.valid.to_vec(), [1, particles], device);
    let particle_x = tensor2::<B>(inputs.particle_x.to_vec(), [1, particles], device);
    let particle_theta = tensor2::<B>(inputs.particle_theta.to_vec(), [1, particles], device);
    let particle_r = tensor2::<B>(inputs.particle_r.to_vec(), [1, particles], device);
    let particle_features = tensor2::<B>(
        packed_particle_features(&inputs),
        [particles, PACKED_FEATURES],
        device,
    );
    // Normalize the actual discrete, cutoff Gaussian over the complete cylindrical
    // grid. This includes x, theta, and r and therefore conserves one unit of mass
    // per valid particle even when the grid is coarse or the radial domain is cut.
    let mut norm = Tensor::<B, 1>::zeros([particles], device);
    for start in (0..grid.len()).step_by(GAUSSIAN_GRID_CHUNK) {
        let chunk = (grid.len() - start).min(GAUSSIAN_GRID_CHUNK);
        let weights = gaussian_weights::<B>(
            grid,
            device_grid,
            start,
            chunk,
            &particle_x,
            &particle_theta,
            &particle_r,
            &valid,
            inputs.sigma,
        );
        let volume_tensor = device_grid
            .volumes
            .clone()
            .slice([start..start + chunk, 0..1]);
        norm = norm + (weights * volume_tensor).sum_dim(0).reshape([particles]);
    }
    // Materialize the normalization before the deposition pass. Without this barrier,
    // fused backends retain the complete first-pass graph (including every
    // grid-by-particle weight chunk) until the final frame readback.
    let normalized = tensor_vector(norm + EPS)?;
    let norm = tensor2::<B>(normalized, [1, particles], device);

    // Read each packed grid chunk directly into the host-side frame accumulator.
    // Building one slice-assigned device tensor retains every chunk's computation
    // graph and creates the largest allocation only at final readback.
    for start in (0..grid.len()).step_by(GAUSSIAN_GRID_CHUNK) {
        let chunk = (grid.len() - start).min(GAUSSIAN_GRID_CHUNK);
        let weights = gaussian_weights::<B>(
            grid,
            device_grid,
            start,
            chunk,
            &particle_x,
            &particle_theta,
            &particle_r,
            &valid,
            inputs.sigma,
        );
        let volume_tensor = device_grid
            .volumes
            .clone()
            .slice([start..start + chunk, 0..1]);
        let mass_weights = weights * volume_tensor / norm.clone();
        let chunk_fields = mass_weights.matmul(particle_features.clone());
        write_packed_chunk(fields, tensor_matrix(chunk_fields)?, start, chunk)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gaussian_weights<B: Backend>(
    grid: &Grid3,
    device_grid: &DeviceGrid<B>,
    start: usize,
    chunk: usize,
    particle_x: &Tensor<B, 2>,
    particle_theta: &Tensor<B, 2>,
    particle_r: &Tensor<B, 2>,
    valid: &Tensor<B, 2>,
    sigma: f32,
) -> Tensor<B, 2> {
    let grid_x = device_grid.x.clone().slice([start..start + chunk, 0..1]);
    let grid_theta = device_grid
        .theta
        .clone()
        .slice([start..start + chunk, 0..1]);
    let grid_r = device_grid.r.clone().slice([start..start + chunk, 0..1]);
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

fn tensor2<B: Backend>(values: Vec<f32>, shape: [usize; 2], device: &B::Device) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data(TensorData::new(values, shape), device)
}

fn tensor_matrix<B: Backend>(tensor: Tensor<B, 2>) -> CoreResult<Vec<f32>> {
    tensor
        .try_into_data()
        .map_err(|error| CoreError::InvalidInput(format!("Burn readback failed: {error}")))?
        .to_vec::<f32>()
        .map_err(|error| CoreError::InvalidInput(format!("Burn data conversion failed: {error}")))
}

fn tensor_vector<B: Backend>(tensor: Tensor<B, 1>) -> CoreResult<Vec<f32>> {
    tensor
        .try_into_data()
        .map_err(|error| CoreError::InvalidInput(format!("Burn readback failed: {error}")))?
        .to_vec::<f32>()
        .map_err(|error| CoreError::InvalidInput(format!("Burn data conversion failed: {error}")))
}

struct DeviceGrid<B: Backend> {
    x: Tensor<B, 2>,
    theta: Tensor<B, 2>,
    r: Tensor<B, 2>,
    volumes: Tensor<B, 2>,
}

impl<B: Backend> DeviceGrid<B> {
    fn new(grid: &Grid3, device: &B::Device) -> Self {
        let (x, theta, r, volumes) = grid_chunk(grid, 0, grid.len());
        Self {
            x: tensor2(x, [grid.len(), 1], device),
            theta: tensor2(theta, [grid.len(), 1], device),
            r: tensor2(r, [grid.len(), 1], device),
            volumes: tensor2(volumes, [grid.len(), 1], device),
        }
    }
}

fn packed_particle_features(inputs: &GaussianFrameInputs<'_>) -> Vec<f32> {
    let particles = inputs.particle_x.len();
    let mut output = vec![0.0; particles * PACKED_FEATURES];
    for particle in 0..particles {
        let row = &mut output[particle * PACKED_FEATURES..(particle + 1) * PACKED_FEATURES];
        row[0] = 1.0;
        row[1] = inputs.psi6[particle];
        row[2..5].copy_from_slice(&[
            inputs.dir[0][particle],
            inputs.dir[1][particle],
            inputs.dir[2][particle],
        ]);
        for q_index in 0..9 {
            row[5 + q_index] = inputs.q_particle[q_index][particle];
        }
        row[14..17].copy_from_slice(&[
            inputs.vel[0][particle],
            inputs.vel[1][particle],
            inputs.vel[2][particle],
        ]);
        for flux in 0..3 {
            for component in 0..3 {
                row[17 + flux * 3 + component] =
                    inputs.vel[flux][particle] * inputs.dir[component][particle];
            }
            for q_index in 0..9 {
                row[26 + flux * 9 + q_index] =
                    inputs.vel[flux][particle] * inputs.q_particle[q_index][particle];
            }
        }
    }
    output
}

fn write_packed_chunk(
    fields: &mut MechanicalFrame,
    values: Vec<f32>,
    start: usize,
    chunk: usize,
) -> CoreResult<()> {
    if values.len() != chunk * PACKED_FEATURES {
        return Err(CoreError::Shape(
            "packed mechanical readback has an unexpected shape".to_string(),
        ));
    }
    for offset in 0..chunk {
        let cell = start + offset;
        let row = &values[offset * PACKED_FEATURES..(offset + 1) * PACKED_FEATURES];
        fields.mass[cell] = row[0] as f64;
        fields.hexatic_mass[cell] = row[1] as f64;
        for component in 0..3 {
            fields.p_mass[component][cell] = row[2 + component] as f64;
            fields.j_rho_mass[component][cell] = row[14 + component] as f64;
        }
        for q_index in 0..9 {
            fields.q_mass[q_index][cell] = row[5 + q_index] as f64;
        }
        for index in 0..9 {
            fields.j_p_mass[index][cell] = row[17 + index] as f64;
        }
        for index in 0..27 {
            fields.j_q_mass[index][cell] = row[26 + index] as f64;
        }
    }
    Ok(())
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
