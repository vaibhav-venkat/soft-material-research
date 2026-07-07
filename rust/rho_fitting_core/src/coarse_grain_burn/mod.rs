mod deposit;
mod frame;
mod grid;
mod smooth;

use burn::prelude::*;
#[cfg(feature = "gpu-cuda")]
use burn_cuda::{Cuda, CudaDevice};
#[cfg(feature = "gpu-metal")]
use burn_wgpu::{graphics, init_setup, Metal, WgpuDevice};
use ndarray::{Array3, Array4, Array5, Array6, ArrayD, ArrayView1, ArrayView2, ArrayView3, IxDyn};
use std::panic::{catch_unwind, set_hook, take_hook, AssertUnwindSafe};

use deposit::{collapse_surface_shell, deposit_mechanical_components, deposit_surface_shell_frame};
use frame::{
    combine_particle_components, frame_component, mechanical_frame_mask, print_conservation,
    sanitized_frame_component, sanitized_frame_component2, sanitized_frame_scalar,
    surface_frame_mask, write_mechanical_frame, FrameFields,
};
use grid::{
    cylindrical_cell_volumes, radial_spacing, single_radial_width, surface_shell_radial_centers,
    Grid3,
};
use smooth::{smooth_fields_2d, smooth_frame_fields};

use crate::coarse_grain;
use crate::mechanics::MechanicalFields;
use crate::{CoreError, CoreResult};

#[cfg(feature = "gpu-cuda")]
type CudaBackend = Cuda<f32, i32>;
#[cfg(feature = "gpu-metal")]
type MetalBackend = Metal<f32, i32, u8>;

const SMOOTHING_PASSES: usize = 1;
const SMOOTHING_STENCIL: [f32; 3] = [1.0, 2.0, 1.0];
const EPS: f32 = 1.0e-12;
const SINGLE_RADIAL_WIDTH_FRACTION: f32 = 0.1;

pub(super) struct CoarseGrainInputs<'a> {
    coords: ArrayView3<'a, f64>,
    p_particles: ArrayView3<'a, f64>,
    shell_mask: ArrayView2<'a, bool>,
    x_centers: ArrayView1<'a, f64>,
    y_centers: ArrayView1<'a, f64>,
    lx: f64,
    ly: f64,
    radius: f64,
}

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
}

/// Coarse-grain density and two-component polarization density with conservative GPU TSC deposition.
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
    let inputs = CoarseGrainInputs {
        coords,
        p_particles,
        shell_mask,
        x_centers,
        y_centers,
        lx,
        ly,
        radius,
    };
    run_coarse_grain(inputs)
}

/// Build mechanical moment fields and current tensors with conservative GPU TSC deposition.
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
            "Burn {backend} TSC coarse-grain initialization panicked"
        ))
    })?
}

fn run_coarse_grain(inputs: CoarseGrainInputs<'_>) -> CoreResult<(Array3<f64>, Array4<f64>)> {
    #[cfg(feature = "gpu-cuda")]
    if <CudaBackend as Backend>::device_count(0) > 0 {
        let device = CudaDevice::new(0);
        println!("[rho_fitting] using Burn CUDA backend for TSC coarse-graining");
        return catch_burn_panic("CUDA", || {
            tsc_coarse_grain_device::<CudaBackend>(inputs, &device)
        });
    }

    #[cfg(feature = "gpu-metal")]
    {
        let device = WgpuDevice::DefaultDevice;
        println!("[rho_fitting] using Burn Metal/WGPU backend for TSC coarse-graining");
        return catch_burn_panic("Metal/WGPU", || {
            init_setup::<graphics::Metal>(&device, Default::default());
            tsc_coarse_grain_device::<MetalBackend>(inputs, &device)
        });
    }

    #[allow(unreachable_code)]
    Err(CoreError::InvalidInput(
        "extension was built without a supported Burn GPU backend".to_string(),
    ))
}

fn run_mechanical(inputs: MechanicalInputs<'_>) -> CoreResult<MechanicalFields> {
    #[cfg(feature = "gpu-cuda")]
    if <CudaBackend as Backend>::device_count(0) > 0 {
        let device = CudaDevice::new(0);
        println!("[rho_fitting] using Burn CUDA backend for mechanical TSC fields");
        return catch_burn_panic("CUDA", || {
            mechanical_tsc_device::<CudaBackend>(inputs, &device)
        });
    }

    #[cfg(feature = "gpu-metal")]
    {
        let device = WgpuDevice::DefaultDevice;
        println!("[rho_fitting] using Burn Metal/WGPU backend for mechanical TSC fields");
        return catch_burn_panic("Metal/WGPU", || {
            init_setup::<graphics::Metal>(&device, Default::default());
            mechanical_tsc_device::<MetalBackend>(inputs, &device)
        });
    }

    #[allow(unreachable_code)]
    Err(CoreError::InvalidInput(
        "extension was built without a supported Burn GPU backend".to_string(),
    ))
}

fn tsc_coarse_grain_device<B: Backend>(
    inputs: CoarseGrainInputs<'_>,
    device: &B::Device,
) -> CoreResult<(Array3<f64>, Array4<f64>)> {
    let CoarseGrainInputs {
        coords,
        p_particles,
        shell_mask,
        x_centers,
        y_centers,
        lx,
        ly,
        radius,
    } = inputs;
    let (frames, particles, _) = coords.dim();
    let nx = x_centers.len();
    let ny = y_centers.len();
    let grid = nx * ny;
    let dx = (lx / nx as f64) as f32;
    let theta_period = (ly / radius) as f32;
    let dtheta = theta_period / ny as f32;
    let radial_width = single_radial_width(radius as f32)?;
    let radial_centers = surface_shell_radial_centers(radius as f32, radial_width)?;
    let volumes = cylindrical_cell_volumes(dx, dtheta, &radial_centers, radial_width)?;
    let column_volume: f32 = volumes.iter().sum();
    let x_values: Vec<f32> = x_centers.iter().map(|value| *value as f32).collect();
    let theta_values: Vec<f32> = y_centers
        .iter()
        .map(|value| (*value / radius) as f32)
        .collect();
    let mut rho = Array3::<f64>::zeros((frames, nx, ny));
    let mut p_density = Array4::<f64>::zeros((frames, nx, ny, 2));

    for t in 0..frames {
        println!(
            "[rho_fitting] GPU TSC coarse-grain frame {}/{}",
            t + 1,
            frames
        );
        let valid = surface_frame_mask(coords, p_particles, shell_mask, t, particles);
        let particle_x = frame_component(coords, t, particles, 0, 1.0);
        let particle_theta = frame_component(coords, t, particles, 1, 1.0);
        let particle_r = frame_component(coords, t, particles, 2, 1.0);
        let px = sanitized_frame_component2(p_particles, t, particles, 0);
        let py = sanitized_frame_component2(p_particles, t, particles, 1);
        let shell_grid = grid * radial_centers.len();
        let mut shell_mass = vec![0.0f32; shell_grid];
        let mut shell_p_num = [vec![0.0f32; shell_grid], vec![0.0f32; shell_grid]];
        let deposited = deposit_surface_shell_frame(
            &x_values,
            &theta_values,
            &radial_centers,
            &particle_x,
            &particle_theta,
            &particle_r,
            &valid,
            &[px, py],
            lx as f32,
            theta_period,
            dx,
            dtheta,
            radial_width,
            &mut shell_mass,
            &mut shell_p_num,
        );
        let (mut mass, mut p_num) =
            collapse_surface_shell(&shell_mass, &shell_p_num, nx, ny, radial_centers.len());
        smooth_fields_2d::<B>(&mut mass, &mut p_num, nx, ny, device)?;
        for flat in 0..grid {
            let ix = flat / ny;
            let iy = flat % ny;
            rho[[t, ix, iy]] = (mass[flat] / column_volume) as f64;
            p_density[[t, ix, iy, 0]] = (p_num[0][flat] / column_volume) as f64;
            p_density[[t, ix, iy, 1]] = (p_num[1][flat] / column_volume) as f64;
        }
        print_conservation("density", &mass, deposited);
    }
    Ok((rho, p_density))
}

fn mechanical_tsc_device<B: Backend>(
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
            "[rho_fitting] GPU TSC mechanical coarse-grain frame {}/{}",
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
        let deposited = deposit_mechanical_components(
            &particle_x,
            &particle_theta,
            &particle_r,
            &valid,
            &dir,
            &vel,
            &psi6,
            &q_particle,
            &grid,
            &mut fields,
        );
        smooth_frame_fields::<B>(&mut fields, &grid, device)?;
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
