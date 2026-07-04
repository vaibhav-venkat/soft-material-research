use cubecl::{prelude::*, server::Handle};
use ndarray::{Array3, Array4, ArrayView1, ArrayView2, ArrayView3};

use crate::{CoreError, CoreResult};

#[cube(launch_unchecked)]
fn coarse_grain_kernel<F: Float>(
    coords: &Array<F>,
    p_particles: &Array<F>,
    shell_mask: &Array<F>,
    x_centers: &Array<F>,
    y_centers: &Array<F>,
    params: &Array<F>,
    rho_out: &mut Array<F>,
    p_out: &mut Array<F>,
    #[comptime] frames: usize,
    #[comptime] particles: usize,
    #[comptime] nx: usize,
    #[comptime] ny: usize,
) {
    let out_index = ABSOLUTE_POS as usize;
    let total = frames * nx * ny;
    if out_index < total {
        let iy = out_index % ny;
        let ix = (out_index / ny) % nx;
        let t = out_index / (nx * ny);
        let lx = params[0];
        let ly = params[1];
        let radius = params[2];
        let sigma = params[3];
        let grid_x = x_centers[ix];
        let grid_y = y_centers[iy];
        let cutoff = F::new(4.0f32) * sigma;
        let cutoff2 = cutoff * cutoff;
        let norm =
            F::new(1.0f32) / (F::new(comptime!(2.0f32 * core::f32::consts::PI)) * sigma * sigma);

        let mut rho = F::new(0.0f32);
        let mut px_sum = F::new(0.0f32);
        let mut py_sum = F::new(0.0f32);

        for i in 0..particles {
            let shell_index = t * particles + i;
            if shell_mask[shell_index] > F::new(0.5f32) {
                let coord_index = (t * particles + i) * 3;
                let p_index = (t * particles + i) * 2;
                let particle_x = coords[coord_index];
                let particle_y = radius * coords[coord_index + 1];
                let px = p_particles[p_index];
                let py = p_particles[p_index + 1];

                let mut dx = grid_x - particle_x;
                dx = dx - lx * (dx / lx).round();
                if dx.abs() <= cutoff {
                    let mut dy = grid_y - particle_y;
                    dy = dy - ly * (dy / ly).round();
                    let dist2 = dx * dx + dy * dy;
                    if dy.abs() <= cutoff && dist2 <= cutoff2 {
                        let weight = norm * (F::new(-0.5f32) * dist2 / (sigma * sigma)).exp();
                        rho += weight;
                        px_sum += weight * px;
                        py_sum += weight * py;
                    }
                }
            }
        }

        rho_out[out_index] = rho;
        p_out[out_index * 2] = px_sum;
        p_out[out_index * 2 + 1] = py_sum;
    }
}

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
    let (frames, particles, coord_components) = coords.dim();
    if coord_components != 3 || p_particles.dim() != (frames, particles, 2) {
        return Err(CoreError::Shape(
            "coords must be (T,N,3) and p_particles must be (T,N,2)".to_string(),
        ));
    }
    if shell_mask.dim() != (frames, particles) {
        return Err(CoreError::Shape(
            "shell_mask must have shape (T,N)".to_string(),
        ));
    }
    let nx = x_centers.len();
    let ny = y_centers.len();
    if frames == 0 || particles == 0 || nx == 0 || ny == 0 {
        return Err(CoreError::InvalidInput(
            "coarse-grain axes must be non-empty".to_string(),
        ));
    }

    run_with_runtime::<cubecl_wgpu::WgpuRuntime>(
        coords,
        p_particles,
        shell_mask,
        x_centers,
        y_centers,
        lx,
        ly,
        radius,
        sigma,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_with_runtime<R: Runtime>(
    coords: ArrayView3<'_, f64>,
    p_particles: ArrayView3<'_, f64>,
    shell_mask: ArrayView2<'_, bool>,
    x_centers: ArrayView1<'_, f64>,
    y_centers: ArrayView1<'_, f64>,
    lx: f64,
    ly: f64,
    radius: f64,
    sigma: f64,
) -> CoreResult<(Array3<f64>, Array4<f64>)>
where
    R::Device: Default,
{
    let (frames, particles, _) = coords.dim();
    let nx = x_centers.len();
    let ny = y_centers.len();
    let output_len = frames * nx * ny;
    let device: R::Device = Default::default();
    let client = R::client(&device);

    let coords_f32 = coords.iter().map(|value| *value as f32).collect::<Vec<_>>();
    let p_f32 = p_particles
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let shell_f32 = shell_mask
        .iter()
        .map(|value| if *value { 1.0f32 } else { 0.0f32 })
        .collect::<Vec<_>>();
    let x_f32 = x_centers
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let y_f32 = y_centers
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let params_f32 = [lx as f32, ly as f32, radius as f32, sigma as f32];

    let coords_handle = client.create_from_slice(f32::as_bytes(&coords_f32));
    let p_handle = client.create_from_slice(f32::as_bytes(&p_f32));
    let shell_handle = client.create_from_slice(f32::as_bytes(&shell_f32));
    let x_handle = client.create_from_slice(f32::as_bytes(&x_f32));
    let y_handle = client.create_from_slice(f32::as_bytes(&y_f32));
    let params_handle = client.create_from_slice(f32::as_bytes(&params_f32));
    let rho_handle = client.empty(output_len * core::mem::size_of::<f32>());
    let p_out_handle = client.empty(output_len * 2 * core::mem::size_of::<f32>());

    launch_kernel::<R>(
        &client,
        coords_handle,
        p_handle,
        shell_handle,
        x_handle,
        y_handle,
        params_handle,
        rho_handle.clone(),
        p_out_handle.clone(),
        coords_f32.len(),
        p_f32.len(),
        shell_f32.len(),
        x_f32.len(),
        y_f32.len(),
        params_f32.len(),
        output_len,
        frames,
        particles,
        nx,
        ny,
    );

    let rho_bytes = client
        .read_one(rho_handle)
        .map_err(|error| CoreError::InvalidInput(format!("GPU rho read failed: {error}")))?;
    let p_bytes = client
        .read_one(p_out_handle)
        .map_err(|error| CoreError::InvalidInput(format!("GPU p read failed: {error}")))?;
    let rho_f32 = f32::from_bytes(&rho_bytes);
    let p_f32 = f32::from_bytes(&p_bytes);
    let rho = Array3::from_shape_vec(
        (frames, nx, ny),
        rho_f32.iter().map(|value| *value as f64).collect(),
    )
    .map_err(|error| CoreError::Shape(error.to_string()))?;
    let p_density = Array4::from_shape_vec(
        (frames, nx, ny, 2),
        p_f32.iter().map(|value| *value as f64).collect(),
    )
    .map_err(|error| CoreError::Shape(error.to_string()))?;

    Ok((rho, p_density))
}

#[allow(clippy::too_many_arguments)]
fn launch_kernel<R: Runtime>(
    client: &ComputeClient<R>,
    coords: Handle,
    p_particles: Handle,
    shell_mask: Handle,
    x_centers: Handle,
    y_centers: Handle,
    params: Handle,
    rho_out: Handle,
    p_out: Handle,
    coords_len: usize,
    p_len: usize,
    shell_len: usize,
    x_len: usize,
    y_len: usize,
    params_len: usize,
    output_len: usize,
    frames: usize,
    particles: usize,
    nx: usize,
    ny: usize,
) {
    let cube_dim = 256u32;
    let cubes = output_len.div_ceil(cube_dim as usize) as u32;
    // SAFETY: every raw handle is allocated above for the exact length passed
    // here, and the kernel only indexes within the frame/particle/grid bounds.
    unsafe {
        coarse_grain_kernel::launch_unchecked::<f32, R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts(coords, coords_len),
            ArrayArg::from_raw_parts(p_particles, p_len),
            ArrayArg::from_raw_parts(shell_mask, shell_len),
            ArrayArg::from_raw_parts(x_centers, x_len),
            ArrayArg::from_raw_parts(y_centers, y_len),
            ArrayArg::from_raw_parts(params, params_len),
            ArrayArg::from_raw_parts(rho_out, output_len),
            ArrayArg::from_raw_parts(p_out, output_len * 2),
            frames,
            particles,
            nx,
            ny,
        )
    }
}
