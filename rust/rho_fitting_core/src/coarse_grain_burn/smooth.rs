use burn::prelude::*;
use burn::tensor::TensorData;

use super::frame::FrameFields;
use super::grid::{flat_index, Grid3};
use super::{EPS, SMOOTHING_PASSES, SMOOTHING_STENCIL};
use crate::{CoreError, CoreResult};

fn tensor3<B: Backend>(values: Vec<f32>, shape: [usize; 3], device: &B::Device) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(TensorData::new(values, shape), device)
}

fn tensor_vec<B: Backend>(tensor: Tensor<B, 1>) -> CoreResult<Vec<f32>> {
    tensor
        .try_into_data()
        .map_err(|error| CoreError::InvalidInput(format!("Burn readback failed: {error}")))?
        .to_vec::<f32>()
        .map_err(|error| CoreError::InvalidInput(format!("Burn data conversion failed: {error}")))
}

pub(super) fn smooth_fields_2d<B: Backend>(
    mass: &mut Vec<f32>,
    component_num: &mut [Vec<f32>; 2],
    nx: usize,
    ny: usize,
    device: &B::Device,
) -> CoreResult<()> {
    for _ in 0..SMOOTHING_PASSES {
        *mass = smooth_2d_component::<B>(mass, nx, ny, device)?;
        for component in component_num.iter_mut() {
            *component = smooth_2d_component::<B>(component, nx, ny, device)?;
        }
    }
    Ok(())
}

pub(super) fn smooth_frame_fields<B: Backend>(
    fields: &mut FrameFields,
    grid: &Grid3,
    device: &B::Device,
) -> CoreResult<()> {
    for _ in 0..SMOOTHING_PASSES {
        fields.mass = smooth_3d_component::<B>(&fields.mass, grid, device)?;
        fields.psi6_num = smooth_3d_component::<B>(&fields.psi6_num, grid, device)?;
        for values in fields.p_num.iter_mut() {
            *values = smooth_3d_component::<B>(values, grid, device)?;
        }
        for values in fields.q_num.iter_mut() {
            *values = smooth_3d_component::<B>(values, grid, device)?;
        }
        for values in fields.j_rho_num.iter_mut() {
            *values = smooth_3d_component::<B>(values, grid, device)?;
        }
        for values in fields.j_p_num.iter_mut() {
            *values = smooth_3d_component::<B>(values, grid, device)?;
        }
        for values in fields.j_q_num.iter_mut() {
            *values = smooth_3d_component::<B>(values, grid, device)?;
        }
    }
    Ok(())
}

fn smooth_2d_component<B: Backend>(
    values: &[f32],
    nx: usize,
    ny: usize,
    device: &B::Device,
) -> CoreResult<Vec<f32>> {
    let norm = SMOOTHING_STENCIL.iter().sum::<f32>();
    let stencil = [
        SMOOTHING_STENCIL[0] / norm,
        SMOOTHING_STENCIL[1] / norm,
        SMOOTHING_STENCIL[2] / norm,
    ];
    let field = tensor3(values.to_vec(), [nx, ny, 1], device);
    let mut out = Tensor::<B, 3>::zeros([nx, ny, 1], device);
    for sx in -1i32..=1 {
        for sy in -1i32..=1 {
            let weight = stencil[(sx + 1) as usize] * stencil[(sy + 1) as usize];
            out = out + (field.clone() * weight).roll(&[sx, sy, 0], &[0, 1, 2]);
        }
    }
    tensor_vec(out.reshape([nx * ny]))
}

fn smooth_3d_component<B: Backend>(
    values: &[f32],
    grid: &Grid3,
    device: &B::Device,
) -> CoreResult<Vec<f32>> {
    let norm = SMOOTHING_STENCIL.iter().sum::<f32>();
    let stencil = [
        SMOOTHING_STENCIL[0] / norm,
        SMOOTHING_STENCIL[1] / norm,
        SMOOTHING_STENCIL[2] / norm,
    ];
    let source_norm = smoothing_source_norms(grid, &stencil);
    let field = tensor3(values.to_vec(), [grid.nx, grid.ntheta, grid.nr], device);
    let norm_tensor = tensor3(source_norm, [grid.nx, grid.ntheta, grid.nr], device);
    let mut out = Tensor::<B, 3>::zeros([grid.nx, grid.ntheta, grid.nr], device);
    for sx in -1i32..=1 {
        for st in -1i32..=1 {
            for sr in -1i32..=1 {
                let mask = tensor3(
                    smoothing_source_mask(grid, sr),
                    [grid.nx, grid.ntheta, grid.nr],
                    device,
                );
                let weight = stencil[(sx + 1) as usize]
                    * stencil[(st + 1) as usize]
                    * stencil[(sr + 1) as usize];
                let contribution = field.clone() * mask * weight / norm_tensor.clone();
                out = out + contribution.roll(&[sx, st, sr], &[0, 1, 2]);
            }
        }
    }
    tensor_vec(out.reshape([grid.len()]))
}

fn smoothing_source_norms(grid: &Grid3, stencil: &[f32; 3]) -> Vec<f32> {
    let mut values = vec![0.0; grid.len()];
    for ix in 0..grid.nx {
        for itheta in 0..grid.ntheta {
            for ir in 0..grid.nr {
                let mut total = 0.0;
                for sx in -1i32..=1 {
                    for st in -1i32..=1 {
                        for sr in -1i32..=1 {
                            let target_r = ir as i32 + sr;
                            if target_r < 0 || target_r >= grid.nr as i32 {
                                continue;
                            }
                            total += stencil[(sx + 1) as usize]
                                * stencil[(st + 1) as usize]
                                * stencil[(sr + 1) as usize];
                        }
                    }
                }
                values[flat_index(ix, itheta, ir, grid.ntheta, grid.nr)] = total.max(EPS);
            }
        }
    }
    values
}

fn smoothing_source_mask(grid: &Grid3, radial_shift: i32) -> Vec<f32> {
    let mut values = vec![0.0; grid.len()];
    for ix in 0..grid.nx {
        for itheta in 0..grid.ntheta {
            for ir in 0..grid.nr {
                let target_r = ir as i32 + radial_shift;
                if target_r >= 0 && target_r < grid.nr as i32 {
                    values[flat_index(ix, itheta, ir, grid.ntheta, grid.nr)] = 1.0;
                }
            }
        }
    }
    values
}
