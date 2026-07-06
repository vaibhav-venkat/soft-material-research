//! Spectral finite-domain operators on the periodic cylinder surface grid.
//!
//! Fields are represented on a two-dimensional periodic domain with axes
//! `(x, y)` where `y = R * theta`. The leading axis is time/frame, so scalar
//! fields have shape `(T, Nx, Ny)` and surface vector fields have shape
//! `(T, Nx, Ny, 2)`. All derivatives are computed frame-by-frame with 2D FFTs
//! and physical domain lengths `lx` and `ly`.

use std::f64::consts::PI;
use std::sync::Arc;

use ndarray::{Array3, Array4, ArrayView3, ArrayView4};
use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};

use crate::{CoreError, CoreResult};

/// Return the surface gradient of a scalar field.
///
/// Input shape is `(T, Nx, Ny)`. Output shape is `(T, Nx, Ny, 2)`, where the
/// final axis stores derivatives with respect to `x` and physical surface
/// coordinate `y = R * theta`.
///
/// Edge cases: this assumes both spatial axes are periodic and non-empty; it
/// does not window or pad non-periodic data.
pub fn gradient_scalar(field: ArrayView3<'_, f64>, lx: f64, ly: f64) -> CoreResult<Array4<f64>> {
    let (frames, nx, ny) = validate_scalar(field, lx, ly)?;
    let kx = wavenumbers(nx, lx);
    let ky = wavenumbers(ny, ly);
    let mut out = Array4::<f64>::zeros((frames, nx, ny, 2));
    for t in 0..frames {
        let spectrum = fft2_real(field, t, nx, ny);
        let dx = inverse_with_multiplier(&spectrum, nx, ny, |ix, _| Complex64::new(0.0, kx[ix]));
        let dy = inverse_with_multiplier(&spectrum, nx, ny, |_, iy| Complex64::new(0.0, ky[iy]));
        for ix in 0..nx {
            for iy in 0..ny {
                out[[t, ix, iy, 0]] = dx[ix * ny + iy];
                out[[t, ix, iy, 1]] = dy[ix * ny + iy];
            }
        }
    }
    Ok(out)
}

/// Return the scalar Laplacian of a periodic scalar field.
///
/// This applies the spectral multiplier `-(kx^2 + ky^2)` independently to each
/// frame.
pub fn laplacian_scalar(field: ArrayView3<'_, f64>, lx: f64, ly: f64) -> CoreResult<Array3<f64>> {
    repeated_laplacian_scalar(field, lx, ly, 1)
}

/// Return the scalar bilaplacian of a periodic scalar field.
///
/// This is equivalent to applying the Laplacian twice.
pub fn bilaplacian_scalar(field: ArrayView3<'_, f64>, lx: f64, ly: f64) -> CoreResult<Array3<f64>> {
    repeated_laplacian_scalar(field, lx, ly, 2)
}

/// Return a scalar field after applying `order` Laplacians.
///
/// `order == 0` returns a copy of the input. Higher orders use the spectral
/// multiplier `[-(kx^2 + ky^2)]^order`.
pub fn repeated_laplacian_scalar(
    field: ArrayView3<'_, f64>,
    lx: f64,
    ly: f64,
    order: usize,
) -> CoreResult<Array3<f64>> {
    // Apply the scalar Laplacian `order` times in spectral space.
    if order == 0 {
        return Ok(field.to_owned());
    }
    scalar_k_power(field, lx, ly, order)
}

/// Return the surface divergence of a two-component vector field.
///
/// Input shape is `(T, Nx, Ny, 2)`, with vector components ordered as
/// `(x, y)`. Output shape is `(T, Nx, Ny)`.
/// The last axis is interpreted as surface flux direction, not a 3D orientation
/// component.
pub fn divergence_vector(field: ArrayView4<'_, f64>, lx: f64, ly: f64) -> CoreResult<Array3<f64>> {
    let (frames, nx, ny) = validate_vector(field, lx, ly)?;
    let kx = wavenumbers(nx, lx);
    let ky = wavenumbers(ny, ly);
    let mut out = Array3::<f64>::zeros((frames, nx, ny));
    for t in 0..frames {
        let spectrum_x = fft2_vector_component(field, t, 0, nx, ny);
        let spectrum_y = fft2_vector_component(field, t, 1, nx, ny);
        let mut spectrum = vec![Complex64::new(0.0, 0.0); nx * ny];
        for ix in 0..nx {
            for iy in 0..ny {
                let index = ix * ny + iy;
                spectrum[index] = Complex64::new(0.0, kx[ix]) * spectrum_x[index]
                    + Complex64::new(0.0, ky[iy]) * spectrum_y[index];
            }
        }
        let div = inverse_complex(&spectrum, nx, ny);
        for ix in 0..nx {
            for iy in 0..ny {
                out[[t, ix, iy]] = div[ix * ny + iy];
            }
        }
    }
    Ok(out)
}

/// Return the componentwise Laplacian of a two-component vector field.
pub fn laplacian_vector(field: ArrayView4<'_, f64>, lx: f64, ly: f64) -> CoreResult<Array4<f64>> {
    vector_k_power(field, lx, ly, 1)
}

/// Return the componentwise bilaplacian of a two-component vector field.
pub fn bilaplacian_vector(field: ArrayView4<'_, f64>, lx: f64, ly: f64) -> CoreResult<Array4<f64>> {
    vector_k_power(field, lx, ly, 2)
}

/// Return `grad(div(field))` for a two-component surface vector field.
pub fn grad_div_vector(field: ArrayView4<'_, f64>, lx: f64, ly: f64) -> CoreResult<Array4<f64>> {
    let div = divergence_vector(field, lx, ly)?;
    gradient_scalar(div.view(), lx, ly)
}

/// Apply the scalar spectral multiplier `[-(kx^2 + ky^2)]^order`.
fn scalar_k_power(
    field: ArrayView3<'_, f64>,
    lx: f64,
    ly: f64,
    order: usize,
) -> CoreResult<Array3<f64>> {
    // Apply `(-k^2)^order` to a scalar spectrum and transform back to real space.
    let (frames, nx, ny) = validate_scalar(field, lx, ly)?;
    let kx = wavenumbers(nx, lx);
    let ky = wavenumbers(ny, ly);
    let mut out = Array3::<f64>::zeros((frames, nx, ny));
    for t in 0..frames {
        let spectrum = fft2_real(field, t, nx, ny);
        let values = inverse_with_multiplier(&spectrum, nx, ny, |ix, iy| {
            let k2 = kx[ix] * kx[ix] + ky[iy] * ky[iy];
            Complex64::new((-k2).powi(order as i32), 0.0)
        });
        for ix in 0..nx {
            for iy in 0..ny {
                out[[t, ix, iy]] = values[ix * ny + iy];
            }
        }
    }
    Ok(out)
}

/// Apply either the vector Laplacian or bilaplacian componentwise.
fn vector_k_power(
    field: ArrayView4<'_, f64>,
    lx: f64,
    ly: f64,
    power: u32,
) -> CoreResult<Array4<f64>> {
    let (frames, nx, ny) = validate_vector(field, lx, ly)?;
    let kx = wavenumbers(nx, lx);
    let ky = wavenumbers(ny, ly);
    let mut out = Array4::<f64>::zeros((frames, nx, ny, 2));
    for t in 0..frames {
        for component in 0..2 {
            let spectrum = fft2_vector_component(field, t, component, nx, ny);
            let values = inverse_with_multiplier(&spectrum, nx, ny, |ix, iy| {
                let k2 = kx[ix] * kx[ix] + ky[iy] * ky[iy];
                if power == 1 {
                    Complex64::new(-k2, 0.0)
                } else {
                    Complex64::new(k2 * k2, 0.0)
                }
            });
            for ix in 0..nx {
                for iy in 0..ny {
                    out[[t, ix, iy, component]] = values[ix * ny + iy];
                }
            }
        }
    }
    Ok(out)
}

/// Validate scalar field shape and physical domain lengths.
fn validate_scalar(
    field: ArrayView3<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<(usize, usize, usize)> {
    let (frames, nx, ny) = field.dim();
    if frames == 0 || nx == 0 || ny == 0 {
        return Err(CoreError::InvalidInput(
            "field axes must be non-empty".to_string(),
        ));
    }
    if !(lx.is_finite() && lx > 0.0 && ly.is_finite() && ly > 0.0) {
        return Err(CoreError::InvalidInput(
            "domain lengths must be positive".to_string(),
        ));
    }
    Ok((frames, nx, ny))
}

/// Validate surface vector field shape and physical domain lengths.
fn validate_vector(
    field: ArrayView4<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<(usize, usize, usize)> {
    let (frames, nx, ny, components) = field.dim();
    if components != 2 {
        return Err(CoreError::Shape(
            "vector field must have shape (T, Nx, Ny, 2)".to_string(),
        ));
    }
    validate_scalar(field.index_axis(ndarray::Axis(3), 0), lx, ly)?;
    Ok((frames, nx, ny))
}

/// Return FFT wavenumbers in rustfft order for a periodic domain.
///
/// The output ordering matches the unshifted discrete Fourier transform:
/// non-negative modes first, then wrapped negative modes.
fn wavenumbers(n: usize, length: f64) -> Vec<f64> {
    let cutoff = (n - 1) / 2;
    (0..n)
        .map(|i| {
            let mode = if i <= cutoff {
                i as f64
            } else {
                i as f64 - n as f64
            };
            2.0 * PI * mode / length
        })
        .collect()
}

/// Compute the 2D forward FFT of one scalar frame.
fn fft2_real(field: ArrayView3<'_, f64>, t: usize, nx: usize, ny: usize) -> Vec<Complex64> {
    let mut data = vec![Complex64::new(0.0, 0.0); nx * ny];
    for ix in 0..nx {
        for iy in 0..ny {
            data[ix * ny + iy] = Complex64::new(field[[t, ix, iy]], 0.0);
        }
    }
    fft2_in_place(&mut data, nx, ny, false);
    data
}

/// Compute the 2D forward FFT of one component of one vector-field frame.
fn fft2_vector_component(
    field: ArrayView4<'_, f64>,
    t: usize,
    component: usize,
    nx: usize,
    ny: usize,
) -> Vec<Complex64> {
    let mut data = vec![Complex64::new(0.0, 0.0); nx * ny];
    for ix in 0..nx {
        for iy in 0..ny {
            data[ix * ny + iy] = Complex64::new(field[[t, ix, iy, component]], 0.0);
        }
    }
    fft2_in_place(&mut data, nx, ny, false);
    data
}

/// Multiply a spectrum by an index-dependent spectral factor and invert it.
fn inverse_with_multiplier<F>(
    spectrum: &[Complex64],
    nx: usize,
    ny: usize,
    multiplier: F,
) -> Vec<f64>
where
    F: Fn(usize, usize) -> Complex64,
{
    let mut data = spectrum.to_vec();
    for ix in 0..nx {
        for iy in 0..ny {
            let index = ix * ny + iy;
            data[index] *= multiplier(ix, iy);
        }
    }
    inverse_complex(&data, nx, ny)
}

/// Compute the inverse 2D FFT and return normalized real values.
fn inverse_complex(spectrum: &[Complex64], nx: usize, ny: usize) -> Vec<f64> {
    let mut data = spectrum.to_vec();
    fft2_in_place(&mut data, nx, ny, true);
    let norm = (nx * ny) as f64;
    data.into_iter().map(|value| value.re / norm).collect()
}

/// Run an in-place separable 2D FFT over row-major `(Nx, Ny)` data.
fn fft2_in_place(data: &mut [Complex64], nx: usize, ny: usize, inverse: bool) {
    let mut planner = FftPlanner::<f64>::new();
    let fft_y = plan_fft(&mut planner, ny, inverse);
    let fft_x = plan_fft(&mut planner, nx, inverse);

    for ix in 0..nx {
        let start = ix * ny;
        fft_y.process(&mut data[start..start + ny]);
    }

    let mut column = vec![Complex64::new(0.0, 0.0); nx];
    for iy in 0..ny {
        for ix in 0..nx {
            column[ix] = data[ix * ny + iy];
        }
        fft_x.process(&mut column);
        for ix in 0..nx {
            data[ix * ny + iy] = column[ix];
        }
    }
}

/// Create a forward or inverse FFT plan.
fn plan_fft(planner: &mut FftPlanner<f64>, len: usize, inverse: bool) -> Arc<dyn Fft<f64>> {
    if inverse {
        planner.plan_fft_inverse(len)
    } else {
        planner.plan_fft_forward(len)
    }
}
