use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayD, ArrayViewD, IxDyn};
use rayon::prelude::*;
use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};

use rho_fitting_types::{CoreError, CoreResult};

#[derive(Clone, Copy)]
enum Direction {
    Axial,
    Angular,
    Radial,
}

pub struct CylindricalSpectralOperators {
    pub lx: f64,
    pub theta_period: f64,
    pub shape: (usize, usize, usize),
    pub radial_nodes: Array1<f64>,
    radial_derivative: Array2<f64>,
    dct: Array2<f64>,
    fft_x_forward: Arc<dyn Fft<f64>>,
    fft_x_inverse: Arc<dyn Fft<f64>>,
    fft_theta_forward: Arc<dyn Fft<f64>>,
    fft_theta_inverse: Arc<dyn Fft<f64>>,
}

impl CylindricalSpectralOperators {
    pub fn new(
        lx: f64,
        theta_period: f64,
        r_min: f64,
        r_max: f64,
        nx: usize,
        ntheta: usize,
        nr: usize,
    ) -> CoreResult<Self> {
        if !(lx.is_finite()
            && lx > 0.0
            && theta_period.is_finite()
            && theta_period > 0.0
            && r_min.is_finite()
            && r_min > 0.0
            && r_max.is_finite()
            && r_max > r_min
            && nx > 1
            && ntheta > 1
            && nr > 1)
        {
            return Err(CoreError::InvalidInput(
                "spectral geometry and dimensions are invalid".to_string(),
            ));
        }
        let center = 0.5 * (r_min + r_max);
        let half_span = 0.5 * (r_max - r_min);
        let angles = (0..nr)
            .map(|index| std::f64::consts::PI * (2 * index + 1) as f64 / (2 * nr) as f64)
            .collect::<Vec<_>>();
        let radial_nodes =
            Array1::from_iter(angles.iter().map(|angle| center + half_span * angle.cos()));
        let barycentric = angles
            .iter()
            .enumerate()
            .map(|(index, angle)| {
                if index % 2 == 0 {
                    angle.sin()
                } else {
                    -angle.sin()
                }
            })
            .collect::<Vec<_>>();
        let mut radial_derivative = Array2::zeros((nr, nr));
        for row in 0..nr {
            for col in 0..nr {
                if row != col {
                    radial_derivative[[row, col]] = barycentric[col]
                        / (barycentric[row] * (radial_nodes[row] - radial_nodes[col]));
                }
            }
            radial_derivative[[row, row]] = -radial_derivative.row(row).sum();
        }
        let dct = Array2::from_shape_fn((nr, nr), |(mode, point)| {
            let scale = if mode == 0 {
                (1.0 / nr as f64).sqrt()
            } else {
                (2.0 / nr as f64).sqrt()
            };
            scale * (std::f64::consts::PI * (point as f64 + 0.5) * mode as f64 / nr as f64).cos()
        });
        let mut planner = FftPlanner::<f64>::new();
        Ok(Self {
            lx,
            theta_period,
            shape: (nx, ntheta, nr),
            radial_nodes,
            radial_derivative,
            dct,
            fft_x_forward: planner.plan_fft_forward(nx),
            fft_x_inverse: planner.plan_fft_inverse(nx),
            fft_theta_forward: planner.plan_fft_forward(ntheta),
            fft_theta_inverse: planner.plan_fft_inverse(ntheta),
        })
    }

    pub fn derivative(
        &self,
        values: ArrayViewD<'_, f64>,
        grid_offset: usize,
        direction: usize,
    ) -> CoreResult<ArrayD<f64>> {
        let direction = match direction {
            0 => Direction::Axial,
            1 => Direction::Angular,
            2 => Direction::Radial,
            _ => {
                return Err(CoreError::InvalidInput(
                    "spectral direction must be 0=x, 1=theta, or 2=r".to_string(),
                ))
            }
        };
        self.require_grid(values.shape(), grid_offset)?;
        let owned = values.to_owned();
        let input = owned.as_slice().ok_or_else(|| {
            CoreError::InvalidInput("spectral input could not be made contiguous".to_string())
        })?;
        let shape = values.shape();
        let (nx, ntheta, nr) = self.shape;
        let prefix = shape[..grid_offset].iter().product::<usize>().max(1);
        let trailing = shape[grid_offset + 3..].iter().product::<usize>().max(1);
        let block = nx * ntheta * nr * trailing;
        let (line_count, line_len) = match direction {
            Direction::Axial => (prefix * ntheta * nr * trailing, nx),
            Direction::Angular => (prefix * nx * nr * trailing, ntheta),
            Direction::Radial => (prefix * nx * ntheta * trailing, nr),
        };
        let lines = (0..line_count)
            .into_par_iter()
            .map(|line| {
                let mut data = vec![0.0; line_len];
                for position in 0..line_len {
                    let index = match direction {
                        Direction::Axial => {
                            let component = line % trailing;
                            let reduced = line / trailing;
                            let radial = reduced % nr;
                            let reduced = reduced / nr;
                            let angular = reduced % ntheta;
                            let outer = reduced / ntheta;
                            outer * block
                                + ((position * ntheta + angular) * nr + radial) * trailing
                                + component
                        }
                        Direction::Angular => {
                            let component = line % trailing;
                            let reduced = line / trailing;
                            let radial = reduced % nr;
                            let reduced = reduced / nr;
                            let axial = reduced % nx;
                            let outer = reduced / nx;
                            outer * block
                                + ((axial * ntheta + position) * nr + radial) * trailing
                                + component
                        }
                        Direction::Radial => {
                            let component = line % trailing;
                            let reduced = line / trailing;
                            let angular = reduced % ntheta;
                            let reduced = reduced / ntheta;
                            let axial = reduced % nx;
                            let outer = reduced / nx;
                            outer * block
                                + ((axial * ntheta + angular) * nr + position) * trailing
                                + component
                        }
                    };
                    data[position] = input[index];
                }
                match direction {
                    Direction::Axial => {
                        fourier_derivative(data, self.lx, &self.fft_x_forward, &self.fft_x_inverse)
                    }
                    Direction::Angular => fourier_derivative(
                        data,
                        self.theta_period,
                        &self.fft_theta_forward,
                        &self.fft_theta_inverse,
                    ),
                    Direction::Radial => (0..nr)
                        .map(|row| {
                            (0..nr)
                                .map(|col| self.radial_derivative[[row, col]] * data[col])
                                .sum()
                        })
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        let mut output = vec![0.0; input.len()];
        for (line, data) in lines.into_iter().enumerate() {
            for position in 0..line_len {
                let index = match direction {
                    Direction::Axial => {
                        let component = line % trailing;
                        let reduced = line / trailing;
                        let radial = reduced % nr;
                        let reduced = reduced / nr;
                        let angular = reduced % ntheta;
                        let outer = reduced / ntheta;
                        outer * block
                            + ((position * ntheta + angular) * nr + radial) * trailing
                            + component
                    }
                    Direction::Angular => {
                        let component = line % trailing;
                        let reduced = line / trailing;
                        let radial = reduced % nr;
                        let reduced = reduced / nr;
                        let axial = reduced % nx;
                        let outer = reduced / nx;
                        outer * block
                            + ((axial * ntheta + position) * nr + radial) * trailing
                            + component
                    }
                    Direction::Radial => {
                        let component = line % trailing;
                        let reduced = line / trailing;
                        let angular = reduced % ntheta;
                        let reduced = reduced / ntheta;
                        let axial = reduced % nx;
                        let outer = reduced / nx;
                        outer * block
                            + ((axial * ntheta + angular) * nr + position) * trailing
                            + component
                    }
                };
                output[index] = data[position];
            }
        }
        ArrayD::from_shape_vec(IxDyn(shape), output).map_err(|error| {
            CoreError::InvalidInput(format!("spectral derivative assembly failed: {error}"))
        })
    }

    pub fn gradient(
        &self,
        values: ArrayViewD<'_, f64>,
        grid_offset: usize,
    ) -> CoreResult<ArrayD<f64>> {
        self.require_grid(values.shape(), grid_offset)?;
        let axial = self.derivative(values.clone(), grid_offset, 0)?;
        let angular = self.derivative(values.clone(), grid_offset, 1)?;
        let radial = self.derivative(values, grid_offset, 2)?;
        let trailing = axial.shape()[grid_offset + 3..]
            .iter()
            .product::<usize>()
            .max(1);
        let grid_points = axial.len() / trailing;
        let mut shape = axial.shape().to_vec();
        shape.insert(grid_offset + 3, 3);
        let axial = axial.as_slice().unwrap();
        let angular = angular.as_slice().unwrap();
        let radial = radial.as_slice().unwrap();
        let nr = self.shape.2;
        let mut output = vec![0.0; grid_points * 3 * trailing];
        for point in 0..grid_points {
            let radius = self.radial_nodes[point % nr];
            for component in 0..trailing {
                let input_index = point * trailing + component;
                output[(point * 3) * trailing + component] = axial[input_index];
                output[(point * 3 + 1) * trailing + component] = angular[input_index] / radius;
                output[(point * 3 + 2) * trailing + component] = radial[input_index];
            }
        }
        ArrayD::from_shape_vec(IxDyn(&shape), output).map_err(|error| {
            CoreError::InvalidInput(format!("spectral gradient assembly failed: {error}"))
        })
    }

    pub fn divergence(
        &self,
        values: ArrayViewD<'_, f64>,
        grid_offset: usize,
    ) -> CoreResult<ArrayD<f64>> {
        self.require_grid(values.shape(), grid_offset)?;
        let direction_axis = grid_offset + 3;
        if values.ndim() <= direction_axis || values.shape()[direction_axis] != 3 {
            return Err(CoreError::Shape(
                "flux direction axis must have length 3".to_string(),
            ));
        }
        let trailing = values.shape()[direction_axis + 1..]
            .iter()
            .product::<usize>()
            .max(1);
        let grid_points = values.len() / (3 * trailing);
        let mut component_shape = values.shape().to_vec();
        component_shape.remove(direction_axis);
        let source = values.to_owned();
        let source = source.as_slice().unwrap();
        let mut components = [
            vec![0.0; grid_points * trailing],
            vec![0.0; grid_points * trailing],
            vec![0.0; grid_points * trailing],
        ];
        for point in 0..grid_points {
            let radius = self.radial_nodes[point % self.shape.2];
            for direction in 0..3 {
                for component in 0..trailing {
                    let mut value = source[(point * 3 + direction) * trailing + component];
                    if direction == 2 {
                        value *= radius;
                    }
                    components[direction][point * trailing + component] = value;
                }
            }
        }
        let arrays = components
            .into_iter()
            .map(|data| ArrayD::from_shape_vec(IxDyn(&component_shape), data).unwrap())
            .collect::<Vec<_>>();
        let dx = self.derivative(arrays[0].view(), grid_offset, 0)?;
        let dt = self.derivative(arrays[1].view(), grid_offset, 1)?;
        let dr = self.derivative(arrays[2].view(), grid_offset, 2)?;
        let mut output = vec![0.0; grid_points * trailing];
        let (dx, dt, dr) = (
            dx.as_slice().unwrap(),
            dt.as_slice().unwrap(),
            dr.as_slice().unwrap(),
        );
        for point in 0..grid_points {
            let radius = self.radial_nodes[point % self.shape.2];
            for component in 0..trailing {
                let index = point * trailing + component;
                output[index] = dx[index] + dt[index] / radius + dr[index] / radius;
            }
        }
        ArrayD::from_shape_vec(IxDyn(&component_shape), output).map_err(|error| {
            CoreError::InvalidInput(format!("spectral divergence assembly failed: {error}"))
        })
    }

    pub fn laplacian(
        &self,
        values: ArrayViewD<'_, f64>,
        grid_offset: usize,
    ) -> CoreResult<ArrayD<f64>> {
        let dx = self.derivative(values.clone(), grid_offset, 0)?;
        let dxx = self.derivative(dx.view(), grid_offset, 0)?;
        let dt = self.derivative(values.clone(), grid_offset, 1)?;
        let dtt = self.derivative(dt.view(), grid_offset, 1)?;
        let dr = self.derivative(values, grid_offset, 2)?;
        let drr = self.derivative(dr.view(), grid_offset, 2)?;
        let trailing = dxx.shape()[grid_offset + 3..]
            .iter()
            .product::<usize>()
            .max(1);
        let grid_points = dxx.len() / trailing;
        let (dxx_s, dtt_s, dr_s, drr_s) = (
            dxx.as_slice().unwrap(),
            dtt.as_slice().unwrap(),
            dr.as_slice().unwrap(),
            drr.as_slice().unwrap(),
        );
        let mut output = vec![0.0; dxx.len()];
        for point in 0..grid_points {
            let radius = self.radial_nodes[point % self.shape.2];
            for component in 0..trailing {
                let index = point * trailing + component;
                output[index] = dxx_s[index]
                    + drr_s[index]
                    + dr_s[index] / radius
                    + dtt_s[index] / (radius * radius);
            }
        }
        ArrayD::from_shape_vec(dxx.raw_dim(), output).map_err(|error| {
            CoreError::InvalidInput(format!("spectral laplacian assembly failed: {error}"))
        })
    }

    pub fn filter_two_thirds(
        &self,
        values: ArrayViewD<'_, f64>,
        grid_offset: usize,
    ) -> CoreResult<ArrayD<f64>> {
        self.require_grid(values.shape(), grid_offset)?;
        let shape = values.shape().to_vec();
        let prefix = shape[..grid_offset].iter().product::<usize>().max(1);
        let trailing = shape[grid_offset + 3..].iter().product::<usize>().max(1);
        let owned = values.to_owned();
        let input = owned.as_slice().unwrap();
        let (nx, ntheta, nr) = self.shape;
        let grid_len = nx * ntheta * nr;
        let fields = (0..prefix * trailing)
            .into_par_iter()
            .map(|field| {
                let outer = field / trailing;
                let component = field % trailing;
                let mut data = vec![Complex64::new(0.0, 0.0); grid_len];
                for ix in 0..nx {
                    for itheta in 0..ntheta {
                        for mode in 0..nr {
                            let value = (0..nr)
                                .map(|point| {
                                    let index = outer * grid_len * trailing
                                        + ((ix * ntheta + itheta) * nr + point) * trailing
                                        + component;
                                    self.dct[[mode, point]] * input[index]
                                })
                                .sum::<f64>();
                            data[(ix * ntheta + itheta) * nr + mode] = Complex64::new(value, 0.0);
                        }
                    }
                }
                for itheta in 0..ntheta {
                    for mode in 0..nr {
                        let mut line = (0..nx)
                            .map(|ix| data[(ix * ntheta + itheta) * nr + mode])
                            .collect::<Vec<_>>();
                        self.fft_x_forward.process(&mut line);
                        for ix in 0..nx {
                            data[(ix * ntheta + itheta) * nr + mode] = line[ix];
                        }
                    }
                }
                for ix in 0..nx {
                    for mode in 0..nr {
                        let mut line = (0..ntheta)
                            .map(|itheta| data[(ix * ntheta + itheta) * nr + mode])
                            .collect::<Vec<_>>();
                        self.fft_theta_forward.process(&mut line);
                        for itheta in 0..ntheta {
                            data[(ix * ntheta + itheta) * nr + mode] = line[itheta];
                        }
                    }
                }
                let radial_cutoff = (2 * nr) / 3;
                for ix in 0..nx {
                    let kx = signed_mode(ix, nx).unsigned_abs();
                    for itheta in 0..ntheta {
                        let kt = signed_mode(itheta, ntheta).unsigned_abs();
                        for mode in 0..nr {
                            if kx > nx / 3 || kt > ntheta / 3 || mode >= radial_cutoff {
                                data[(ix * ntheta + itheta) * nr + mode] = Complex64::new(0.0, 0.0);
                            }
                        }
                    }
                }
                for ix in 0..nx {
                    for mode in 0..nr {
                        let mut line = (0..ntheta)
                            .map(|itheta| data[(ix * ntheta + itheta) * nr + mode])
                            .collect::<Vec<_>>();
                        self.fft_theta_inverse.process(&mut line);
                        for itheta in 0..ntheta {
                            data[(ix * ntheta + itheta) * nr + mode] = line[itheta];
                        }
                    }
                }
                for itheta in 0..ntheta {
                    for mode in 0..nr {
                        let mut line = (0..nx)
                            .map(|ix| data[(ix * ntheta + itheta) * nr + mode])
                            .collect::<Vec<_>>();
                        self.fft_x_inverse.process(&mut line);
                        for ix in 0..nx {
                            data[(ix * ntheta + itheta) * nr + mode] = line[ix];
                        }
                    }
                }
                let fft_scale = 1.0 / (nx * ntheta) as f64;
                let mut output = vec![0.0; grid_len];
                for ix in 0..nx {
                    for itheta in 0..ntheta {
                        for point in 0..nr {
                            output[(ix * ntheta + itheta) * nr + point] = (0..nr)
                                .map(|mode| {
                                    self.dct[[mode, point]]
                                        * data[(ix * ntheta + itheta) * nr + mode].re
                                        * fft_scale
                                })
                                .sum();
                        }
                    }
                }
                output
            })
            .collect::<Vec<_>>();
        let mut output = vec![0.0; values.len()];
        for (field, data) in fields.into_iter().enumerate() {
            let outer = field / trailing;
            let component = field % trailing;
            for point in 0..grid_len {
                output[outer * grid_len * trailing + point * trailing + component] = data[point];
            }
        }
        ArrayD::from_shape_vec(IxDyn(&shape), output).map_err(|error| {
            CoreError::InvalidInput(format!("spectral filter assembly failed: {error}"))
        })
    }

    fn require_grid(&self, shape: &[usize], offset: usize) -> CoreResult<()> {
        if shape.len() < offset + 3
            || shape[offset] != self.shape.0
            || shape[offset + 1] != self.shape.1
            || shape[offset + 2] != self.shape.2
        {
            return Err(CoreError::Shape(format!(
                "expected cylindrical grid {:?} at axis {offset}; got {shape:?}",
                self.shape
            )));
        }
        Ok(())
    }
}

fn fourier_derivative(
    values: Vec<f64>,
    period: f64,
    forward: &Arc<dyn Fft<f64>>,
    inverse: &Arc<dyn Fft<f64>>,
) -> Vec<f64> {
    let size = values.len();
    let mut spectrum = values
        .into_iter()
        .map(|value| Complex64::new(value, 0.0))
        .collect::<Vec<_>>();
    forward.process(&mut spectrum);
    for (index, value) in spectrum.iter_mut().enumerate() {
        let mode = signed_mode(index, size);
        if size % 2 == 0 && index == size / 2 {
            *value = Complex64::new(0.0, 0.0);
        } else {
            *value *= Complex64::new(0.0, 2.0 * std::f64::consts::PI * mode as f64 / period);
        }
    }
    inverse.process(&mut spectrum);
    spectrum
        .into_iter()
        .map(|value| value.re / size as f64)
        .collect()
}

fn signed_mode(index: usize, size: usize) -> isize {
    if index <= size / 2 {
        index as isize
    } else {
        index as isize - size as isize
    }
}
