use ndarray::ArrayView1;

use crate::{CoreError, CoreResult};

pub(super) struct Grid3 {
    pub(super) nx: usize,
    pub(super) ntheta: usize,
    pub(super) nr: usize,
    pub(super) x_centers: Vec<f32>,
    pub(super) theta_centers: Vec<f32>,
    pub(super) r_centers: Vec<f32>,
    pub(super) volumes: Vec<f32>,
    pub(super) lx: f32,
    pub(super) theta_period: f32,
    #[allow(dead_code)]
    pub(super) dr: f32,
}

impl Grid3 {
    pub(super) fn new(
        x_centers: ArrayView1<'_, f64>,
        theta_centers: ArrayView1<'_, f64>,
        r_centers: ArrayView1<'_, f64>,
        lx: f64,
        theta_period: f64,
    ) -> CoreResult<Self> {
        let nx = x_centers.len();
        let ntheta = theta_centers.len();
        let nr = r_centers.len();
        if nx == 0 || ntheta == 0 || nr == 0 {
            return Err(CoreError::InvalidInput(
                "grid centers must be non-empty".to_string(),
            ));
        }
        let dx = (lx / nx as f64) as f32;
        let dtheta = (theta_period / ntheta as f64) as f32;
        let dr = radial_spacing(r_centers)?;
        let r0 = r_centers[0] as f32;
        let mut volumes = Vec::with_capacity(nx * ntheta * nr);
        for _ix in 0..nx {
            for _itheta in 0..ntheta {
                for ir in 0..nr {
                    let inner = r0 - 0.5 * dr + ir as f32 * dr;
                    let outer = inner + dr;
                    let volume = dx * dtheta * (outer * outer - inner * inner) * 0.5;
                    if !(volume.is_finite() && volume > 0.0 && inner >= 0.0) {
                        return Err(CoreError::InvalidInput(
                            "cylindrical grid contains a non-positive finite-volume cell"
                                .to_string(),
                        ));
                    }
                    volumes.push(volume);
                }
            }
        }
        Ok(Self {
            nx,
            ntheta,
            nr,
            x_centers: x_centers.iter().map(|value| *value as f32).collect(),
            theta_centers: theta_centers.iter().map(|value| *value as f32).collect(),
            r_centers: r_centers.iter().map(|value| *value as f32).collect(),
            volumes,
            lx: lx as f32,
            theta_period: theta_period as f32,
            dr,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.nx * self.ntheta * self.nr
    }
}

pub(super) fn radial_spacing(r_centers: ArrayView1<'_, f64>) -> CoreResult<f32> {
    if r_centers.len() == 1 {
        let dr = (r_centers[0] * 0.1) as f32;
        if dr.is_finite() && dr > 0.0 {
            return Ok(dr);
        }
        return Err(CoreError::InvalidInput(
            "single radial bin center must be positive".to_string(),
        ));
    }
    let dr = r_centers[1] - r_centers[0];
    if !(dr.is_finite() && dr > 0.0) {
        return Err(CoreError::InvalidInput(
            "radial centers must be increasing".to_string(),
        ));
    }
    for index in 1..r_centers.len() - 1 {
        let next = r_centers[index + 1] - r_centers[index];
        if (next - dr).abs() > 1.0e-8 * dr.abs().max(1.0) {
            return Err(CoreError::InvalidInput(
                "radial centers must be uniformly spaced".to_string(),
            ));
        }
    }
    Ok(dr as f32)
}

pub(super) fn flat_index(ix: usize, itheta: usize, ir: usize, ntheta: usize, nr: usize) -> usize {
    ix * ntheta * nr + itheta * nr + ir
}
