use crate::{CoreError, CoreResult};
use ndarray::{
    Array1, Array4, Array5, Array6, ArrayD, ArrayView1, ArrayView2, ArrayView3, ArrayView5,
    ArrayView6, IxDyn,
};

use crate::fields::ParticleFieldSet;
pub use crate::fields::{CurrentQField, MechanicalFieldSet};

pub const PHYSICAL_COMPONENT_COUNT: usize = 3;

/// Components are always ordered as `(x, e_theta, e_r)` throughout the core.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalComponent {
    Axial,
    Azimuthal,
    Radial,
}

impl PhysicalComponent {
    pub const ALL: [Self; PHYSICAL_COMPONENT_COUNT] = [Self::Axial, Self::Azimuthal, Self::Radial];

    pub const fn index(self) -> usize {
        match self {
            Self::Axial => 0,
            Self::Azimuthal => 1,
            Self::Radial => 2,
        }
    }
}

/// Row-major tensor component order derived from the same physical frame.
/// xx, xtheta, xr, thetax, ...
pub const TENSOR_COMPONENTS: [(PhysicalComponent, PhysicalComponent); 9] = [
    (PhysicalComponent::Axial, PhysicalComponent::Axial),
    (PhysicalComponent::Axial, PhysicalComponent::Azimuthal),
    (PhysicalComponent::Axial, PhysicalComponent::Radial),
    (PhysicalComponent::Azimuthal, PhysicalComponent::Axial),
    (PhysicalComponent::Azimuthal, PhysicalComponent::Azimuthal),
    (PhysicalComponent::Azimuthal, PhysicalComponent::Radial),
    (PhysicalComponent::Radial, PhysicalComponent::Axial),
    (PhysicalComponent::Radial, PhysicalComponent::Azimuthal),
    (PhysicalComponent::Radial, PhysicalComponent::Radial),
];

/// Owned cylindrical grid
#[derive(Clone)]
pub struct CylindricalGrid {
    pub x: Array1<f64>,
    pub theta: Array1<f64>,
    pub r: Array1<f64>,
    pub lx: f64,
    pub theta_period: f64,
    pub cell_volumes: Array1<f64>,
}

impl CylindricalGrid {
    pub fn new(
        x: ArrayView1<'_, f64>,
        theta: ArrayView1<'_, f64>,
        r: ArrayView1<'_, f64>,
        lx: f64,
        theta_period: f64,
    ) -> CoreResult<Self> {
        if x.is_empty() || theta.is_empty() || r.is_empty() {
            return Err(CoreError::InvalidInput(
                "grid centers must be non-empty".to_string(),
            ));
        }
        if !x.iter().all(|value| value.is_finite())
            || !theta.iter().all(|value| value.is_finite())
            || !lx.is_finite()
            || lx <= 0.0
            || !theta_period.is_finite()
            || theta_period <= 0.0
        {
            return Err(CoreError::InvalidInput(
                "grid coordinates and periods must be finite and positive".to_string(),
            ));
        }
        if !r.iter().all(|value| value.is_finite() && *value > 0.0) {
            return Err(CoreError::InvalidInput(
                "radial centers must be finite and positive".to_string(),
            ));
        }
        let dr = radial_spacing(r)?;
        let dx = lx / x.len() as f64;
        let dtheta = theta_period / theta.len() as f64;
        let r0 = r[0];
        let mut cell_volumes = Vec::with_capacity(x.len() * theta.len() * r.len());
        for _ in 0..x.len() {
            for _ in 0..theta.len() {
                for ir in 0..r.len() {
                    let inner = r0 - 0.5 * dr + ir as f64 * dr;
                    let outer = inner + dr;
                    let volume = dx * dtheta * (outer * outer - inner * inner) * 0.5;
                    if !(volume.is_finite() && volume > 0.0 && inner >= 0.0) {
                        return Err(CoreError::InvalidInput(
                            "cylindrical grid contains a non-positive finite-volume cell"
                                .to_string(),
                        ));
                    }
                    cell_volumes.push(volume);
                }
            }
        }
        Ok(Self {
            x: x.to_owned(),
            theta: theta.to_owned(),
            r: r.to_owned(),
            lx,
            theta_period,
            cell_volumes: Array1::from_vec(cell_volumes),
        })
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.x.len(), self.theta.len(), self.r.len())
    }

    pub fn flat_index(&self, ix: usize, itheta: usize, ir: usize) -> usize {
        ix * self.theta.len() * self.r.len() + itheta * self.r.len() + ir
    }
}

pub fn radial_spacing(r: ArrayView1<'_, f64>) -> CoreResult<f64> {
    if r.len() == 1 {
        let dr = r[0] * 0.1;
        return if dr.is_finite() && dr > 0.0 {
            Ok(dr)
        } else {
            Err(CoreError::InvalidInput(
                "single radial bin center must be positive".to_string(),
            ))
        };
    }
    let dr = r[1] - r[0];
    if !(dr.is_finite() && dr > 0.0) {
        return Err(CoreError::InvalidInput(
            "radial centers must be increasing".to_string(),
        ));
    }
    for idx in 1..r.len() - 1 {
        let next = r[idx + 1] - r[idx];
        if (next - dr).abs() > 1.0e-8 * dr.abs().max(1.0) {
            return Err(CoreError::InvalidInput(
                "radial centers must be uniformly spaced".to_string(),
            ));
        }
    }
    Ok(dr)
}

pub fn relative_mass_error(mass: &[f64], expected: usize) -> f64 {
    let total: f64 = mass.iter().sum();
    if expected > 0 {
        (total - expected as f64).abs() / expected as f64
    } else {
        0.0
    }
}

/// Validated particle arrays plus the canonical grid
#[derive(Clone)]
pub struct MechanicalInputViews<'a> {
    pub particles: ParticleFieldSet<'a>,
    pub grid: CylindricalGrid,
    pub sigma: f64,
}

impl<'a> MechanicalInputViews<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
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
        gamma: f64,
        u0: f64,
    ) -> CoreResult<Self> {
        let (frames, particles, components) = coords.dim();
        if components != PHYSICAL_COMPONENT_COUNT {
            return Err(CoreError::Shape(
                "coords must have shape (T,N,3)".to_string(),
            ));
        }
        if directions.dim() != (frames, particles, PHYSICAL_COMPONENT_COUNT)
            || velocities.dim() != (frames, particles, PHYSICAL_COMPONENT_COUNT)
        {
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
        if !(sigma.is_finite() && sigma > 0.0 && gamma.is_finite() && u0.is_finite() && u0 != 0.0) {
            return Err(CoreError::InvalidInput(
                "sigma and gamma must be finite and positive where required; u0 must be nonzero"
                    .to_string(),
            ));
        }
        Ok(Self {
            particles: ParticleFieldSet {
                coords,
                directions,
                velocities,
                hexatic_order: psi6_abs,
                mask,
            },
            grid: CylindricalGrid::new(x_centers, theta_centers, r_centers, lx, theta_period)?,
            sigma,
        })
    }
}

impl MechanicalFieldSet {
    pub fn zeros(frames: usize, grid: &CylindricalGrid) -> Self {
        let (nx, ntheta, nr) = grid.shape();
        Self {
            rho: Array4::zeros((frames, nx, ntheta, nr)),
            p: Array5::zeros((frames, nx, ntheta, nr, PHYSICAL_COMPONENT_COUNT)),
            q: Array6::zeros((
                frames,
                nx,
                ntheta,
                nr,
                PHYSICAL_COMPONENT_COUNT,
                PHYSICAL_COMPONENT_COUNT,
            )),
            a: Array6::zeros((
                frames,
                nx,
                ntheta,
                nr,
                PHYSICAL_COMPONENT_COUNT,
                PHYSICAL_COMPONENT_COUNT,
            )),
            hexatic_order: Array4::from_elem((frames, nx, ntheta, nr), f32::NAN),
            j_rho: Array5::zeros((frames, nx, ntheta, nr, PHYSICAL_COMPONENT_COUNT)),
            j_p: Array6::zeros((
                frames,
                nx,
                ntheta,
                nr,
                PHYSICAL_COMPONENT_COUNT,
                PHYSICAL_COMPONENT_COUNT,
            )),
            j_q: ArrayD::zeros(IxDyn(&[frames, nx, ntheta, nr, 3, 3, 3])),
        }
    }
}

/// Mass-weighted frame accumulator used by both CPU and Burn deposition.
pub struct MechanicalFrame {
    pub mass: Vec<f64>,
    pub hexatic_mass: Vec<f64>,
    pub p_mass: [Vec<f64>; 3],
    pub q_mass: [Vec<f64>; 9],
    pub j_rho_mass: [Vec<f64>; 3],
    pub j_p_mass: [Vec<f64>; 9],
    pub j_q_mass: [Vec<f64>; 27],
}

impl MechanicalFrame {
    pub fn new(grid_len: usize) -> Self {
        Self {
            mass: vec![0.0; grid_len],
            hexatic_mass: vec![0.0; grid_len],
            p_mass: std::array::from_fn(|_| vec![0.0; grid_len]),
            q_mass: std::array::from_fn(|_| vec![0.0; grid_len]),
            j_rho_mass: std::array::from_fn(|_| vec![0.0; grid_len]),
            j_p_mass: std::array::from_fn(|_| vec![0.0; grid_len]),
            j_q_mass: std::array::from_fn(|_| vec![0.0; grid_len]),
        }
    }

    pub fn write_into(
        &self,
        frame: usize,
        grid: &CylindricalGrid,
        fields: &mut MechanicalFieldSet,
    ) {
        let (nx, ntheta, nr) = grid.shape();
        for ix in 0..nx {
            for itheta in 0..ntheta {
                for ir in 0..nr {
                    let flat = grid.flat_index(ix, itheta, ir);
                    let volume = grid.cell_volumes[flat];
                    let mass = self.mass[flat];
                    let density = mass / volume;
                    fields.rho[[frame, ix, itheta, ir]] = density as f32;
                    if mass <= 1.0e-12 {
                        continue;
                    }
                    let psi = self.hexatic_mass[flat] / mass;
                    // TODO: Make the value just psi, not psi^2. Update it across the report and all conventions as well.
                    fields.hexatic_order[[frame, ix, itheta, ir]] = (psi * psi) as f32;
                    for component in PhysicalComponent::ALL {
                        let index = component.index();
                        fields.p[[frame, ix, itheta, ir, index]] =
                            (self.p_mass[index][flat] / volume) as f32;
                        fields.j_rho[[frame, ix, itheta, ir, index]] =
                            (self.j_rho_mass[index][flat] / volume) as f32;
                    }
                    for (tensor_index, (row, col)) in TENSOR_COMPONENTS.into_iter().enumerate() {
                        let row_index = row.index();
                        let col_index = col.index();
                        let q_value = self.q_mass[tensor_index][flat] / volume;
                        fields.q[[frame, ix, itheta, ir, row_index, col_index]] = q_value as f32;
                        // TODO: Add d = 3 to the constants.rs, and use it instead of 3.0
                        fields.a[[frame, ix, itheta, ir, row_index, col_index]] =
                            (q_value + if row == col { density / 3.0 } else { 0.0 }) as f32;
                        for flux in PhysicalComponent::ALL {
                            let flux_index = flux.index();
                            fields.j_p[[frame, ix, itheta, ir, flux_index, row_index]] =
                                (self.j_p_mass[flux_index * 3 + row_index][flat] / volume) as f32;
                            fields.j_q[[frame, ix, itheta, ir, flux_index, row_index, col_index]] =
                                (self.j_q_mass[flux_index * 9 + tensor_index][flat] / volume) as f32;
                        }
                    }
                }
            }
        }
    }
}

pub struct MechanicalTargets {
    pub y_rho: Array5<f64>,
    pub y_p: Array6<f64>,
}

pub fn build_targets(
    p: ArrayView5<'_, f64>,
    j_rho: ArrayView5<'_, f64>,
    j_p: ArrayView6<'_, f64>,
    gamma: f64,
    u0: f64,
) -> CoreResult<MechanicalTargets> {
    let shape = p.shape();
    if shape[4] != PHYSICAL_COMPONENT_COUNT {
        return Err(CoreError::Shape(
            "P must have shape (T,Nx,Ntheta,Nr,3)".to_string(),
        ));
    }
    if j_rho.shape() != shape {
        return Err(CoreError::Shape(
            "J_rho must have shape (T,Nx,Ntheta,Nr,3)".to_string(),
        ));
    }
    if j_p.shape() != [shape[0], shape[1], shape[2], shape[3], 3, 3] {
        return Err(CoreError::Shape(
            "J_P must have shape (T,Nx,Ntheta,Nr,3,3)".to_string(),
        ));
    }
    if !(gamma.is_finite() && u0.is_finite() && u0 != 0.0) {
        return Err(CoreError::InvalidInput(
            "gamma must be finite and u0 must be nonzero".to_string(),
        ));
    }
    let y_rho = (&j_rho - &(&p * u0)) * gamma;
    Ok(MechanicalTargets {
        y_rho,
        y_p: j_p.to_owned() / u0,
    })
}
