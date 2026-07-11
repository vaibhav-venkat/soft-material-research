use crate::mechanics::CylindricalGrid;

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
    pub(super) domain: CylindricalGrid,
}

impl Grid3 {
    pub(super) fn from_domain(domain: CylindricalGrid) -> Self {
        let (nx, ntheta, nr) = domain.shape();
        Self {
            nx,
            ntheta,
            nr,
            x_centers: domain.x.iter().map(|value| *value as f32).collect(),
            theta_centers: domain.theta.iter().map(|value| *value as f32).collect(),
            r_centers: domain.r.iter().map(|value| *value as f32).collect(),
            volumes: domain
                .cell_volumes
                .iter()
                .map(|value| *value as f32)
                .collect(),
            lx: domain.lx as f32,
            theta_period: domain.theta_period as f32,
            domain,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.nx * self.ntheta * self.nr
    }
}
