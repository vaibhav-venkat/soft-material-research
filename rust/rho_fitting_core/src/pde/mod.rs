use ndarray::{ArrayD, ArrayView1, ArrayViewD, Axis, IxDyn};

use crate::constants::{P_RELAXATION_COEFFICIENT, Q_RELAXATION_COEFFICIENT};
use crate::fitting;
use crate::interpolation::RadialTransfer;
use crate::spectral::CylindricalSpectralOperators;
use crate::{CoreError, CoreResult};

pub(crate) struct ValidationResult {
    pub(crate) rho: ArrayD<f64>,
    pub(crate) p: ArrayD<f64>,
    pub(crate) q: ArrayD<f64>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_validation(
    rho_cache: ArrayViewD<'_, f64>,
    p_cache: ArrayViewD<'_, f64>,
    q_cache: ArrayViewD<'_, f64>,
    psi6_cache: ArrayViewD<'_, f64>,
    y_p_cache: ArrayViewD<'_, f64>,
    times: ArrayView1<'_, f64>,
    c_rho: ArrayView1<'_, f64>,
    c_p: ArrayView1<'_, f64>,
    c_q: ArrayView1<'_, f64>,
    r_centers: ArrayView1<'_, f64>,
    r_edges: ArrayView1<'_, f64>,
    lx: f64,
    theta_period: f64,
    u0: f64,
    gamma: f64,
    frames: usize,
    dt_max: f64,
    mode: u8,
    ubar_source: u8,
) -> CoreResult<ValidationResult> {
    if rho_cache.ndim() != 4
        || p_cache.shape() != [rho_cache.shape(), &[3]].concat()
        || q_cache.shape() != [rho_cache.shape(), &[3, 3]].concat()
        || psi6_cache.shape() != rho_cache.shape()
        || y_p_cache.shape() != [rho_cache.shape(), &[3, 3]].concat()
        || times.len() != rho_cache.shape()[0]
        || r_centers.len() != rho_cache.shape()[3]
        || r_edges.len() != r_centers.len() + 1
        || c_rho.len() != 3
        || c_p.len() != 3
        || c_q.len() != 3
    {
        return Err(CoreError::Shape(
            "PDE validation inputs have incompatible shapes".to_string(),
        ));
    }
    if frames < 2
        || frames > times.len()
        || !(dt_max.is_finite() && dt_max > 0.0 && gamma.is_finite() && gamma != 0.0)
        || mode > 3
        || ubar_source > 1
    {
        return Err(CoreError::InvalidInput(
            "PDE validation options are invalid".to_string(),
        ));
    }
    let (nx, ntheta, nr) = (
        rho_cache.shape()[1],
        rho_cache.shape()[2],
        rho_cache.shape()[3],
    );
    let operators = CylindricalSpectralOperators::new(
        lx,
        theta_period,
        r_edges[0],
        r_edges[r_edges.len() - 1],
        nx,
        ntheta,
        nr,
    )?;
    let to_spectral = RadialTransfer::new(r_centers, operators.radial_nodes.view())?;
    let to_cache = RadialTransfer::new(operators.radial_nodes.view(), r_centers)?;
    let frame = |values: ArrayViewD<'_, f64>, index: usize| {
        values.index_axis(Axis(0), index).to_owned().into_dyn()
    };
    let mut rho = to_spectral.apply(frame(rho_cache.clone(), 0).view(), 2)?;
    let mut p = to_spectral.apply(frame(p_cache.clone(), 0).view(), 2)?;
    let mut q = to_spectral.apply(frame(q_cache.clone(), 0).view(), 2)?;
    let psi6 = to_spectral.apply(frame(psi6_cache, 0).view(), 2)?;
    let mut rho_output = ArrayD::zeros(IxDyn(&[frames, nx, ntheta, nr]));
    let mut p_output = ArrayD::zeros(IxDyn(&[frames, nx, ntheta, nr, 3]));
    let mut q_output = ArrayD::zeros(IxDyn(&[frames, nx, ntheta, nr, 3, 3]));
    rho_output
        .index_axis_mut(Axis(0), 0)
        .assign(&rho_cache.index_axis(Axis(0), 0));
    p_output
        .index_axis_mut(Axis(0), 0)
        .assign(&p_cache.index_axis(Axis(0), 0));
    q_output
        .index_axis_mut(Axis(0), 0)
        .assign(&q_cache.index_axis(Axis(0), 0));

    for saved_frame in 0..frames - 1 {
        let interval = times[saved_frame + 1] - times[saved_frame];
        if !(interval.is_finite() && interval > 0.0) {
            return Err(CoreError::InvalidInput(
                "PDE validation times must increase strictly".to_string(),
            ));
        }
        let steps = (interval / dt_max).ceil().max(1.0) as usize;
        let dt = interval / steps as f64;
        let rho_refs = transfer_pair(&to_spectral, rho_cache.clone(), saved_frame)?;
        let p_refs = transfer_pair(&to_spectral, p_cache.clone(), saved_frame)?;
        let q_refs = transfer_pair(&to_spectral, q_cache.clone(), saved_frame)?;
        let y_p_refs = transfer_pair(&to_spectral, y_p_cache.clone(), saved_frame)?;
        println!(
            "[rho_fitting.pde] frame={}/{} substeps={steps}",
            saved_frame + 1,
            frames - 1
        );
        for substep in 0..steps {
            let weight = substep as f64 / steps as f64;
            let state = State { rho, p, q };
            let k1 = rollout_rhs(
                &state,
                weight,
                &rho_refs,
                &p_refs,
                &q_refs,
                &y_p_refs,
                &psi6,
                c_rho,
                c_p,
                c_q,
                &operators,
                u0,
                gamma,
                mode,
                ubar_source,
            )?;
            let stage2 = state.add_scaled(&k1, 0.5 * dt)?;
            let k2 = rollout_rhs(
                &stage2,
                weight + 0.5 / steps as f64,
                &rho_refs,
                &p_refs,
                &q_refs,
                &y_p_refs,
                &psi6,
                c_rho,
                c_p,
                c_q,
                &operators,
                u0,
                gamma,
                mode,
                ubar_source,
            )?;
            let stage3 = state.add_scaled(&k2, 0.5 * dt)?;
            let k3 = rollout_rhs(
                &stage3,
                weight + 0.5 / steps as f64,
                &rho_refs,
                &p_refs,
                &q_refs,
                &y_p_refs,
                &psi6,
                c_rho,
                c_p,
                c_q,
                &operators,
                u0,
                gamma,
                mode,
                ubar_source,
            )?;
            let stage4 = state.add_scaled(&k3, dt)?;
            let k4 = rollout_rhs(
                &stage4,
                weight + 1.0 / steps as f64,
                &rho_refs,
                &p_refs,
                &q_refs,
                &y_p_refs,
                &psi6,
                c_rho,
                c_p,
                c_q,
                &operators,
                u0,
                gamma,
                mode,
                ubar_source,
            )?;
            let next = state.rk4(&k1, &k2, &k3, &k4, dt)?;
            let next_weight = weight + 1.0 / steps as f64;
            rho = if mode == 0 || mode == 1 {
                next.rho
            } else {
                interpolate(&rho_refs.0, &rho_refs.1, next_weight)
            };
            p = if mode == 0 || mode == 2 {
                next.p
            } else {
                interpolate(&p_refs.0, &p_refs.1, next_weight)
            };
            q = if mode == 0 || mode == 3 {
                next.q
            } else {
                interpolate(&q_refs.0, &q_refs.1, next_weight)
            };
            rho = operators.filter_two_thirds(rho.view(), 0)?;
            p = operators.filter_two_thirds(p.view(), 0)?;
            q = operators.filter_two_thirds(q.view(), 0)?;
            if !rho
                .iter()
                .chain(p.iter())
                .chain(q.iter())
                .all(|value| value.is_finite())
            {
                return Err(CoreError::InvalidInput(format!(
                    "PDE rollout became non-finite at frame {saved_frame}, substep {substep}"
                )));
            }
        }
        println!(
            "[rho_fitting.pde] frame={}/{} complete",
            saved_frame + 1,
            frames - 1
        );
        rho_output
            .index_axis_mut(Axis(0), saved_frame + 1)
            .assign(&to_cache.apply(rho.view(), 2)?);
        p_output
            .index_axis_mut(Axis(0), saved_frame + 1)
            .assign(&to_cache.apply(p.view(), 2)?);
        q_output
            .index_axis_mut(Axis(0), saved_frame + 1)
            .assign(&to_cache.apply(q.view(), 2)?);
    }
    Ok(ValidationResult {
        rho: rho_output,
        p: p_output,
        q: q_output,
    })
}

struct State {
    rho: ArrayD<f64>,
    p: ArrayD<f64>,
    q: ArrayD<f64>,
}

impl State {
    fn add_scaled(&self, rhs: &Self, scale: f64) -> CoreResult<Self> {
        Ok(Self {
            rho: add_scaled(&self.rho, &rhs.rho, scale)?,
            p: add_scaled(&self.p, &rhs.p, scale)?,
            q: add_scaled(&self.q, &rhs.q, scale)?,
        })
    }
    fn rk4(&self, k1: &Self, k2: &Self, k3: &Self, k4: &Self, dt: f64) -> CoreResult<Self> {
        let combine = |value: &ArrayD<f64>,
                       a: &ArrayD<f64>,
                       b: &ArrayD<f64>,
                       c: &ArrayD<f64>,
                       d: &ArrayD<f64>|
         -> CoreResult<ArrayD<f64>> {
            if value.shape() != a.shape()
                || a.shape() != b.shape()
                || b.shape() != c.shape()
                || c.shape() != d.shape()
            {
                return Err(CoreError::Shape("RK4 stage shapes differ".to_string()));
            }
            Ok(value + &((a + &(b * 2.0) + &(c * 2.0) + d) * (dt / 6.0)))
        };
        Ok(Self {
            rho: combine(&self.rho, &k1.rho, &k2.rho, &k3.rho, &k4.rho)?,
            p: combine(&self.p, &k1.p, &k2.p, &k3.p, &k4.p)?,
            q: combine(&self.q, &k1.q, &k2.q, &k3.q, &k4.q)?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn rollout_rhs(
    state: &State,
    weight: f64,
    rho_refs: &(ArrayD<f64>, ArrayD<f64>),
    p_refs: &(ArrayD<f64>, ArrayD<f64>),
    q_refs: &(ArrayD<f64>, ArrayD<f64>),
    y_p_refs: &(ArrayD<f64>, ArrayD<f64>),
    psi6: &ArrayD<f64>,
    c_rho: ArrayView1<'_, f64>,
    c_p: ArrayView1<'_, f64>,
    c_q: ArrayView1<'_, f64>,
    operators: &CylindricalSpectralOperators,
    u0: f64,
    gamma: f64,
    mode: u8,
    ubar_source: u8,
) -> CoreResult<State> {
    let rho_ref = interpolate(&rho_refs.0, &rho_refs.1, weight);
    let p_ref = interpolate(&p_refs.0, &p_refs.1, weight);
    let q_ref = interpolate(&q_refs.0, &q_refs.1, weight);
    let y_p_ref = interpolate(&y_p_refs.0, &y_p_refs.1, weight);
    let rho_eval = if mode == 0 || mode == 1 {
        &state.rho
    } else {
        &rho_ref
    };
    let p_eval = if mode == 0 || mode == 2 {
        &state.p
    } else {
        &p_ref
    };
    let q_eval = if mode == 0 || mode == 3 {
        &state.q
    } else {
        &q_ref
    };
    let a_ref = fitting::alignment_tensor(rho_ref.view(), q_ref.view())?;
    let ubar = if ubar_source == 0 {
        Some(fitting::estimate_ubar(y_p_ref.view(), a_ref.view())?)
    } else {
        None
    };
    let (f_rho, f_p, f_q) = closure_fields(
        rho_eval,
        p_eval,
        q_eval,
        psi6,
        c_rho,
        c_p,
        c_q,
        operators,
        ubar.as_ref(),
    )?;
    let rho_flux = combine_scaled(p_eval, u0, &f_rho, 1.0 / gamma)?;
    let d_rho = if mode == 0 || mode == 1 {
        operators.divergence(rho_flux.view(), 0)?.mapv(|v| -v)
    } else {
        ArrayD::zeros(rho_ref.raw_dim())
    };
    let d_p = if mode == 0 || mode == 2 {
        operators.divergence(f_p.view(), 0)?.mapv(|v| -u0 * v)
            + &(p_eval * P_RELAXATION_COEFFICIENT)
    } else {
        ArrayD::zeros(p_ref.raw_dim())
    };
    let d_q = if mode == 0 || mode == 3 {
        operators.divergence(f_q.view(), 0)?.mapv(|v| -v) + &(q_eval * Q_RELAXATION_COEFFICIENT)
    } else {
        ArrayD::zeros(q_ref.raw_dim())
    };
    Ok(State {
        rho: d_rho,
        p: d_p,
        q: d_q,
    })
}

#[allow(clippy::too_many_arguments)]
fn closure_fields(
    rho: &ArrayD<f64>,
    p: &ArrayD<f64>,
    q: &ArrayD<f64>,
    psi6: &ArrayD<f64>,
    c_rho: ArrayView1<'_, f64>,
    c_p: ArrayView1<'_, f64>,
    c_q: ArrayView1<'_, f64>,
    operators: &CylindricalSpectralOperators,
    ubar_override: Option<&ArrayD<f64>>,
) -> CoreResult<(ArrayD<f64>, ArrayD<f64>, ArrayD<f64>)> {
    let a = fitting::alignment_tensor(rho.view(), q.view())?;
    let grad_rho = operators.gradient(rho.view(), 0)?;
    let a_grad = fitting::alignment_dot_gradient(a.view(), grad_rho.view())?;
    let f_rho =
        fitting::weighted_linear_combination(&[grad_rho.view(), a_grad.view(), p.view()], c_rho)?;
    let grad_p = operators.gradient(p.view(), 0)?;
    let psi_a = fitting::scale_by_scalar(psi6.view(), a.view())?;
    let f_p = fitting::weighted_linear_combination(&[a.view(), psi_a.view(), grad_p.view()], c_p)?;
    let ubar = match ubar_override {
        Some(value) => value.clone(),
        None => fitting::estimate_ubar(f_p.view(), a.view())?,
    };
    let p_alpha = fitting::p_alignment_traceless(p.view())?;
    let ubar_p_alpha = fitting::scale_by_scalar(ubar.view(), p_alpha.view())?;
    let grad_q = operators.gradient(q.view(), 0)?;
    let tangential = fitting::project_flux_directions(ubar_p_alpha.view(), 0)?;
    let radial = fitting::project_flux_directions(ubar_p_alpha.view(), 1)?;
    let radial_grad = fitting::project_flux_directions(grad_q.view(), 1)?;
    let f_q = fitting::weighted_linear_combination(
        &[tangential.view(), radial.view(), radial_grad.view()],
        c_q,
    )?;
    Ok((f_rho, f_p, f_q))
}

fn transfer_pair(
    transfer: &RadialTransfer,
    values: ArrayViewD<'_, f64>,
    frame: usize,
) -> CoreResult<(ArrayD<f64>, ArrayD<f64>)> {
    Ok((
        transfer.apply(values.index_axis(Axis(0), frame).into_dyn(), 2)?,
        transfer.apply(values.index_axis(Axis(0), frame + 1).into_dyn(), 2)?,
    ))
}

fn interpolate(left: &ArrayD<f64>, right: &ArrayD<f64>, weight: f64) -> ArrayD<f64> {
    left * (1.0 - weight) + right * weight
}

fn add_scaled(left: &ArrayD<f64>, right: &ArrayD<f64>, scale: f64) -> CoreResult<ArrayD<f64>> {
    if left.shape() != right.shape() {
        return Err(CoreError::Shape(
            "PDE state/update shapes differ".to_string(),
        ));
    }
    Ok(left + &(right * scale))
}

fn combine_scaled(
    left: &ArrayD<f64>,
    left_scale: f64,
    right: &ArrayD<f64>,
    right_scale: f64,
) -> CoreResult<ArrayD<f64>> {
    if left.shape() != right.shape() {
        return Err(CoreError::Shape("PDE flux shapes differ".to_string()));
    }
    Ok(left * left_scale + right * right_scale)
}
