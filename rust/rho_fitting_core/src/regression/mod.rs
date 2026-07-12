use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, NonnegativeConeT, SolverStatus,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::{CoreError, CoreResult};

pub(crate) struct ConstrainedLassoResult {
    pub(crate) coefficients: Array1<f64>,
    pub(crate) normalized_coefficients: Array1<f64>,
    pub(crate) column_norms: Array1<f64>,
    pub(crate) objective: f64,
    pub(crate) status: String,
    pub(crate) iterations: usize,
    pub(crate) primal_residual: f64,
    pub(crate) dual_residual: f64,
    pub(crate) gap: f64,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_constrained_lasso(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    lambda: f64,
    tolerance: f64,
    max_iterations: usize,
    non_positive: ArrayView1<'_, i64>,
    non_negative: ArrayView1<'_, i64>,
) -> CoreResult<ConstrainedLassoResult> {
    let (rows, features) = x.dim();
    if rows == 0 || features == 0 || y.len() != rows {
        return Err(CoreError::Shape(
            "constrained L1 regression requires X=(rows,features) and y=(rows,)".to_string(),
        ));
    }
    if !x.iter().chain(y.iter()).all(|value| value.is_finite()) {
        return Err(CoreError::InvalidInput(
            "regression inputs must be finite".to_string(),
        ));
    }
    if !(lambda.is_finite() && lambda >= 0.0 && tolerance.is_finite() && tolerance > 0.0)
        || max_iterations == 0
    {
        return Err(CoreError::InvalidInput(
            "lambda, tolerance, and maximum iterations are invalid".to_string(),
        ));
    }
    let non_positive = validate_indices(non_positive, features, "non-positive")?;
    let non_negative = validate_indices(non_negative, features, "non-negative")?;
    if non_positive
        .iter()
        .any(|index| non_negative.contains(index))
    {
        return Err(CoreError::InvalidInput(
            "a coefficient cannot have both sign constraints".to_string(),
        ));
    }

    let column_norms = Array1::from_iter((0..features).map(|feature| {
        x.column(feature)
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
    }));
    let safe_norms = column_norms.mapv(|norm| if norm > 0.0 { norm } else { 1.0 });
    let normalized = Array2::from_shape_fn((rows, features), |(row, feature)| {
        x[[row, feature]] / safe_norms[feature]
    });
    let mut gram = vec![0.0; features * features];
    let mut xty = vec![0.0; features];
    for left in 0..features {
        xty[left] = normalized.column(left).dot(&y);
        for right in 0..features {
            gram[left * features + right] = normalized.column(left).dot(&normalized.column(right));
        }
    }

    let variables = 2 * features;
    let mut hessian = vec![0.0; variables * variables];
    for row in 0..features {
        for col in 0..features {
            hessian[row * variables + col] = 2.0 * gram[row * features + col];
        }
    }
    let p = dense_csc(&hessian, variables, variables).to_triu();
    let mut q = vec![0.0; variables];
    for feature in 0..features {
        q[feature] = -2.0 * xty[feature];
        q[features + feature] = lambda;
    }

    let constraint_rows = 3 * features + non_positive.len() + non_negative.len();
    let mut constraints = vec![0.0; constraint_rows * variables];
    let mut row = 0;
    for feature in 0..features {
        constraints[row * variables + feature] = 1.0;
        constraints[row * variables + features + feature] = -1.0;
        row += 1;
        constraints[row * variables + feature] = -1.0;
        constraints[row * variables + features + feature] = -1.0;
        row += 1;
        constraints[row * variables + features + feature] = -1.0;
        row += 1;
    }
    for &feature in &non_positive {
        constraints[row * variables + feature] = 1.0;
        row += 1;
    }
    for &feature in &non_negative {
        constraints[row * variables + feature] = -1.0;
        row += 1;
    }
    let a = dense_csc(&constraints, constraint_rows, variables);
    let b = vec![0.0; constraint_rows];
    let cones = [NonnegativeConeT(constraint_rows)];
    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .max_iter(max_iterations as u32)
        .tol_gap_abs(tolerance)
        .tol_gap_rel(tolerance)
        .tol_feas(tolerance)
        .build()
        .map_err(|error| CoreError::InvalidInput(format!("Clarabel settings failed: {error}")))?;
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings)
        .map_err(|error| CoreError::InvalidInput(format!("Clarabel setup failed: {error}")))?;
    solver.solve();
    let info = &solver.info;
    let acceptable = matches!(
        info.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    ) || (matches!(info.status, SolverStatus::MaxIterations)
        && info.res_primal <= 10.0 * tolerance
        && info.res_dual <= 10.0 * tolerance
        && info.gap_abs <= 10.0 * tolerance);
    if !acceptable {
        return Err(CoreError::InvalidInput(format!(
            "Clarabel regression failed with status {:?}, primal={:.3e}, dual={:.3e}, gap={:.3e}",
            info.status, info.res_primal, info.res_dual, info.gap_abs
        )));
    }
    let normalized_coefficients = Array1::from_vec(solver.solution.x[..features].to_vec());
    let coefficients = &normalized_coefficients / &safe_norms;
    let residual_sum = (0..rows)
        .map(|sample| {
            let predicted = (0..features)
                .map(|feature| x[[sample, feature]] * coefficients[feature])
                .sum::<f64>();
            let residual = y[sample] - predicted;
            residual * residual
        })
        .sum::<f64>();
    let objective = residual_sum
        + lambda
            * normalized_coefficients
                .iter()
                .map(|value| value.abs())
                .sum::<f64>();
    Ok(ConstrainedLassoResult {
        coefficients,
        normalized_coefficients,
        column_norms,
        objective,
        status: format!("{:?}", info.status),
        iterations: info.iterations as usize,
        primal_residual: info.res_primal,
        dual_residual: info.res_dual,
        gap: info.gap_abs,
    })
}

fn validate_indices(
    values: ArrayView1<'_, i64>,
    features: usize,
    label: &str,
) -> CoreResult<Vec<usize>> {
    let mut output = Vec::with_capacity(values.len());
    for value in values {
        if *value < 0 || *value as usize >= features {
            return Err(CoreError::InvalidInput(format!(
                "{label} coefficient index {value} is out of range"
            )));
        }
        let index = *value as usize;
        if !output.contains(&index) {
            output.push(index);
        }
    }
    Ok(output)
}

fn dense_csc(values: &[f64], rows: usize, columns: usize) -> CscMatrix<f64> {
    let mut colptr = Vec::with_capacity(columns + 1);
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    colptr.push(0);
    for column in 0..columns {
        for row in 0..rows {
            let value = values[row * columns + column];
            if value != 0.0 {
                rowval.push(row);
                nzval.push(value);
            }
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(rows, columns, colptr, rowval, nzval)
}
