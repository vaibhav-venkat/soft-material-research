# Additional instructions after converting to 3d

Move the actual STLSQ and subsequent looping algorithms for the regression to Rust. 

You will use `burn` for storing all the tensors, but install `faer` in Rust for the actual solver. The core loop of STLSQ is:
[
\Xi^{(0)} = \arg\min_{\Xi} |\Theta \Xi - Y|_2
]
Where `\Xi` is the coefficient matrix. An AI-recommended implementation of this is 

```rust
use burn::tensor::{backend::Backend, Tensor, TensorData};
use faer::prelude::*;
use faer::Mat;

#[derive(Debug, Clone)]
pub struct StlsqConfig {
    pub lambda: f64,
    pub max_iter: usize,
}

pub fn stlsq_sindy<B>(
    y: Tensor<B, 2>,
    f: Tensor<B, 2>,
    config: StlsqConfig,
    device: &B::Device,
) -> Tensor<B, 2>
where
    B: Backend,
{
    let [n_samples, n_features] = y.dims();
    let [f_rows, n_targets] = f.dims();

    assert_eq!(
        n_samples, f_rows,
        "Y and F must have the same number of rows"
    );

    let y_vec: Vec<f64> = y
        .to_data()
        .convert::<f64>()
        .into_vec::<f64>()
        .expect("failed to convert Y tensor to Vec<f64>");

    let f_vec: Vec<f64> = f
        .to_data()
        .convert::<f64>()
        .into_vec::<f64>()
        .expect("failed to convert F tensor to Vec<f64>");

    let y_mat = row_major_to_faer(&y_vec, n_samples, n_features);
    let f_mat = row_major_to_faer(&f_vec, n_samples, n_targets);

    let xi = stlsq_faer(&y_mat, &f_mat, config.lambda, config.max_iter);

    let xi_vec = faer_to_row_major(&xi);

    Tensor::<B, 2>::from_data(
        TensorData::new(xi_vec, [n_features, n_targets]),
        device,
    )
}

fn stlsq_faer(
    y: &Mat<f64>,
    f: &Mat<f64>,
    lambda: f64,
    max_iter: usize,
) -> Mat<f64> {
    let n_samples = y.nrows();
    let n_features = y.ncols();
    let n_targets = f.ncols();

    assert_eq!(f.nrows(), n_samples);

    // Initial dense least-squares solve: Y Xi ≈ F
    let mut xi = y.qr().solve_lstsq(f);

    let mut prev_support = vec![true; n_features * n_targets];

    for _ in 0..max_iter {
        let mut support_changed = false;

        // Threshold small coefficients.
        let mut support = vec![false; n_features * n_targets];

        for j in 0..n_targets {
            for i in 0..n_features {
                let keep = xi[(i, j)].abs() >= lambda;
                support[i * n_targets + j] = keep;

                if !keep {
                    xi[(i, j)] = 0.0;
                }

                if keep != prev_support[i * n_targets + j] {
                    support_changed = true;
                }
            }
        }

        if !support_changed {
            break;
        }

        prev_support = support.clone();

        // Refit each target column on its active feature subset.
        for j in 0..n_targets {
            let active: Vec<usize> = (0..n_features)
                .filter(|&i| support[i * n_targets + j])
                .collect();

            // If everything was thresholded out, leave this target as all zeros.
            if active.is_empty() {
                for i in 0..n_features {
                    xi[(i, j)] = 0.0;
                }
                continue;
            }

            let y_active = Mat::from_fn(n_samples, active.len(), |r, c| {
                y[(r, active[c])]
            });

            let f_col = Mat::from_fn(n_samples, 1, |r, _| f[(r, j)]);

            let xi_active = y_active.qr().solve_lstsq(&f_col);

            // Clear old column, then write active coefficients.
            for i in 0..n_features {
                xi[(i, j)] = 0.0;
            }

            for (local_i, &global_i) in active.iter().enumerate() {
                xi[(global_i, j)] = xi_active[(local_i, 0)];
            }
        }
    }

    xi
}

fn row_major_to_faer(data: &[f64], rows: usize, cols: usize) -> Mat<f64> {
    assert_eq!(data.len(), rows * cols);

    Mat::from_fn(rows, cols, |r, c| {
        data[r * cols + c]
    })
}

fn faer_to_row_major(mat: &Mat<f64>) -> Vec<f64> {
    let rows = mat.nrows();
    let cols = mat.ncols();

    let mut out = Vec::with_capacity(rows * cols);

    for r in 0..rows {
        for c in 0..cols {
            out.push(mat[(r, c)]);
        }
    }

    out
}
```

But do not follow it strictly, make it simpler and check more for validity. It may not apply in our case
