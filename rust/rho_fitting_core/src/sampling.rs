use ndarray::{Array2, ArrayView3};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{CoreError, CoreResult};

/// Sample valid `(frame, ix, iy)` rows from a boolean mask.
///
/// `valid_mask` has shape `(T, Nx, Ny)`, `nd` is the requested number of
/// rows, and `replace` controls whether repeated rows are allowed.
///
/// Edge cases: without replacement, fewer than `nd` rows can be returned if
/// the mask has too few valid entries; Python validates that policy.
pub fn sample_rows(
    valid_mask: ArrayView3<'_, bool>,
    nd: usize,
    seed: u64,
    replace: bool,
) -> CoreResult<Array2<i64>> {
    if nd == 0 {
        return Err(CoreError::InvalidInput("nd must be positive".to_string()));
    }
    let mut valid = valid_indices(valid_mask);
    if valid.is_empty() {
        return Err(CoreError::InvalidInput(
            "no valid rows to sample".to_string(),
        ));
    }

    let mut rng = seeded_rng(seed);
    let sampled = if replace {
        (0..nd)
            .map(|_| {
                let index = rng.gen_range(0..valid.len());
                valid[index]
            })
            .collect()
    } else {
        valid.shuffle(&mut rng);
        valid.truncate(nd.min(valid.len()));
        valid
    };

    let mut out = Array2::<i64>::zeros((sampled.len(), 3));
    for (row, (t, ix, iy)) in sampled.into_iter().enumerate() {
        out[[row, 0]] = t as i64;
        out[[row, 1]] = ix as i64;
        out[[row, 2]] = iy as i64;
    }
    Ok(out)
}

/// Construct the deterministic RNG used for regression row sampling.
pub fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

fn valid_indices(valid_mask: ArrayView3<'_, bool>) -> Vec<(usize, usize, usize)> {
    let (frames, nx, ny) = valid_mask.dim();
    let mut out = Vec::new();
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                if valid_mask[[t, ix, iy]] {
                    out.push((t, ix, iy));
                }
            }
        }
    }
    out
}
