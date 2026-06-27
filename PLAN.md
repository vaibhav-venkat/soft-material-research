# Plan — Add P, chiral·P_perp, STLSQ, and modularize fitting

## Goal

Extend the fitting submodule to include polarization `P` and chiral-coupled perpendicular polarization `chiral * P_perp` as candidate fields in the regression. Switch from plain least squares to STLSQ. Modularize `fitting/` so adding a new field or output is a one-place change.

## New fields

**Polarization `P`:** The active_matter npz already stores `polar_mean` (gaussian-kernel smoothed per-particle polarization) and `polar_cylindrical`. Bin these onto the (x, θ) grid the same way rho is binned. `P_x` uses the x-component; `P_y` uses the θ-cylindrical component (since `y = R * θ`). No extra gaussian kernel needed — the active_matter computation already applies one.

**Chirality χ:** Load from an existing chirality npz (the `xtheta_values` field for `instant_helix_relative`). If missing, recompute via `compute_chirality_fields()` from `hexatic.chirality.compute`. Must match the fitting grid dimensions.

**chiral · P_perp:** Given χ and P = (P_x, P_y), compute `P_perp = (-P_y, P_x)`. Then `chiral_P_perp_x = χ * (-P_y)` and `chiral_P_perp_y = χ * P_x`. All frame fields get averaged to mid-transitions via a shared `_mid()` helper.

## STLSQ regression

Replace `scipy.linalg.lstsq` with Sequential Thresholded Least Squares: solve, zero out coefficients below a threshold ε, re-solve on the active set, repeat until the active set stabilizes. Uses `scipy.linalg.lstsq` internally. Config exposes `threshold` and `max_iter`.

## Modularization via `fitting/types.py`

Create a new file with a `FieldSpec` dataclass (name, role, label, components, at_frames flag) and a `FieldRegistry` that holds all specs. Candidate fields, targets, and auxiliary fields are all registered here. Other modules look up specs by name instead of hardcoding field lists.

`FittingFields` becomes dict-based: `frame_fields: dict[str, ndarray]` and `mid_fields: dict[str, ndarray]` keyed by field name. `FittingResult` stores coefficients as `coef_map: dict[str, ndarray]` and `coef_global: dict[str, float]` instead of separate `c_x`/`c_y` attributes.

## Variable selection in `run_fitting.py`

Top-level tuples `CANDIDATES` define which fields go into the regression. Passed into `FittingConfig`. No terminal flags — edit the code to change the set.

## Files

- `fitting/types.py` — **new**: field registry, specs, defaults
- `fitting/config.py` — add candidate_names, exclude_candidates, stlsq params
- `fitting/fields.py` — rewrite to dict-based fields; add P and chiral·P_perp computation
- `fitting/fit.py` — rewrite to STLSQ; dict-based FittingResult
- `fitting/plots.py` — iterate over candidate names from result instead of hardcoded list
- `fitting/io_cache.py` — flatten/reconstruct dicts for npz serialization
- `fitting/__init__.py` — update exports
- `run_fitting.py` — add CANDIDATES/EXCLUDE block
