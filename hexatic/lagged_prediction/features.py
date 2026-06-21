from pathlib import Path
import warnings

import gsd.hoomd
import numpy as np

from .types import FeatureFamily, LaggedPredictionConfig


def require_file(filename: str | Path) -> Path:
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Required saved analysis output is missing: {path}")
    return path


def _bin_indices(values: np.ndarray, low: float, high: float, n_bins: int) -> np.ndarray:
    span = high - low
    wrapped = np.mod(values - low, span)
    indices = np.floor(wrapped / span * n_bins).astype(np.int64)
    return np.clip(indices, 0, n_bins - 1)


def shell_xtheta_grid(
    steps: np.ndarray,
    coords: np.ndarray,
    shell_mask: np.ndarray,
    values: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> np.ndarray:
    n_frames = len(steps)
    n_x = len(x_edges) - 1
    n_theta = len(theta_edges) - 1
    grids = np.full((n_frames, n_x, n_theta), np.nan, dtype=np.float64)
    n_bins = n_x * n_theta

    for frame_idx in range(n_frames):
        mask = shell_mask[frame_idx] & np.isfinite(values[frame_idx])
        frame_coords = coords[frame_idx, mask]
        frame_values = values[frame_idx, mask]
        if frame_values.size == 0:
            continue

        x_idx = _bin_indices(frame_coords[:, 0], x_edges[0], x_edges[-1], n_x)
        theta_idx = _bin_indices(
            frame_coords[:, 1],
            theta_edges[0],
            theta_edges[-1],
            n_theta,
        )
        groups = x_idx * n_theta + theta_idx
        counts = np.bincount(groups, minlength=n_bins)
        sums = np.bincount(groups, weights=frame_values, minlength=n_bins)
        flat = np.full(n_bins, np.nan, dtype=np.float64)
        occupied = counts > 0
        flat[occupied] = sums[occupied] / counts[occupied]
        grids[frame_idx] = flat.reshape((n_x, n_theta))

    return grids


def field_family(
    name: str,
    steps: np.ndarray,
    grids: np.ndarray,
    max_modes: int,
    variance_threshold: float,
    scalar_prefix: str | None = None,
) -> FeatureFamily:
    flat = grids.reshape((grids.shape[0], -1)).astype(np.float64)
    finite_counts = np.sum(np.isfinite(flat), axis=1)
    scalars = np.nanmean(flat, axis=1)
    scalars[finite_counts == 0] = np.nan
    column_means = np.nanmean(flat, axis=0)
    column_means = np.where(np.isfinite(column_means), column_means, 0.0)
    filled = np.where(np.isfinite(flat), flat, column_means)
    centered = filled - np.mean(filled, axis=0, keepdims=True)

    mode_count = 0
    mode_scores = np.empty((len(steps), 0), dtype=np.float64)
    explained = np.empty(0, dtype=np.float64)
    if centered.shape[0] >= 2 and centered.shape[1] >= 1:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        variance = singular_values**2
        total_variance = float(np.sum(variance))
        if total_variance > 0.0:
            explained_all = variance / total_variance
            cumulative = np.cumsum(explained_all)
            threshold_count = int(np.searchsorted(cumulative, variance_threshold) + 1)
            mode_count = min(max_modes, threshold_count, vt.shape[0])
            components = vt[:mode_count]
            mode_scores = centered @ components.T
            explained = explained_all[:mode_count]

    scalar_name = scalar_prefix or f"{name}_mean"
    values = np.column_stack((scalars, mode_scores))
    feature_names = (scalar_name,) + tuple(
        f"{name}_mode_{mode_idx + 1}" for mode_idx in range(mode_count)
    )
    return FeatureFamily(
        name=name,
        steps=np.asarray(steps, dtype=np.int64),
        values=values,
        feature_names=feature_names,
        mode_variance=explained,
    )


def load_active_families(
    filename: str | Path,
    config: LaggedPredictionConfig,
) -> tuple[dict[str, FeatureFamily], dict[str, np.ndarray]]:
    path = require_file(filename)
    with np.load(path) as data:
        steps = np.asarray(data["steps"], dtype=np.int64)
        coords = np.asarray(data["coords"], dtype=np.float64)
        shell_mask = np.asarray(data["shell_mask"], dtype=bool)
        x_edges = np.asarray(data["x_edges"], dtype=np.float64)
        theta_edges = np.asarray(data["theta_edges"], dtype=np.float64)
        px = np.asarray(data["polar_cylindrical"], dtype=np.float64)[..., 0]
        rho = np.asarray(data["rho"], dtype=np.float64)

    px_grid = shell_xtheta_grid(steps, coords, shell_mask, px, x_edges, theta_edges)
    rho_grid = shell_xtheta_grid(steps, coords, shell_mask, rho, x_edges, theta_edges)
    families = {
        "px": field_family(
            "px",
            steps,
            px_grid,
            config.max_modes,
            config.variance_threshold,
            scalar_prefix="P_x_shell_mean",
        ),
        "rho": field_family(
            "rho",
            steps,
            rho_grid,
            config.max_modes,
            config.variance_threshold,
            scalar_prefix="rho_shell_mean",
        ),
    }
    return families, {
        "steps": steps,
        "coords": coords,
        "shell_mask": shell_mask,
        "x_edges": x_edges,
        "theta_edges": theta_edges,
    }


def load_hexatic_families(
    filename: str | Path,
    active_context: dict[str, np.ndarray],
    config: LaggedPredictionConfig,
) -> dict[str, FeatureFamily]:
    path = require_file(filename)
    active_steps = np.asarray(active_context["steps"], dtype=np.int64)
    coords = np.asarray(active_context["coords"], dtype=np.float64)
    shell_mask = np.asarray(active_context["shell_mask"], dtype=bool)
    x_edges = np.asarray(active_context["x_edges"], dtype=np.float64)
    theta_edges = np.asarray(active_context["theta_edges"], dtype=np.float64)
    step_to_idx = {int(step): idx for idx, step in enumerate(active_steps)}

    psi6 = np.full(shell_mask.shape, np.nan, dtype=np.float64)
    charge = np.full(shell_mask.shape, np.nan, dtype=np.float64)
    dislocation = np.full(shell_mask.shape, np.nan, dtype=np.float64)

    with gsd.hoomd.open(name=str(path), mode="r") as trajectory:
        for frame in trajectory:
            step = int(frame.configuration.step)
            if step not in step_to_idx:
                continue
            idx = step_to_idx[step]
            velocity = np.asarray(frame.particles.velocity, dtype=np.float64)
            psi6[idx] = velocity[:, 0]
            charge[idx] = velocity[:, 2]
            orientation = frame.particles.orientation
            if orientation is not None:
                dislocation[idx] = np.asarray(orientation, dtype=np.float64)[:, 0]

    psi6_grid = shell_xtheta_grid(
        active_steps,
        coords,
        shell_mask,
        psi6,
        x_edges,
        theta_edges,
    )
    charge_grid = shell_xtheta_grid(
        active_steps,
        coords,
        shell_mask,
        charge,
        x_edges,
        theta_edges,
    )
    dislocation_grid = shell_xtheta_grid(
        active_steps,
        coords,
        shell_mask,
        dislocation,
        x_edges,
        theta_edges,
    )

    defect_scalars = np.full((len(active_steps), 6), np.nan, dtype=np.float64)
    for frame_idx in range(len(active_steps)):
        mask = shell_mask[frame_idx] & np.isfinite(charge[frame_idx])
        if not np.any(mask):
            continue
        frame_charge = charge[frame_idx, mask]
        frame_dislocation = dislocation[frame_idx, mask]
        defect_scalars[frame_idx] = (
            np.mean(frame_charge),
            np.mean(np.abs(frame_charge)),
            np.count_nonzero(frame_charge == 1),
            np.count_nonzero(frame_charge == -1),
            np.count_nonzero(frame_charge == 1) - np.count_nonzero(frame_charge == -1),
            np.nanmean(frame_dislocation),
        )

    charge_family = field_family(
        "defect_charge",
        active_steps,
        charge_grid,
        config.max_modes,
        config.variance_threshold,
        scalar_prefix="defect_charge_shell_mean",
    )
    dislocation_family = field_family(
        "dislocation",
        active_steps,
        dislocation_grid,
        config.max_modes,
        config.variance_threshold,
        scalar_prefix="dislocation_shell_mean",
    )
    defect_values = np.column_stack(
        (defect_scalars, charge_family.values, dislocation_family.values)
    )
    defect_names = (
        "defect_charge_mean",
        "defect_abs_charge_mean",
        "plus_disclination_count",
        "minus_disclination_count",
        "defect_count_asymmetry",
        "dislocation_fraction",
    ) + tuple(f"q_{name}" for name in charge_family.feature_names) + tuple(
        f"dislocation_{name}" for name in dislocation_family.feature_names
    )
    return {
        "psi6": field_family(
            "psi6",
            active_steps,
            psi6_grid,
            config.max_modes,
            config.variance_threshold,
            scalar_prefix="psi6_shell_mean",
        ),
        "defects": FeatureFamily(
            name="defects",
            steps=active_steps,
            values=defect_values,
            feature_names=defect_names,
            mode_variance=np.concatenate(
                (charge_family.mode_variance, dislocation_family.mode_variance)
            ),
        ),
    }


def load_ccm_family(
    filename: str | Path,
    config: LaggedPredictionConfig,
) -> FeatureFamily:
    path = require_file(filename)
    with np.load(path) as data:
        steps = np.asarray(data["steps"], dtype=np.int64)
        metric_names = tuple(str(name) for name in data["metric_names"])
        ccm_idx = metric_names.index("ccm")
        global_ccm = np.asarray(data["global_values"], dtype=np.float64)[ccm_idx]
        ccm_grid = np.asarray(data["xtheta_values"], dtype=np.float64)[ccm_idx]

    ccm_modes = field_family(
        "ccm",
        steps,
        ccm_grid,
        config.max_modes,
        config.variance_threshold,
        scalar_prefix="ccm_xtheta_mean",
    )
    return FeatureFamily(
        name="ccm",
        steps=steps,
        values=np.column_stack((global_ccm, ccm_modes.values)),
        feature_names=("ccm_global",) + ccm_modes.feature_names,
        mode_variance=ccm_modes.mode_variance,
    )


def _vector_xtheta_grids_from_points(
    steps: np.ndarray,
    grid_coords: np.ndarray,
    vectors: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_frames = len(steps)
    n_x = len(x_edges) - 1
    n_theta = len(theta_edges) - 1
    x_grids = np.full((n_frames, n_x, n_theta), np.nan, dtype=np.float64)
    theta_grids = np.full_like(x_grids, np.nan)
    mag_grids = np.full_like(x_grids, np.nan)
    n_bins = n_x * n_theta

    for frame_idx in range(n_frames):
        coords = grid_coords[frame_idx]
        values = vectors[frame_idx]
        finite = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 2])
        finite &= np.isfinite(values[:, 0]) & np.isfinite(values[:, 2])
        if not np.any(finite):
            continue

        x_idx = _bin_indices(coords[finite, 0], x_edges[0], x_edges[-1], n_x)
        theta_idx = _bin_indices(
            coords[finite, 2],
            theta_edges[0],
            theta_edges[-1],
            n_theta,
        )
        groups = x_idx * n_theta + theta_idx
        counts = np.bincount(groups, minlength=n_bins)
        occupied = counts > 0

        x_sums = np.bincount(groups, weights=values[finite, 0], minlength=n_bins)
        theta_sums = np.bincount(groups, weights=values[finite, 2], minlength=n_bins)
        magnitudes = np.hypot(values[finite, 0], values[finite, 2])
        mag_sums = np.bincount(groups, weights=magnitudes, minlength=n_bins)

        for target, sums in (
            (x_grids[frame_idx], x_sums),
            (theta_grids[frame_idx], theta_sums),
            (mag_grids[frame_idx], mag_sums),
        ):
            flat = np.full(n_bins, np.nan, dtype=np.float64)
            flat[occupied] = sums[occupied] / counts[occupied]
            target[:] = flat.reshape((n_x, n_theta))

    return x_grids, theta_grids, mag_grids


def _as_time_series(values: np.ndarray, n_points: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 2 and arr.shape == (n_points, 3):
        return arr[np.newaxis, ...]
    if arr.ndim == 3 and arr.shape[1:] == (n_points, 3):
        return arr
    raise ValueError(f"Expected vector field with shape ({n_points}, 3) or (T, {n_points}, 3).")


def load_stress_families(
    filename: str | Path,
    config: LaggedPredictionConfig,
) -> dict[str, FeatureFamily]:
    path = require_file(filename)
    with np.load(path) as data:
        grid_coords = np.asarray(data["grid_coords"], dtype=np.float64)
        x_edges = np.asarray(data["x_edges"], dtype=np.float64)
        theta_edges = np.asarray(data["theta_edges"], dtype=np.float64)
        if "steps" in data:
            steps = np.asarray(data["steps"], dtype=np.int64)
        elif "step" in data:
            steps = np.asarray([int(np.asarray(data["step"]))], dtype=np.int64)
        else:
            raise KeyError(f"{path} has no 'steps' or 'step' array.")

        if grid_coords.ndim == 2:
            grid_coords = grid_coords[np.newaxis, ...]
        n_points = grid_coords.shape[1]
        fields = {
            "normal_stress_divergence": _as_time_series(
                data["div_sigma_normal"],
                n_points,
            ),
            "shear_stress_divergence": _as_time_series(
                data["div_sigma_shear"],
                n_points,
            ),
            "wall_force": _as_time_series(data["wall_force_density"], n_points),
        }

    if steps.size < 2:
        warnings.warn(
            f"{path} contains only one stress/decomposition frame; skipping "
            "stress and wall-force lagged predictors because no time series is available.",
            RuntimeWarning,
        )
        return {}
    if grid_coords.shape[0] == 1 and steps.size > 1:
        grid_coords = np.repeat(grid_coords, steps.size, axis=0)

    families: dict[str, FeatureFamily] = {}
    for name, vectors in fields.items():
        if vectors.shape[0] != steps.size:
            warnings.warn(
                f"Skipping {name}: field has {vectors.shape[0]} frames but steps has "
                f"{steps.size}.",
                RuntimeWarning,
            )
            continue
        x_grid, theta_grid, mag_grid = _vector_xtheta_grids_from_points(
            steps,
            grid_coords,
            vectors,
            x_edges,
            theta_edges,
        )
        component_families = [
            field_family(
                f"{name}_x",
                steps,
                x_grid,
                config.max_modes,
                config.variance_threshold,
                scalar_prefix=f"{name}_x_mean",
            ),
            field_family(
                f"{name}_theta",
                steps,
                theta_grid,
                config.max_modes,
                config.variance_threshold,
                scalar_prefix=f"{name}_theta_mean",
            ),
            field_family(
                f"{name}_magnitude",
                steps,
                mag_grid,
                config.max_modes,
                config.variance_threshold,
                scalar_prefix=f"{name}_magnitude_mean",
            ),
        ]
        families[name] = FeatureFamily(
            name=name,
            steps=steps,
            values=np.column_stack([family.values for family in component_families]),
            feature_names=tuple(
                feature
                for family in component_families
                for feature in family.feature_names
            ),
            mode_variance=np.concatenate(
                [family.mode_variance for family in component_families]
            ),
        )
    return families
