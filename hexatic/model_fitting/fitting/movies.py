"""x-theta movies for Model 3 stochastic density/source comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.animation import FFMpegWriter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .fit import FittingResult
from .library import build_current_library
from .plots import _robust_color_limits
from .stochastic import (
    SOURCE_STOCHASTIC_POWER_FRACTION,
    SOURCE_STOCHASTIC_ROLLOUT_SEED,
    STOCHASTIC_ETA_POWER_FRACTION,
    STOCHASTIC_ROLLOUT_SEED,
    _adaptive_eta_power_mode_mask,
    _ar1_rollout,
    _ar1_rollout_scalar,
    _masked_time_mean,
)
from .write_report import _current_from_fit, _divergence, _fit_without_force_density


MODEL_FITTING_MOVIE_FPS = 4
MODEL_FITTING_MOVIE_SECONDS_PER_FRAME = 5.0


@dataclass(frozen=True)
class Model3MovieFields:
    """Fields used by the Model 3 x-theta movie comparisons."""

    model3_no_source: np.ndarray
    source_pred_stochastic: np.ndarray
    full_source_pred_stochastic: np.ndarray
    density_minus_source: np.ndarray
    inferred_source: np.ndarray


def write_model3_xtheta_movies(
    result: FittingResult,
    output_dir: str | Path,
    *,
    case_id: str = "radius_15D",
    drop_no_force_low_k_terms: tuple[str, ...] = (),
    included_bonus_terms: tuple[str, ...] = (),
    rho_N_power: int = 0,
    fps: int = MODEL_FITTING_MOVIE_FPS,
    seconds_per_frame: float = MODEL_FITTING_MOVIE_SECONDS_PER_FRAME,
) -> list[Path]:
    """Write the requested Model 3 x-theta comparison movies."""
    if fps < 1:
        raise ValueError("fps must be at least 1.")
    if seconds_per_frame <= 0.0:
        raise ValueError("seconds_per_frame must be positive.")
    dest = Path(output_dir) / "movies"
    dest.mkdir(parents=True, exist_ok=True)
    fields = _model3_movie_fields(
        result,
        drop_no_force_low_k_terms=drop_no_force_low_k_terms,
        included_bonus_terms=included_bonus_terms,
        rho_N_power=rho_N_power,
    )

    specs = (
        (
            result.fields.partial_t_rho,
            fields.full_source_pred_stochastic,
            "partial_t rho",
            (
                "partial_t rho_pred = -div(J_fit_no_force + J_sys_no_force) "
                "+ S_cross_pred + eta_AR1 + zeta_AR1"
            ),
            dest / f"{case_id}_partial_t_rho_vs_model3_source_ar1.mp4",
            "left",
        ),
        (
            fields.model3_no_source,
            fields.density_minus_source,
            "Model 3 without S_cross",
            "partial_t rho - S_cross",
            dest / f"{case_id}_model3_vs_partial_t_rho_minus_s_cross.mp4",
            "right",
        ),
        (
            fields.source_pred_stochastic,
            fields.inferred_source,
            "S_cross_pred + zeta_AR1",
            "partial_t rho + div(J_m)",
            dest / f"{case_id}_s_cross_pred_ar1_vs_inferred_source.mp4",
            "right",
        ),
    )

    saved: list[Path] = []
    for left, right, left_label, right_label, path, r2_target in specs:
        _write_pair_movie(
            result,
            left,
            right,
            left_label,
            right_label,
            path,
            case_id=case_id,
            r2_target=r2_target,
            fps=fps,
            seconds_per_frame=seconds_per_frame,
        )
        saved.append(path)
    print(f"[fitting] Model 3 x-theta movies saved to {dest}")
    return saved


def _model3_movie_fields(
    result: FittingResult,
    *,
    drop_no_force_low_k_terms: tuple[str, ...],
    included_bonus_terms: tuple[str, ...] = (),
    rho_N_power: int = 0,
) -> Model3MovieFields:
    current_lib = build_current_library(
        result.fields,
        included_bonus_terms=included_bonus_terms,
        rho_N_power=rho_N_power,
    )
    no_force_fit = _fit_without_force_density(
        result,
        current_lib,
        drop_low_k_terms=drop_no_force_low_k_terms,
    )
    if no_force_fit is None:
        raise ValueError("Model 3 no-force fit is unavailable for this result.")

    no_force_result, no_force_lib = no_force_fit
    j_fit_no_force = _current_from_fit(
        no_force_lib,
        no_force_result.coefficients,
    )
    j_res = result.fields.material_current - j_fit_no_force
    j_sys = _masked_time_mean(j_res, result.mask)
    xi = j_res - j_sys

    eta_empirical = -_divergence(result, xi)
    eta_mode_mask = _adaptive_eta_power_mode_mask(
        eta_empirical,
        result.mask,
        keep_fraction=STOCHASTIC_ETA_POWER_FRACTION,
    )
    xi_ar1, *_ = _ar1_rollout(
        xi,
        mode_mask=eta_mode_mask,
        seed=STOCHASTIC_ROLLOUT_SEED,
    )
    if xi_ar1 is None:
        raise ValueError("Model 3 eta_AR1 rollout could not be generated.")
    eta_ar1 = -_divergence(result, xi_ar1)

    source_residual = result.fields.S_cross - result.source.prediction
    source_mode_mask = _adaptive_eta_power_mode_mask(
        source_residual,
        result.mask,
        keep_fraction=SOURCE_STOCHASTIC_POWER_FRACTION,
    )
    zeta_ar1, *_ = _ar1_rollout_scalar(
        source_residual,
        mask=result.mask,
        mode_mask=source_mode_mask,
        seed=SOURCE_STOCHASTIC_ROLLOUT_SEED,
    )
    if zeta_ar1 is None:
        raise ValueError("Model 3 zeta_AR1 rollout could not be generated.")

    model3_no_source = -_divergence(result, j_fit_no_force + j_sys) + eta_ar1
    source_pred_stochastic = result.source.prediction + zeta_ar1
    full_source_pred_stochastic = model3_no_source + source_pred_stochastic
    density_minus_source = result.fields.partial_t_rho - result.fields.S_cross
    inferred_source = (
        result.fields.partial_t_rho
        + _divergence(result, result.fields.material_current)
    )
    return Model3MovieFields(
        model3_no_source=model3_no_source,
        source_pred_stochastic=source_pred_stochastic,
        full_source_pred_stochastic=full_source_pred_stochastic,
        density_minus_source=density_minus_source,
        inferred_source=inferred_source,
    )


def _write_pair_movie(
    result: FittingResult,
    left: np.ndarray,
    right: np.ndarray,
    left_label: str,
    right_label: str,
    path: Path,
    *,
    case_id: str,
    r2_target: str,
    fps: int,
    seconds_per_frame: float,
) -> None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.shape != result.mask.shape:
        raise ValueError(
            "movie fields must both match result.mask shape "
            f"{result.mask.shape}; got {left.shape} and {right.shape}."
        )

    valid = np.asarray(result.mask, dtype=bool)
    combined = np.concatenate((left[valid], right[valid]))
    vmin, vmax = _robust_color_limits(combined, symmetric=True)
    if r2_target == "left":
        global_r2 = _r2(left[valid], right[valid])
    elif r2_target == "right":
        global_r2 = _r2(right[valid], left[valid])
    else:
        raise ValueError("r2_target must be 'left' or 'right'.")
    frame_repeats = max(1, int(round(float(fps) * float(seconds_per_frame))))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.85")
    x_grid, theta_grid = np.meshgrid(
        result.fields.x_edges,
        result.fields.theta_edges,
        indexing="ij",
    )

    fig = plt.figure(figsize=(14, 5))
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, str(path), dpi=160):
        for frame_idx in range(left.shape[0]):
            fig.clear()
            axes = fig.subplots(1, 2, sharex=True, sharey=True)
            for axis, values, label in (
                (axes[0], left, left_label),
                (axes[1], right, right_label),
            ):
                frame_values = np.where(valid[frame_idx], values[frame_idx], np.nan)
                mesh = axis.pcolormesh(
                    x_grid,
                    theta_grid,
                    np.ma.masked_invalid(frame_values),
                    shading="auto",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                fig.colorbar(mesh, ax=axis, label=label)
                axis.set_xlabel("x")
                axis.set_ylabel("theta")
                axis.set_title(label)
            step0, step1 = result.transition_steps[frame_idx]
            fig.suptitle(
                f"case {case_id} coarse frame {frame_idx + 1}/{left.shape[0]}, "
                f"transition {int(step0)}-{int(step1)}, R2={global_r2:.4g}"
            )
            fig.tight_layout()
            for _ in range(frame_repeats):
                writer.grab_frame()
    plt.close(fig)


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    finite = np.isfinite(target) & np.isfinite(prediction)
    if not np.any(finite):
        return float("nan")
    y = target[finite]
    p = prediction[finite]
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y - p) ** 2)) / ss_tot
