from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import fft


DENSITY_ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_ID = "radius_15D"
FITTING_OUTPUT_DIR = DENSITY_ANALYSIS_DIR / "output" / "fitting"
OUTPUT_DIR = DENSITY_ANALYSIS_DIR / "output" / "m4_analysis"
ROBUST_PERCENTILE = 98.0


@dataclass(frozen=True)
class FittingCache:
    transition_steps: np.ndarray
    cylinder_radius: float
    lx: float
    x_edges: np.ndarray
    x_centers: np.ndarray
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    J: np.ndarray
    fitted: np.ndarray
    mask: np.ndarray
    candidate_names: tuple[str, ...]
    mid_fields: dict[str, np.ndarray]
    coef_global: dict[str, float]


@dataclass(frozen=True)
class FluxOperatorComparison:
    case_id: str
    coefficient_mode: str
    transition_steps: np.ndarray
    x_edges: np.ndarray
    x_centers: np.ndarray
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    analysis_mask: np.ndarray
    J_measured: np.ndarray
    J_M4: np.ndarray
    J_residual: np.ndarray
    curl_J_measured: np.ndarray
    curl_J_M4: np.ndarray
    curl_residual: np.ndarray
    div_J_measured: np.ndarray
    div_J_M4: np.ndarray
    div_residual: np.ndarray
    curl_residual_mean_abs: float
    curl_residual_median_abs: float
    div_residual_mean_abs: float
    div_residual_median_abs: float
    curl_normalized_mean_abs: float
    div_normalized_mean_abs: float
    J_x_residual_mean_abs: float
    J_y_residual_mean_abs: float
    J_x_normalized_mean_abs: float
    J_y_normalized_mean_abs: float

    def as_cache_arrays(self) -> dict[str, object]:
        return {
            "case_id": np.asarray(self.case_id),
            "coefficient_mode": np.asarray(self.coefficient_mode),
            "transition_steps": self.transition_steps,
            "x_edges": self.x_edges,
            "x_centers": self.x_centers,
            "theta_edges": self.theta_edges,
            "theta_centers": self.theta_centers,
            "analysis_mask": self.analysis_mask,
            "J_measured": self.J_measured,
            "J_M4": self.J_M4,
            "J_residual": self.J_residual,
            "curl_J_measured": self.curl_J_measured,
            "curl_J_M4": self.curl_J_M4,
            "curl_residual": self.curl_residual,
            "div_J_measured": self.div_J_measured,
            "div_J_M4": self.div_J_M4,
            "div_residual": self.div_residual,
            "curl_residual_mean_abs": self.curl_residual_mean_abs,
            "curl_residual_median_abs": self.curl_residual_median_abs,
            "div_residual_mean_abs": self.div_residual_mean_abs,
            "div_residual_median_abs": self.div_residual_median_abs,
            "curl_normalized_mean_abs": self.curl_normalized_mean_abs,
            "div_normalized_mean_abs": self.div_normalized_mean_abs,
            "J_x_residual_mean_abs": self.J_x_residual_mean_abs,
            "J_y_residual_mean_abs": self.J_y_residual_mean_abs,
            "J_x_normalized_mean_abs": self.J_x_normalized_mean_abs,
            "J_y_normalized_mean_abs": self.J_y_normalized_mean_abs,
        }

    def summary(self) -> str:
        return (
            f"J_x residual mean abs={self.J_x_residual_mean_abs:.6g}, "
            f"normalized mean abs={self.J_x_normalized_mean_abs:.6g}; "
            f"J_y residual mean abs={self.J_y_residual_mean_abs:.6g}, "
            f"normalized mean abs={self.J_y_normalized_mean_abs:.6g}; "
            f"curl residual mean abs={self.curl_residual_mean_abs:.6g}, "
            f"median abs={self.curl_residual_median_abs:.6g}, "
            f"normalized mean abs={self.curl_normalized_mean_abs:.6g}; "
            f"div residual mean abs={self.div_residual_mean_abs:.6g}, "
            f"median abs={self.div_residual_median_abs:.6g}, "
            f"normalized mean abs={self.div_normalized_mean_abs:.6g}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare FFT curl/divergence of measured J and M4-predicted J."
    )
    parser.add_argument("--case", default=DEFAULT_CASE_ID)
    parser.add_argument(
        "--coefficient-mode",
        choices=("global", "local"),
        default="global",
        help="Use cached global M4 coefficients or cached local coefficient maps.",
    )
    parser.add_argument("--fitting-cache", default=None)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--frame-idx", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fitting_cache = (
        Path(args.fitting_cache)
        if args.fitting_cache is not None
        else FITTING_OUTPUT_DIR / f"{args.case}_fitting.npz"
    )
    output_dir = Path(args.output_dir)
    output_stem = f"{args.case}_M4_{args.coefficient_mode}_fft_flux_operators"
    npz_path = output_dir / f"{output_stem}.npz"
    plot_path = output_dir / f"{output_stem}.png"

    print(f"[M4] Loading fitting cache from {fitting_cache}...")
    fitting = load_fitting_cache(fitting_cache)
    result = compare_fft_flux_operators(
        fitting,
        case_id=args.case,
        coefficient_mode=args.coefficient_mode,
    )
    write_cache(npz_path, overwrite=args.overwrite, **result.as_cache_arrays())
    print(f"[M4] Wrote analysis arrays to {npz_path}.")

    if not args.no_plot:
        write_operator_plot(result, plot_path, frame_idx=args.frame_idx)
        print(f"[M4] Wrote operator comparison plot to {plot_path}.")

    print(f"[M4] {result.summary()}.")
    return 0


def compare_fft_flux_operators(
    fitting: FittingCache,
    *,
    case_id: str,
    coefficient_mode: str,
) -> FluxOperatorComparison:
    J_measured = np.asarray(fitting.J, dtype=float)
    J_M4 = predicted_flux(fitting, coefficient_mode)
    if J_M4.shape != J_measured.shape:
        raise ValueError(f"J_M4 shape {J_M4.shape} does not match {J_measured.shape}.")
    analysis_mask = analysis_mask_for(fitting, J_measured, J_M4, coefficient_mode)
    J_residual = np.where(analysis_mask[..., None], J_measured - J_M4, np.nan)
    J_measured_fft = np.where(analysis_mask[..., None], J_measured, 0.0)
    J_M4_fft = np.where(analysis_mask[..., None], J_M4, 0.0)

    div_measured = div_fft(
        J_measured_fft,
        lx=fitting.lx,
        cylinder_radius=fitting.cylinder_radius,
        theta_period=float(fitting.theta_edges[-1] - fitting.theta_edges[0]),
    )
    div_M4 = div_fft(
        J_M4_fft,
        lx=fitting.lx,
        cylinder_radius=fitting.cylinder_radius,
        theta_period=float(fitting.theta_edges[-1] - fitting.theta_edges[0]),
    )
    curl_measured = curl_fft(
        J_measured_fft,
        lx=fitting.lx,
        cylinder_radius=fitting.cylinder_radius,
        theta_period=float(fitting.theta_edges[-1] - fitting.theta_edges[0]),
    )
    curl_M4 = curl_fft(
        J_M4_fft,
        lx=fitting.lx,
        cylinder_radius=fitting.cylinder_radius,
        theta_period=float(fitting.theta_edges[-1] - fitting.theta_edges[0]),
    )
    div_residual = div_measured - div_M4
    curl_residual = curl_measured - curl_M4
    J_measured_masked = np.where(analysis_mask[..., None], J_measured, np.nan)
    J_M4_masked = np.where(analysis_mask[..., None], J_M4, np.nan)
    div_measured_masked = np.where(analysis_mask, div_measured, np.nan)
    div_M4_masked = np.where(analysis_mask, div_M4, np.nan)
    div_residual_masked = np.where(analysis_mask, div_residual, np.nan)
    curl_measured_masked = np.where(analysis_mask, curl_measured, np.nan)
    curl_M4_masked = np.where(analysis_mask, curl_M4, np.nan)
    curl_residual_masked = np.where(analysis_mask, curl_residual, np.nan)

    curl_scale = mean_abs_finite(curl_measured_masked)
    div_scale = mean_abs_finite(div_measured_masked)
    if not np.isfinite(curl_scale) or curl_scale == 0.0:
        curl_scale = 1.0
    if not np.isfinite(div_scale) or div_scale == 0.0:
        div_scale = 1.0
    J_x_scale = mean_abs_finite(J_measured_masked[..., 0])
    J_y_scale = mean_abs_finite(J_measured_masked[..., 1])
    if not np.isfinite(J_x_scale) or J_x_scale == 0.0:
        J_x_scale = 1.0
    if not np.isfinite(J_y_scale) or J_y_scale == 0.0:
        J_y_scale = 1.0

    return FluxOperatorComparison(
        case_id=case_id,
        coefficient_mode=coefficient_mode,
        transition_steps=np.asarray(fitting.transition_steps),
        x_edges=np.asarray(fitting.x_edges),
        x_centers=np.asarray(fitting.x_centers),
        theta_edges=np.asarray(fitting.theta_edges),
        theta_centers=np.asarray(fitting.theta_centers),
        analysis_mask=analysis_mask,
        J_measured=J_measured_masked,
        J_M4=J_M4_masked,
        J_residual=J_residual,
        curl_J_measured=curl_measured_masked,
        curl_J_M4=curl_M4_masked,
        curl_residual=curl_residual_masked,
        div_J_measured=div_measured_masked,
        div_J_M4=div_M4_masked,
        div_residual=div_residual_masked,
        curl_residual_mean_abs=mean_abs_finite(curl_residual_masked),
        curl_residual_median_abs=median_abs_finite(curl_residual_masked),
        div_residual_mean_abs=mean_abs_finite(div_residual_masked),
        div_residual_median_abs=median_abs_finite(div_residual_masked),
        curl_normalized_mean_abs=mean_abs_finite(curl_residual_masked / curl_scale),
        div_normalized_mean_abs=mean_abs_finite(div_residual_masked / div_scale),
        J_x_residual_mean_abs=mean_abs_finite(J_residual[..., 0]),
        J_y_residual_mean_abs=mean_abs_finite(J_residual[..., 1]),
        J_x_normalized_mean_abs=mean_abs_finite(J_residual[..., 0] / J_x_scale),
        J_y_normalized_mean_abs=mean_abs_finite(J_residual[..., 1] / J_y_scale),
    )


def analysis_mask_for(
    fitting: FittingCache,
    J_measured: np.ndarray,
    J_M4: np.ndarray,
    coefficient_mode: str,
) -> np.ndarray:
    finite = np.all(np.isfinite(J_measured), axis=-1) & np.all(
        np.isfinite(J_M4),
        axis=-1,
    )
    if coefficient_mode == "local":
        mask = np.asarray(fitting.mask, dtype=bool) & finite
        if not np.any(mask):
            raise ValueError("Local M4 has no finite bins inside the fitting mask.")
        return mask
    return finite


def predicted_flux(fitting: FittingCache, coefficient_mode: str) -> np.ndarray:
    if coefficient_mode == "local":
        return np.asarray(fitting.fitted, dtype=float)
    if coefficient_mode != "global":
        raise ValueError(f"Unsupported coefficient mode: {coefficient_mode!r}.")
    if any(name.startswith("bulk__") for name in fitting.coef_global):
        return np.asarray(fitting.fitted, dtype=float)

    first = fitting.mid_fields[fitting.candidate_names[0]]
    predicted = np.zeros_like(first, dtype=float)
    for name in fitting.candidate_names:
        for component_idx, component in enumerate(("x", "y")):
            coefficient = fitting.coef_global[f"{name}_{component}"]
            predicted[..., component_idx] += (
                coefficient * fitting.mid_fields[name][..., component_idx]
            )
    return predicted


def load_fitting_cache(path: str | Path) -> FittingCache:
    arrays = load_npz_arrays(path)
    mid_fields = reconstruct_array_dict(arrays, "mid_fields")
    coef_global = reconstruct_float_dict(arrays, "coef_global")
    required = (
        "transition_steps",
        "cylinder_radius",
        "lx",
        "x_edges",
        "x_centers",
        "theta_edges",
        "theta_centers",
        "J",
        "fitted",
        "mask",
        "candidate_names",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise KeyError("Fitting cache is missing: " + ", ".join(missing))
    candidate_names = tuple(str(name) for name in np.asarray(arrays["candidate_names"]))
    for name in candidate_names:
        if name not in mid_fields:
            raise KeyError(f"Fitting cache is missing mid_fields for {name!r}.")
    return FittingCache(
        transition_steps=np.asarray(arrays["transition_steps"]),
        cylinder_radius=float(np.asarray(arrays["cylinder_radius"])),
        lx=float(np.asarray(arrays["lx"])),
        x_edges=np.asarray(arrays["x_edges"], dtype=float),
        x_centers=np.asarray(arrays["x_centers"], dtype=float),
        theta_edges=np.asarray(arrays["theta_edges"], dtype=float),
        theta_centers=np.asarray(arrays["theta_centers"], dtype=float),
        J=np.asarray(arrays["J"], dtype=float),
        fitted=np.asarray(arrays["fitted"], dtype=float),
        mask=np.asarray(arrays["mask"], dtype=bool),
        candidate_names=candidate_names,
        mid_fields=mid_fields,
        coef_global=coef_global,
    )


def load_npz_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def write_cache(
    path: str | Path,
    *,
    overwrite: bool = False,
    **arrays: object,
) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"{destination} already exists; pass --overwrite to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return destination


def reconstruct_array_dict(
    arrays: dict[str, np.ndarray],
    prefix: str,
) -> dict[str, np.ndarray]:
    names_key = f"{prefix}__names"
    if names_key not in arrays:
        return {}
    names = tuple(str(name) for name in np.asarray(arrays[names_key]))
    return {
        name: np.asarray(arrays[f"{prefix}__{name}"], dtype=float)
        for name in names
        if f"{prefix}__{name}" in arrays
    }


def reconstruct_float_dict(
    arrays: dict[str, np.ndarray],
    prefix: str,
) -> dict[str, float]:
    names_key = f"{prefix}__names"
    values_key = f"{prefix}__values"
    if names_key not in arrays or values_key not in arrays:
        return {}
    names = tuple(str(name) for name in np.asarray(arrays[names_key]))
    values = np.asarray(arrays[values_key], dtype=float)
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def div_fft(
    J: np.ndarray,
    *,
    lx: float,
    cylinder_radius: float,
    theta_period: float,
) -> np.ndarray:
    dJx_dx, _ = fft_gradients(
        J[..., 0],
        lx=lx,
        cylinder_radius=cylinder_radius,
        theta_period=theta_period,
    )
    _, dJy_dy = fft_gradients(
        J[..., 1],
        lx=lx,
        cylinder_radius=cylinder_radius,
        theta_period=theta_period,
    )
    return dJx_dx + dJy_dy


def curl_fft(
    J: np.ndarray,
    *,
    lx: float,
    cylinder_radius: float,
    theta_period: float,
) -> np.ndarray:
    dJy_dx, _ = fft_gradients(
        J[..., 1],
        lx=lx,
        cylinder_radius=cylinder_radius,
        theta_period=theta_period,
    )
    _, dJx_dy = fft_gradients(
        J[..., 0],
        lx=lx,
        cylinder_radius=cylinder_radius,
        theta_period=theta_period,
    )
    return dJy_dx - dJx_dy


def fft_gradients(
    values: np.ndarray,
    *,
    lx: float,
    cylinder_radius: float,
    theta_period: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 3:
        raise ValueError("values must have shape (transitions, nx, ntheta).")
    if lx <= 0.0:
        raise ValueError("lx must be positive.")
    if cylinder_radius <= 0.0:
        raise ValueError("cylinder_radius must be positive.")
    if theta_period <= 0.0:
        raise ValueError("theta_period must be positive.")

    _, nx, ntheta = values.shape
    ly = float(cylinder_radius) * float(theta_period)
    kx = 2.0 * np.pi * fft.fftfreq(nx, d=float(lx) / nx)
    ky = 2.0 * np.pi * fft.fftfreq(ntheta, d=ly / ntheta)
    values_hat = fft.fft2(values, axes=(1, 2))
    grad_x = fft.ifft2(1j * kx[None, :, None] * values_hat, axes=(1, 2)).real
    grad_y = fft.ifft2(1j * ky[None, None, :] * values_hat, axes=(1, 2)).real
    return grad_x, grad_y


def write_operator_plot(
    result: FluxOperatorComparison,
    output_path: str | Path,
    *,
    frame_idx: int | None,
) -> Path:
    import matplotlib.pyplot as plt

    panels = (
        ("J_x measured", result.J_measured[..., 0]),
        ("J_x M4", result.J_M4[..., 0]),
        ("J_x residual", result.J_residual[..., 0]),
        ("J_y measured", result.J_measured[..., 1]),
        ("J_y M4", result.J_M4[..., 1]),
        ("J_y residual", result.J_residual[..., 1]),
        ("curl J_measured", result.curl_J_measured),
        ("curl J_M4", result.curl_J_M4),
        ("curl residual", result.curl_residual),
        ("div J_measured", result.div_J_measured),
        ("div J_M4", result.div_J_M4),
        ("div residual", result.div_residual),
    )
    fig, axes = plt.subplots(4, 3, figsize=(15, 16), constrained_layout=True)
    for axis, (title, values) in zip(axes.ravel(), panels, strict=True):
        selected = select_transition_values(values, frame_idx)
        vmin, vmax = robust_color_limits(selected, symmetric=True)
        mesh = axis.pcolormesh(
            result.x_centers,
            result.theta_centers,
            selected.T,
            shading="auto",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(mesh, ax=axis, label=title)
        axis.set_xlabel("x")
        axis.set_ylabel("theta")
        axis.set_title(title)
    fig.suptitle(
        f"M4 {result.coefficient_mode} FFT flux operators "
        f"{frame_title_suffix(result, frame_idx)}"
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200)
    plt.close(fig)
    return destination


def select_transition_values(values: np.ndarray, frame_idx: int | None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if frame_idx is None:
        return nanmean_no_warning(values, axis=0)
    if frame_idx < 0 or frame_idx >= values.shape[0]:
        raise IndexError(
            f"frame_idx={frame_idx} is outside transition range 0..{values.shape[0] - 1}."
        )
    return values[frame_idx]


def robust_color_limits(
    values: np.ndarray,
    *,
    symmetric: bool,
    percentile: float = ROBUST_PERCENTILE,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    if symmetric:
        limit = float(np.nanpercentile(np.abs(finite), percentile))
        if not np.isfinite(limit) or limit == 0.0:
            limit = float(np.nanmax(np.abs(finite)))
        if not np.isfinite(limit) or limit == 0.0:
            limit = 1.0
        return -limit, limit

    low = float(np.nanpercentile(finite, 100.0 - percentile))
    high = float(np.nanpercentile(finite, percentile))
    if not np.isfinite(low) or not np.isfinite(high) or np.isclose(low, high):
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
    if not np.isfinite(low) or not np.isfinite(high) or np.isclose(low, high):
        return 0.0, 1.0
    return low, high


def nanmean_no_warning(values: np.ndarray, *, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    summed = np.sum(np.where(finite, values, 0.0), axis=axis)
    counts = np.sum(finite, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = summed / counts
    return np.where(counts > 0, mean, np.nan)


def mean_abs_finite(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = np.abs(finite[np.isfinite(finite)])
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def median_abs_finite(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = np.abs(finite[np.isfinite(finite)])
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def frame_title_suffix(
    result: FluxOperatorComparison,
    frame_idx: int | None,
) -> str:
    if frame_idx is None:
        return "(transition mean)"
    steps = np.asarray(result.transition_steps)
    if steps.ndim == 2 and frame_idx < steps.shape[0]:
        return f"(steps {steps[frame_idx, 0]} -> {steps[frame_idx, 1]})"
    return f"(transition {frame_idx})"


if __name__ == "__main__":
    raise SystemExit(main())
