"""Intermediate scattering functions for bulk and cylindrical trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from numpy.typing import NDArray
import seaborn as sns

from .plotting import _save_figure


ISF_Q_VALUES = np.asarray([0.1, 0.3, 1.0, 3.0, 10.0], dtype=np.float64)
PARTICLE_SAMPLE_SEED = 20240717


@dataclass(frozen=True)
class IsfSeedResult:
    """One seed's lag-averaged COM and sampled-particle estimates."""

    com: NDArray[np.complex128]
    single: NDArray[np.complex128]
    com_msd: NDArray[np.float64]
    origin_counts: NDArray[np.int64]


@dataclass(frozen=True)
class IsfResult:
    """Seed-averaged curves for one spatial estimator."""

    key: str
    title: str
    dimensions: int
    com: NDArray[np.complex128]
    single: NDArray[np.complex128]
    gaussian: NDArray[np.float64]
    com_msd: NDArray[np.float64]
    origin_counts: NDArray[np.int64]


@dataclass(frozen=True)
class IsfDiagnostics:
    """Strict-validation diagnostics reported by the CLI."""

    maximum_imaginary_leakage: float
    maximum_small_k_msd_error: float | None


def select_particle_ids(
    n_particles: int,
    requested: int,
    *,
    seed: int = PARTICLE_SAMPLE_SEED,
) -> NDArray[np.int64]:
    """Select a deterministic, sorted particle subset without replacement."""
    if n_particles < 1:
        raise ValueError("particle count must be positive")
    if requested < 1:
        raise ValueError("requested particle sample must be positive")
    count = min(n_particles, requested)
    generator = np.random.default_rng(seed)
    return np.sort(
        generator.choice(n_particles, size=count, replace=False)
    ).astype(np.int64)


def lag_origin_counts(n_frames: int, max_lag: int) -> NDArray[np.int64]:
    """Return the number of usable time origins at every inclusive lag."""
    if n_frames < 1:
        raise ValueError("at least one frame is required")
    if max_lag < 0 or max_lag >= n_frames:
        raise ValueError(f"max_lag {max_lag} out of range for {n_frames} frames")
    return np.arange(n_frames, n_frames - max_lag - 1, -1, dtype=np.int64)


def _validate_inputs(
    com: NDArray[np.float64],
    particles: NDArray[np.float64],
    k_values: NDArray[np.float64],
    particle_count: int,
    max_lag: int,
) -> None:
    if com.ndim != 2 or len(com) < 1:
        raise ValueError(f"expected COM shape (frames, dimensions), got {com.shape}")
    if particles.ndim != 3 or particles.shape[0] != len(com):
        raise ValueError(
            "expected sampled tracks shape (frames, particles, dimensions) "
            f"matching the COM, got {particles.shape}"
        )
    if particles.shape[1] < 1 or particles.shape[2] != com.shape[1]:
        raise ValueError("sampled tracks and COM must have matching dimensions")
    if particle_count < 1:
        raise ValueError("particle count must be positive")
    if k_values.ndim != 1 or len(k_values) < 1:
        raise ValueError("k values must be a nonempty vector")
    if not np.all(np.isfinite(k_values)) or np.any(k_values <= 0.0):
        raise ValueError("k values must be finite and positive")
    if not np.all(np.isfinite(com)) or not np.all(np.isfinite(particles)):
        raise ValueError("ISF trajectories contain non-finite coordinates")
    lag_origin_counts(len(com), max_lag)


def isotropic_3d_seed(
    com: NDArray[np.float64],
    particles: NDArray[np.float64],
    k_values: NDArray[np.float64],
    particle_count: int,
    max_lag: int,
) -> IsfSeedResult:
    """Direction-averaged 3D ISFs using ``sin(kr)/(kr)``."""
    _validate_inputs(com, particles, k_values, particle_count, max_lag)
    if com.shape[1] != 3:
        raise ValueError("the isotropic bulk estimator requires three dimensions")
    counts = lag_origin_counts(len(com), max_lag)
    com_isf = np.empty((max_lag + 1, len(k_values)), dtype=np.complex128)
    single_isf = np.empty_like(com_isf)
    msd = np.empty(max_lag + 1, dtype=np.float64)
    scale = np.sqrt(float(particle_count))
    for lag in range(max_lag + 1):
        stop = len(com) - lag
        com_delta = scale * (com[lag:] - com[:stop])
        particle_delta = particles[lag:] - particles[:stop]
        com_radius = np.linalg.norm(com_delta, axis=1)
        particle_radius = np.linalg.norm(particle_delta, axis=2)
        com_isf[lag] = np.mean(
            np.sinc(com_radius[:, None] * k_values[None, :] / np.pi), axis=0
        )
        single_isf[lag] = np.mean(
            np.sinc(
                particle_radius[:, :, None] * k_values[None, None, :] / np.pi
            ),
            axis=(0, 1),
        )
        msd[lag] = np.mean(np.sum(com_delta**2, axis=1))
    return IsfSeedResult(com_isf, single_isf, msd, counts)


def component_seed(
    com_coordinate: NDArray[np.float64],
    particle_coordinates: NDArray[np.float64],
    k_values: NDArray[np.float64],
    particle_count: int,
    max_lag: int,
) -> IsfSeedResult:
    """One-dimensional component ISFs using ``exp(-ik delta_x)``."""
    com = np.asarray(com_coordinate, dtype=np.float64).reshape(-1, 1)
    particles = np.asarray(particle_coordinates, dtype=np.float64)
    if particles.ndim == 2:
        particles = particles[:, :, None]
    _validate_inputs(com, particles, k_values, particle_count, max_lag)
    counts = lag_origin_counts(len(com), max_lag)
    com_isf = np.empty((max_lag + 1, len(k_values)), dtype=np.complex128)
    single_isf = np.empty_like(com_isf)
    msd = np.empty(max_lag + 1, dtype=np.float64)
    scale = np.sqrt(float(particle_count))
    com_values = com[:, 0]
    particle_values = particles[:, :, 0]
    for lag in range(max_lag + 1):
        stop = len(com_values) - lag
        com_delta = scale * (com_values[lag:] - com_values[:stop])
        particle_delta = particle_values[lag:] - particle_values[:stop]
        com_isf[lag] = np.mean(
            np.exp(-1j * com_delta[:, None] * k_values[None, :]), axis=0
        )
        single_isf[lag] = np.mean(
            np.exp(
                -1j
                * particle_delta[:, :, None]
                * k_values[None, None, :]
            ),
            axis=(0, 1),
        )
        msd[lag] = np.mean(com_delta**2)
    return IsfSeedResult(com_isf, single_isf, msd, counts)


def gaussian_reference(
    msd: NDArray[np.float64],
    k_values: NDArray[np.float64],
    dimensions: int,
) -> NDArray[np.float64]:
    """Return the MSD-derived Gaussian characteristic function."""
    if dimensions < 1:
        raise ValueError("Gaussian-reference dimensions must be positive")
    if msd.ndim != 1 or not np.all(np.isfinite(msd)) or np.any(msd < 0.0):
        raise ValueError("MSD must be a finite, nonnegative vector")
    return np.exp(-msd[:, None] * k_values[None, :] ** 2 / (2.0 * dimensions))


def average_seed_results(
    key: str,
    title: str,
    dimensions: int,
    seeds: list[IsfSeedResult],
    k_values: NDArray[np.float64],
) -> IsfResult:
    """Average already time-origin-averaged seed estimators equally."""
    if not seeds:
        raise ValueError("cannot average an empty ISF seed list")
    reference_counts = seeds[0].origin_counts
    for seed in seeds[1:]:
        if not np.array_equal(seed.origin_counts, reference_counts):
            raise ValueError("ISF seed origin counts do not match")
        if (
            seed.com.shape != seeds[0].com.shape
            or seed.single.shape != seeds[0].single.shape
        ):
            raise ValueError("ISF seed curve shapes do not match")
    com = np.mean(np.asarray([seed.com for seed in seeds]), axis=0)
    single = np.mean(np.asarray([seed.single for seed in seeds]), axis=0)
    msd = np.mean(np.asarray([seed.com_msd for seed in seeds]), axis=0)
    return IsfResult(
        key=key,
        title=title,
        dimensions=dimensions,
        com=np.asarray(com, dtype=np.complex128),
        single=np.asarray(single, dtype=np.complex128),
        gaussian=gaussian_reference(msd, k_values, dimensions),
        com_msd=np.asarray(msd, dtype=np.float64),
        origin_counts=reference_counts.copy(),
    )


def validate_results(
    results: list[IsfResult],
    k_values: NDArray[np.float64],
    *,
    msd_relative_tolerance: float = 0.10,
    minimum_origins: int = 50,
) -> IsfDiagnostics:
    """Validate ISFs while retaining drift and short-run diagnostics.

    A component ISF can legitimately have a large imaginary part when its
    displacement distribution has drift or lacks inversion symmetry. Its
    magnitude is therefore reported but is not a validation failure. Likewise,
    the small-k approximation is checked only when at least one lag lies in its
    numerical validation window.
    """
    if not results:
        raise ValueError("no ISF estimators to validate")
    if k_values.ndim != 1 or len(k_values) < 1:
        raise ValueError("validation requires a nonempty k grid")
    if not np.all(np.isfinite(k_values)) or np.any(k_values <= 0.0):
        raise ValueError("validation k values must be finite and positive")
    if minimum_origins < 1:
        raise ValueError("minimum origin count must be positive")
    maximum_imaginary = 0.0
    maximum_msd_error: float | None = None
    bound_tolerance = 64.0 * np.finfo(np.float64).eps
    for result in results:
        if result.com.shape != result.single.shape:
            raise ValueError(
                f"{result.key}: COM and single-particle shapes differ"
            )
        if result.com.shape != (len(result.origin_counts), len(k_values)):
            raise ValueError(f"{result.key}: ISF shape does not match lag/k grids")
        if result.gaussian.shape != result.com.shape:
            raise ValueError(f"{result.key}: Gaussian-reference shape differs")
        if (
            result.com_msd.shape != result.origin_counts.shape
            or not np.all(np.isfinite(result.com_msd))
            or np.any(result.com_msd < 0.0)
        ):
            raise ValueError(f"{result.key}: invalid direct COM MSD")
        if (
            not np.all(np.isfinite(result.gaussian))
            or not np.allclose(
                result.gaussian[0], 1.0, rtol=0.0, atol=bound_tolerance
            )
            or np.any(result.gaussian > 1.0 + bound_tolerance)
            or np.any(result.gaussian < 0.0)
        ):
            raise ValueError(f"{result.key}: invalid Gaussian-reference ISF")
        if np.any(result.origin_counts < 1):
            raise ValueError(f"{result.key}: origin counts must be positive")
        if np.any(np.diff(result.origin_counts) >= 0):
            raise ValueError(f"{result.key}: origin counts are not decreasing")
        for name, values in (
            ("COM", result.com),
            ("single-particle", result.single),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{result.key} {name} ISF contains non-finite values"
                )
            if not np.allclose(values[0], 1.0, rtol=0.0, atol=bound_tolerance):
                raise ValueError(
                    f"{result.key} {name} ISF does not satisfy F(k, 0)=1"
                )
            if np.any(np.abs(values) > 1.0 + bound_tolerance):
                raise ValueError(f"{result.key} {name} ISF exceeds |F|=1")
            eligible = result.origin_counts >= minimum_origins
            if np.any(eligible):
                leakage = float(np.max(np.abs(values.imag[eligible])))
                maximum_imaginary = max(maximum_imaginary, leakage)
        loss = 1.0 - result.com[:, 0].real
        window = (loss >= 1.0e-6) & (loss <= 0.1)
        if not np.any(window):
            continue
        inferred = 2.0 * result.dimensions * loss[window] / k_values[0] ** 2
        direct = result.com_msd[window]
        error = np.abs(inferred - direct) / np.maximum(direct, np.finfo(float).tiny)
        estimator_error = float(np.max(error))
        if maximum_msd_error is None:
            maximum_msd_error = estimator_error
        else:
            maximum_msd_error = max(maximum_msd_error, estimator_error)
        if estimator_error > msd_relative_tolerance:
            raise ValueError(
                f"{result.key}: q=0.1 ISF/MSD relative error "
                f"{estimator_error:.4g} exceeds {msd_relative_tolerance:.4g}"
            )
    return IsfDiagnostics(maximum_imaginary, maximum_msd_error)


def plot_summary(
    lag_over_tau_r: NDArray[np.float64],
    q_values: NDArray[np.float64],
    results: list[IsfResult],
    label: str,
    seed_count: int,
    output: Path,
) -> None:
    """Plot COM, non-Gaussian residual, and single/COM comparisons."""
    sns.set_theme(context="paper", style="ticks", font_scale=1.0)
    figure, axes = plt.subplots(
        len(results),
        3,
        figsize=(13.2, 4.0 * len(results)),
        squeeze=False,
        sharex=True,
        constrained_layout=True,
    )
    colors = sns.color_palette("viridis", n_colors=len(q_values))
    for row, result in enumerate(results):
        for index, (q_value, color) in enumerate(
            zip(q_values, colors, strict=True)
        ):
            axes[row, 0].plot(
                lag_over_tau_r,
                result.com[:, index].real,
                color=color,
                lw=1.8,
                label=rf"$k\ell_p={q_value:g}$",
            )
            axes[row, 1].plot(
                lag_over_tau_r,
                result.com[:, index].real - result.gaussian[:, index],
                color=color,
                lw=1.8,
            )
            axes[row, 2].plot(
                lag_over_tau_r,
                result.com[:, index].real,
                color=color,
                lw=1.8,
            )
            axes[row, 2].plot(
                lag_over_tau_r,
                result.single[:, index].real,
                color=color,
                ls="--",
                lw=1.5,
            )
        axes[row, 0].set_ylabel(rf"{result.title}  $\mathrm{{Re}}\,F$")
        axes[row, 0].set_title("rescaled COM")
        axes[row, 1].set_title(r"$F_{\rm COM}-F_{\rm Gaussian}$")
        axes[row, 2].set_title("COM (solid) vs sampled particles (dashed)")
        for axis in axes[row]:
            axis.axhline(0.0, color="0.65", lw=0.7)
            axis.grid(color="0.92", lw=0.6)
            sns.despine(ax=axis)
    for axis in axes[-1]:
        axis.set_xlabel(r"lag $\tau/\tau_r$")
    axes[0, 0].legend(frameon=False, ncol=2)
    figure.suptitle(
        f"Intermediate scattering functions — {label} ({seed_count} seeds)"
    )
    _save_figure(figure, output)


def plot_heatmap(
    lag_over_tau_r: NDArray[np.float64],
    q_values: NDArray[np.float64],
    results: list[IsfResult],
    label: str,
    output: Path,
) -> None:
    """Plot signed COM ISFs over lag and dimensionless wavenumber."""
    sns.set_theme(context="paper", style="ticks", font_scale=1.0)
    figure, axes = plt.subplots(
        1,
        len(results),
        figsize=(6.2 * len(results), 4.6),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, result in zip(axes[0], results, strict=True):
        values = result.com.real.T
        extent = float(np.max(np.abs(values)))
        if extent == 0.0:
            extent = 1.0
        image = axis.pcolormesh(
            lag_over_tau_r,
            q_values,
            values,
            shading="nearest",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent),
        )
        axis.set_yscale("log")
        axis.set_xlabel(r"lag $\tau/\tau_r$")
        axis.set_ylabel(r"$q=k\ell_p$")
        axis.set_title(result.title)
        figure.colorbar(image, ax=axis, label=r"$\mathrm{Re}\,F_{\rm COM}$")
    figure.suptitle(f"Signed COM intermediate scattering function — {label}")
    _save_figure(figure, output)
