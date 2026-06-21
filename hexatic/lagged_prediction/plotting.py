from pathlib import Path

import numpy as np

from .types import LAGGED_PREDICTION_IMAGE_DIR, LaggedPredictionResult


def plot_lagged_predictive_decomposition(
    result: LaggedPredictionResult,
    image_dir: str | Path = LAGGED_PREDICTION_IMAGE_DIR,
) -> None:
    import matplotlib.pyplot as plt

    image_path = Path(image_dir)
    image_path.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(result.lag_frames, result.full_r2, marker="o", color="tab:blue")
    axis.axhline(0.0, color="0.35", linewidth=1.0)
    axis.set_xlabel("Lag (frames)")
    axis.set_ylabel(r"cross-validated $R^2$")
    axis.set_title(f"Lagged prediction of future shell {result.target_name}")
    axis.grid(True, ls="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(image_path / "lagged_prediction_r2.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for family_idx, family_name in enumerate(result.family_names):
        axis.plot(
            result.lag_frames,
            result.family_delta_r2[:, family_idx],
            marker="o",
            label=str(family_name),
        )
    axis.axhline(0.0, color="0.35", linewidth=1.0)
    axis.set_xlabel("Lag (frames)")
    axis.set_ylabel(r"drop-family $\Delta R^2$")
    axis.set_title("Unique predictive power by family")
    axis.grid(True, ls="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(image_path / "family_delta_r2.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    im = axis.imshow(
        result.family_coefficient_norms.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    axis.set_xticks(np.arange(len(result.lag_frames)))
    axis.set_xticklabels(result.lag_frames)
    axis.set_yticks(np.arange(len(result.family_names)))
    axis.set_yticklabels(result.family_names)
    axis.set_xlabel("Lag (frames)")
    axis.set_title("Elastic-net coefficient norm by family")
    fig.colorbar(im, ax=axis, label="coefficient norm")
    fig.tight_layout()
    fig.savefig(image_path / "family_coefficient_norms.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        result.lag_frames,
        result.mediation_r2_without_px,
        marker="o",
        label="defects + CCM",
    )
    axis.plot(
        result.lag_frames,
        result.mediation_r2_with_px,
        marker="o",
        label="defects + CCM + P_x",
    )
    axis.axhline(0.0, color="0.35", linewidth=1.0)
    axis.set_xlabel("Lag (frames)")
    axis.set_ylabel(r"cross-validated $R^2$")
    axis.set_title("Mediation-style comparison")
    axis.grid(True, ls="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(image_path / "mediation_r2.png", dpi=200)
    plt.close(fig)

    finite = np.isfinite(result.full_r2)
    if np.any(finite):
        best_idx = int(np.nanargmax(result.full_r2))
        y = result.actual[best_idx]
        pred = result.predictions[best_idx]
        mask = np.isfinite(y) & np.isfinite(pred)
        fig, axis = plt.subplots(figsize=(5.5, 5.5))
        axis.scatter(y[mask], pred[mask], s=28, color="tab:blue", alpha=0.75)
        if np.any(mask):
            low = float(min(np.min(y[mask]), np.min(pred[mask])))
            high = float(max(np.max(y[mask]), np.max(pred[mask])))
            axis.plot([low, high], [low, high], color="0.25", linestyle="--")
        axis.set_xlabel(f"actual future {result.target_name}")
        axis.set_ylabel(f"predicted future {result.target_name}")
        axis.set_title(
            f"{result.target_name} best lag: {int(result.lag_frames[best_idx])} frames"
        )
        axis.grid(True, ls="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(image_path / "best_lag_predicted_vs_actual.png", dpi=200)
        plt.close(fig)
