from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent
FIT = BASE / "output" / "fitting" / "radius_15D_fitting.npz"
OUT = BASE / "output" / "m4_analysis" / "radius_15D_M4_local_coefficients.png"
FIELDS = (
    ("P", "β_P"),
    ("chiral_P_perp", "β_chiral"),
    ("force_density", "β_force"),
    ("grad_rho", "β_gradrho"),
    ("grad_hexatic_order", "β_gradpsi"),
    ("grad_D", "β_gradD"),
    ("D_P", "β_DP"),
    ("D_chiral_P_perp", "β_Dchiral"),
)


def main() -> None:
    with np.load(FIT, allow_pickle=False) as data:
        x = data["x_centers"]
        theta = data["theta_centers"]
        fig, axes = plt.subplots(len(FIELDS), 2, figsize=(11, 18), constrained_layout=True)
        for row, (field, label) in enumerate(FIELDS):
            beta = data[f"coef_map__{field}"]
            for component, name in enumerate(("x", "y")):
                axis = axes[row, component]
                values = beta[..., component]
                limit = robust_limit(values)
                mesh = axis.pcolormesh(
                    x,
                    theta,
                    values.T,
                    shading="auto",
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                )
                axis.set_title(f"{label}_{name}(x, θ)")
                axis.set_xlabel("x")
                axis.set_ylabel("θ")
                fig.colorbar(mesh, ax=axis)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"Wrote {OUT}")


def robust_limit(values: np.ndarray) -> float:
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(finite, 98.0))
    return limit if np.isfinite(limit) and limit > 0.0 else 1.0


if __name__ == "__main__":
    main()
