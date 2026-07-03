from __future__ import annotations

from pathlib import Path
import json

import numpy as np


def gradient_scalar(field: np.ndarray, lx: float, ly: float) -> np.ndarray:
    kx = 2.0 * np.pi * np.fft.fftfreq(field.shape[1], d=lx / field.shape[1])
    ky = 2.0 * np.pi * np.fft.fftfreq(field.shape[2], d=ly / field.shape[2])
    field_hat = np.fft.fftn(field, axes=(1, 2))
    dx = np.fft.ifftn(1j * kx[None, :, None] * field_hat, axes=(1, 2)).real
    dy = np.fft.ifftn(1j * ky[None, None, :] * field_hat, axes=(1, 2)).real
    return np.stack((dx, dy), axis=-1)


def laplacian_scalar(field: np.ndarray, lx: float, ly: float) -> np.ndarray:
    kx = 2.0 * np.pi * np.fft.fftfreq(field.shape[1], d=lx / field.shape[1])
    ky = 2.0 * np.pi * np.fft.fftfreq(field.shape[2], d=ly / field.shape[2])
    k2 = kx[None, :, None] ** 2 + ky[None, None, :] ** 2
    field_hat = np.fft.fftn(field, axes=(1, 2))
    return np.fft.ifftn(-k2 * field_hat, axes=(1, 2)).real


def divergence_vector(field: np.ndarray, lx: float, ly: float) -> np.ndarray:
    return gradient_scalar(field[..., 0], lx, ly)[..., 0] + gradient_scalar(field[..., 1], lx, ly)[..., 1]


def div_div_q_surface(q: np.ndarray, lx: float, ly: float) -> np.ndarray:
    div_q = np.empty(q.shape[:3] + (2,), dtype=float)
    for a in range(2):
        div_q[..., a] = divergence_vector(q[..., :, a], lx, ly)
    return divergence_vector(div_q, lx, ly)


def corr(target: np.ndarray, feature: np.ndarray) -> float:
    mask = np.isfinite(target) & np.isfinite(feature)
    y = target[mask]
    x = feature[mask]
    y = y - y.mean()
    x = x - x.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if denom == 0.0:
        return np.nan
    return float(np.dot(x, y) / denom)


def divide_by_density(numerator: np.ndarray, density: np.ndarray) -> np.ndarray:
    assert numerator.shape == density.shape
    return np.divide(
        numerator,
        density,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=np.isfinite(density) & (density > 0.0),
    )


def main() -> None:
    path = Path(__file__).resolve().parent / "output" / "radius_15D_fit_result.npz"
    data = np.load(path, allow_pickle=True)
    gaussian_path = (
        Path(__file__).resolve().parents[1]
        / "model_fitting"
        / "output"
        / "fitting"
        / "radius_15D_gaussian_fields.npz"
    )
    gaussian = np.load(gaussian_path, allow_pickle=True)
    metadata = json.loads(str(data["metadata_json"]))
    lx = float(metadata["lx"])
    ly = float(metadata["ly"])

    rho = np.asarray(data["rho"], dtype=float)
    p = np.asarray(data["P"], dtype=float)
    q = np.asarray(data["Q"], dtype=float)
    y_p = np.asarray(data["Y_P"], dtype=float)
    a = np.asarray(data["A"], dtype=float)
    a_surface = a[..., :2, :]

    assert y_p.shape == a_surface.shape

    numerator = np.einsum("...ka,...ka->...", y_p, a_surface)
    denominator = np.einsum("...ka,...ka->...", a_surface, a_surface)
    u = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=u, where=denominator > 0.0)

    finite = np.isfinite(u)
    print(f"path={path}")
    print(f"Y_P shape={y_p.shape}")
    print(f"A_surface shape={a_surface.shape}")
    print(f"finite={finite.sum()} / {u.size}")
    print(f"min={np.nanmin(u):.12g}")
    print(f"p01={np.nanpercentile(u, 1):.12g}")
    print(f"p05={np.nanpercentile(u, 5):.12g}")
    print(f"median={np.nanmedian(u):.12g}")
    print(f"mean={np.nanmean(u):.12g}")
    print(f"p95={np.nanpercentile(u, 95):.12g}")
    print(f"p99={np.nanpercentile(u, 99):.12g}")
    print(f"max={np.nanmax(u):.12g}")

    grad_rho = gradient_scalar(rho, lx, ly)
    grad_rho_norm2 = np.einsum("...k,...k->...", grad_rho, grad_rho)
    lap_rho = laplacian_scalar(rho, lx, ly)
    div_p = divergence_vector(p[..., :2], lx, ly)
    lap_div_p = laplacian_scalar(div_p, lx, ly)
    div_div_q = div_div_q_surface(q[..., :2, :2], lx, ly)

    candidates = {
        "rho": rho,
        "lap rho": lap_rho,
        "bilap rho": laplacian_scalar(lap_rho, lx, ly),
        "|grad rho|": np.sqrt(grad_rho_norm2),
        "|grad rho|^2": grad_rho_norm2,
        "div P": div_p,
        "lap div P": lap_div_p,
        "bilap div P": laplacian_scalar(lap_div_p, lx, ly),
        "div div Q": div_div_q,
        "lap div div Q": laplacian_scalar(div_div_q, lx, ly),
        "bilap div div Q": laplacian_scalar(laplacian_scalar(div_div_q, lx, ly), lx, ly),
        "tr(Q^2)": np.einsum("...ab,...ab->...", q, q),
        "lap tr(Q^2)": laplacian_scalar(np.einsum("...ab,...ab->...", q, q), lx, ly),
    }

    print("\ncorrelations with projected Y_P/A")
    for name, value in sorted(candidates.items(), key=lambda item: abs(corr(u, item[1])), reverse=True):
        print(f"{name}: {corr(u, value): .12g}")

    assert np.array_equal(np.asarray(gaussian["x_edges"]), np.linspace(-lx / 2.0, lx / 2.0, rho.shape[1] + 1))
    assert np.asarray(gaussian["theta_edges"]).shape == (rho.shape[2] + 1,)
    gaussian_density = np.asarray(gaussian["rho_gaussian"], dtype=float)
    hexatic_order = divide_by_density(np.asarray(gaussian["hexatic_order_numerator"], dtype=float), gaussian_density)
    disclination_strength = divide_by_density(np.asarray(gaussian["D_numerator"], dtype=float), gaussian_density)
    disclination_fraction = divide_by_density(
        np.asarray(gaussian["disclination_numerator"], dtype=float),
        gaussian_density,
    )
    chirality = divide_by_density(np.asarray(gaussian["chirality_numerator"], dtype=float), gaussian_density)
    gaussian_candidates = {
        "gaussian |psi6|": hexatic_order,
        "gaussian lap |psi6|": laplacian_scalar(hexatic_order, lx, ly),
        "gaussian |grad |psi6||": np.sqrt(
            np.einsum("...k,...k->...", gradient_scalar(hexatic_order, lx, ly), gradient_scalar(hexatic_order, lx, ly))
        ),
        "gaussian D=(6-n)^2": disclination_strength,
        "gaussian disclination fraction |6-n|=1": disclination_fraction,
        "gaussian chirality": chirality,
        "gaussian |chirality|": np.abs(chirality),
        "gaussian lap chirality": laplacian_scalar(chirality, lx, ly),
        "gaussian |grad chirality|": np.sqrt(
            np.einsum("...k,...k->...", gradient_scalar(chirality, lx, ly), gradient_scalar(chirality, lx, ly))
        ),
    }

    print(f"\nmodel_fitting gaussian cache={gaussian_path}")
    print("correlations with gaussian-smoothed hexatic/disclination/chirality scalars")
    for name, value in sorted(gaussian_candidates.items(), key=lambda item: abs(corr(u, item[1])), reverse=True):
        print(f"{name}: {corr(u, value): .12g}")


if __name__ == "__main__":
    main()
