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


def gradient_q(q: np.ndarray, lx: float, ly: float) -> np.ndarray:
    out = np.empty(q.shape[:3] + (2, 3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            out[..., :, a, b] = gradient_scalar(q[..., a, b], lx, ly)
    return out


def q_grad_rho(q: np.ndarray, grad_rho: np.ndarray) -> np.ndarray:
    out = np.zeros(q.shape[:3] + (2, 3, 3), dtype=float)
    for k in range(2):
        for a in range(3):
            for b in range(2):
                out[..., k, a, b] = q[..., k, a] * grad_rho[..., b]
    return out


def grad_rho_q(q: np.ndarray, grad_rho: np.ndarray) -> np.ndarray:
    return grad_rho[..., :, None, None] * q[..., None, :, :]


def p_q_symmetrized(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    out = np.zeros(q.shape[:3] + (2, 3, 3), dtype=float)
    for k in range(2):
        for a in range(3):
            for b in range(3):
                out[..., k, a, b] = (
                    p[..., k] * q[..., a, b]
                    + p[..., a] * q[..., k, b]
                    + p[..., b] * q[..., k, a]
                )
    return out


def ppp_traceless(p: np.ndarray) -> np.ndarray:
    p2 = np.einsum("...a,...a->...", p, p)
    out = np.zeros(p.shape[:3] + (2, 3, 3), dtype=float)
    for k in range(2):
        for a in range(3):
            for b in range(3):
                trace = (p[..., k] * p2 / 3.0) if a == b else 0.0
                out[..., k, a, b] = p[..., k] * p[..., a] * p[..., b] - trace
    return out


def corr(target: np.ndarray, feature: np.ndarray) -> float:
    assert target.shape == feature.shape
    mask = np.isfinite(target) & np.isfinite(feature)
    y = target[mask]
    x = feature[mask]
    y = y - y.mean()
    x = x - x.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if denom == 0.0:
        return np.nan
    return float(np.dot(x, y) / denom)


def decode_names(values: np.ndarray) -> tuple[str, ...]:
    return tuple(value.decode() if isinstance(value, bytes) else str(value) for value in values)


def main() -> None:
    path = Path(__file__).resolve().parent / "output" / "radius_15D_fit_result.npz"
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        lx = float(metadata["lx"])
        ly = float(metadata["ly"])
        rho = np.asarray(data["rho"], dtype=float)
        p = np.asarray(data["P"], dtype=float)
        q = np.asarray(data["Q"], dtype=float)
        y_q = np.asarray(data["Y_Q"], dtype=float)
        y_q_library = np.asarray(data["Y_Q_library"], dtype=float)
        y_q_names = decode_names(data["Y_Q_names"])

    assert y_q_library.shape[1:] == y_q.shape
    assert y_q_names[0] == "Ubar_P_dot_alpha_traceless"
    y_q_res = y_q - y_q_library[0]
    grad_rho = gradient_scalar(rho, lx, ly)

    candidates = {
        "grad Q": gradient_q(q, lx, ly),
        "Q tensor grad rho": q_grad_rho(q, grad_rho),
        "P tensor Q symmetrized": p_q_symmetrized(p, q),
        "P tensor P tensor P traceless": ppp_traceless(p),
        "grad rho tensor Q": grad_rho_q(q, grad_rho),
    }

    print(f"path={path}")
    print(f"residual=Y_Q - {y_q_names[0]}")
    print(f"Y_Q shape={y_q.shape}")
    print("\ncorrelations with Y_Q residual")
    for name, field in sorted(candidates.items(), key=lambda item: abs(corr(y_q_res, item[1])), reverse=True):
        print(f"{name}: {corr(y_q_res, field): .12g}")


if __name__ == "__main__":
    main()
