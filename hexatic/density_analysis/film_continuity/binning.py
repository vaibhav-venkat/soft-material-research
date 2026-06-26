from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised on environments without numba
    njit = None


@dataclass(frozen=True)
class BinnedTransitionSums:
    counts: np.ndarray
    sum_vx: np.ndarray
    sum_vy: np.ndarray
    n_in: np.ndarray
    n_out: np.ndarray
    x_face_crossings: np.ndarray
    theta_face_crossings: np.ndarray
    film_count_per_bin_frame: np.ndarray


def particle_bin_indices(
    x: np.ndarray,
    theta: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if x.shape != theta.shape:
        raise ValueError("x and theta must have matching shapes.")
    x_edges = np.asarray(x_edges, dtype=np.float64)
    theta_edges = np.asarray(theta_edges, dtype=np.float64)
    if _particle_bin_indices_numba is not None:
        return _particle_bin_indices_numba(x, theta, x_edges, theta_edges)
    return _particle_bin_indices_numpy(x, theta, x_edges, theta_edges)


def accumulate_counts_and_sums(
    coords: np.ndarray,
    shell_mask: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> BinnedTransitionSums:
    coords = np.asarray(coords, dtype=np.float64)
    shell_mask = np.asarray(shell_mask, dtype=bool)
    vx = np.asarray(vx, dtype=np.float64)
    vy = np.asarray(vy, dtype=np.float64)
    x_edges = np.asarray(x_edges, dtype=np.float64)
    theta_edges = np.asarray(theta_edges, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[2] < 2:
        raise ValueError("coords must have shape (frames, particles, >=2).")
    if shell_mask.shape != coords.shape[:2]:
        raise ValueError("shell_mask must match coords frame/particle axes.")
    if vx.shape != (coords.shape[0] - 1, coords.shape[1]):
        raise ValueError("vx must have shape (frames - 1, particles).")
    if vy.shape != vx.shape:
        raise ValueError("vy must match vx shape.")

    if _accumulate_counts_and_sums_numba is not None:
        arrays = _accumulate_counts_and_sums_numba(
            coords,
            shell_mask,
            vx,
            vy,
            x_edges,
            theta_edges,
        )
    else:
        arrays = _accumulate_counts_and_sums_numpy(
            coords,
            shell_mask,
            vx,
            vy,
            x_edges,
            theta_edges,
        )
    return BinnedTransitionSums(*arrays)


def _wrapped_values(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    period = edges[-1] - edges[0]
    return np.mod(values - edges[0], period) + edges[0]


def _particle_bin_indices_numpy(
    x: np.ndarray,
    theta: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_wrapped = _wrapped_values(x, x_edges)
    theta_wrapped = _wrapped_values(theta, theta_edges)
    ix = np.searchsorted(x_edges, x_wrapped, side="right") - 1
    itheta = np.searchsorted(theta_edges, theta_wrapped, side="right") - 1
    return np.mod(ix, x_edges.size - 1), np.mod(itheta, theta_edges.size - 1)


def _accumulate_counts_and_sums_numpy(
    coords: np.ndarray,
    shell_mask: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_frames, n_particles = coords.shape[:2]
    nx = x_edges.size - 1
    nt = theta_edges.size - 1
    counts = np.zeros((n_frames - 1, nx, nt), dtype=np.int64)
    sum_vx = np.zeros((n_frames - 1, nx, nt), dtype=np.float64)
    sum_vy = np.zeros((n_frames - 1, nx, nt), dtype=np.float64)
    n_in = np.zeros((n_frames - 1, nx, nt), dtype=np.int64)
    n_out = np.zeros((n_frames - 1, nx, nt), dtype=np.int64)
    x_faces = np.zeros((n_frames - 1, nx, nt), dtype=np.int64)
    theta_faces = np.zeros((n_frames - 1, nx, nt), dtype=np.int64)
    frame_counts = np.zeros((n_frames, nx, nt), dtype=np.int64)

    for frame in range(n_frames):
        ix, itheta = particle_bin_indices(
            coords[frame, :, 0],
            coords[frame, :, 1],
            x_edges,
            theta_edges,
        )
        for particle in range(n_particles):
            if shell_mask[frame, particle]:
                frame_counts[frame, ix[particle], itheta[particle]] += 1

    for frame in range(n_frames - 1):
        ix_t, itheta_t = particle_bin_indices(
            coords[frame, :, 0],
            coords[frame, :, 1],
            x_edges,
            theta_edges,
        )
        ix_next, itheta_next = particle_bin_indices(
            coords[frame + 1, :, 0],
            coords[frame + 1, :, 1],
            x_edges,
            theta_edges,
        )
        for particle in range(n_particles):
            in_film_t = shell_mask[frame, particle]
            in_film_next = shell_mask[frame + 1, particle]
            if in_film_t:
                ixp = ix_t[particle]
                itp = itheta_t[particle]
                counts[frame, ixp, itp] += 1
                sum_vx[frame, ixp, itp] += vx[frame, particle]
                sum_vy[frame, ixp, itp] += vy[frame, particle]
            if in_film_t and in_film_next:
                _accumulate_crossing_path_numpy(
                    x_faces[frame],
                    theta_faces[frame],
                    ix_t[particle],
                    itheta_t[particle],
                    ix_next[particle],
                    itheta_next[particle],
                )
            if (not in_film_t) and in_film_next:
                n_in[frame, ix_next[particle], itheta_next[particle]] += 1
            elif in_film_t and (not in_film_next):
                n_out[frame, ix_t[particle], itheta_t[particle]] += 1
    return counts, sum_vx, sum_vy, n_in, n_out, x_faces, theta_faces, frame_counts


def _signed_bin_delta(start: int, stop: int, n_bins: int) -> int:
    delta = int(stop) - int(start)
    if delta > n_bins // 2:
        delta -= n_bins
    elif delta < -n_bins // 2:
        delta += n_bins
    return delta


def _accumulate_crossing_path_numpy(
    x_faces: np.ndarray,
    theta_faces: np.ndarray,
    ix_start: int,
    itheta_start: int,
    ix_stop: int,
    itheta_stop: int,
) -> None:
    nx, nt = x_faces.shape
    ix = int(ix_start)
    itheta = int(itheta_start)

    dx_bins = _signed_bin_delta(ix, int(ix_stop), nx)
    step_x = 1 if dx_bins > 0 else -1
    for _ in range(abs(dx_bins)):
        if step_x > 0:
            x_faces[ix, itheta] += 1
            ix = (ix + 1) % nx
        else:
            face_ix = (ix - 1) % nx
            x_faces[face_ix, itheta] -= 1
            ix = face_ix

    dtheta_bins = _signed_bin_delta(itheta, int(itheta_stop), nt)
    step_theta = 1 if dtheta_bins > 0 else -1
    for _ in range(abs(dtheta_bins)):
        if step_theta > 0:
            theta_faces[ix, itheta] += 1
            itheta = (itheta + 1) % nt
        else:
            face_itheta = (itheta - 1) % nt
            theta_faces[ix, face_itheta] -= 1
            itheta = face_itheta


if njit is not None:

    @njit(cache=True)
    def _wrap_scalar(value: float, start: float, period: float) -> float:
        return ((value - start) % period) + start

    @njit(cache=True)
    def _bin_index(value: float, edges: np.ndarray) -> int:
        n_bins = edges.size - 1
        wrapped = _wrap_scalar(value, edges[0], edges[-1] - edges[0])
        lo = 0
        hi = edges.size
        while lo < hi:
            mid = (lo + hi) // 2
            if wrapped < edges[mid]:
                hi = mid
            else:
                lo = mid + 1
        idx = lo - 1
        return idx % n_bins

    @njit(cache=True)
    def _particle_bin_indices_numba(
        x: np.ndarray,
        theta: np.ndarray,
        x_edges: np.ndarray,
        theta_edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = x.size
        ix = np.empty(n, dtype=np.int64)
        itheta = np.empty(n, dtype=np.int64)
        for particle in range(n):
            ix[particle] = _bin_index(x[particle], x_edges)
            itheta[particle] = _bin_index(theta[particle], theta_edges)
        return ix, itheta

    @njit(cache=True)
    def _signed_bin_delta_numba(start: int, stop: int, n_bins: int) -> int:
        delta = stop - start
        if delta > n_bins // 2:
            delta -= n_bins
        elif delta < -n_bins // 2:
            delta += n_bins
        return delta

    @njit(cache=True)
    def _accumulate_crossing_path_numba(
        x_faces: np.ndarray,
        theta_faces: np.ndarray,
        ix_start: int,
        itheta_start: int,
        ix_stop: int,
        itheta_stop: int,
    ) -> None:
        nx = x_faces.shape[0]
        nt = x_faces.shape[1]
        ix = ix_start
        itheta = itheta_start

        dx_bins = _signed_bin_delta_numba(ix, ix_stop, nx)
        if dx_bins > 0:
            for _ in range(dx_bins):
                x_faces[ix, itheta] += 1
                ix = (ix + 1) % nx
        elif dx_bins < 0:
            for _ in range(-dx_bins):
                face_ix = (ix - 1) % nx
                x_faces[face_ix, itheta] -= 1
                ix = face_ix

        dtheta_bins = _signed_bin_delta_numba(itheta, itheta_stop, nt)
        if dtheta_bins > 0:
            for _ in range(dtheta_bins):
                theta_faces[ix, itheta] += 1
                itheta = (itheta + 1) % nt
        elif dtheta_bins < 0:
            for _ in range(-dtheta_bins):
                face_itheta = (itheta - 1) % nt
                theta_faces[ix, face_itheta] -= 1
                itheta = face_itheta

    @njit(cache=True)
    def _accumulate_counts_and_sums_numba(
        coords: np.ndarray,
        shell_mask: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        x_edges: np.ndarray,
        theta_edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_frames = coords.shape[0]
        n_particles = coords.shape[1]
        nx = x_edges.size - 1
        nt = theta_edges.size - 1
        counts = np.zeros((n_frames - 1, nx, nt), dtype=np.int64)
        sum_vx = np.zeros((n_frames - 1, nx, nt), dtype=np.float64)
        sum_vy = np.zeros((n_frames - 1, nx, nt), dtype=np.float64)
        n_in = np.zeros((n_frames - 1, nx, nt), dtype=np.int64)
        n_out = np.zeros((n_frames - 1, nx, nt), dtype=np.int64)
        x_faces = np.zeros((n_frames - 1, nx, nt), dtype=np.int64)
        theta_faces = np.zeros((n_frames - 1, nx, nt), dtype=np.int64)
        frame_counts = np.zeros((n_frames, nx, nt), dtype=np.int64)

        for frame in range(n_frames):
            for particle in range(n_particles):
                if shell_mask[frame, particle]:
                    ix = _bin_index(coords[frame, particle, 0], x_edges)
                    itheta = _bin_index(coords[frame, particle, 1], theta_edges)
                    frame_counts[frame, ix, itheta] += 1

        for frame in range(n_frames - 1):
            for particle in range(n_particles):
                in_film_t = shell_mask[frame, particle]
                in_film_next = shell_mask[frame + 1, particle]
                ix_t = _bin_index(coords[frame, particle, 0], x_edges)
                itheta_t = _bin_index(coords[frame, particle, 1], theta_edges)
                if in_film_t:
                    counts[frame, ix_t, itheta_t] += 1
                    sum_vx[frame, ix_t, itheta_t] += vx[frame, particle]
                    sum_vy[frame, ix_t, itheta_t] += vy[frame, particle]
                if in_film_t and in_film_next:
                    ix_next_for_path = _bin_index(coords[frame + 1, particle, 0], x_edges)
                    itheta_next_for_path = _bin_index(
                        coords[frame + 1, particle, 1],
                        theta_edges,
                    )
                    _accumulate_crossing_path_numba(
                        x_faces[frame],
                        theta_faces[frame],
                        ix_t,
                        itheta_t,
                        ix_next_for_path,
                        itheta_next_for_path,
                    )
                if (not in_film_t) and in_film_next:
                    ix_next = _bin_index(coords[frame + 1, particle, 0], x_edges)
                    itheta_next = _bin_index(coords[frame + 1, particle, 1], theta_edges)
                    n_in[frame, ix_next, itheta_next] += 1
                elif in_film_t and (not in_film_next):
                    n_out[frame, ix_t, itheta_t] += 1
        return counts, sum_vx, sum_vy, n_in, n_out, x_faces, theta_faces, frame_counts

else:
    _particle_bin_indices_numba = None
    _accumulate_counts_and_sums_numba = None
