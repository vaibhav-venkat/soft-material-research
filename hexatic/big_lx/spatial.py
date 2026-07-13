from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class PeriodicXTree:
    """A 3D cKDTree with explicit periodic images along x only."""

    positions: np.ndarray
    source_indices: np.ndarray
    box_length_x: float
    tree: cKDTree

    @classmethod
    def build(cls, positions: np.ndarray, box_length_x: float) -> PeriodicXTree:
        points = np.asarray(positions, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if box_length_x <= 0.0:
            raise ValueError("box_length_x must be positive")
        count = len(points)
        augmented = np.concatenate(
            (
                points,
                points + np.asarray((box_length_x, 0.0, 0.0)),
                points - np.asarray((box_length_x, 0.0, 0.0)),
            ),
            axis=0,
        )
        source_indices = np.tile(np.arange(count, dtype=np.int64), 3)
        return cls(
            positions=augmented,
            source_indices=source_indices,
            box_length_x=float(box_length_x),
            tree=cKDTree(augmented),
        )

    def nearest_bonds(
        self,
        query_positions: np.ndarray,
        n_neighbors: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(query_positions, dtype=np.float64)
        distances, hits = self.tree.query(
            queries,
            k=n_neighbors + 1,
            workers=-1,
        )
        if distances.ndim == 1:
            distances = distances[:, np.newaxis]
            hits = hits[:, np.newaxis]
        if not np.allclose(distances[:, 0], 0.0, atol=1e-10):
            raise ValueError("nearest-neighbor query did not find each particle itself")
        neighbor_hits = hits[:, 1 : n_neighbors + 1]
        bonds = self.positions[neighbor_hits] - queries[:, np.newaxis, :]
        return self.source_indices[neighbor_hits], bonds

    def radius_block(
        self,
        query_positions: np.ndarray,
        radius: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return padded source IDs, bond vectors, and a validity mask."""
        queries = np.asarray(query_positions, dtype=np.float64)
        hit_lists = self.tree.query_ball_point(
            queries,
            radius,
            workers=-1,
            return_sorted=True,
        )
        width = max((len(hits) for hits in hit_lists), default=0)
        width = max(1, width)
        source_ids = np.zeros((len(queries), width), dtype=np.int64)
        bonds = np.zeros((len(queries), width, 3), dtype=np.float32)
        valid = np.zeros((len(queries), width), dtype=np.bool_)
        for row, hits in enumerate(hit_lists):
            if not hits:
                continue
            hit_array = np.asarray(hits, dtype=np.int64)
            count = len(hit_array)
            source_ids[row, :count] = self.source_indices[hit_array]
            bonds[row, :count] = (
                self.positions[hit_array] - queries[row]
            ).astype(np.float32)
            valid[row, :count] = True
        return source_ids, bonds, valid


def exclude_self(
    source_ids: np.ndarray,
    valid: np.ndarray,
    query_source_ids: np.ndarray,
) -> np.ndarray:
    return valid & (source_ids != np.asarray(query_source_ids)[:, np.newaxis])
