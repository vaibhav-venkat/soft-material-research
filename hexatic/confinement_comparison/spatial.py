from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

from .cases import ComparisonCase


@dataclass(frozen=True)
class PeriodicTree:
    """KD-tree using the minimum image in each periodic Cartesian axis."""

    positions: np.ndarray
    transformed_positions: np.ndarray
    box_lengths: np.ndarray
    tree: KDTree

    @classmethod
    def build(cls, positions: np.ndarray, case: ComparisonCase) -> PeriodicTree:
        points = np.asarray(positions, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        lengths = np.zeros(3, dtype=np.float64)
        lengths[0] = case.lx
        if case.is_sandwich:
            lengths[1] = case.transverse_span
        transformed = points.copy()
        for axis, length in enumerate(lengths):
            if length > 0.0:
                transformed[:, axis] = np.mod(transformed[:, axis] + 0.5 * length, length)
        return cls(
            positions=points,
            transformed_positions=transformed,
            box_lengths=lengths,
            tree=KDTree(transformed, boxsize=lengths),
        )

    def _transform_queries(self, queries: np.ndarray) -> np.ndarray:
        transformed = np.asarray(queries, dtype=np.float64).copy()
        for axis, length in enumerate(self.box_lengths):
            if length > 0.0:
                transformed[:, axis] = np.mod(transformed[:, axis] + 0.5 * length, length)
        return transformed

    def _bonds(self, source_ids: np.ndarray, queries: np.ndarray) -> np.ndarray:
        bonds = self.positions[source_ids] - queries[:, np.newaxis, :]
        for axis, length in enumerate(self.box_lengths):
            if length > 0.0:
                bonds[..., axis] -= length * np.rint(bonds[..., axis] / length)
        return bonds.astype(np.float32)

    def nearest_bonds(
        self,
        query_positions: np.ndarray,
        n_neighbors: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(query_positions, dtype=np.float64)
        distances, hits = self.tree.query(
            self._transform_queries(queries),
            k=n_neighbors + 1,
            workers=-1,
        )
        if distances.ndim == 1:
            distances = distances[:, np.newaxis]
            hits = hits[:, np.newaxis]
        if not np.allclose(distances[:, 0], 0.0, atol=1e-10):
            raise ValueError("nearest-neighbor query did not find each particle itself")
        source_ids = np.asarray(hits[:, 1 : n_neighbors + 1], dtype=np.int64)
        return source_ids, self._bonds(source_ids, queries)

    def radius_block(
        self,
        query_positions: np.ndarray,
        radius: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        queries = np.asarray(query_positions, dtype=np.float64)
        hit_lists = self.tree.query_ball_point(
            self._transform_queries(queries),
            radius,
            workers=-1,
            return_sorted=True,
        )
        width = max(1, max((len(hits) for hits in hit_lists), default=0))
        source_ids = np.zeros((len(queries), width), dtype=np.int64)
        valid = np.zeros((len(queries), width), dtype=np.bool_)
        for row, hits in enumerate(hit_lists):
            count = len(hits)
            if count:
                source_ids[row, :count] = np.asarray(hits, dtype=np.int64)
                valid[row, :count] = True
        return source_ids, self._bonds(source_ids, queries), valid


def exclude_self(
    source_ids: np.ndarray,
    valid: np.ndarray,
    query_source_ids: np.ndarray,
) -> np.ndarray:
    return valid & (source_ids != np.asarray(query_source_ids)[:, np.newaxis])
