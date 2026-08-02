from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label


@dataclass(frozen=True)
class BandComponent:
    label: int
    area_cells: int
    winds_x: bool
    winds_s: bool


def label_dilute_bands(
    dilute: np.ndarray,
    *,
    minimum_area: int = 1,
) -> tuple[np.ndarray, list[BandComponent]]:
    """Label toroidal components that wind in either periodic direction."""
    nx, ns = dilute.shape
    ordinary_labels, count = label(
        dilute, structure=np.ones((3, 3), dtype=np.int8)
    )
    parents = np.arange(count + 1)

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = int(parents[value])
        return value

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for i, j in np.argwhere(dilute):
        source = int(ordinary_labels[i, j])
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                neighbor = int(ordinary_labels[(i + di) % nx, (j + dj) % ns])
                if neighbor:
                    union(source, neighbor)

    tiled = np.tile(dilute, (3, 3))
    tiled_labels, _ = label(tiled, structure=np.ones((3, 3), dtype=np.int8))
    roots = np.zeros_like(ordinary_labels)
    for component_label in range(1, count + 1):
        roots[ordinary_labels == component_label] = find(component_label)
    output = np.zeros_like(ordinary_labels, dtype=np.int32)
    components: list[BandComponent] = []

    for root in np.unique(roots):
        if root == 0:
            continue
        cells = roots == root
        area = int(np.count_nonzero(cells))
        if area < minimum_area:
            continue
        ii, jj = np.nonzero(cells)
        winds_x = bool(
            np.any(
                tiled_labels[ii + nx, jj + ns]
                == tiled_labels[ii + 2 * nx, jj + ns]
            )
        )
        winds_s = bool(
            np.any(
                tiled_labels[ii + nx, jj + ns]
                == tiled_labels[ii + nx, jj + 2 * ns]
            )
        )
        if not (winds_x or winds_s):
            continue
        band_label = len(components) + 1
        output[cells] = band_label
        components.append(BandComponent(band_label, area, winds_x, winds_s))
    return output, components
