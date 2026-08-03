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
    """Label dilute components that wind around ``s`` but not around ``x``."""
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

    # Interior connections are already represented by ``ordinary_labels``.
    # Only the two periodic seams can join otherwise distinct labels. Gather
    # those label pairs with vectorized rolls, then union each unique pair.
    seam_pairs: list[np.ndarray] = []
    first_x = ordinary_labels[0]
    last_x = ordinary_labels[-1]
    for shift in (-1, 0, 1):
        across_x = np.roll(last_x, -shift)
        valid = (first_x != 0) & (across_x != 0)
        seam_pairs.append(np.column_stack((first_x[valid], across_x[valid])))
    first_s = ordinary_labels[:, 0]
    last_s = ordinary_labels[:, -1]
    for shift in (-1, 0, 1):
        across_s = np.roll(last_s, -shift)
        valid = (first_s != 0) & (across_s != 0)
        seam_pairs.append(np.column_stack((first_s[valid], across_s[valid])))
    nonempty_pairs = [pairs for pairs in seam_pairs if len(pairs)]
    if nonempty_pairs:
        for first, second in np.unique(np.concatenate(nonempty_pairs), axis=0):
            union(int(first), int(second))

    tiled = np.tile(dilute, (3, 3))
    tiled_labels, _ = label(tiled, structure=np.ones((3, 3), dtype=np.int8))
    root_lookup = np.asarray([find(value) for value in range(count + 1)])
    roots = root_lookup[ordinary_labels]
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
        # Only a circumferential winding describes an axially localized band.
        # Axial and doubly winding components have no well-defined axial width
        # or center and are deliberately absent from every downstream output.
        if winds_x or not winds_s:
            continue
        band_label = len(components) + 1
        output[cells] = band_label
        components.append(BandComponent(band_label, area, winds_x, winds_s))
    return output, components
