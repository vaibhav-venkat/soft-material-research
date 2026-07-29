"""Spectral entropy of the zero-decay-coordinate Laplace spectrum."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import seaborn as sns

from hexatic.confinement_comparison.cases import GeometryKind
from hexatic.new_sims.cases import CaseKind

from .plotting import _save_figure


LX_JITTER = 0.0

# Explicit mapping from every known geometry_kind to its plot group. Built from
# the two enums so a new member fails loudly here instead of being silently
# swallowed by a substring test or a catch-all default. big_lx serialises no
# geometry_kind at all and is always a cylinder.
BIG_LX_GEOMETRY_KIND = "cylinder"
GEOMETRY_GROUPS: dict[str, str] = {
    BIG_LX_GEOMETRY_KIND: "cylinder",
    CaseKind.IDEAL_ABP_CYLINDER: "cylinder",
    CaseKind.SINGLE_ACTIVE_TRACER_CYLINDER: "cylinder",
    CaseKind.INVERSION_ACTIVE_PAIR_CYLINDER: "cylinder",
    CaseKind.X_WALLED_PRISM: "prism",
    CaseKind.PERIODIC_2D_PLANE: "2d_plane",
    CaseKind.IDEAL_2D_X_WALLS: "2d_plane",
    CaseKind.IDEAL_2D_PERIODIC: "2d_plane",
    CaseKind.PERIODIC_3D_BULK: "3d_bulk",
    CaseKind.IDEAL_3D_BULK: "3d_bulk",
    CaseKind.IDEAL_3D_BULK_PERIOD_10: "3d_bulk",
    CaseKind.IDEAL_3D_BULK_PERIOD_1: "3d_bulk",
    GeometryKind.PRISM_VOLUME: "prism",
    GeometryKind.PRISM_SURFACE_AREA: "prism",
    GeometryKind.SANDWICH_VOLUME: "sandwich",
    GeometryKind.SANDWICH_SURFACE_AREA: "sandwich",
    GeometryKind.TWO_DIMENSION: "2d_plane",
    GeometryKind.CYLINDER_RATTLE: "cylinder_rattle",
    GeometryKind.CYLINDER_RATTLE_TANGENT: "cylinder_rattle",
}

# Marker fill disambiguates cases that share a geometry group and an interaction
# class. Geometry drives the marker and interaction drives the color, but those
# two channels are coarser than the case list -- the three ideal_3d_bulk
# diffusion periods, the two rattle variants, and the three prisms each collapse
# into one (marker, color) pair -- so without a third channel roughly two thirds
# of the cases would be drawn identically on top of each other at Lx = 1.
FILL_STYLES = ("full", "none", "left", "right", "top", "bottom")


@dataclass(frozen=True)
class EntropyPoint:
    key: str
    case_id: str
    family: str
    geometry_group: str
    interaction_class: str
    lx_multiplier: int
    entropy: float


def spectral_entropy(values_at_r0: NDArray[np.complex128]) -> float:
    power = np.abs(values_at_r0) ** 2
    total = float(np.sum(power))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("spectral entropy requires positive finite total power")
    probabilities = power / total
    positive = probabilities > 0.0
    return float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))


def geometry_group(geometry_kind: str, *, tracer: bool = False) -> str:
    if tracer:
        return "tracer"
    try:
        return GEOMETRY_GROUPS[geometry_kind]
    except KeyError:
        raise ValueError(
            f"unknown geometry_kind {geometry_kind!r}; add it to GEOMETRY_GROUPS"
        ) from None


def _fill_styles(points: list[EntropyPoint]) -> dict[str, str]:
    """Give every case a distinct fill within its (Lx, marker, color) bucket.

    Bucketing per Lx column keeps the fill meaning as narrow as possible: cases
    at different Lx are already separated along x, so the big-Lx sweep stays one
    consistently drawn series instead of cycling fills along its own line.
    """
    buckets: dict[tuple[int, str, str], list[str]] = {}
    for point in sorted(points, key=lambda point: point.case_id):
        buckets.setdefault(
            (point.lx_multiplier, point.geometry_group, point.interaction_class),
            [],
        ).append(point.key)
    fills: dict[str, str] = {}
    for (multiplier, group, interaction), keys in buckets.items():
        if len(keys) > len(FILL_STYLES):
            raise ValueError(
                f"{len(keys)} cases share Lx={multiplier}, geometry {group!r} "
                f"and interaction {interaction!r}, but only "
                f"{len(FILL_STYLES)} fill styles exist to separate them: {keys}"
            )
        for slot, key in enumerate(keys):
            fills[key] = FILL_STYLES[slot]
    return fills


def plot_entropy(points: list[EntropyPoint], variant_label: str, output: Path) -> None:
    if not points:
        raise ValueError("no spectral entropy points to plot")
    sns.set_theme(context="paper", style="ticks")
    palette = sns.color_palette("colorblind", 3)
    colors = {
        "interacting": palette[0],
        "non-interacting": palette[1],
        "tracer": palette[2],
    }
    markers = {
        "cylinder": "o",
        "cylinder_rattle": "s",
        "prism": "^",
        "sandwich": "v",
        "2d_plane": "D",
        "3d_bulk": "P",
        "tracer": "X",
    }
    fills = _fill_styles(points)
    # Jitter slots are taken within the Lx = 1 subset alone, so no two crowded
    # points can land on the same offset.
    crowded = sorted(point.key for point in points if point.lx_multiplier == 1)
    jitter_slots = {key: slot for slot, key in enumerate(crowded)}
    figure, axis = plt.subplots(figsize=(10.6, 5.6), constrained_layout=True)

    sweep = sorted(
        (point for point in points if point.family == "big_lx"),
        key=lambda point: point.lx_multiplier,
    )
    if len(sweep) > 1:
        axis.plot(
            [point.lx_multiplier for point in sweep],
            [point.entropy for point in sweep],
            color="0.55",
            lw=1.0,
            zorder=1,
        )

    handles = []
    for point in sorted(
        points,
        key=lambda point: (
            point.geometry_group,
            point.interaction_class,
            point.case_id,
        ),
    ):
        jitter = 0.0
        if point.lx_multiplier == 1 and LX_JITTER and len(crowded) > 1:
            fraction = jitter_slots[point.key] / (len(crowded) - 1)
            jitter = LX_JITTER * 2.0 * (fraction - 0.5)
        color = colors[point.interaction_class]
        style = {
            "marker": markers[point.geometry_group],
            "fillstyle": fills[point.key],
            "markerfacecolor": color,
            "markeredgecolor": color,
            "markeredgewidth": 1.3,
            "markersize": 8.0,
            "linestyle": "none",
        }
        axis.plot(
            [point.lx_multiplier * (2.0**jitter)],
            [point.entropy],
            zorder=2,
            **style,
        )
        handles.append(plt.Line2D([], [], label=point.case_id, **style))

    axis.legend(
        handles=handles,
        frameon=False,
        fontsize=7,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="marker: geometry, color: interaction, fill: case",
        title_fontsize=8,
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 2, 4, 8, 16], labels=["1", "2", "4", "8", "16"])
    axis.set_xlabel(r"axial length multiplier $L_x$")
    axis.set_ylabel(r"spectral entropy $H_{\mathrm{spec}}$")
    axis.set_title(rf"Spectral entropy of $L(0,\omega)$ — {variant_label}")
    axis.grid(axis="y", color="0.9", lw=0.7)
    sns.despine(ax=axis)
    _save_figure(figure, output)
