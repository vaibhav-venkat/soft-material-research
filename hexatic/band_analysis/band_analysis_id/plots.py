"""Seaborn SVG plots of identified band areas and lifetimes."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .statistics import (
    BandStatistics,
    binned_moment,
    probability_points,
    survival_points,
)


logger = logging.getLogger(__name__)


def _points(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    label: str,
    marker: str = "o",
) -> None:
    sns.scatterplot(
        x=x,
        y=y,
        color=color,
        marker=marker,
        s=48,
        facecolor="none",
        edgecolor=color,
        linewidth=1.4,
        label=label,
        ax=axis,
    )


def plot_area_dynamics(stats: BandStatistics, output_dir: Path) -> Path:
    if stats.delta_areas.size == 0:
        raise ValueError("no tracked band survives for the requested time increment")
    sns.set_theme(style="ticks")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    logger.info("plotting joint band-area and area-change distribution")
    sns.histplot(
        x=stats.delta_start_areas,
        y=stats.delta_areas,
        bins=40,
        stat="probability",
        cbar=True,
        cmap="viridis",
        ax=axes[0],
    )
    axes[0].set(xlabel=r"$A/\sigma^2$", ylabel=r"$\Delta A/\sigma^2$")

    logger.info("plotting conditional area-change moments")
    area, first = binned_moment(stats.delta_start_areas, stats.delta_areas)
    _, second = binned_moment(stats.delta_start_areas, stats.delta_areas**2)
    _points(axes[1], area, first, "red", r"$\langle\Delta A\rangle$")
    axes[1].set(xlabel=r"$A/\sigma^2$", ylabel=r"$\langle\Delta A\rangle/\sigma^2$")
    second_axis = axes[1].twinx()
    _points(
        second_axis,
        area,
        second / np.sqrt(area),
        "blue",
        r"$\langle\Delta A^2\rangle/\sqrt{A}$",
    )
    second_axis.set_ylabel(r"$\langle\Delta A^2\rangle/\sqrt{A}$")
    axes[1].get_legend().remove()
    second_axis.get_legend().remove()
    axes[1].legend(
        [axes[1].collections[0], second_axis.collections[0]],
        [
            r"$\langle\Delta A\rangle$",
            r"$\langle\Delta A^2\rangle/\sqrt{A}$",
        ],
    )
    path = output_dir / "band_area_dynamics.svg"
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)
    logger.info("saved area-dynamics plots: %s", path)
    return path


def plot_lifetimes(stats: BandStatistics, output_dir: Path) -> Path:
    if stats.lifetimes.size == 0:
        raise ValueError("no complete persistent band lifetimes were observed")
    sns.set_theme(style="ticks")
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    logger.info("plotting stationary band-area distribution")
    area, probability = probability_points(stats.areas)
    _points(axes[0, 0], area, probability, "red", "bands")
    axes[0, 0].set(xlabel=r"$A/\sigma^2$", ylabel=r"$P(A)$")
    axes[0, 0].get_legend().remove()

    logger.info("plotting mean first-passage time by band area")
    area, passage = binned_moment(stats.passage_areas, stats.passage_times)
    _points(axes[0, 1], area, passage, "red", "disappearance")
    axes[0, 1].set(xlabel=r"$A/\sigma^2$", ylabel=r"$T(A)$")
    axes[0, 1].get_legend().remove()

    logger.info("plotting lifetime distributions by birth-area quartile")
    quartiles = np.quantile(stats.initial_areas, [0.25, 0.5, 0.75])
    groups = np.digitize(stats.initial_areas, quartiles)
    colors = sns.color_palette("colorblind", 4)
    for group, color in enumerate(colors):
        selected = groups == group
        if not np.any(selected):
            continue
        lifetime, probability = probability_points(stats.lifetimes[selected], bins=20)
        representative = np.median(stats.initial_areas[selected])
        _points(
            axes[1, 0],
            lifetime,
            probability,
            color,
            rf"$A_0={representative:.2f}\,\sigma^2$",
        )
    axes[1, 0].set(xlabel=r"lifetime $\tau$", ylabel=r"$P(\tau\mid A_0)$")

    logger.info("plotting disappearance lifetime distribution and survival")
    lifetime, probability = probability_points(stats.lifetimes)
    _points(axes[1, 1], lifetime, probability, "red", r"$P(\tau)$")
    axes[1, 1].set(xlabel=r"lifetime $\tau$", ylabel=r"$P(\tau)$")
    axes[1, 1].set_yscale("log")
    survival_axis = axes[1, 1].twinx()
    tau, survival = survival_points(stats.lifetimes)
    positive = survival > 0.0
    _points(survival_axis, tau[positive], survival[positive], "blue", r"$S(\tau)$")
    survival_axis.set_ylabel(r"$S(\tau)=P(T>\tau)$")
    survival_axis.set_yscale("log")
    handles_left, labels_left = axes[1, 1].get_legend_handles_labels()
    handles_right, labels_right = survival_axis.get_legend_handles_labels()
    axes[1, 1].get_legend().remove()
    survival_axis.get_legend().remove()
    axes[1, 1].legend(handles_left + handles_right, labels_left + labels_right)

    path = output_dir / "band_lifetimes.svg"
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)
    logger.info("saved lifetime plots: %s", path)
    return path


def _topology_lifetime_plot(
    axis: plt.Axes,
    values: np.ndarray,
    marker: str,
    event: str,
) -> None:
    lifetime, probability = probability_points(values)
    _points(axis, lifetime, probability, "red", rf"$P_{{{event}}}(\tau)$", marker)
    axis.set(xlabel=r"lifetime $\tau$", ylabel=rf"$P_{{{event}}}(\tau)$")
    axis.set_yscale("log")
    survival_axis = axis.twinx()
    tau, survival = survival_points(values)
    positive = survival > 0.0
    _points(
        survival_axis,
        tau[positive],
        survival[positive],
        "blue",
        rf"$S_{{{event}}}(\tau)$",
        marker,
    )
    survival_axis.set_ylabel(rf"$S_{{{event}}}(\tau)$")
    survival_axis.set_yscale("log")
    left_handles, left_labels = axis.get_legend_handles_labels()
    right_handles, right_labels = survival_axis.get_legend_handles_labels()
    if legend := axis.get_legend():
        legend.remove()
    if legend := survival_axis.get_legend():
        legend.remove()
    axis.legend(left_handles + right_handles, left_labels + right_labels)


def plot_topology_events(stats: BandStatistics, output_dir: Path) -> Path:
    sns.set_theme(style="ticks")
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    logger.info("plotting split first-passage time")
    if stats.split_passage_areas.size:
        area, passage = binned_moment(
            stats.split_passage_areas, stats.split_passage_times
        )
        _points(axes[0, 0], area, passage, "red", "split", "^")
        axes[0, 0].get_legend().remove()
    axes[0, 0].set(xlabel=r"$A/\sigma^2$", ylabel=r"$T_{split}(A)$")

    logger.info("plotting merge first-passage time")
    if stats.merge_passage_areas.size:
        area, passage = binned_moment(
            stats.merge_passage_areas, stats.merge_passage_times
        )
        _points(axes[0, 1], area, passage, "red", "merge", "s")
        axes[0, 1].get_legend().remove()
    axes[0, 1].set(xlabel=r"$A/\sigma^2$", ylabel=r"$T_{merge}(A)$")

    logger.info("plotting split lifetime distribution and survival")
    if stats.split_lifetimes.size:
        _topology_lifetime_plot(axes[1, 0], stats.split_lifetimes, "^", "split")
    logger.info("plotting merge lifetime distribution and survival")
    if stats.merge_lifetimes.size:
        _topology_lifetime_plot(axes[1, 1], stats.merge_lifetimes, "s", "merge")

    path = output_dir / "band_topology_events.svg"
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)
    logger.info("saved topology-event plots: %s", path)
    return path
