import numpy as np
from matplotlib.ticker import FuncFormatter

from .config import (
    ACTIVE_FLUX_PLOT_THETA_BINS,
    ACTIVE_FLUX_PLOT_X_BINS,
    ACTIVE_RADIAL_BIN_WIDTH,
    CYLINDER,
    ActiveMatterFields,
)

def _x_edges_and_centers(box_length_x: float, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    assert box_length_x > 0.0
    assert n_bins > 0
    edges = np.linspace(-0.5 * box_length_x, 0.5 * box_length_x, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _theta_edges_and_centers(n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    assert n_bins > 0
    edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _x_bin_indices(x_positions: np.ndarray, box_length_x: float, n_bins: int) -> np.ndarray:
    wrapped = np.mod(x_positions + 0.5 * box_length_x, box_length_x)
    indices = np.floor(wrapped / box_length_x * n_bins).astype(np.int64)
    return np.clip(indices, 0, n_bins - 1)


def _theta_bin_indices(theta: np.ndarray, n_bins: int) -> np.ndarray:
    wrapped = np.mod(theta, 2.0 * np.pi)
    indices = np.floor(wrapped / (2.0 * np.pi) * n_bins).astype(np.int64)
    return np.clip(indices, 0, n_bins - 1)

def _radial_integral_mean(
    coords: np.ndarray,
    values: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    box_length_x: float,
    radial_bin_width: float = ACTIVE_RADIAL_BIN_WIDTH,
    average_particles: bool = True,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    x_bins = len(x_edges) - 1
    theta_bins = len(theta_edges) - 1
    assert radial_bin_width > 0.0

    radial_min = 0.0
    radial_max = CYLINDER.cylinder_radius
    radial_span = radial_max - radial_min
    assert radial_span > 0.0
    n_radial_bins = int(np.ceil(radial_span / radial_bin_width))
    radial_edges = radial_min + radial_bin_width * np.arange(n_radial_bins + 1)
    radial_edges[-1] = radial_max
    radial_widths = np.diff(radial_edges)

    valid = (coords[:, 2] >= radial_min) & (coords[:, 2] <= radial_max)
    coords = coords[valid]
    values = values[valid]

    x_indices = _x_bin_indices(coords[:, 0], box_length_x, x_bins)
    theta_indices = _theta_bin_indices(coords[:, 1], theta_bins)
    radial_indices = np.searchsorted(radial_edges, coords[:, 2], side="right") - 1
    radial_indices = np.clip(radial_indices, 0, n_radial_bins - 1)
    counts = np.zeros((x_bins, theta_bins, n_radial_bins), dtype=np.float64)
    np.add.at(counts, (x_indices, theta_indices, radial_indices), 1.0)

    if values.ndim == 1:
        sums = np.zeros((x_bins, theta_bins, n_radial_bins), dtype=np.float64)
        np.add.at(sums, (x_indices, theta_indices, radial_indices), values)
        radial_values = sums
        if average_particles:
            radial_values = np.divide(
                sums,
                counts,
                out=np.zeros_like(sums),
                where=counts > 0,
            )
        return np.sum(radial_values * radial_widths, axis=2) / radial_span

    sums = np.zeros(
        (x_bins, theta_bins, n_radial_bins, values.shape[1]),
        dtype=np.float64,
    )
    for component in range(values.shape[1]):
        np.add.at(
            sums[..., component],
            (x_indices, theta_indices, radial_indices),
            values[:, component],
        )
    radial_values = sums
    if average_particles:
        radial_values = np.divide(
            sums,
            counts[..., np.newaxis],
            out=np.zeros_like(sums),
            where=counts[..., np.newaxis] > 0,
        )
    return np.sum(radial_values * radial_widths[:, np.newaxis], axis=2) / radial_span


def _xytheta_mean(
    coords: np.ndarray,
    values: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    box_length_x: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    x_bins = len(x_edges) - 1
    theta_bins = len(theta_edges) - 1
    x_indices = _x_bin_indices(coords[:, 0], box_length_x, x_bins)
    theta_indices = _theta_bin_indices(coords[:, 1], theta_bins)
    counts = np.zeros((x_bins, theta_bins), dtype=np.float64)
    np.add.at(counts, (x_indices, theta_indices), 1.0)

    if values.ndim == 1:
        sums = np.zeros((x_bins, theta_bins), dtype=np.float64)
        np.add.at(sums, (x_indices, theta_indices), values)
        return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    sums = np.zeros((x_bins, theta_bins, values.shape[1]), dtype=np.float64)
    for component in range(values.shape[1]):
        np.add.at(sums[..., component], (x_indices, theta_indices), values[:, component])
    return np.divide(
        sums,
        counts[..., np.newaxis],
        out=np.zeros_like(sums),
        where=counts[..., np.newaxis] > 0,
    )


def _xytheta_occupied(
    coords: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    box_length_x: float,
) -> np.ndarray:
    x_bins = len(x_edges) - 1
    theta_bins = len(theta_edges) - 1
    x_indices = _x_bin_indices(coords[:, 0], box_length_x, x_bins)
    theta_indices = _theta_bin_indices(coords[:, 1], theta_bins)
    counts = np.zeros((x_bins, theta_bins), dtype=np.int64)
    np.add.at(counts, (x_indices, theta_indices), 1)
    return counts > 0

def _color_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0

    vmin, vmax = np.percentile(finite, [2.0, 98.0])
    if np.isclose(vmin, vmax):
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def _format_theta_axis(axis) -> None:
    axis.set_ylabel("theta")
    axis.set_ylim(0.0, 2.0 * np.pi)
    axis.set_yticks(
        [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi]
    )
    axis.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: {
                0.0: "0",
                0.5: r"$\pi/2$",
                1.0: r"$\pi$",
                1.5: r"$3\pi/2$",
                2.0: r"$2\pi$",
            }.get(round(value / np.pi, 1), "")
        )
    )


def _fixed_length_quiver_components(
    x_components: np.ndarray,
    theta_components: np.ndarray,
    x_span: float,
    n_x_bins: int,
    n_theta_bins: int,
    theta_span: float = 2.0 * np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    x_fraction = x_components / x_span
    theta_fraction = theta_components / theta_span
    lengths = np.hypot(x_fraction, theta_fraction)
    arrow_fraction = 0.35 * min(1.0 / n_x_bins, 1.0 / n_theta_bins)

    unit_x = np.divide(
        x_fraction,
        lengths,
        out=np.zeros_like(x_fraction, dtype=np.float64),
        where=lengths > 0.0,
    )
    unit_theta = np.divide(
        theta_fraction,
        lengths,
        out=np.zeros_like(theta_fraction, dtype=np.float64),
        where=lengths > 0.0,
    )
    return unit_x * x_span * arrow_fraction, unit_theta * theta_span * arrow_fraction


def _frame_index(index: int, n_frames: int) -> int:
    if index < 0:
        index += n_frames
    assert 0 <= index < n_frames
    return index


def _coarse_vector_density_grid(
    fields: ActiveMatterFields,
    vectors: np.ndarray,
    frame_idx: int,
    shell_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x_span = fields.x_edges[-1] - fields.x_edges[0]
    x_edges, x_centers = _x_edges_and_centers(x_span, ACTIVE_FLUX_PLOT_X_BINS)
    theta_edges, theta_centers = _theta_edges_and_centers(ACTIVE_FLUX_PLOT_THETA_BINS)
    mask = fields.shell_mask[frame_idx] if shell_only else np.ones(
        fields.coords.shape[1],
        dtype=np.bool_,
    )
    if shell_only:
        vector_grid = _xytheta_mean(
            fields.coords[frame_idx, mask],
            vectors[frame_idx, mask],
            x_edges,
            theta_edges,
            x_span,
        )
    else:
        vector_grid = _radial_integral_mean(
            fields.coords[frame_idx],
            vectors[frame_idx],
            x_edges,
            theta_edges,
            x_span,
        )
    x_grid, theta_grid = np.meshgrid(x_centers, theta_centers, indexing="ij")
    return x_grid, theta_grid, vector_grid, x_span

def _time_edges(steps: np.ndarray) -> np.ndarray:
    steps = np.asarray(steps, dtype=np.float64)
    if len(steps) == 1:
        return np.asarray([steps[0] - 0.5, steps[0] + 0.5], dtype=np.float64)

    mids = 0.5 * (steps[:-1] + steps[1:])
    edges = np.empty(len(steps) + 1, dtype=np.float64)
    edges[1:-1] = mids
    edges[0] = steps[0] - (mids[0] - steps[0])
    edges[-1] = steps[-1] + (steps[-1] - mids[-1])
    return edges


def _write_full_view_html(output_path, div_id: str, controls: str, plot_html: str, script: str) -> None:
    output_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <style>
    html, body {{
      height: 100%;
      width: 100%;
      margin: 0;
      overflow: hidden;
    }}
    body {{
      display: flex;
      flex-direction: column;
      font-family: sans-serif;
    }}
    .controls {{
      flex: 0 0 auto;
      padding: 10px 12px;
    }}
    #{div_id} {{
      flex: 1 1 auto;
      width: 100% !important;
      height: calc(100vh - 48px) !important;
    }}
  </style>
</head>
<body>{controls}{plot_html}{script}</body>
</html>"""
    )
