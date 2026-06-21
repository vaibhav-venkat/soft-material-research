from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import TwoSlopeNorm

try:
    from scipy.integrate import trapezoid
except ImportError:
    trapezoid = np.trapz

from .common import _write_full_view_html
from .config import ACTIVE_IMAGE_DIR, CYLINDER, CYLINDER_SIM
from .shear_types import ShearFluxDecomposition

def _field_component_values(field: np.ndarray, component: str) -> np.ndarray:
    if component == "magnitude":
        return np.linalg.norm(field, axis=1)
    component_index = {"x": 0, "r": 1, "theta": 2}[component]
    return field[:, component_index]


def _valid_plot_indices(
    decomposition: ShearFluxDecomposition,
    max_points: int,
) -> np.ndarray:
    valid = np.isfinite(decomposition.grid_coords).all(axis=1) & np.isfinite(decomposition.rho_density)
    valid &= decomposition.rho_density > 0.0
    indices = np.flatnonzero(valid)
    if len(indices) > max_points:
        indices = indices[np.linspace(0, len(indices) - 1, max_points).astype(np.int64)]
    return indices


def _trace_map_literal(trace_indices: dict[str, int]) -> str:
    return "{" + ",".join(f'"{key}": {value}' for key, value in trace_indices.items()) + "}"


def _select_options(values: tuple[str, ...], selected: str) -> str:
    return "\n".join(
        f'<option value="{value}"{" selected" if value == selected else ""}>{value}</option>'
        for value in values
    )


def _symmetric_color_limit(values: np.ndarray, percentile: float | None = None) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(np.abs(finite), percentile)) if percentile else float(np.max(np.abs(finite)))
    return 1.0 if np.isclose(limit, 0.0) else limit


def _shear_scene_layout(coords: np.ndarray) -> dict:
    return {
        "xaxis": {"title": "x", "range": [float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))]},
        "yaxis": {"title": "r", "range": [0.0, CYLINDER.cylinder_radius]},
        "zaxis": {"title": "theta", "range": [0.0, 2.0 * np.pi]},
        "aspectmode": "data",
    }


def _grid_marker_trace(
    coords: np.ndarray,
    values: np.ndarray,
    name: str,
    colorbar_title: str,
    visible: bool,
    cmin: float,
    cmax: float,
    colorscale: str,
    hover_label: str = "value",
) -> go.Scatter3d:
    return go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker={
            "size": 3,
            "color": values,
            "colorscale": colorscale,
            "cmin": cmin,
            "cmax": cmax,
            "opacity": 0.8,
            "colorbar": {"title": colorbar_title},
        },
        name=name,
        visible=visible,
        customdata=values,
        hovertemplate=(
            "x=%{x:.3g}<br>r=%{y:.3g}<br>theta=%{z:.3g}"
            f"<br>{hover_label}=%{{customdata:.3g}}<extra></extra>"
        ),
    )


def _grid_shape(decomposition: ShearFluxDecomposition) -> tuple[int, int, int]:
    return (
        len(decomposition.x_centers),
        len(decomposition.r_centers),
        len(decomposition.theta_centers),
    )


def _reshape_grid_field(decomposition: ShearFluxDecomposition, values: np.ndarray) -> np.ndarray:
    return np.asarray(values).reshape(_grid_shape(decomposition) + np.asarray(values).shape[1:])


def _radial_trapezoid(values: np.ndarray, radii: np.ndarray, include_jacobian: bool) -> np.ndarray:
    integrand = values
    if include_jacobian:
        shape = (1, len(radii)) + (1,) * (values.ndim - 2)
        integrand = values * radii.reshape(shape)
    return trapezoid(integrand, x=radii, axis=1)


def _direct_radial_j_integral(decomposition: ShearFluxDecomposition) -> np.ndarray:
    values = _reshape_grid_field(decomposition, decomposition.j_total_with_wall)
    return _radial_trapezoid(values, decomposition.r_centers, include_jacobian=True)


def _formula_radial_j_integral(decomposition: ShearFluxDecomposition) -> np.ndarray:
    radii = decomposition.r_centers
    sigma = _reshape_grid_field(decomposition, decomposition.sigma_full)
    deriv_sigma = _reshape_grid_field(decomposition, decomposition.deriv_sigma_full)
    nonstress_j = _reshape_grid_field(
        decomposition,
        decomposition.j_active + decomposition.j_wall,
    )

    def sigma_int(a: int, b: int, include_jacobian: bool) -> np.ndarray:
        return _radial_trapezoid(sigma[..., a, b], radii, include_jacobian)

    def deriv_int(direction: int, a: int, b: int, include_jacobian: bool) -> np.ndarray:
        return _radial_trapezoid(deriv_sigma[..., direction, a, b], radii, include_jacobian)

    def radial_boundary(a: int, b: int) -> np.ndarray:
        return radii[-1] * sigma[:, -1, :, a, b] - radii[0] * sigma[:, 0, :, a, b]

    nx, _, ntheta = _grid_shape(decomposition)
    div_integral = np.empty((nx, ntheta, 3), dtype=np.float64)
    div_integral[..., 0] = (
        deriv_int(0, 0, 0, True)
        + deriv_int(2, 0, 2, False)
        + radial_boundary(0, 1)
    )
    div_integral[..., 1] = (
        deriv_int(0, 1, 0, True)
        + deriv_int(2, 1, 2, False)
        + radial_boundary(1, 1)
        - sigma_int(2, 2, False)
    )
    div_integral[..., 2] = (
        deriv_int(0, 2, 0, True)
        + deriv_int(2, 2, 2, False)
        + radial_boundary(2, 1)
        + sigma_int(1, 2, False)
    )

    return (
        _radial_trapezoid(nonstress_j, radii, include_jacobian=True)
        + div_integral / CYLINDER_SIM.gamma
    )


def _plot_radial_j_integral_components(
    decomposition: ShearFluxDecomposition,
    values: np.ndarray,
    filename: Path,
    title: str,
) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    components = (("x", 0), ("theta", 2), ("r", 1))
    limit = _symmetric_color_limit(values)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, (label, component_index) in zip(axes, components):
        mesh = axis.pcolormesh(
            decomposition.x_edges,
            decomposition.theta_edges,
            values[..., component_index].T,
            shading="auto",
            cmap="coolwarm",
            norm=norm,
        )
        fig.colorbar(mesh, ax=axis, label=rf"$\int r J_{label}\,dr$")
        axis.set_xlabel("x")
        axis.set_title(f"{label} component")
    axes[0].set_ylabel("theta")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def plot_radial_j_integral_comparison(
    decomposition: ShearFluxDecomposition,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
) -> None:
    image_path = Path(image_dir) / "shear"
    _plot_radial_j_integral_components(
        decomposition,
        _direct_radial_j_integral(decomposition),
        image_path / "hardy_j_radial_integral_direct.png",
        (
            "Direct radial integral of "
            r"$J_\mathrm{total}=U_0P+\gamma^{-1}\nabla\cdot\sigma+J_\mathrm{wall}$"
        ),
    )
    _plot_radial_j_integral_components(
        decomposition,
        _formula_radial_j_integral(decomposition),
        image_path / "hardy_j_radial_integral_formula.png",
        "Radial integral from integrated stress-divergence identities plus active and wall terms",
    )


def plot_shear_flux_decomposition(
    decomposition: ShearFluxDecomposition,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
    max_points: int = 9000,
) -> None:
    output_path = Path(image_dir) / "shear" / "hardy_shear_flux_decomposition.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = {
        "J_active": decomposition.j_active,
        "J_normal": decomposition.j_normal,
        "J_shear": decomposition.j_shear,
        "J_wall": decomposition.j_wall,
        "J_pair_total": decomposition.j_total,
        "J_total_with_wall": decomposition.j_total_with_wall,
        "J_force_baseline": decomposition.j_force_baseline,
    }
    components = ("x", "r", "theta", "magnitude")
    indices = _valid_plot_indices(decomposition, max_points)

    if len(indices) == 0:
        fig = go.Figure()
        fig.update_layout(title="flux decomposition: no occupied grid points")
        fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
        return

    coords = decomposition.grid_coords[indices]
    traces = []
    trace_indices: dict[str, int] = {}
    for field_name, field_values in fields.items():
        for component in components:
            key = f"{field_name}:{component}"
            trace_indices[key] = len(traces)
            values = _field_component_values(field_values[indices], component)
            finite = values[np.isfinite(values)]
            if component == "magnitude":
                cmin = 0.0
                cmax = float(np.max(finite)) if finite.size else 1.0
                colorscale = "Plasma"
            else:
                limit = _symmetric_color_limit(values)
                cmin = -limit
                cmax = limit
                colorscale = "RdBu"
            traces.append(
                _grid_marker_trace(
                    coords,
                    values,
                    name=key,
                    colorbar_title=component,
                    visible=(field_name == "J_total_with_wall" and component == "magnitude"),
                    cmin=cmin,
                    cmax=cmax,
                    colorscale=colorscale,
                )
            )

    labels = {
        "J_active": "active",
        "J_normal": "normal stress",
        "J_shear": "shear stress",
        "J_wall": "wall force",
        "J_pair_total": "active + normal + shear",
        "J_total_with_wall": "active + normal + shear + wall",
        "J_force_baseline": "force-density baseline",
    }
    field_options = "\n".join(
        f'<option value="{name}"{" selected" if name == "J_total_with_wall" else ""}>{label}</option>'
        for name, label in labels.items()
    )
    component_options = _select_options(components, "magnitude")
    trace_map = _trace_map_literal(trace_indices)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Flux decomposition: J_total_with_wall magnitude, step {decomposition.step}",
        scene=_shear_scene_layout(coords),
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
    )

    div_id = "hardy-shear-flux-decomposition-plot"
    controls = f"""
<div class="controls">
  <label for="shear-field">Field:&nbsp;</label>
  <select id="shear-field">{field_options}</select>
  <label for="shear-component" style="margin-left: 18px;">Component:&nbsp;</label>
  <select id="shear-component">{component_options}</select>
</div>
"""
    script = f"""
<script>
const shearTraceMap = {trace_map};
const shearTraceCount = {len(traces)};
const shearStep = {decomposition.step};
function updateShearTrace() {{
  const field = document.getElementById("shear-field").value;
  const component = document.getElementById("shear-component").value;
  const key = field + ":" + component;
  const visible = Array(shearTraceCount).fill(false);
  visible[shearTraceMap[key]] = true;
  Plotly.restyle("{div_id}", {{"visible": visible}});
  Plotly.relayout(
    "{div_id}",
    {{"title.text": "Flux decomposition: "
      + field + " " + component + ", step " + shearStep}}
  );
}}
document.getElementById("shear-field").addEventListener("change", updateShearTrace);
document.getElementById("shear-component").addEventListener("change", updateShearTrace);
</script>
"""
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id=div_id,
        default_width="100%",
        default_height="100%",
    )
    _write_full_view_html(output_path, div_id, controls, plot_html, script)


def plot_shear_stress_tensor_components(
    decomposition: ShearFluxDecomposition,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
    max_points: int = 9000,
) -> None:
    output_path = Path(image_dir) / "shear" / "hardy_sigma_shear_tensor.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ("x", "r", "theta")
    indices = _valid_plot_indices(decomposition, max_points)
    if len(indices) == 0:
        fig = go.Figure()
        fig.update_layout(title="Hardy stress tensor: no occupied grid points")
        fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
        return

    coords = decomposition.grid_coords[indices]
    traces = []
    trace_indices: dict[str, int] = {}
    for component_index, component_label in enumerate(labels):
        for normal_index, normal_label in enumerate(labels):
            key = f"{component_label}:{normal_label}"
            trace_indices[key] = len(traces)
            values = decomposition.sigma_full[indices, component_index, normal_index]
            limit = _symmetric_color_limit(values)
            traces.append(
                _grid_marker_trace(
                    coords,
                    values,
                    name=f"sigma_{component_label}{normal_label}",
                    visible=(component_label == "x" and normal_label == "r"),
                    colorbar_title="sigma",
                    cmin=-limit,
                    cmax=limit,
                    colorscale="RdBu",
                    hover_label="sigma",
                )
            )

    options = _select_options(labels, "x")
    normal_options = _select_options(labels, "r")
    trace_map = _trace_map_literal(trace_indices)
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Hardy stress tensor: sigma_xr, step {decomposition.step}",
        scene=_shear_scene_layout(coords),
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
    )

    div_id = "hardy-sigma-shear-tensor-plot"
    controls = f"""
<div class="controls">
  <label for="sigma-component">Component a:&nbsp;</label>
  <select id="sigma-component">{options}</select>
  <label for="sigma-normal" style="margin-left: 18px;">Normal b:&nbsp;</label>
  <select id="sigma-normal">{normal_options}</select>
</div>
"""
    script = f"""
<script>
const sigmaTraceMap = {trace_map};
const sigmaTraceCount = {len(traces)};
const sigmaStep = {decomposition.step};
function updateSigmaTrace() {{
  const component = document.getElementById("sigma-component").value;
  const normal = document.getElementById("sigma-normal").value;
  const key = component + ":" + normal;
  const visible = Array(sigmaTraceCount).fill(false);
  visible[sigmaTraceMap[key]] = true;
  Plotly.restyle("{div_id}", {{"visible": visible}});
  Plotly.relayout(
    "{div_id}",
    {{"title.text": "Hardy stress tensor: sigma_" + component + normal + ", step " + sigmaStep}}
  );
}}
document.getElementById("sigma-component").addEventListener("change", updateSigmaTrace);
document.getElementById("sigma-normal").addEventListener("change", updateSigmaTrace);
</script>
"""
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id=div_id,
        default_width="100%",
        default_height="100%",
    )
    _write_full_view_html(output_path, div_id, controls, plot_html, script)


def plot_shear_flux_fraction(
    decomposition: ShearFluxDecomposition,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
    max_points: int = 9000,
) -> None:
    output_path = Path(image_dir) / "shear" / "hardy_j_shear_fraction.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    components = ("x", "r", "theta", "magnitude")
    indices = _valid_plot_indices(decomposition, max_points)
    if len(indices) == 0:
        fig = go.Figure()
        fig.update_layout(title="J_shear/J_total: no occupied grid points")
        fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
        return

    coords = decomposition.grid_coords[indices]
    shear = decomposition.j_shear[indices]
    total = decomposition.j_total[indices]
    traces = []
    trace_indices: dict[str, int] = {}
    fraction_means: dict[str, float] = {}
    for component in components:
        if component == "magnitude":
            numerator = np.linalg.norm(shear, axis=1)
            denominator = np.linalg.norm(total, axis=1)
        else:
            component_index = {"x": 0, "r": 1, "theta": 2}[component]
            numerator = shear[:, component_index]
            denominator = total[:, component_index]
        values = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=np.float64),
            where=np.abs(denominator) > 1e-12,
        )
        finite = values[np.isfinite(values)]
        fraction_means[component] = float(np.mean(finite)) if finite.size else float("nan")
        if component == "magnitude":
            cmin = 0.0
            cmax = float(np.nanpercentile(finite, 99.0)) if finite.size else 1.0
            if np.isclose(cmax, 0.0):
                cmax = 1.0
            colorscale = "Plasma"
        else:
            limit = _symmetric_color_limit(values, percentile=99.0)
            cmin = -limit
            cmax = limit
            colorscale = "RdBu"
        trace_indices[component] = len(traces)
        traces.append(
            _grid_marker_trace(
                coords,
                values,
                name=f"J_shear/J_total {component}",
                visible=(component == "magnitude"),
                colorbar_title="J_shear/J_total",
                cmin=cmin,
                cmax=cmax,
                colorscale=colorscale,
                hover_label="fraction",
            )
        )

    component_options = _select_options(components, "magnitude")
    trace_map = _trace_map_literal(trace_indices)
    mean_map = "{" + ",".join(
        f'"{key}": {value:.17g}' if np.isfinite(value) else f'"{key}": NaN'
        for key, value in fraction_means.items()
    ) + "}"
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Hardy J_shear / (J_active + J_normal + J_shear): magnitude, step {decomposition.step}",
        scene=_shear_scene_layout(coords),
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
    )

    div_id = "hardy-j-shear-fraction-plot"
    controls = f"""
<div class="controls">
  <label for="fraction-component">Component:&nbsp;</label>
  <select id="fraction-component">{component_options}</select>
  <span id="fraction-mean" style="margin-left: 18px;">mean = {fraction_means["magnitude"]:.6g}</span>
</div>
"""
    script = f"""
<script>
const fractionTraceMap = {trace_map};
const fractionMeanMap = {mean_map};
const fractionTraceCount = {len(traces)};
const fractionStep = {decomposition.step};
function updateFractionTrace() {{
  const component = document.getElementById("fraction-component").value;
  const visible = Array(fractionTraceCount).fill(false);
  visible[fractionTraceMap[component]] = true;
  Plotly.restyle("{div_id}", {{"visible": visible}});
  Plotly.relayout(
    "{div_id}",
    {{"title.text": "Hardy J_shear / (J_active + J_normal + J_shear): " + component + ", step " + fractionStep}}
  );
  const meanValue = fractionMeanMap[component];
  document.getElementById("fraction-mean").textContent =
    Number.isFinite(meanValue) ? "mean = " + meanValue.toPrecision(6) : "mean = NaN";
}}
document.getElementById("fraction-component").addEventListener("change", updateFractionTrace);
</script>
"""
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id=div_id,
        default_width="100%",
        default_height="100%",
    )
    _write_full_view_html(output_path, div_id, controls, plot_html, script)
