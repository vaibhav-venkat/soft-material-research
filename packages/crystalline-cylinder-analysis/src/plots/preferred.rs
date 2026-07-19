//! Kuva preferred-coordinate summary interface.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use crystalline_cylinder_analysis::model::{PreferredAxis, PreferredEstimate};
use crystalline_cylinder_analysis::pipeline::CaseAnalysis;
use crystalline_cylinder_analysis::CaseSchema;
use kuva::backend::svg::SvgBackend;
use kuva::plot::scatter::{MarkerShape, ScatterPlot};
use kuva::plot::LinePlot;
use kuva::render::figure::Figure;
use kuva::render::layout::Layout;
use kuva::render::plots::Plot;

/// Write preferred-coordinate summaries versus axial-length multiplier.
pub fn write_preferred_plot(
    analyses: &[CaseAnalysis],
    axis: PreferredAxis,
    output: &Path,
) -> PathBuf {
    assert!(!analyses.is_empty(), "no preferred estimates to plot");
    assert_eq!(
        output.extension().and_then(|suffix| suffix.to_str()),
        Some("svg"),
        "preferred output must be SVG"
    );
    let family_colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44"];
    let confinement_colors = ["#AA3377", "#44AA99", "#882255", "#999933", "#332288"];
    let mut plots = Vec::new();

    let mut circumferences = analyses
        .iter()
        .filter(|analysis| analysis.group.schema == CaseSchema::BigLx)
        .filter_map(|analysis| analysis.group.case.circumference_diameters)
        .collect::<Vec<_>>();
    circumferences.sort_by(f64::total_cmp);
    circumferences.dedup_by(|left, right| left.to_bits() == right.to_bits());
    for (index, circumference) in circumferences.iter().enumerate() {
        let family = analyses
            .iter()
            .filter(|analysis| {
                analysis.group.schema == CaseSchema::BigLx
                    && analysis
                        .group
                        .case
                        .circumference_diameters
                        .map(f64::to_bits)
                        == Some(circumference.to_bits())
            })
            .collect::<Vec<_>>();
        let color = family_colors[index % family_colors.len()];
        let points = family
            .iter()
            .map(|analysis| {
                (
                    multiplier_position(analysis.group.case.lx_multiplier),
                    estimate(analysis, axis).coordinate,
                )
            })
            .collect::<Vec<_>>();
        let errors = family
            .iter()
            .map(|analysis| estimate(analysis, axis).coordinate_std)
            .collect::<Vec<_>>();
        plots.push(Plot::Line(
            LinePlot::new()
                .with_data(points.iter().copied())
                .with_color(color)
                .with_stroke_width(1.7),
        ));
        plots.push(Plot::Scatter(
            ScatterPlot::new()
                .with_data(points)
                .with_y_err(errors)
                .with_color(color)
                .with_size(5.5)
                .with_marker_stroke_width(0.8)
                .with_legend(format!("regular cylinder, C = {circumference}D")),
        ));
    }

    for (index, analysis) in analyses
        .iter()
        .filter(|analysis| analysis.group.schema == CaseSchema::Confinement)
        .enumerate()
    {
        let preferred = estimate(analysis, axis);
        let label = analysis
            .group
            .case
            .label
            .as_deref()
            .unwrap_or(&analysis.group.case.case_id);
        plots.push(Plot::Scatter(
            ScatterPlot::new()
                .with_data([(
                    multiplier_position(analysis.group.case.lx_multiplier),
                    preferred.coordinate,
                )])
                .with_y_err([preferred.coordinate_std])
                .with_color(confinement_colors[index % confinement_colors.len()])
                .with_size(7.0)
                .with_marker(confinement_marker(
                    analysis.group.case.geometry_kind.as_deref(),
                ))
                .with_marker_stroke_width(0.9)
                .with_legend(label),
        ));
    }
    assert!(!plots.is_empty(), "no preferred estimates to plot");

    if axis == PreferredAxis::R {
        plots.insert(
            0,
            Plot::Line(
                LinePlot::new()
                    .with_data([(0.75, 0.0), (5.25, 0.0)])
                    .with_color("#555555")
                    .with_stroke_width(1.0)
                    .with_dashed(),
            ),
        );
    }
    let (title, y_label, figure_title) = match axis {
        PreferredAxis::Omega => (
            "Maximum at r = 0 over positive frequencies",
            "preferred omega*",
            "Preferred angular frequency of the axial velocity correlation",
        ),
        PreferredAxis::R => (
            "Maximum at omega = 0 over negative real coordinates",
            "preferred r*",
            "Preferred real coordinate of the axial velocity correlation",
        ),
    };
    let layout = Layout::auto_from_plots(&plots)
        .with_x_axis_min(0.75)
        .with_x_axis_max(5.25)
        .with_x_categories(
            ["1x", "2x", "4x", "8x", "16x"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        )
        .with_title(title)
        .with_x_label("axial length multiplier")
        .with_y_label(y_label);
    let scene = Figure::new(1, 1)
        .with_plots(vec![plots])
        .with_layouts(vec![layout])
        .with_title(figure_title)
        .with_shared_legend_right_middle()
        .with_figure_size(1300.0, 760.0)
        .render();
    let svg = SvgBackend::new()
        .with_pretty(true)
        .with_embedded_font(true)
        .render_scene(&scene);
    write_atomic(output, svg.as_bytes());
    output.to_path_buf()
}

fn estimate(analysis: &CaseAnalysis, axis: PreferredAxis) -> &PreferredEstimate {
    analysis
        .preferred
        .iter()
        .find(|estimate| estimate.axis == axis)
        .expect("analysis has no preferred estimate")
}

fn multiplier_position(multiplier: i64) -> f64 {
    let multiplier = u64::try_from(multiplier).expect("multiplier must be positive");
    assert!(
        multiplier.is_power_of_two(),
        "multiplier must be a power of two"
    );
    multiplier.ilog2() as f64 + 1.0
}

fn confinement_marker(geometry: Option<&str>) -> MarkerShape {
    match geometry {
        Some("sandwich_volume") => MarkerShape::Diamond,
        Some("sandwich_surface_area") => MarkerShape::Square,
        Some("two_dimension") => MarkerShape::Triangle,
        Some("cylinder_rattle") => MarkerShape::Cross,
        Some("cylinder_rattle_tangent") => MarkerShape::Plus,
        _ => MarkerShape::Circle,
    }
}

fn write_atomic(output: &Path, contents: &[u8]) {
    let parent = output.parent().expect("output has no parent");
    std::fs::create_dir_all(parent).expect("create output directory");
    let file_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .expect("output filename is invalid");
    let temporary = parent.join(format!(".{file_name}.{}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .expect("open temporary output");
    file.write_all(contents).expect("write temporary output");
    file.sync_all().expect("sync temporary output");
    drop(file);
    std::fs::rename(&temporary, output).expect("move output");
}
