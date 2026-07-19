//! Kuva damped-cosine fit panels.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use crystalline_cylinder_analysis::pipeline::CaseAnalysis;
use kuva::backend::svg::SvgBackend;
use kuva::plot::{LegendPosition, LinePlot};
use kuva::render::figure::Figure;
use kuva::render::layout::Layout;
use kuva::render::plots::Plot;

/// Write measured-correlation and fitted-model overlays.
pub fn write_fit_plot(analyses: &[CaseAnalysis], output: &Path) -> PathBuf {
    assert!(!analyses.is_empty(), "no damped-cosine fits to plot");
    assert_eq!(
        output.extension().and_then(|suffix| suffix.to_str()),
        Some("svg"),
        "fit output must be SVG"
    );
    let columns = 2.min(analyses.len());
    let rows = analyses.len().div_ceil(columns);
    let mut panels = Vec::with_capacity(analyses.len());
    let mut layouts = Vec::with_capacity(analyses.len());
    let maximum_time = analyses
        .iter()
        .filter_map(|analysis| analysis.correlation.as_ref())
        .filter_map(|correlation| correlation.lag_times.last().copied())
        .reduce(f64::max)
        .expect("no fit lag times");

    for analysis in analyses {
        let correlation = analysis
            .correlation
            .as_ref()
            .expect("analysis has no correlation");
        let fit = analysis.fit.as_ref().expect("analysis has no fit");
        assert_eq!(
            fit.prediction.len(),
            correlation.lag_times.len(),
            "fit prediction length differs"
        );
        let label = analysis
            .group
            .case
            .label
            .as_deref()
            .unwrap_or(&analysis.group.case.case_id);
        let measured = LinePlot::new()
            .with_data(
                correlation
                    .lag_times
                    .iter()
                    .copied()
                    .zip(correlation.pearson_mean.iter().copied()),
            )
            .with_color("#4477AA")
            .with_stroke_width(1.7)
            .with_legend("data: mean(C_v)");
        let fitted = LinePlot::new()
            .with_data(
                correlation
                    .lag_times
                    .iter()
                    .copied()
                    .zip(fit.prediction.iter().copied()),
            )
            .with_color("#EE6677")
            .with_stroke_width(2.1)
            .with_dashed()
            .with_legend(format!(
                "fit [A,r,omega,phi]: A={:.2e}, r={:.2e}, omega={:.2e}, phi={:.2e}; B={:.2e}, R2={:.3}",
                fit.amplitude, fit.rate, fit.omega, fit.phase, fit.offset, fit.r_squared
            ));
        let zero = LinePlot::new()
            .with_data([(0.0, 0.0), (maximum_time, 0.0)])
            .with_color("#777777")
            .with_stroke_width(0.8);
        let plots = vec![Plot::Line(zero), Plot::Line(measured), Plot::Line(fitted)];
        layouts.push(
            Layout::auto_from_plots(&plots)
                .with_x_axis_min(0.0)
                .with_x_axis_max(maximum_time)
                .with_title(label)
                .with_x_label("lag time")
                .with_y_label("Pearson coefficient")
                .with_legend_position(LegendPosition::OutsideBottomCenter)
                .with_legend_wrap(48)
                .with_legend_box(false),
        );
        panels.push(plots);
    }
    let structure = panel_structure(analyses.len(), rows, columns);
    let scene = Figure::new(rows, columns)
        .with_structure(structure)
        .with_plots(panels)
        .with_layouts(layouts)
        .with_title("Robust constrained damped-cosine fits")
        .with_figure_size(columns as f64 * 780.0, rows as f64 * 590.0 + 70.0)
        .render();
    let svg = SvgBackend::new()
        .with_pretty(true)
        .with_embedded_font(true)
        .render_scene(&scene);
    write_atomic(output, svg.as_bytes());
    output.to_path_buf()
}

/// Give an unpaired final panel the full bottom row instead of drawing an empty axis.
fn panel_structure(count: usize, rows: usize, columns: usize) -> Vec<Vec<usize>> {
    let mut structure = (0..count).map(|cell| vec![cell]).collect::<Vec<_>>();
    if columns == 2 && count % 2 == 1 {
        structure[count - 1] = vec![rows * columns - 2, rows * columns - 1];
    }
    structure
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
