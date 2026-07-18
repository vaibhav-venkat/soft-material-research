//! Kuva velocity-Pearson-correlation figure.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use crystalline_cylinder_analysis::pipeline::CaseAnalysis;
use kuva::backend::svg::SvgBackend;
use kuva::plot::LinePlot;
use kuva::render::figure::Figure;
use kuva::render::layout::Layout;
use kuva::render::plots::Plot;

/// Write mean Pearson curves and a separate replicate-deviation panel.
pub fn write_correlation_plot(analyses: &[CaseAnalysis], output: &Path) -> PathBuf {
    assert!(!analyses.is_empty(), "no correlations");
    assert_eq!(
        output.extension().and_then(|suffix| suffix.to_str()),
        Some("svg"),
        "bad output"
    );
    let colors = [
        "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB", "#000000",
        "#44AA99", "#DDCC77",
    ];
    let mut mean_plots = Vec::with_capacity(analyses.len() + 2);
    let mut deviation_plots = Vec::with_capacity(analyses.len());
    let maximum_lag = analyses
        .iter()
        .filter_map(|analysis| analysis.correlation.as_ref())
        .filter_map(|series| series.lag_times.last().copied())
        .fold(0.0_f64, f64::max);
    mean_plots.push(Plot::Line(reference_line(maximum_lag, 0.0, false)));
    mean_plots.push(Plot::Line(reference_line(maximum_lag, 1.0, true)));

    for (index, analysis) in analyses.iter().enumerate() {
        let correlation = analysis.correlation.as_ref().expect("no correlation");
        validate_series(correlation);
        let color = colors[index % colors.len()];
        let label = analysis
            .group
            .case
            .label
            .as_deref()
            .unwrap_or(&analysis.group.case.case_id);
        mean_plots.push(Plot::Line(series_line(
            &correlation.lag_times,
            &correlation.pearson_mean,
            color,
            Some(label),
        )));
        deviation_plots.push(Plot::Line(series_line(
            &correlation.lag_times,
            &correlation.pearson_std,
            color,
            None,
        )));
    }

    let plots = vec![mean_plots, deviation_plots];
    let layouts = vec![
        Layout::new((0.0, maximum_lag), (-1.05, 1.05))
            .with_title("Axial COM-velocity lagged Pearson correlation")
            .with_y_label("Pearson coefficient"),
        Layout::auto_from_plots(&plots[1])
            .with_title("Replicate sample standard deviation")
            .with_x_label("lag time")
            .with_y_label("std(Pearson coefficient)"),
    ];
    let scene = Figure::new(2, 1)
        .with_plots(plots)
        .with_layouts(layouts)
        .with_title("Crystalline-cylinder axial velocity Pearson correlation")
        .with_shared_x_all()
        .with_shared_legend_right_middle()
        .with_figure_size(1400.0, 900.0)
        .render();
    let svg = SvgBackend::new()
        .with_pretty(true)
        .with_embedded_font(true)
        .render_scene(&scene);
    write_atomic(output, svg.as_bytes());
    output.to_path_buf()
}

fn series_line(x: &[f64], values: &[f64], color: &str, label: Option<&str>) -> LinePlot {
    let line = LinePlot::new()
        .with_data(x.iter().copied().zip(values.iter().copied()))
        .with_color(color)
        .with_stroke_width(1.8);
    if let Some(label) = label {
        line.with_legend(label)
    } else {
        line
    }
}

fn reference_line(maximum_lag: f64, value: f64, dashed: bool) -> LinePlot {
    let line = LinePlot::new()
        .with_data([(0.0, value), (maximum_lag, value)])
        .with_color("#555555")
        .with_stroke_width(1.0);
    if dashed {
        line.with_dashed()
    } else {
        line
    }
}

fn validate_series(series: &crystalline_cylinder_analysis::CorrelationSeries) {
    let count = series.lag_times.len();
    assert!(count >= 1, "short correlation");
    assert_eq!(series.lag_indices.len(), count, "bad shape");
    assert_eq!(series.pearson_mean.len(), count, "bad shape");
    assert_eq!(series.pearson_std.len(), count, "bad shape");
    assert_eq!(series.origin_counts.len(), count, "bad shape");
    assert_eq!(series.pearson_mean[0], 1.0, "bad lag zero");
}

fn write_atomic(output: &Path, contents: &[u8]) {
    let parent = output.parent().expect("bad output");
    std::fs::create_dir_all(parent).expect("make output");
    let file_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .expect("bad output");
    let temporary = parent.join(format!(".{file_name}.{}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .expect("open output");
    file.write_all(contents).expect("write output");
    file.sync_all().expect("sync output");
    drop(file);
    std::fs::rename(&temporary, output).expect("move output");
}
