//! Kuva COM and velocity figure interface.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use crystalline_cylinder_analysis::pipeline::CaseAnalysis;
use kuva::backend::svg::SvgBackend;
use kuva::plot::LinePlot;
use kuva::render::figure::Figure;
use kuva::render::layout::Layout;
use kuva::render::plots::Plot;

/// Write the two-panel COM and velocity SVG with replicate bands.
pub fn write_com_plot(analyses: &[CaseAnalysis], output: &Path) -> PathBuf {
    assert!(!analyses.is_empty(), "no COM");
    assert_eq!(
        output.extension().and_then(|suffix| suffix.to_str()),
        Some("svg"),
        "bad output"
    );
    let colors = [
        "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000",
    ];
    let mut center_plots = Vec::with_capacity(analyses.len());
    let mut velocity_plots = Vec::with_capacity(analyses.len() + 1);
    let mut maximum_time = 0.0_f64;

    for (index, analysis) in analyses.iter().enumerate() {
        let com = analysis.com.as_ref().expect("no COM");
        validate_series(com);
        maximum_time = maximum_time.max(*com.elapsed_time.last().unwrap_or(&0.0));
        let color = colors[index % colors.len()];
        let label = analysis
            .group
            .case
            .label
            .as_deref()
            .unwrap_or(&analysis.group.case.case_id);
        center_plots.push(Plot::Line(series_line(
            &com.elapsed_time,
            &com.x_center_mean,
            &com.x_center_std,
            com.replicate_count,
            color,
            label,
        )));
        velocity_plots.push(Plot::Line(series_line(
            &com.elapsed_time,
            &com.x_velocity_mean,
            &com.x_velocity_std,
            com.replicate_count,
            color,
            label,
        )));
    }
    velocity_plots.insert(
        0,
        Plot::Line(
            LinePlot::new()
                .with_data([(0.0, 0.0), (maximum_time, 0.0)])
                .with_color("#555555")
                .with_stroke_width(1.0)
                .with_dashed(),
        ),
    );

    let plots = vec![center_plots, velocity_plots];
    let layouts = vec![
        Layout::auto_from_plots(&plots[0])
            .with_title("Unwrapped axial center of mass")
            .with_y_label("unwrapped x_COM"),
        Layout::auto_from_plots(&plots[1])
            .with_title("Axial center-of-mass velocity")
            .with_x_label("elapsed simulation time")
            .with_y_label("v_x,COM = dx_COM/dt"),
    ];
    let scene = Figure::new(2, 1)
        .with_plots(plots)
        .with_layouts(layouts)
        .with_title("Crystalline-cylinder axial COM — per-particle minimum-image unwrapping")
        .with_shared_x(0)
        .with_figure_size(1100.0, 820.0)
        .render();
    let svg = SvgBackend::new()
        .with_pretty(true)
        .with_embedded_font(true)
        .render_scene(&scene);
    write_atomic(output, svg.as_bytes());
    output.to_path_buf()
}

fn series_line(
    x: &[f64],
    mean: &[f64],
    std: &[f64],
    replicate_count: usize,
    color: &str,
    label: &str,
) -> LinePlot {
    let mut line = LinePlot::new()
        .with_data(x.iter().copied().zip(mean.iter().copied()))
        .with_color(color)
        .with_stroke_width(1.8)
        .with_legend(label);
    if replicate_count > 1 {
        line = line.with_band(
            mean.iter().zip(std).map(|(&value, &spread)| value - spread),
            mean.iter().zip(std).map(|(&value, &spread)| value + spread),
        );
    }
    line
}

fn validate_series(com: &crystalline_cylinder_analysis::ComSeries) {
    let count = com.elapsed_time.len();
    assert!(count >= 2, "short COM");
    assert_eq!(com.x_center_mean.len(), count, "bad shape");
    assert_eq!(com.x_center_std.len(), count, "bad shape");
    assert_eq!(com.x_velocity_mean.len(), count, "bad shape");
    assert_eq!(com.x_velocity_std.len(), count, "bad shape");
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
