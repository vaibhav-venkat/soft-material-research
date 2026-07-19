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

/// Write mean and standard-deviation COM diagnostics without filled bands.
pub fn write_com_plot(analyses: &[CaseAnalysis], output: &Path) -> PathBuf {
    assert!(!analyses.is_empty(), "no COM");
    assert_eq!(
        output.extension().and_then(|suffix| suffix.to_str()),
        Some("svg"),
        "bad output"
    );
    let colors = [
        "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB", "#000000",
        "#44AA99", "#DDCC77",
    ];
    let mut center_plots = Vec::with_capacity(analyses.len());
    let mut center_std_plots = Vec::with_capacity(analyses.len());
    let mut velocity_plots = Vec::with_capacity(analyses.len() + 1);
    let mut velocity_std_plots = Vec::with_capacity(analyses.len());
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
            color,
            Some(label),
        )));
        center_std_plots.push(Plot::Line(series_line(
            &com.elapsed_time,
            &com.x_center_std,
            color,
            None,
        )));
        velocity_plots.push(Plot::Line(series_line(
            &com.elapsed_time,
            &com.x_velocity_mean,
            color,
            None,
        )));
        velocity_std_plots.push(Plot::Line(series_line(
            &com.elapsed_time,
            &com.x_velocity_std,
            color,
            None,
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

    let plots = vec![
        center_plots,
        center_std_plots,
        velocity_plots,
        velocity_std_plots,
    ];
    let layouts = vec![
        Layout::auto_from_plots(&plots[0])
            .with_title("Mean unwrapped axial center of mass")
            .with_y_label("unwrapped x_COM"),
        Layout::auto_from_plots(&plots[1])
            .with_title("COM sample standard deviation")
            .with_y_label("std(x_COM)"),
        Layout::auto_from_plots(&plots[2])
            .with_title("Mean axial center-of-mass velocity")
            .with_x_label("elapsed simulation time")
            .with_y_label("v_x,COM = dx_COM/dt"),
        Layout::auto_from_plots(&plots[3])
            .with_title("Velocity sample standard deviation")
            .with_x_label("elapsed simulation time")
            .with_y_label("std(v_x,COM)"),
    ];
    let scene = Figure::new(2, 2)
        .with_plots(plots)
        .with_layouts(layouts)
        .with_title("Crystalline-cylinder axial COM — per-particle minimum-image unwrapping")
        .with_shared_x_all()
        .with_shared_legend_right_middle()
        .with_figure_size(1500.0, 900.0)
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
