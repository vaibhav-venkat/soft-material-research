//! Kuva cylinder cluster area-fraction probability histograms.

use std::fmt::Write as FmtWrite;
use std::fs::OpenOptions;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};

use crystalline_cylinder_analysis::{ClusterHistogram, ClusterSnapshot};
use kuva::backend::svg::SvgBackend;
use kuva::plot::Histogram;
use kuva::render::figure::Figure;
use kuva::render::layout::Layout;
use kuva::render::plots::Plot;

/// One physical case and its pooled cluster histogram.
#[derive(Clone, Debug)]
pub struct ClusterPlotCase {
    pub label: String,
    pub histogram: ClusterHistogram,
}

/// Write one panel per selected case with shared probability axes.
pub fn write_cluster_histogram_plot(
    cases: &[ClusterPlotCase],
    title: &str,
    x_label: &str,
    y_label: &str,
    log_x: bool,
    color: &str,
    output: &Path,
) -> PathBuf {
    assert!(!cases.is_empty(), "no cluster histograms");
    assert_eq!(
        output.extension().and_then(|suffix| suffix.to_str()),
        Some("svg"),
        "cluster output must be SVG"
    );
    let mut plots = Vec::with_capacity(cases.len());
    let mut layouts = Vec::with_capacity(cases.len());
    for case in cases {
        let histogram = &case.histogram;
        assert_eq!(
            histogram.bin_edges.len(),
            histogram.probabilities.len() + 1,
            "bad histogram shape"
        );
        let panel_title = if histogram.sample_count == 0 {
            format!("{} — no clusters", case.label)
        } else {
            format!("{} — n={}", case.label, histogram.sample_count)
        };
        let panel = vec![Plot::Histogram(
            Histogram::from_bins(histogram.bin_edges.clone(), histogram.probabilities.clone())
                .with_color(color),
        )];
        let layout = Layout::auto_from_plots(&panel)
            .with_title(panel_title)
            .with_x_label(x_label)
            .with_y_label(y_label);
        layouts.push(if log_x { layout.with_log_x() } else { layout });
        plots.push(panel);
    }
    let columns = cases.len().min(3);
    let rows = cases.len().div_ceil(columns);
    let scene = Figure::new(rows, columns)
        .with_plots(plots)
        .with_layouts(layouts)
        .with_title(title)
        .with_shared_x_all()
        .with_shared_y_all()
        .with_figure_size(columns as f64 * 520.0, rows as f64 * 420.0 + 80.0)
        .render();
    let svg = SvgBackend::new()
        .with_pretty(true)
        .with_embedded_font(true)
        .render_scene(&scene);
    write_atomic(output, svg.as_bytes());
    output.to_path_buf()
}

const CLUSTER_COLORS: [&str; 16] = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB", "#EE7733",
    "#0077BB", "#33BBEE", "#009988", "#EE3377", "#CC3311", "#EECC66", "#332288", "#44AA99",
];

/// Render structural and motion assignments on a cylinder from three azimuths.
pub fn write_cluster_snapshot_plot(
    snapshot: &ClusterSnapshot,
    case_label: &str,
    cylinder_radius: f64,
    axial_length: f64,
    output: &Path,
) -> PathBuf {
    assert!(
        cylinder_radius.is_finite() && cylinder_radius > 0.0,
        "bad radius"
    );
    assert!(
        axial_length.is_finite() && axial_length > 0.0,
        "bad axial length"
    );
    assert_eq!(snapshot.positions.len(), snapshot.eligible.len());
    assert_eq!(
        snapshot.positions.len(),
        snapshot.structural_cluster_ids.len()
    );
    assert_eq!(snapshot.positions.len(), snapshot.motion_cluster_ids.len());
    assert_eq!(
        output.extension().and_then(|value| value.to_str()),
        Some("svg")
    );

    let width = 1800.0;
    let height = 900.0;
    let mut svg = String::with_capacity(snapshot.positions.len() * 220);
    write!(
        svg,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0} {height:.0}" width="{width:.0}" height="{height:.0}"><rect width="100%" height="100%" fill="#ffffff"/><style>text{{font-family:Inter,Arial,sans-serif;fill:#17212b}}.title{{font-size:25px;font-weight:600}}.label{{font-size:18px;font-weight:600}}.note{{font-size:14px;fill:#52606d}}.cage{{fill:#eef4f8;fill-opacity:.42;stroke:#8295a5;stroke-width:1.2}}</style>"##
    )
    .expect("format snapshot SVG");
    write!(
        svg,
        r#"<text class="title" x="36" y="39">{} — cylinder clusters, frame {} (step {})</text><text class="note" x="36" y="65">Colored particles belong to clusters of at least two particles; gray particles are unclustered. Views rotate about the cylinder axis.</text>"#,
        escape_xml(case_label),
        snapshot.frame_index,
        snapshot.step,
    )
    .expect("format snapshot title");

    let views = [0.0_f64, 120.0, 240.0];
    let rows = [
        ("Structural / crystalline", &snapshot.structural_cluster_ids),
        ("Coherent motion", &snapshot.motion_cluster_ids),
    ];
    for (row, (row_label, assignments)) in rows.into_iter().enumerate() {
        let top = 90.0 + row as f64 * 400.0;
        let motion_suffix = if row == 1 {
            snapshot.motion_destination_frame_index.map_or_else(
                || " — no lag destination available".to_owned(),
                |destination| format!(" — frame {} → {}", snapshot.frame_index, destination),
            )
        } else {
            String::new()
        };
        write!(
            svg,
            r#"<text class="label" x="36" y="{:.1}">{}{}</text>"#,
            top + 20.0,
            row_label,
            motion_suffix,
        )
        .expect("format row title");
        for (column, &degrees) in views.iter().enumerate() {
            render_cylinder_view(
                &mut svg,
                snapshot,
                assignments,
                cylinder_radius,
                axial_length,
                degrees.to_radians(),
                20.0 + column as f64 * 590.0,
                top + 38.0,
                570.0,
                330.0,
            );
            write!(
                svg,
                r#"<text class="note" text-anchor="middle" x="{:.1}" y="{:.1}">azimuth {:.0}°</text>"#,
                20.0 + column as f64 * 590.0 + 285.0,
                top + 390.0,
                degrees,
            )
            .expect("format view label");
        }
    }
    svg.push_str("</svg>");
    write_atomic(output, svg.as_bytes());
    output.to_path_buf()
}

#[allow(clippy::too_many_arguments)]
fn render_cylinder_view(
    svg: &mut String,
    snapshot: &ClusterSnapshot,
    assignments: &[Option<usize>],
    radius: f64,
    lx: f64,
    azimuth: f64,
    left: f64,
    top: f64,
    width: f64,
    height: f64,
) {
    let plot_left = left + 25.0;
    let plot_width = width - 50.0;
    let center_y = top + height * 0.5;
    let radial_scale = (height - 42.0) / (2.0 * radius);
    let axial_scale = plot_width / lx;
    let x_center = snapshot
        .positions
        .iter()
        .zip(&snapshot.eligible)
        .filter(|(_, eligible)| **eligible)
        .map(|(position, _)| position[0])
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
            (low.min(value), high.max(value))
        });
    let center_x = if x_center.0.is_finite() {
        0.5 * (x_center.0 + x_center.1)
    } else {
        0.0
    };
    let x_min = center_x - 0.5 * lx;
    let end_cap_width = 13.0;
    write!(
        svg,
        r##"<g><rect x="{left:.1}" y="{top:.1}" width="{width:.1}" height="{height:.1}" rx="8" fill="#fbfcfd" stroke="#d8e0e7"/><path class="cage" d="M {x0:.1} {yt:.1} L {x1:.1} {yt:.1} A {cap:.1} {rad:.1} 0 0 1 {x1:.1} {yb:.1} L {x0:.1} {yb:.1} A {cap:.1} {rad:.1} 0 0 1 {x0:.1} {yt:.1} Z"/><ellipse cx="{x0:.1}" cy="{cy:.1}" rx="{cap:.1}" ry="{rad:.1}" fill="none" stroke="#8295a5"/><ellipse cx="{x1:.1}" cy="{cy:.1}" rx="{cap:.1}" ry="{rad:.1}" fill="none" stroke="#8295a5"/></g>"##,
        x0 = plot_left,
        x1 = plot_left + plot_width,
        yt = center_y - radius * radial_scale,
        yb = center_y + radius * radial_scale,
        cap = end_cap_width,
        rad = radius * radial_scale,
        cy = center_y,
    )
    .expect("format cylinder cage");

    let cosine = azimuth.cos();
    let sine = azimuth.sin();
    let mut particles = snapshot
        .positions
        .iter()
        .enumerate()
        .filter(|(particle, _)| snapshot.eligible[*particle])
        .map(|(particle, position)| {
            let visible = position[1].mul_add(cosine, position[2] * sine);
            let depth = (-position[1]).mul_add(sine, position[2] * cosine);
            (particle, position[0], visible, depth)
        })
        .collect::<Vec<_>>();
    particles.sort_by(|left, right| left.3.total_cmp(&right.3));
    for (particle, x, visible, depth) in particles {
        let screen_x = plot_left + (x - x_min) * axial_scale;
        if !(plot_left - 2.0..=plot_left + plot_width + 2.0).contains(&screen_x) {
            continue;
        }
        let screen_y = center_y - visible * radial_scale;
        let depth_fraction = (0.5 + 0.5 * depth / radius).clamp(0.0, 1.0);
        let (fill, point_radius, opacity) = assignments[particle].map_or(
            ("#9eacb8", 0.9, 0.28 + 0.32 * depth_fraction),
            |cluster| {
                (
                    CLUSTER_COLORS[cluster % CLUSTER_COLORS.len()],
                    2.2,
                    0.58 + 0.42 * depth_fraction,
                )
            },
        );
        write!(
            svg,
            r#"<circle cx="{screen_x:.2}" cy="{screen_y:.2}" r="{point_radius:.2}" fill="{fill}" fill-opacity="{opacity:.2}"/>"#,
        )
        .expect("format particle");
    }
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn write_atomic(output: &Path, contents: &[u8]) {
    let parent = output.parent().expect("output has no parent");
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
    std::fs::rename(&temporary, output).expect("publish output");
}
