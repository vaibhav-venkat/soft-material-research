//! Kuva complex-Laplace heatmap interface.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use crystalline_cylinder_analysis::pipeline::CaseAnalysis;
use crystalline_cylinder_analysis::CaseSchema;
use kuva::backend::svg::SvgBackend;
use kuva::plot::{ColorMap, Heatmap};
use kuva::render::figure::Figure;
use kuva::render::layout::Layout;
use kuva::render::plots::Plot;

/// Write one multi-panel heatmap figure per dataset schema.
pub fn write_laplace_plots(
    analyses: &[CaseAnalysis],
    output_dir: &Path,
    overwrite: bool,
) -> Vec<PathBuf> {
    assert!(!analyses.is_empty(), "no Laplace grids to plot");
    let groups = [
        (
            CaseSchema::BigLx,
            "big_lx_laplace.svg",
            "Big-Lx axial velocity-correlation complex Laplace transforms",
        ),
        (
            CaseSchema::Confinement,
            "confinement_laplace.svg",
            "Confined axial velocity-correlation complex Laplace transforms",
        ),
    ];
    groups
        .into_iter()
        .filter_map(|(schema, file_name, title)| {
            let cases = analyses
                .iter()
                .filter(|analysis| analysis.group.schema == schema)
                .collect::<Vec<_>>();
            if cases.is_empty() {
                None
            } else {
                Some(write_schema_panel(
                    &cases,
                    &output_dir.join(file_name),
                    title,
                    overwrite,
                ))
            }
        })
        .collect()
}

fn write_schema_panel(
    analyses: &[&CaseAnalysis],
    output: &Path,
    title: &str,
    overwrite: bool,
) -> PathBuf {
    assert!(
        !output.exists() || overwrite,
        "output exists; use --overwrite"
    );
    let mut plots = Vec::with_capacity(analyses.len());
    let mut layouts = Vec::with_capacity(analyses.len());
    for analysis in analyses {
        let grid = analysis
            .laplace
            .as_ref()
            .expect("analysis has no Laplace grid");
        validate_grid(grid);
        let r_min = grid.r[0];
        let r_max = *grid.r.last().expect("r grid is empty");
        let omega_min = grid.omega[0];
        let omega_max = *grid.omega.last().expect("omega grid is empty");
        let data = grid
            .values
            .chunks_exact(grid.r.len())
            .map(|row| {
                row.iter()
                    .map(|value| value.norm().max(f64::MIN_POSITIVE).log10())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let heatmap = Heatmap::new()
            .with_data(data)
            .with_color_map(ColorMap::Viridis)
            .with_legend("log10 |C_hat_v|")
            .with_x_range(r_min, r_max)
            .with_y_range(omega_min, omega_max)
            .with_cell_size(1.0);
        let case_plots = vec![Plot::Heatmap(heatmap)];
        let label = analysis
            .group
            .case
            .label
            .as_deref()
            .unwrap_or(&analysis.group.case.case_id);
        let layout = Layout::auto_from_plots(&case_plots)
            .with_x_axis_min(r_min)
            .with_x_axis_max(r_max)
            .with_y_axis_min(omega_min)
            .with_y_axis_max(omega_max)
            .with_title(label)
            .with_x_label("r")
            .with_y_label("omega");
        plots.push(case_plots);
        layouts.push(layout);
    }

    let columns = analyses.len().min(3);
    let rows = analyses.len().div_ceil(columns);
    let scene = Figure::new(rows, columns)
        .with_plots(plots)
        .with_layouts(layouts)
        .with_title(title)
        .with_shared_x_all()
        .with_shared_y_all()
        .with_figure_size(columns as f64 * 520.0, rows as f64 * 420.0 + 70.0)
        .render();
    let svg = SvgBackend::new()
        .with_pretty(true)
        .with_embedded_font(true)
        .render_scene(&scene);
    write_atomic(output, svg.as_bytes());
    output.to_path_buf()
}

fn validate_grid(grid: &crystalline_cylinder_analysis::LaplaceGrid) {
    assert!(grid.r.len() >= 2, "r grid needs two points");
    assert!(grid.omega.len() >= 2, "omega grid needs two points");
    assert_eq!(
        grid.shape,
        [grid.omega.len(), grid.r.len()],
        "Laplace shape differs"
    );
    assert_eq!(
        grid.values.len(),
        grid.r.len() * grid.omega.len(),
        "Laplace value count differs"
    );
    assert!(
        grid.r.windows(2).all(|pair| pair[1] > pair[0]),
        "r grid is not increasing"
    );
    assert!(
        grid.omega.windows(2).all(|pair| pair[1] > pair[0]),
        "omega grid is not increasing"
    );
    assert!(
        grid.values
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite()),
        "Laplace grid is non-finite"
    );
}

fn write_atomic(output: &Path, contents: &[u8]) {
    std::fs::create_dir_all(output.parent().expect("output has no parent"))
        .expect("create output directory");
    let file_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .expect("output filename is invalid");
    let temporary = output
        .parent()
        .expect("output has no parent")
        .join(format!(".{file_name}.{}.tmp", std::process::id()));
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
