//! Kuva complex-Laplace heatmap interface.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use crystalline_cylinder_analysis::pipeline::CaseAnalysis;
use kuva::backend::svg::SvgBackend;
use kuva::plot::{ColorMap, Heatmap};
use kuva::render::figure::Figure;
use kuva::render::layout::Layout;
use kuva::render::plots::Plot;

/// Write one tightly bounded log-magnitude transform heatmap per case.
pub fn write_laplace_plots(
    analyses: &[CaseAnalysis],
    output_dir: &Path,
    overwrite: bool,
) -> Vec<PathBuf> {
    assert!(!analyses.is_empty(), "no Laplace grids to plot");
    analyses
        .iter()
        .map(|analysis| {
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
            let plots = vec![Plot::Heatmap(heatmap)];
            let label = analysis
                .group
                .case
                .label
                .as_deref()
                .unwrap_or(&analysis.group.case.case_id);
            let layout = Layout::auto_from_plots(&plots)
                .with_x_axis_min(r_min)
                .with_x_axis_max(r_max)
                .with_y_axis_min(omega_min)
                .with_y_axis_max(omega_max)
                .with_title(label)
                .with_x_label("r")
                .with_y_label("omega");
            let scene = Figure::new(1, 1)
                .with_plots(vec![plots])
                .with_layouts(vec![layout])
                .with_title("Axial velocity-correlation complex Laplace transform")
                .with_figure_size(1100.0, 800.0)
                .render();
            let svg = SvgBackend::new()
                .with_pretty(true)
                .with_embedded_font(true)
                .render_scene(&scene);
            let output = output_dir.join(format!("{}_laplace.svg", analysis.group.case.case_id));
            assert!(
                !output.exists() || overwrite,
                "output exists; use --overwrite"
            );
            write_atomic(&output, svg.as_bytes());
            output
        })
        .collect()
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
