from __future__ import annotations

import argparse
import json

import gsd.hoomd
import numpy as np

from hexatic.active_matter_cylinder import (
    ACTIVE_FIELD_THETA_BINS,
    ACTIVE_FIELD_X_BINS,
    ACTIVE_FLUX_PLOT_THETA_BINS,
    ACTIVE_GRID_DX,
    ACTIVE_GRID_DY,
    ACTIVE_GRID_DZ,
    LOCAL_POCKET_RADIUS,
    active_matter_field_series,
    compute_cartesian_flux_comparison,
    compute_shear_flux_decomposition,
    compute_shear_flux_decomposition_series,
    save_active_matter_fields,
    save_cartesian_flux_comparison,
    save_shear_flux_decomposition,
    save_shear_flux_decomposition_series,
)
from hexatic.chirality import (
    compute_translation_chirality_trajectory,
    shell_bond_translation_chirality_series,
)
from hexatic.constants import cylinder
from hexatic.hexatic_cylinder_calculation import calculate_cylinder_hexatic

from .cases import HEXATIC_OUTPUT_DIR, NPZ_FIELDS_DIR, RadiusCase, ensure_output_dirs, get_case


def _trajectory_frame_count(case: RadiusCase) -> int:
    with gsd.hoomd.open(name=str(case.trajectory_gsd), mode="r") as source:
        return len(source)


def _shear_frame_indices(case: RadiusCase, stride: int) -> range:
    if stride <= 0:
        raise ValueError("shear-series-stride must be positive.")
    return range(0, _trajectory_frame_count(case), stride)


def analyze_case(
    case: RadiusCase,
    overwrite: bool = False,
    shear_series_stride: int = 1,
    skip_shear_series: bool = False,
) -> None:
    ensure_output_dirs()
    if not case.trajectory_gsd.exists():
        raise FileNotFoundError(f"Missing trajectory GSD: {case.trajectory_gsd}")

    output_paths = (
        HEXATIC_OUTPUT_DIR / f"{case.case_id}_hexatic_order.txt",
        HEXATIC_OUTPUT_DIR / f"{case.case_id}_neighbor_counts.txt",
        HEXATIC_OUTPUT_DIR / f"{case.case_id}_hexatic_velocity.gsd",
        NPZ_FIELDS_DIR / f"{case.case_id}_active_matter_fields.npz",
        NPZ_FIELDS_DIR / f"{case.case_id}_cartesian_flux_comparison.npz",
        NPZ_FIELDS_DIR / f"{case.case_id}_shear_flux_decomposition.npz",
        NPZ_FIELDS_DIR / f"{case.case_id}_translation_chirality_fields.npz",
        NPZ_FIELDS_DIR / f"{case.case_id}_shell_bond_translation_chirality.npz",
        NPZ_FIELDS_DIR / f"{case.case_id}_analysis_metadata.json",
    )
    if not skip_shear_series:
        output_paths += (
            NPZ_FIELDS_DIR / f"{case.case_id}_shear_flux_decomposition_series.npz",
        )
    if not overwrite:
        existing = [path for path in output_paths if path.exists()]
        if existing:
            text = "\n".join(str(path) for path in existing)
            raise FileExistsError(f"Refusing to overwrite existing file(s):\n{text}")

    calculate_cylinder_hexatic(
        input_gsd=case.trajectory_gsd,
        cylinder_radius=case.radius,
        output_dir=HEXATIC_OUTPUT_DIR,
        case_id=case.case_id,
        write_ovito_gsd=True,
    )

    fields = active_matter_field_series(
        case.trajectory_gsd,
        pocket_radius=LOCAL_POCKET_RADIUS,
        n_x_bins=ACTIVE_FIELD_X_BINS,
        n_theta_bins=ACTIVE_FIELD_THETA_BINS,
        cylinder_radius=case.radius,
    )
    save_active_matter_fields(
        fields,
        NPZ_FIELDS_DIR / f"{case.case_id}_active_matter_fields.npz",
        pocket_radius=LOCAL_POCKET_RADIUS,
    )

    comparison = compute_cartesian_flux_comparison(
        case.trajectory_gsd,
        pocket_radius=LOCAL_POCKET_RADIUS,
        dx=ACTIVE_GRID_DX,
        dy=ACTIVE_GRID_DY,
        dz=ACTIVE_GRID_DZ,
        frame_index=-2,
        cylinder_radius=case.radius,
    )
    save_cartesian_flux_comparison(
        comparison,
        NPZ_FIELDS_DIR / f"{case.case_id}_cartesian_flux_comparison.npz",
    )

    shear = compute_shear_flux_decomposition(
        case.trajectory_gsd,
        pocket_radius=LOCAL_POCKET_RADIUS,
        dx=ACTIVE_GRID_DX,
        dr=ACTIVE_GRID_DY,
        n_theta_bins=ACTIVE_FLUX_PLOT_THETA_BINS,
        frame_index=-2,
        cylinder_radius=case.radius,
    )
    save_shear_flux_decomposition(
        shear,
        NPZ_FIELDS_DIR / f"{case.case_id}_shear_flux_decomposition.npz",
    )

    if not skip_shear_series:
        shear_series = compute_shear_flux_decomposition_series(
            case.trajectory_gsd,
            frame_indices=_shear_frame_indices(case, shear_series_stride),
            pocket_radius=LOCAL_POCKET_RADIUS,
            dx=ACTIVE_GRID_DX,
            dr=ACTIVE_GRID_DY,
            n_theta_bins=ACTIVE_FLUX_PLOT_THETA_BINS,
            cylinder_radius=case.radius,
        )
        save_shear_flux_decomposition_series(
            shear_series,
            NPZ_FIELDS_DIR / f"{case.case_id}_shear_flux_decomposition_series.npz",
        )

    translation_chirality = compute_translation_chirality_trajectory(
        case.trajectory_gsd,
        neighborhood_radius=cylinder.ANALYSIS.neighbor_count_radius,
    )
    np.savez_compressed(
        NPZ_FIELDS_DIR / f"{case.case_id}_translation_chirality_fields.npz",
        steps=translation_chirality.steps,
        chirality=translation_chirality.chirality,
        neighborhood_radius=cylinder.ANALYSIS.neighbor_count_radius,
    )

    shell_steps, shell_values, shell_counts = shell_bond_translation_chirality_series(
        case.trajectory_gsd,
        neighborhood_radius=cylinder.ANALYSIS.neighbor_count_radius,
        cylinder_radius=case.radius,
        shell_delta=cylinder.ANALYSIS.shell_delta,
    )
    np.savez_compressed(
        NPZ_FIELDS_DIR / f"{case.case_id}_shell_bond_translation_chirality.npz",
        steps=shell_steps,
        mean_abs_bond_translation_chirality=shell_values,
        bond_counts=shell_counts,
        neighborhood_radius=cylinder.ANALYSIS.neighbor_count_radius,
        cylinder_radius=case.radius,
    )

    metadata = case.as_metadata()
    metadata.update(
        shear_series_stride=shear_series_stride,
        skip_shear_series=skip_shear_series,
    )
    (NPZ_FIELDS_DIR / f"{case.case_id}_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-long-axis", action="store_true")
    parser.add_argument("--shear-series-stride", type=int, default=1)
    parser.add_argument("--skip-shear-series", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    analyze_case(
        get_case(args.case, include_long_axis=args.include_long_axis),
        overwrite=args.overwrite,
        shear_series_stride=args.shear_series_stride,
        skip_shear_series=args.skip_shear_series,
    )


if __name__ == "__main__":
    main()

