#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
echo "$ROOT_DIR"
ANALYSIS_WORKERS="${ANALYSIS_WORKERS:-3}"
SHEAR_SERIES_STRIDE="${SHEAR_SERIES_STRIDE:-1}"

pixi run python -m hexatic.radii_analysis.gpu_check
pixi run python -m hexatic.radii_analysis.run_sweep --all
pixi run python -m hexatic.radii_analysis.run_analysis \
  --all \
  --workers "$ANALYSIS_WORKERS" \
  --shear-series-stride "$SHEAR_SERIES_STRIDE"
