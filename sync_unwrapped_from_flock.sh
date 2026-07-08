#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${REMOTE_HOST:-flock}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/drive3/vaibhav_data/structures_and_functions}"
DRY_RUN="${DRY_RUN:-0}"

RSYNC_ARGS=(
  -avh
  --progress
  --partial
  --ignore-existing
  --exclude
  "*_shear_flux_decomposition_series.npz"
)

if [[ "$DRY_RUN" == "1" ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

SYNC_DIRS=(
  "hexatic/unwrapped_analysis/initial"
  "hexatic/unwrapped_analysis/gsd"
  "hexatic/unwrapped_analysis/hexatic_output"
  "hexatic/unwrapped_analysis/npz_fields"
)

echo "Syncing unwrapped-analysis outputs from ${REMOTE_HOST}:${REMOTE_ROOT}"
echo "Local destination: ${LOCAL_ROOT}"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: showing what would be copied without writing files."
fi

for rel_dir in "${SYNC_DIRS[@]}"; do
  mkdir -p "${LOCAL_ROOT}/${rel_dir}"
  rsync "${RSYNC_ARGS[@]}" \
    "${REMOTE_HOST}:${REMOTE_ROOT}/${rel_dir}/" \
    "${LOCAL_ROOT}/${rel_dir}/"
done
