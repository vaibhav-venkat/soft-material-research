#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

trap 'status=$?; printf "run_new_sims.sh failed at line %s (status %s)\n" "$LINENO" "$status" >&2; exit "$status"' ERR

usage() {
    printf 'Usage: %s OUTPUT_ROOT [GPU_IDS]\n' "$0" >&2
    printf 'Example: %s /mnt/drive3/vaibhav_data/new_sims 0,1\n' "$0" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
cd "$repo_root"

output_root="$1"
gpu_ids="${2:-0,1}"
run_steps="${RUN_STEPS:-100000000}"
write_period="${TRAJECTORY_WRITE_PERIOD:-100000}"
frames_per_shard="${FRAMES_PER_SHARD:-100}"
cuda_version="${CONDA_OVERRIDE_CUDA:-12.9}"

if [[ "${OVERWRITE:-0}" == 1 && "${RESUME_ANALYSIS:-0}" == 1 ]]; then
    printf 'OVERWRITE and RESUME_ANALYSIS cannot both be 1\n' >&2
    exit 2
fi

mkdir -p "$output_root/logs"
launcher_log="$output_root/logs/launcher.log"

command=(
    pixi run -e big-lx-cuda12 new-sims-pipeline
    --gpu-ids "$gpu_ids"
    --output-root "$output_root"
    --run-steps "$run_steps"
    --trajectory-write-period "$write_period"
    --frames-per-shard "$frames_per_shard"
)

if [[ "${OVERWRITE:-0}" == 1 ]]; then
    command+=(--overwrite)
fi
if [[ "${RESUME_ANALYSIS:-0}" == 1 ]]; then
    command+=(--resume-analysis)
fi

printf 'Starting new_sims pipeline\n'
printf '  output_root=%s\n  gpu_ids=%s\n  run_steps=%s\n  write_period=%s\n  frames_per_shard=%s\n' \
    "$output_root" "$gpu_ids" "$run_steps" "$write_period" "$frames_per_shard"
printf '  launcher_log=%s\n' "$launcher_log"

export CONDA_OVERRIDE_CUDA="$cuda_version"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CUDA_VISIBLE_DEVICES="$gpu_ids" \
    pixi run -e big-lx-cuda12 new-sims-gpu-check

"${command[@]}" 2>&1 | tee -a "$launcher_log"
