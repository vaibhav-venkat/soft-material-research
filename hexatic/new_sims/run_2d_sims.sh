#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

trap 'status=$?; printf "run_2d_sims.sh failed at line %s (status %s)\n" "$LINENO" "$status" >&2; exit "$status"' ERR

usage() {
    printf 'Usage: %s OUTPUT_ROOT [GPU_IDS]\n' "$0" >&2
    printf 'Example: %s /mnt/drive3/vaibhav_data/new_sims_2d 0,1\n' "$0" >&2
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
if [[ ! "$gpu_ids" =~ ^[0-9]+,[0-9]+$ ]]; then
    printf 'GPU_IDS must contain exactly two comma-separated non-negative IDs\n' >&2
    exit 2
fi
gpu_walled="${gpu_ids%%,*}"
gpu_periodic="${gpu_ids##*,}"
if [[ "$gpu_walled" == "$gpu_periodic" ]]; then
    printf 'GPU_IDS must name two distinct GPUs\n' >&2
    exit 2
fi

run_steps="${RUN_STEPS:-100000000}"
write_period="${TRAJECTORY_WRITE_PERIOD:-100000}"
frames_per_shard="${FRAMES_PER_SHARD:-100}"
plot_frames="${PLOT_FRAMES:-1000}"
plot_max_lag="${PLOT_MAX_LAG:-900}"
cuda_version="${CONDA_OVERRIDE_CUDA:-12.9}"

mkdir -p "$output_root/logs" "$output_root/plots"

simulation_flags=()
analysis_flags=()
if [[ "${OVERWRITE:-0}" == 1 ]]; then
    simulation_flags+=(--overwrite)
    analysis_flags+=(--overwrite)
elif [[ "${RESUME_ANALYSIS:-0}" == 1 ]]; then
    analysis_flags+=(--resume)
fi

export CONDA_OVERRIDE_CUDA="$cuda_version"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CUDA_VISIBLE_DEVICES="$gpu_ids" \
    pixi run -e big-lx-cuda12 new-sims-gpu-check

run_case_pipeline() {
    local case_id="$1"
    local physical_gpu="$2"
    local simulation_log="$output_root/logs/${case_id}_simulation.log"
    local analysis_log="$output_root/logs/${case_id}_analysis.log"

    printf 'Starting %s on physical GPU %s\n' "$case_id" "$physical_gpu"
    CUDA_VISIBLE_DEVICES="$physical_gpu" \
        pixi run -e big-lx-cuda12 python -u -m hexatic.new_sims.simulate_case \
        --case "$case_id" \
        --output-root "$output_root" \
        --run-steps "$run_steps" \
        --trajectory-write-period "$write_period" \
        --device gpu \
        --gpu-id 0 \
        "${simulation_flags[@]}" \
        >"$simulation_log" 2>&1

    CUDA_VISIBLE_DEVICES="$physical_gpu" \
        pixi run -e big-lx-cuda12 python -u -m hexatic.new_sims.analyze_case \
        --case "$case_id" \
        --output-root "$output_root" \
        --backend jax \
        --require-gpu \
        --frames-per-shard "$frames_per_shard" \
        "${analysis_flags[@]}" \
        >"$analysis_log" 2>&1
    printf 'Completed simulation and analysis for %s\n' "$case_id"
}

run_case_pipeline ideal_2d_x_walls "$gpu_walled" &
walled_pid=$!
run_case_pipeline ideal_2d_periodic "$gpu_periodic" &
periodic_pid=$!

pipeline_status=0
if ! wait "$walled_pid"; then
    printf 'ideal_2d_x_walls failed; inspect %s/logs\n' "$output_root" >&2
    pipeline_status=1
fi
if ! wait "$periodic_pid"; then
    printf 'ideal_2d_periodic failed; inspect %s/logs\n' "$output_root" >&2
    pipeline_status=1
fi
if [[ "$pipeline_status" -ne 0 ]]; then
    exit "$pipeline_status"
fi

pixi run -e big-lx-cuda12 python -u \
    -m hexatic.new_sims_analysis.plot_2d_correlation \
    --walled-shard-dir "$output_root/safetensors_output/ideal_2d_x_walls" \
    --periodic-shard-dir "$output_root/safetensors_output/ideal_2d_periodic" \
    --frames "$plot_frames" \
    --max-lag "$plot_max_lag" \
    --output "$output_root/plots/2d_orientation_correlation.svg" \
    --report "$output_root/plots/2d_orientation_correlation_report.md"

printf 'All 2D runs, analyses, and correlation plotting completed\n'
printf 'Output root: %s\n' "$output_root"
