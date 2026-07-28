#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

trap 'status=$?; printf "run_2d_sims.sh failed at line %s (status %s)\n" "$LINENO" "$status" >&2; exit "$status"' ERR

usage() {
    printf 'Usage: %s (--x-wall | --periodic) --output-root PATH --seed INTEGER [--gpu-id INTEGER]\n' "$0" >&2
    printf 'Example: %s --periodic --output-root /mnt/drive3/vaibhav_data/ideal_seed_7 --seed 7 --gpu-id 0\n' "$0" >&2
}

output_root=""
seed=""
gpu_id="0"
run_x_wall=0
run_periodic=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --x-wall)
            run_x_wall=1
            shift
            ;;
        --periodic)
            run_periodic=1
            shift
            ;;
        --output-root)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            output_root="$2"
            shift 2
            ;;
        --seed)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            seed="$2"
            shift 2
            ;;
        --gpu-id)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            gpu_id="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$output_root" || -z "$seed" ]]; then
    usage
    exit 2
fi
if (( run_x_wall + run_periodic != 1 )); then
    printf 'Specify exactly one of --x-wall or --periodic\n' >&2
    usage
    exit 2
fi
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    printf -- '--seed must be a non-negative integer\n' >&2
    exit 2
fi
if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
    printf -- '--gpu-id must be a non-negative integer\n' >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
cd "$repo_root"

run_steps="${RUN_STEPS:-100000000}"
write_period="${TRAJECTORY_WRITE_PERIOD:-100000}"
frames_per_shard="${FRAMES_PER_SHARD:-100}"
plot_frames="${PLOT_FRAMES:-1000}"
plot_max_lag="${PLOT_MAX_LAG:-900}"
cuda_version="${CONDA_OVERRIDE_CUDA:-12.9}"

mkdir -p "$output_root/logs" "$output_root/plots"

if [[ "${OVERWRITE:-0}" == 1 && "${RESUME_ANALYSIS:-0}" == 1 ]]; then
    printf 'OVERWRITE and RESUME_ANALYSIS cannot both be 1\n' >&2
    exit 2
fi

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

CUDA_VISIBLE_DEVICES="$gpu_id" \
    pixi run -e big-lx-cuda12 new-sims-gpu-check

run_case_pipeline() {
    local case_id="$1"
    local simulation_log="$output_root/logs/${case_id}_simulation.log"
    local analysis_log="$output_root/logs/${case_id}_analysis.log"

    printf 'Starting %s for seed %s on physical GPU %s\n' \
        "$case_id" "$seed" "$gpu_id"
    CUDA_VISIBLE_DEVICES="$gpu_id" \
        pixi run -e big-lx-cuda12 python -u -m hexatic.new_sims.simulate_case \
        --case "$case_id" \
        --output-root "$output_root" \
        --seed "$seed" \
        --run-steps "$run_steps" \
        --trajectory-write-period "$write_period" \
        --device gpu \
        --gpu-id 0 \
        "${simulation_flags[@]}" \
        >"$simulation_log" 2>&1

    CUDA_VISIBLE_DEVICES="$gpu_id" \
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

if (( run_x_wall )); then
    case_id="ideal_2d_x_walls"
    plot_input_flag="--walled-shard-dir"
else
    case_id="ideal_2d_periodic"
    plot_input_flag="--periodic-shard-dir"
fi

run_case_pipeline "$case_id"

pixi run -e big-lx-cuda12 python -u \
    -m hexatic.new_sims_analysis.plot_2d_correlation \
    "$plot_input_flag" "$output_root/safetensors_output/$case_id" \
    --frames "$plot_frames" \
    --max-lag "$plot_max_lag" \
    --output "$output_root/plots/2d_orientation_correlation.svg" \
    --report "$output_root/plots/2d_orientation_correlation_report.md"

printf '%s simulation, analysis, and correlation plotting completed for seed %s\n' \
    "$case_id" "$seed"
printf 'Output root: %s\n' "$output_root"
