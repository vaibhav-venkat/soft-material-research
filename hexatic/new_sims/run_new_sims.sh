#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

trap 'status=$?; printf "run_new_sims.sh failed at line %s (status %s)\n" "$LINENO" "$status" >&2; exit "$status"' ERR

usage() {
    printf 'Usage: %s --output-root PATH --seed INTEGER [--gpu-id INTEGER]\n' "$0" >&2
    printf 'Example: %s --output-root /mnt/drive3/vaibhav_data/ideal_cylinder_seed_7 --seed 7 --gpu-id 0\n' "$0" >&2
}

output_root=""
seed=""
gpu_id="0"
while [[ $# -gt 0 ]]; do
    case "$1" in
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

run_steps="300000000"
write_period="${TRAJECTORY_WRITE_PERIOD:-100000}"
frames_per_shard="${FRAMES_PER_SHARD:-100}"
cuda_version="${CONDA_OVERRIDE_CUDA:-12.9}"
case_id="ideal_abp_cylinder"
period="1"
tau_r="1"

if [[ "${OVERWRITE:-0}" == 1 && "${RESUME_ANALYSIS:-0}" == 1 ]]; then
    printf 'OVERWRITE and RESUME_ANALYSIS cannot both be 1\n' >&2
    exit 2
fi

mkdir -p "$output_root/logs"
simulation_log="$output_root/logs/${case_id}_simulation.log"
analysis_log="$output_root/logs/${case_id}_analysis.log"

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

printf 'Starting ideal non-interacting ABP cylinder run\n'
printf '  case=%s\n  output_root=%s\n  seed=%s\n  period=%s\n  tau_r=%s\n  gpu_id=%s\n' \
    "$case_id" "$output_root" "$seed" "$period" "$tau_r" "$gpu_id"
printf '  run_steps=%s\n  write_period=%s\n  frames_per_shard=%s\n' \
    "$run_steps" "$write_period" "$frames_per_shard"

CUDA_VISIBLE_DEVICES="$gpu_id" \
    pixi run -e big-lx-cuda12 new-sims-gpu-check

CUDA_VISIBLE_DEVICES="$gpu_id" \
    pixi run -e big-lx-cuda12 python -u -m hexatic.new_sims.simulate_case \
    --case "$case_id" \
    --output-root "$output_root" \
    --seed "$seed" \
    --trajectory-write-period "$write_period" \
    --device gpu \
    --gpu-id 0 \
    "${simulation_flags[@]}" \
    2>&1 | tee "$simulation_log"

CUDA_VISIBLE_DEVICES="$gpu_id" \
    pixi run -e big-lx-cuda12 python -u -m hexatic.new_sims.analyze_case \
    --case "$case_id" \
    --output-root "$output_root" \
    --backend jax \
    --require-gpu \
    --frames-per-shard "$frames_per_shard" \
    "${analysis_flags[@]}" \
    2>&1 | tee "$analysis_log"

printf 'Completed ideal cylinder simulation and analysis for seed %s (period=%s, tau_r=%s)\n' \
    "$seed" "$period" "$tau_r"
