#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASAP_ROOT="${ASAP_ROOT:-$PROJECT_ROOT/third_party/ASAP}"
CONDA_ROOT="${CONDA_ROOT:-/home/ubuntu/miniconda3}"
ENV_NAME="${G1UB_ENV_NAME:-g1ub_isaacgym}"
ENV_DIR="$CONDA_ROOT/envs/$ENV_NAME"
PYTHON="$ENV_DIR/bin/python"
OUTPUT_DIR="$PROJECT_ROOT/outputs/G1UB-01_IsaacGym_capacity_audit"
MOTION_FILE="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl"
ENV_COUNTS=(64 128 256 512)

bash "$PROJECT_ROOT/scripts/setup_g1ub_isaacgym.sh"
mkdir -p "$OUTPUT_DIR"

SUMMARY="$OUTPUT_DIR/capacity_summary.csv"
printf 'num_envs,status,duration_seconds,peak_gpu_mib,peak_wsl_used_mib,exit_code\n' > "$SUMMARY"

for num_envs in "${ENV_COUNTS[@]}"; do
  log_file="$OUTPUT_DIR/num_envs_${num_envs}.log"
  start_time=$(date +%s)
  peak_gpu=0
  peak_ram=0

  echo "Starting capacity probe: num_envs=$num_envs"
  (
    cd "$ASAP_ROOT"
    export PATH="$ENV_DIR/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
    export HYDRA_FULL_ERROR=1
    "$PYTHON" humanoidverse/train_agent.py \
      +simulator=isaacgym \
      +exp=motion_tracking \
      +domain_rand=NO_domain_rand \
      +rewards=motion_tracking/reward_motion_tracking_dm_2real \
      +robot=g1/g1_29dof_anneal_23dof \
      +terrain=terrain_locomotion_plane \
      +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
      num_envs="$num_envs" \
      seed=0 \
      headless=True \
      use_wandb=False \
      base_dir="$OUTPUT_DIR/training_logs" \
      project_name=G1UB-01 \
      experiment_name="capacity_${num_envs}" \
      robot.motion.motion_file="$MOTION_FILE" \
      rewards.reward_penalty_curriculum=True \
      rewards.reward_penalty_degree=0.00001 \
      env.config.resample_motion_when_training=False \
      env.config.termination.terminate_when_motion_far=True \
      env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
      env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
      env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
      robot.asset.self_collisions=0 \
      algo.config.num_learning_iterations=3 \
      algo.config.save_interval=100000
  ) >"$log_file" 2>&1 &
  train_pid=$!

  while kill -0 "$train_pid" 2>/dev/null; do
    gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')
    mem_total=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
    mem_available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    ram_used=$(( (mem_total - mem_available) / 1024 ))
    (( gpu_used > peak_gpu )) && peak_gpu=$gpu_used
    (( ram_used > peak_ram )) && peak_ram=$ram_used
    sleep 1
  done

  set +e
  wait "$train_pid"
  exit_code=$?
  set -e
  duration=$(( $(date +%s) - start_time ))

  if [[ $exit_code -eq 0 ]]; then
    status=passed
  else
    status=failed
  fi

  printf '%s,%s,%s,%s,%s,%s\n' \
    "$num_envs" "$status" "$duration" "$peak_gpu" "$peak_ram" "$exit_code" \
    | tee -a "$SUMMARY"

  if [[ $exit_code -ne 0 ]]; then
    echo "Probe failed at num_envs=$num_envs. See $log_file" >&2
    break
  fi
done

echo
echo "G1UB-01 result:"
column -s, -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo "Saved to: $OUTPUT_DIR"
