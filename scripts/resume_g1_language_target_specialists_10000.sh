#!/usr/bin/env bash
set -euo pipefail

# Resume each audited 5K target specialist with its saved optimizer, scheduler,
# and RNG state. New output directories preserve the selected 5K checkpoints.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
PYTHON="${LEROBOT_PYTHON:-/home/ubuntu/miniconda3/envs/lerobot/bin/python}"
SOURCE_STEP="${SOURCE_STEP:-5000}"
TARGET_STEP="${TARGET_STEP:-10000}"

if ! [[ "$SOURCE_STEP" =~ ^[1-9][0-9]*$ && "$TARGET_STEP" =~ ^[1-9][0-9]*$ ]]; then
  echo "SOURCE_STEP and TARGET_STEP must be positive integers" >&2
  exit 2
fi
if (( TARGET_STEP <= SOURCE_STEP )); then
  echo "TARGET_STEP must be greater than SOURCE_STEP" >&2
  exit 2
fi

export PYTHONPATH="$LEROBOT_ROOT/src:$PROJECT_ROOT/scripts"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

resume_specialist() {
  local target="$1"
  local source_dir="$PROJECT_ROOT/outputs/G1FINAL-01_target_specialist_${target}_${SOURCE_STEP}step"
  local source_checkpoint="$source_dir/checkpoints/$(printf '%06d' "$SOURCE_STEP")"
  local config_path="$source_checkpoint/pretrained_model/train_config.json"
  local state_path="$source_checkpoint/training_state/training_step.json"
  local output_dir="$PROJECT_ROOT/outputs/G1FINAL-01_target_specialist_${target}_${TARGET_STEP}step"
  local final_model="$output_dir/checkpoints/$(printf '%06d' "$TARGET_STEP")/pretrained_model/model.safetensors"
  local log_path="$PROJECT_ROOT/outputs/G1FINAL-01_target_specialist_${target}_${TARGET_STEP}step.log"

  if [[ -f "$final_model" ]]; then
    echo "SPECIALIST_RESUME_EXISTS target=$target checkpoint=$final_model"
    return
  fi
  if [[ ! -f "$config_path" || ! -f "$state_path" ]]; then
    echo "Missing resumable checkpoint for $target: $source_checkpoint" >&2
    exit 2
  fi
  if ! grep -Eq "\"step\"[[:space:]]*:[[:space:]]*$SOURCE_STEP" "$state_path"; then
    echo "Unexpected source training step for $target: $state_path" >&2
    exit 2
  fi
  if [[ -e "$output_dir" ]]; then
    echo "Output exists without a complete checkpoint: $output_dir" >&2
    exit 2
  fi

  echo "SPECIALIST_RESUME_START target=$target source_step=$SOURCE_STEP target_step=$TARGET_STEP"
  "$PYTHON" -m lerobot.scripts.train \
    --config_path="$config_path" \
    --resume=true \
    --steps="$TARGET_STEP" \
    --save_freq="$TARGET_STEP" \
    --log_freq=20 \
    --eval_freq=0 \
    --wandb.enable=false \
    --output_dir="$output_dir" \
    2>&1 | tee "$log_path"
  if [[ ! -f "$final_model" ]]; then
    echo "Resume did not produce the expected checkpoint: $final_model" >&2
    exit 3
  fi
  echo "SPECIALIST_RESUME_COMPLETE target=$target checkpoint=$final_model"
}

resume_specialist red_triangle
resume_specialist yellow_rod
resume_specialist green_cube
echo "G1_LANGUAGE_TARGET_SPECIALISTS_10000_COMPLETE=1"
