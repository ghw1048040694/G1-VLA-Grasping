#!/usr/bin/env bash
set -euo pipefail

# Fixed fallback after the yellow specialist's seamless high-LR resume became
# non-finite. Resume model weights at 5K but intentionally reset optimization.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
PYTHON="${LEROBOT_PYTHON:-/home/ubuntu/miniconda3/envs/lerobot/bin/python}"
SOURCE_STEP=5000
MINIMUM_TARGET_UPDATES=10000
MAX_SKIPPED_BATCHES="${MAX_SKIPPED_BATCHES:-25}"
TARGET_STEP=$((MINIMUM_TARGET_UPDATES + MAX_SKIPPED_BATCHES))
CONTINUATION_UPDATES=$((TARGET_STEP - SOURCE_STEP))
PEAK_LR="${PEAK_LR:-2.5e-5}"
DECAY_LR="${DECAY_LR:-2.5e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-250}"
TARGETS="${TARGETS:-yellow_rod green_cube}"

export PYTHONPATH="$LEROBOT_ROOT/src:$PROJECT_ROOT/scripts"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LEROBOT_DISABLE_CUDNN_BENCHMARK="${LEROBOT_DISABLE_CUDNN_BENCHMARK:-1}"
export G1_NONFINITE_MAX_RETRIES="${G1_NONFINITE_MAX_RETRIES:-3}"
export G1_NONFINITE_MAX_SKIPPED_BATCHES="$MAX_SKIPPED_BATCHES"
export G1_MINIMUM_FINITE_UPDATES=$((MINIMUM_TARGET_UPDATES - SOURCE_STEP))
export G1_FORCE_EXPERT_FP32="${G1_FORCE_EXPERT_FP32:-1}"
export G1_TRAIN_STATE_PROJ="${G1_TRAIN_STATE_PROJ:-false}"
export G1_BOUNDED_FLOW="${G1_BOUNDED_FLOW:-1}"

for target in $TARGETS; do
  case "$target" in
    yellow_rod|green_cube) ;;
    *) echo "Unsupported stable-continuation target: $target" >&2; exit 2 ;;
  esac
  source_checkpoint="$PROJECT_ROOT/outputs/G1FINAL-01_target_specialist_${target}_${SOURCE_STEP}step/checkpoints/005000"
  config_path="$source_checkpoint/pretrained_model/train_config.json"
  output_dir="$PROJECT_ROOT/outputs/G1FINAL-01_target_specialist_${target}_${TARGET_STEP}step_stable"
  final_checkpoint="$output_dir/checkpoints/$(printf '%06d' "$TARGET_STEP")/pretrained_model/model.safetensors"
  log_path="$PROJECT_ROOT/outputs/G1FINAL-01_target_specialist_${target}_${TARGET_STEP}step_stable.log"
  contract_path="$output_dir/continuation_contract.json"

  if [[ -f "$final_checkpoint" && -f "$contract_path" ]]; then
    echo "STABLE_CONTINUATION_EXISTS target=$target checkpoint=$final_checkpoint"
    continue
  fi
  if [[ ! -f "$config_path" ]]; then
    echo "Missing source checkpoint config: $config_path" >&2
    exit 2
  fi
  if [[ -e "$output_dir" ]]; then
    echo "Output exists without a complete stable checkpoint: $output_dir" >&2
    exit 2
  fi

  export G1_CONTINUATION_CONTRACT_PATH="$contract_path"
  export G1_CONTINUATION_TARGET="$target"
  export G1_CONTINUATION_SOURCE="$source_checkpoint"
  export G1_CONTINUATION_SOURCE_STEP="$SOURCE_STEP"
  export G1_CONTINUATION_UPDATES="$CONTINUATION_UPDATES"
  export G1_CONTINUATION_TARGET_STEP="$TARGET_STEP"
  export G1_CONTINUATION_PEAK_LR="$PEAK_LR"
  export G1_CONTINUATION_WARMUP="$WARMUP_STEPS"

  echo "STABLE_CONTINUATION_START target=$target source_step=$SOURCE_STEP target_step=$TARGET_STEP peak_lr=$PEAK_LR"
  "$PYTHON" "$PROJECT_ROOT/scripts/train_g1_smolvla_stable_continuation.py" \
    --config_path="$config_path" \
    --resume=true \
    --policy.optimizer_lr="$PEAK_LR" \
    --policy.scheduler_warmup_steps="$WARMUP_STEPS" \
    --policy.scheduler_decay_steps="$CONTINUATION_UPDATES" \
    --policy.scheduler_decay_lr="$DECAY_LR" \
    --policy.train_state_proj="$G1_TRAIN_STATE_PROJ" \
    --optimizer.lr="$PEAK_LR" \
    --scheduler.peak_lr="$PEAK_LR" \
    --scheduler.decay_lr="$DECAY_LR" \
    --scheduler.num_warmup_steps="$WARMUP_STEPS" \
    --scheduler.num_decay_steps="$CONTINUATION_UPDATES" \
    --steps="$TARGET_STEP" \
    --save_freq="$TARGET_STEP" \
    --log_freq=20 \
    --eval_freq=0 \
    --wandb.enable=false \
    --output_dir="$output_dir" \
    2>&1 | tee "$log_path"
  if [[ ! -f "$final_checkpoint" || ! -f "$contract_path" ]]; then
    echo "Stable continuation did not produce complete evidence: $output_dir" >&2
    exit 3
  fi
  echo "STABLE_CONTINUATION_COMPLETE target=$target checkpoint=$final_checkpoint"
done

echo "G1_LANGUAGE_TARGET_STABLE_CONTINUATIONS_COMPLETE=1"
