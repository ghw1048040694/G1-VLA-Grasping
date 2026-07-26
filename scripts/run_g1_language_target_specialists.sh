#!/usr/bin/env bash
set -euo pipefail

# Train one visual-action specialist per object target. The episode lists are
# fixed by the exact-scene triplet contract: red, yellow, green repeat in order.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
PYTHON="${LEROBOT_PYTHON:-/home/ubuntu/miniconda3/envs/lerobot/bin/python}"
TRAIN_ROOT="${TRAIN_ROOT:-$LEROBOT_ROOT/datasets/local/g1_language_paired_train}"
REPO_ID="${REPO_ID:-local/g1_language_paired_train}"
SOURCE_POLICY="${SOURCE_POLICY:-$PROJECT_ROOT/outputs/G1LANG-12_paired_smolvla_20000step/checkpoints/020000/pretrained_model}"
STEPS="${STEPS:-10000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"

if [[ ! -f "$SOURCE_POLICY/model.safetensors" ]]; then
  echo "Missing source policy: $SOURCE_POLICY" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_ROOT/meta/info.json" ]]; then
  echo "Missing LeRobot dataset metadata: $TRAIN_ROOT" >&2
  exit 2
fi
if ! [[ "$STEPS" =~ ^[1-9][0-9]*$ && "$SAVE_FREQ" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEPS and SAVE_FREQ must be positive integers" >&2
  exit 2
fi

export PYTHONPATH="$LEROBOT_ROOT/src:$PROJECT_ROOT/scripts"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# 50 episodes per target in the 150-episode training split.
RED_EPISODES='[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,84,87,90,93,96,99,102,105,108,111,114,117,120,123,126,129,132,135,138,141,144,147]'
YELLOW_EPISODES='[1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52,55,58,61,64,67,70,73,76,79,82,85,88,91,94,97,100,103,106,109,112,115,118,121,124,127,130,133,136,139,142,145,148]'
GREEN_EPISODES='[2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50,53,56,59,62,65,68,71,74,77,80,83,86,89,92,95,98,101,104,107,110,113,116,119,122,125,128,131,134,137,140,143,146,149]'

train_specialist() {
  local target="$1"
  local episodes="$2"
  local output_dir="$PROJECT_ROOT/outputs/G1FINAL-01_target_specialist_${target}_${STEPS}step"
  local log_path="$PROJECT_ROOT/outputs/G1FINAL-01_target_specialist_${target}.log"
  local final_checkpoint="$output_dir/checkpoints/$(printf '%06d' "$STEPS")/pretrained_model/model.safetensors"

  if [[ -f "$final_checkpoint" ]]; then
    echo "SPECIALIST_EXISTS target=$target checkpoint=$final_checkpoint"
    return
  fi
  if [[ -e "$output_dir" ]]; then
    echo "Output exists without a complete checkpoint: $output_dir" >&2
    echo "Remove only this incomplete experiment directory before retrying." >&2
    exit 2
  fi

  echo "SPECIALIST_TRAIN_START target=$target steps=$STEPS episodes=$episodes"
  set -o pipefail
  "$PYTHON" -m lerobot.scripts.train \
    --policy.path="$SOURCE_POLICY" \
    --policy.device=cuda \
    --policy.use_amp=false \
    --policy.push_to_hub=false \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$TRAIN_ROOT" \
    --dataset.episodes="$episodes" \
    --dataset.video_backend=pyav \
    --batch_size=2 \
    --steps="$STEPS" \
    --num_workers=0 \
    --log_freq=20 \
    --save_checkpoint=true \
    --save_freq="$SAVE_FREQ" \
    --eval_freq=0 \
    --wandb.enable=false \
    --output_dir="$output_dir" \
    2>&1 | tee "$log_path"
  if [[ ! -f "$final_checkpoint" ]]; then
    echo "Specialist did not produce the expected checkpoint: $final_checkpoint" >&2
    exit 3
  fi
  echo "SPECIALIST_TRAIN_COMPLETE target=$target checkpoint=$final_checkpoint"
}

mkdir -p "$PROJECT_ROOT/outputs"
train_specialist red_triangle "$RED_EPISODES"
train_specialist yellow_rod "$YELLOW_EPISODES"
train_specialist green_cube "$GREEN_EPISODES"
echo "G1_LANGUAGE_TARGET_SPECIALISTS_COMPLETE=1"
