#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-/home/ubuntu/miniconda3/envs/lerobot/bin/python}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$PROJECT_ROOT/outputs/G1LANG-30_action_expert_targeted_33000step/checkpoints/033000/pretrained_model}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/G1LANG-31_target_suffix_expert_34000step}"
LOG_PATH="${LOG_PATH:-$PROJECT_ROOT/outputs/G1LANG-31_target_suffix_expert.log}"

export PYTHONPATH="$LEROBOT_ROOT/src:$PROJECT_ROOT/scripts"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/g1_hf_datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$PROJECT_ROOT/outputs"
exec > >(tee -a "$LOG_PATH") 2>&1

"$LEROBOT_PYTHON" "$PROJECT_ROOT/scripts/train_g1_language_triplet_curriculum.py" \
  --checkpoint "$SOURCE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --total-steps 34000 \
  --max-frame-exclusive 16 \
  --save-freq 1000 \
  --log-freq 20 \
  --contrastive-margin 0.01 \
  --contrastive-weight 1.0 \
  --target-classification-weight 1.0

echo "G1_LANGUAGE_TARGET_SUFFIX_EXPERT_COMPLETE=1"
