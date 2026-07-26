#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-/home/ubuntu/miniconda3/envs/lerobot/bin/python}"
SOURCE_CHECKPOINT="$PROJECT_ROOT/outputs/G1LANG-12_paired_smolvla_20000step/checkpoints/020000/pretrained_model"
OUTPUT_DIR="$PROJECT_ROOT/outputs/G1LANG-16_triplet_initial_curriculum_22000step"
LOG_PATH="$PROJECT_ROOT/outputs/G1LANG-16_triplet_initial_curriculum.log"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$LEROBOT_ROOT/src:$PROJECT_ROOT/scripts"

mkdir -p "$PROJECT_ROOT/outputs"
exec > >(tee -a "$LOG_PATH") 2>&1

"$LEROBOT_PYTHON" "$PROJECT_ROOT/scripts/train_g1_language_triplet_curriculum.py" \
  --checkpoint "$SOURCE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --total-steps 22000 \
  --max-frame-exclusive 16 \
  --save-freq 2000 \
  --log-freq 20

echo "G1_LANGUAGE_TRIPLET_CURRICULUM_COMPLETE=1"
