#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-/home/ubuntu/miniconda3/envs/lerobot/bin/python}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/outputs/G1LANG-31_target_suffix_expert_34000step/checkpoints/034000/pretrained_model}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/G1LANG-32A_forward_path_audit_034000}"
LOG_PATH="${LOG_PATH:-$PROJECT_ROOT/outputs/G1LANG-32A_forward_path_audit.log}"

export PYTHONPATH="$LEROBOT_ROOT/src:$PROJECT_ROOT/scripts"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/g1_hf_datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$PROJECT_ROOT/outputs"
if [[ -e "$OUTPUT_DIR/summary.json" ]]; then
  echo "Refusing to overwrite existing audit: $OUTPUT_DIR/summary.json" >&2
  exit 1
fi
exec > >(tee -a "$LOG_PATH") 2>&1

"$LEROBOT_PYTHON" "$PROJECT_ROOT/scripts/audit_g1_language_forward_path.py" \
  --checkpoint "$CHECKPOINT" \
  --train-root /home/ubuntu/lerobot/datasets/local/g1_language_paired_train \
  --val-root /home/ubuntu/lerobot/datasets/local/g1_language_paired_val \
  --output-dir "$OUTPUT_DIR" \
  --scene-index 0 \
  --frame-index 0 \
  --seed 20260727 \
  --time 1.0

echo "G1_LANGUAGE_FORWARD_AUDIT_COMPLETE=1"
