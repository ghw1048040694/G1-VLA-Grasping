#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
PYTHON="${LEROBOT_PYTHON:-/home/ubuntu/miniconda3/envs/lerobot/bin/python}"
TRAIN_ROOT="$LEROBOT_ROOT/datasets/local/g1_assisted_lift_train"
VAL_ROOT="$LEROBOT_ROOT/datasets/local/g1_assisted_lift_val"
TRAIN_OUTPUT="$PROJECT_ROOT/outputs/G1WH-25_smolvla_upper_body_1000step"

export PYTHONPATH="$LEROBOT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$PROJECT_ROOT/scripts/evaluate_g1wh_smolvla_heldout.py" \
  --base-policy "$PROJECT_ROOT/outputs/G1WH-25_adapted_smolvla_policy" \
  --checkpoint-500 "$TRAIN_OUTPUT/checkpoints/000500/pretrained_model" \
  --checkpoint-1000 "$TRAIN_OUTPUT/checkpoints/001000/pretrained_model" \
  --train-root "$TRAIN_ROOT" \
  --val-root "$VAL_ROOT" \
  --samples 128 \
  --batch-size 2 \
  --device cuda \
  --output-dir "$PROJECT_ROOT/outputs/G1WH-26_heldout_action_evaluation"
