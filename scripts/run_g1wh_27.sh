#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
TRAIN_OUTPUT="$PROJECT_ROOT/outputs/G1WH-25_smolvla_upper_body_1000step"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LEROBOT_SITE_PACKAGES="${LEROBOT_SITE_PACKAGES:-/home/ubuntu/miniconda3/envs/lerobot/lib/python3.10/site-packages}"
export PYTHONPATH="$LEROBOT_ROOT/src:$PROJECT_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$PROJECT_ROOT/scripts/evaluate_g1wh_smolvla_closed_loop.py" \
  --asset "$PROJECT_ROOT/outputs/G1WH-15_continuous_pregrasp_trajectory/x055_simultaneous/g1_warehouse_pregrasp.xml" \
  --source-summary "$PROJECT_ROOT/outputs/G1WH-23R_bounded_assisted_lift_dataset/assisted_lift_dataset_summary.json" \
  --checkpoint-500 "$TRAIN_OUTPUT/checkpoints/000500/pretrained_model" \
  --checkpoint-1000 "$TRAIN_OUTPUT/checkpoints/001000/pretrained_model" \
  --train-root "$LEROBOT_ROOT/datasets/local/g1_assisted_lift_train" \
  --validation-start 16 \
  --episodes 4 \
  --control-fps 15 \
  --replan-steps 5 \
  --duration-s 10 \
  --device cuda \
  --output-dir "$PROJECT_ROOT/outputs/G1WH-27_smolvla_closed_loop_validation"
