#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LEROBOT_SITE_PACKAGES="${LEROBOT_SITE_PACKAGES:-/home/ubuntu/miniconda3/envs/lerobot/lib/python3.10/site-packages}"
export PYTHONPATH="$LEROBOT_ROOT/src:$PROJECT_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$PROJECT_ROOT/scripts/collect_g1wh_policy_recovery_dataset.py" \
  --asset "$PROJECT_ROOT/outputs/G1WH-15_continuous_pregrasp_trajectory/x055_simultaneous/g1_warehouse_pregrasp.xml" \
  --source-summary "$PROJECT_ROOT/outputs/G1WH-23R_bounded_assisted_lift_dataset/assisted_lift_dataset_summary.json" \
  --checkpoint "$PROJECT_ROOT/outputs/G1WH-29_phase_smolvla_1000step/checkpoints/000500/pretrained_model" \
  --train-repo-id local/g1_assisted_lift_phase_train \
  --train-root "$LEROBOT_ROOT/datasets/local/g1_assisted_lift_phase_train" \
  --source-episodes 16 \
  --seed-variants 2 \
  --control-fps 15 \
  --replan-steps 5 \
  --duration-s 8 \
  --takeover-delay-s 0.67 \
  --recovery-blend-s 2.0 \
  --device cuda \
  --overwrite \
  --output-dir "$PROJECT_ROOT/outputs/G1WH-33_policy_recovery_dataset"
