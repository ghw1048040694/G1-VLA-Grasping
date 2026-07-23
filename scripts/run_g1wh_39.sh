#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
BASELINE_POLICY="$PROJECT_ROOT/outputs/G1WH-29_phase_smolvla_1000step/checkpoints/000500/pretrained_model"
OUTPUT_DIR="$PROJECT_ROOT/outputs/G1WH-39_terminal_hold_controller"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LEROBOT_SITE_PACKAGES="${LEROBOT_SITE_PACKAGES:-/home/ubuntu/miniconda3/envs/lerobot/lib/python3.10/site-packages}"
export PYTHONPATH="$LEROBOT_ROOT/src:$PROJECT_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$PROJECT_ROOT/scripts/evaluate_g1wh_smolvla_closed_loop.py" \
  --asset "$PROJECT_ROOT/outputs/G1WH-15_continuous_pregrasp_trajectory/x055_simultaneous/g1_warehouse_pregrasp.xml" \
  --source-summary "$PROJECT_ROOT/outputs/G1WH-23R_bounded_assisted_lift_dataset/assisted_lift_dataset_summary.json" \
  --checkpoint "baseline500=$BASELINE_POLICY" \
  --experiment-id G1WH-39-terminal-hold-controller \
  --train-repo-id local/g1_assisted_lift_phase_recovery_train \
  --train-root "$LEROBOT_ROOT/datasets/local/g1_assisted_lift_phase_recovery_train" \
  --validation-start 16 \
  --episodes 4 \
  --control-fps 15 \
  --replan-steps 5 \
  --duration-s 10 \
  --phase-language-scheduler \
  --terminal-hold-controller \
  --hold-gain-scale 0.67 \
  --assist-solref-timeconst 0.05 \
  --device cuda \
  --output-dir "$OUTPUT_DIR"
