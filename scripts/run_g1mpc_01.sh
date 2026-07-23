#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LEROBOT_SITE_PACKAGES="${LEROBOT_SITE_PACKAGES:-/home/ubuntu/miniconda3/envs/lerobot/lib/python3.10/site-packages}"
export PYTHONPATH="$LEROBOT_ROOT/src:$PROJECT_ROOT/scripts"

"$PYTHON" "$PROJECT_ROOT/scripts/evaluate_g1wh_smolvla_closed_loop.py" \
  --asset "$PROJECT_ROOT/outputs/G1WH-15_continuous_pregrasp_trajectory/x055_simultaneous/g1_warehouse_pregrasp.xml" \
  --source-summary "$PROJECT_ROOT/outputs/G1WH-40_scaled_expert_dataset/assisted_lift_dataset_summary.json" \
  --checkpoint "scaled20000=$PROJECT_ROOT/outputs/G1WH-40_scaled_smolvla_20000step/checkpoints/020000/pretrained_model" \
  --experiment-id G1MPC-01-vla-world-model-candidate-ranking \
  --train-repo-id local/g1_scaled_phase_recovery_train \
  --train-root "$LEROBOT_ROOT/datasets/local/g1_scaled_phase_recovery_train" \
  --validation-start 100 \
  --episodes 4 \
  --control-fps 15 \
  --replan-steps 5 \
  --duration-s 10 \
  --phase-language-scheduler \
  --world-model-checkpoint "$PROJECT_ROOT/outputs/G1WM-03_hybrid_constraint_world_model/best_model.pt" \
  --world-model-candidates 4 \
  --world-model-horizon 5 \
  --compare-world-model-planner \
  --device cuda \
  --output-dir "$PROJECT_ROOT/outputs/G1MPC-01_vla_world_model_screening" \
  2>&1 | tee "$PROJECT_ROOT/outputs/G1MPC-01.log"
