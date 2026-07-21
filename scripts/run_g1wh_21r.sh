#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
ASSET="$PROJECT_ROOT/outputs/G1WH-15_continuous_pregrasp_trajectory/x055_simultaneous/g1_warehouse_pregrasp.xml"
TARGET_REPORT="$PROJECT_ROOT/outputs/G1WH-17_ik_task_weight_dynamic_sweep/orientation_weight_008/ik_target_report.json"
OUTPUT_DIR="$PROJECT_ROOT/outputs/G1WH-21R_randomized_expert_dataset"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

"$PYTHON" "$PROJECT_ROOT/scripts/collect_g1_randomized_pregrasp_dataset.py" \
  --python "$PYTHON" \
  --collector "$PROJECT_ROOT/scripts/run_g1_safe_reset_expert.py" \
  --asset "$ASSET" \
  --reachability-report "$TARGET_REPORT" \
  --output-dir "$OUTPUT_DIR" \
  --episodes 20 \
  --seed 2107 \
  --tote-x-min 0.52 \
  --tote-x-max 0.55 \
  --experiment-id G1WH-21R-randomized-expert-dataset
