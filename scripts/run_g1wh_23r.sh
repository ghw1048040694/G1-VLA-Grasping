#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
ASSET="$PROJECT_ROOT/outputs/G1WH-15_continuous_pregrasp_trajectory/x055_simultaneous/g1_warehouse_pregrasp.xml"
OUTPUT_DIR="$PROJECT_ROOT/outputs/G1WH-23R_bounded_assisted_lift_dataset"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

"$PYTHON" "$PROJECT_ROOT/scripts/collect_g1_assisted_lift_dataset.py" \
  --python "$PYTHON" \
  --collector "$PROJECT_ROOT/scripts/run_g1_assisted_tote_lift.py" \
  --asset "$ASSET" \
  --output-dir "$OUTPUT_DIR" \
  --episodes 20 \
  --seed 2207 \
  --tote-x-min 0.43 \
  --tote-x-max 0.45 \
  --experiment-id G1WH-23R-bounded-assisted-lift-dataset
