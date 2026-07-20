#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
ASSET="$PROJECT_ROOT/outputs/G1WH-15_continuous_pregrasp_trajectory/x055_simultaneous/g1_warehouse_pregrasp.xml"
OUTPUT_DIR="$PROJECT_ROOT/outputs/G1WH-17_ik_task_weight_dynamic_sweep"

if [[ ! -f "$ASSET" ]]; then
  echo "Missing G1WH-15 0.55 m pregrasp scene" >&2
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"

"$PYTHON" "$PROJECT_ROOT/scripts/run_g1_ik_weight_hold_sweep.py" \
  --python "$PYTHON" \
  --hold-runner "$PROJECT_ROOT/scripts/validate_g1_pregrasp_hold.py" \
  --asset "$ASSET" \
  --output-dir "$OUTPUT_DIR"
