#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
ASSET="$PROJECT_ROOT/outputs/G1WH-15_continuous_pregrasp_trajectory/x055_simultaneous/g1_warehouse_pregrasp.xml"
REPORT="$PROJECT_ROOT/outputs/G1WH-14_bimanual_reachability_scan/bimanual_reachability_audit.json"
OUTPUT_DIR="$PROJECT_ROOT/outputs/G1WH-16_pregrasp_hold_control_ablation"

if [[ ! -f "$ASSET" || ! -f "$REPORT" ]]; then
  echo "Missing G1WH-15 selected scene or G1WH-14 reachability report" >&2
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"

"$PYTHON" "$PROJECT_ROOT/scripts/run_g1_pregrasp_hold_ablation.py" \
  --python "$PYTHON" \
  --runner "$PROJECT_ROOT/scripts/validate_g1_pregrasp_hold.py" \
  --asset "$ASSET" \
  --reachability-report "$REPORT" \
  --output-dir "$OUTPUT_DIR"
