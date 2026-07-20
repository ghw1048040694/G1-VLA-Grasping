#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
ASSET="$PROJECT_ROOT/outputs/G1WH-13_dynamic_tote_scene_contract/g1_warehouse_dynamic_tote.xml"
OUTPUT_DIR="$PROJECT_ROOT/outputs/G1WH-14_bimanual_reachability_scan"

if [[ ! -f "$ASSET" ]]; then
  echo "Missing G1WH-13 dynamic tote asset: $ASSET" >&2
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"

"$PYTHON" "$PROJECT_ROOT/scripts/audit_g1_bimanual_reachability.py" \
  --source-asset "$ASSET" \
  --output-dir "$OUTPUT_DIR"
