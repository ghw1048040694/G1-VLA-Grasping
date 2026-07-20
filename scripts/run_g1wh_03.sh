#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
ASSET="$PROJECT_ROOT/outputs/G1WH-02_three_camera_interface/g1_warehouse_cameras.xml"
OUTPUT_DIR="$PROJECT_ROOT/outputs/G1WH-03_bimanual_actuation_smoke_test"

if [[ ! -f "$ASSET" ]]; then
  echo "Missing G1WH-02 asset: $ASSET" >&2
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"

"$PYTHON" "$PROJECT_ROOT/scripts/validate_g1_bimanual_actuation.py" \
  --asset "$ASSET" \
  --output-dir "$OUTPUT_DIR"
