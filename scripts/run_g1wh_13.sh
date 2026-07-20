#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
ASSET="$PROJECT_ROOT/outputs/G1WH-08_staged_realism_validation/g1_warehouse_fixed_base.xml"
OUTPUT_DIR="$PROJECT_ROOT/outputs/G1WH-13_dynamic_tote_scene_contract"

if [[ ! -f "$ASSET" ]]; then
  echo "Missing G1WH-08 fixed-base asset: $ASSET" >&2
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"

"$PYTHON" "$PROJECT_ROOT/scripts/build_g1_dynamic_tote_scene.py" \
  --source-asset "$ASSET" \
  --output-dir "$OUTPUT_DIR"
