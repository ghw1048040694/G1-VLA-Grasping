#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
ASSET="${G1_ASSET:-/home/ubuntu/unitree_lerobot/unitree_lerobot/eval_robot/assets/g1/g1_body29_hand14.xml}"
OUTPUT_DIR="$PROJECT_ROOT/outputs/G1WH-01_asset_contract_audit"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

"$PYTHON" "$PROJECT_ROOT/scripts/audit_g1_warehouse_asset.py" \
  --asset "$ASSET" \
  --output-dir "$OUTPUT_DIR"
