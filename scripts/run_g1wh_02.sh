#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
SOURCE_ASSET="${G1_ASSET:-/home/ubuntu/unitree_lerobot/unitree_lerobot/eval_robot/assets/g1/g1_body29_hand14.xml}"
OUTPUT_DIR="$PROJECT_ROOT/outputs/G1WH-02_three_camera_interface"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

"$PYTHON" "$PROJECT_ROOT/scripts/build_g1_warehouse_cameras.py" \
  --source-asset "$SOURCE_ASSET" \
  --output-dir "$OUTPUT_DIR"
