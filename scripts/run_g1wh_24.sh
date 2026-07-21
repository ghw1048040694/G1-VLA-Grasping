#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
PYTHON="${LEROBOT_PYTHON:-/home/ubuntu/miniconda3/envs/lerobot/bin/python}"

export PYTHONPATH="$LEROBOT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$PROJECT_ROOT/scripts/convert_g1_assisted_lift_to_lerobot.py" \
  --source-root "$PROJECT_ROOT/outputs/G1WH-23R_bounded_assisted_lift_dataset" \
  --train-root "$LEROBOT_ROOT/datasets/local/g1_assisted_lift_train" \
  --val-root "$LEROBOT_ROOT/datasets/local/g1_assisted_lift_val" \
  --train-episodes 16 \
  --fps 15 \
  --output-dir "$PROJECT_ROOT/outputs/G1WH-24_lerobot_vla_dataset_contract"
