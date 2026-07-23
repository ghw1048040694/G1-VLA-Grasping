#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
OUTPUT_DIR="$PROJECT_ROOT/outputs/G1WM-01_action_conditioned_world_model"

"$PYTHON" "$PROJECT_ROOT/scripts/train_g1_world_model.py" \
  --source-summary "$PROJECT_ROOT/outputs/G1WH-40_scaled_expert_dataset/assisted_lift_dataset_summary.json" \
  --output-dir "$OUTPUT_DIR" \
  --experiment-id G1WM-01-action-conditioned-state-world-model \
  --train-episodes 100 \
  --val-episodes 20 \
  --steps 10000 \
  --batch-size 256 \
  --train-horizon 10 \
  --rollout-horizons 1 5 10 20 \
  --hidden-dim 512 \
  --depth 4 \
  --learning-rate 3e-4 \
  --eval-freq 500 \
  --device cuda \
  2>&1 | tee "$PROJECT_ROOT/outputs/G1WM-01.log"
