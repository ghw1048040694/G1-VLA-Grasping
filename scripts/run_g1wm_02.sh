#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
TRANSITIONS="$PROJECT_ROOT/outputs/G1WM-02_control_rate_interventions"
OUTPUT="$PROJECT_ROOT/outputs/G1WM-02_intervention_world_model"

"$PYTHON" "$PROJECT_ROOT/scripts/collect_g1_world_model_transitions.py" \
  --source-summary "$PROJECT_ROOT/outputs/G1WH-40_scaled_expert_dataset/assisted_lift_dataset_summary.json" \
  --output-dir "$TRANSITIONS" \
  --episodes 120 \
  --control-fps 15 \
  --noise-correlation 0.90 \
  --seed 4201 \
  --resume \
  --experiment-id G1WM-02-control-rate-action-interventions

"$PYTHON" "$PROJECT_ROOT/scripts/train_g1_world_model_interventions.py" \
  --transition-summary "$TRANSITIONS/summary.json" \
  --baseline-checkpoint "$PROJECT_ROOT/outputs/G1WM-01_action_conditioned_world_model/best_model.pt" \
  --output-dir "$OUTPUT" \
  --experiment-id G1WM-02-intervention-data-aggregation \
  --train-sources 100 \
  --steps 15000 \
  --batch-size 256 \
  --train-horizon 10 \
  --rollout-horizons 1 5 10 20 \
  --hidden-dim 512 \
  --depth 4 \
  --learning-rate 3e-4 \
  --eval-freq 500 \
  --device cuda \
  2>&1 | tee "$PROJECT_ROOT/outputs/G1WM-02.log"
