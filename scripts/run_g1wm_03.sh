#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"

"$PYTHON" "$PROJECT_ROOT/scripts/train_g1_hybrid_world_model.py" \
  --transition-summary "$PROJECT_ROOT/outputs/G1WM-02_control_rate_interventions/summary.json" \
  --joint-limit-scene "$PROJECT_ROOT/outputs/G1WH-40_scaled_expert_dataset/episode_0000/g1_assisted_tote_lift.xml" \
  --baseline-checkpoint "$PROJECT_ROOT/outputs/G1WM-02_intervention_world_model/best_model.pt" \
  --output-dir "$PROJECT_ROOT/outputs/G1WM-03_hybrid_constraint_world_model" \
  --experiment-id G1WM-03-hybrid-constraint-aware-dynamics \
  --train-sources 100 \
  --steps 20000 \
  --batch-size 128 \
  --train-horizon 20 \
  --rollout-horizons 1 5 10 20 \
  --hidden-dim 512 \
  --depth 4 \
  --learning-rate 3e-4 \
  --eval-freq 500 \
  --device cuda \
  2>&1 | tee "$PROJECT_ROOT/outputs/G1WM-03.log"
