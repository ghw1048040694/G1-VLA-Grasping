#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
ROOT="$PROJECT_ROOT/outputs/G1WM-04_deep_ensemble_uncertainty"
TRANSITIONS="$PROJECT_ROOT/outputs/G1WM-02_control_rate_interventions/summary.json"
SCENE="$PROJECT_ROOT/outputs/G1WH-40_scaled_expert_dataset/episode_0000/g1_assisted_tote_lift.xml"
BASELINE="$PROJECT_ROOT/outputs/G1WM-02_intervention_world_model/best_model.pt"

mkdir -p "$ROOT"

for SEED in 4304 4305; do
  MEMBER="$ROOT/member_seed${SEED}"
  if [[ ! -f "$MEMBER/best_model.pt" ]]; then
    "$PYTHON" "$PROJECT_ROOT/scripts/train_g1_hybrid_world_model.py" \
      --transition-summary "$TRANSITIONS" \
      --joint-limit-scene "$SCENE" \
      --baseline-checkpoint "$BASELINE" \
      --output-dir "$MEMBER" \
      --experiment-id "G1WM-04-ensemble-member-${SEED}" \
      --train-sources 100 \
      --steps 12000 \
      --batch-size 128 \
      --train-horizon 20 \
      --rollout-horizons 1 5 10 20 \
      --hidden-dim 512 \
      --depth 4 \
      --learning-rate 3e-4 \
      --eval-freq 500 \
      --seed "$SEED" \
      --device cuda \
      2>&1 | tee "$ROOT/member_seed${SEED}.log"
  fi
done

"$PYTHON" "$PROJECT_ROOT/scripts/evaluate_g1_world_model_ensemble.py" \
  --transition-summary "$TRANSITIONS" \
  --checkpoint "$PROJECT_ROOT/outputs/G1WM-03_hybrid_constraint_world_model/best_model.pt" \
  --checkpoint "$ROOT/member_seed4304/best_model.pt" \
  --checkpoint "$ROOT/member_seed4305/best_model.pt" \
  --train-sources 100 \
  --horizons 5 20 \
  --max-starts 1024 \
  --device cuda \
  --output "$ROOT/ensemble_summary.json" \
  2>&1 | tee "$ROOT/ensemble_evaluation.log"
