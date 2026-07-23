#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
BASELINE_POLICY="$PROJECT_ROOT/outputs/G1WH-29_phase_smolvla_1000step/checkpoints/000500/pretrained_model"
RECOVERY_POLICY="$PROJECT_ROOT/outputs/G1WH-34R_recovery_smolvla_3000step/checkpoints/003000/pretrained_model"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/G1WH-39_constraint_softness_ablation"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LEROBOT_SITE_PACKAGES="${LEROBOT_SITE_PACKAGES:-/home/ubuntu/miniconda3/envs/lerobot/lib/python3.10/site-packages}"
export PYTHONPATH="$LEROBOT_ROOT/src:$PROJECT_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

run_condition() {
  local name="$1"
  local timeconst="$2"

  "$PYTHON" "$PROJECT_ROOT/scripts/evaluate_g1wh_smolvla_closed_loop.py" \
    --asset "$PROJECT_ROOT/outputs/G1WH-15_continuous_pregrasp_trajectory/x055_simultaneous/g1_warehouse_pregrasp.xml" \
    --source-summary "$PROJECT_ROOT/outputs/G1WH-23R_bounded_assisted_lift_dataset/assisted_lift_dataset_summary.json" \
    --checkpoint "baseline500=$BASELINE_POLICY" \
    --checkpoint "recovery3000=$RECOVERY_POLICY" \
    --hybrid "baseline_to_recovery3000_smooth025_${name}=baseline500,recovery3000" \
    --skip-standalone \
    --experiment-id "G1WH-39-constraint-softness-ablation-${name}" \
    --train-repo-id local/g1_assisted_lift_phase_recovery_train \
    --train-root "$LEROBOT_ROOT/datasets/local/g1_assisted_lift_phase_recovery_train" \
    --validation-start 16 \
    --episodes 4 \
    --control-fps 15 \
    --replan-steps 5 \
    --duration-s 10 \
    --phase-language-scheduler \
    --hold-action-blend-alpha 0.25 \
    --hold-gain-scale 1.0 \
    --assist-solref-timeconst "$timeconst" \
    --device cuda \
    --output-dir "$OUTPUT_ROOT/$name"
}

run_condition soft_constraint050 0.05
run_condition soft_constraint100 0.10
