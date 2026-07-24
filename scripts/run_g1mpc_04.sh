#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
HELDOUT_ROOT="${HELDOUT_ROOT:-$PROJECT_ROOT/outputs/G1MPC-04_heldout_source}"
SOURCE_SUMMARY="${SOURCE_SUMMARY:-$HELDOUT_ROOT/assisted_lift_dataset_summary.json}"
EPISODES="${EPISODES:-20}"
VALIDATION_START="${VALIDATION_START:-0}"
DURATION_S="${DURATION_S:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/G1MPC-04_rank_filtered_heldout}"
LOG_PATH="${LOG_PATH:-$PROJECT_ROOT/outputs/G1MPC-04.log}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LEROBOT_SITE_PACKAGES="${LEROBOT_SITE_PACKAGES:-/home/ubuntu/miniconda3/envs/lerobot/lib/python3.10/site-packages}"
export PYTHONPATH="$LEROBOT_ROOT/src:$PROJECT_ROOT/scripts"

if [[ "${SKIP_COLLECTION:-0}" != "1" && ! -f "$SOURCE_SUMMARY" ]]; then
  "$PYTHON" "$PROJECT_ROOT/scripts/collect_g1_assisted_lift_dataset.py" \
    --python "$PYTHON" \
    --collector "$PROJECT_ROOT/scripts/run_g1_assisted_tote_lift.py" \
    --asset "$PROJECT_ROOT/outputs/G1WH-15_continuous_pregrasp_trajectory/x055_simultaneous/g1_warehouse_pregrasp.xml" \
    --output-dir "$HELDOUT_ROOT" \
    --episodes 20 \
    --seed 6107 \
    --tote-x-min 0.43 \
    --tote-x-max 0.45 \
    --resume-successful \
    --experiment-id G1MPC-04-independent-heldout-source
fi

"$PYTHON" "$PROJECT_ROOT/scripts/evaluate_g1wh_smolvla_closed_loop.py" \
  --asset "$PROJECT_ROOT/outputs/G1WH-15_continuous_pregrasp_trajectory/x055_simultaneous/g1_warehouse_pregrasp.xml" \
  --source-summary "$SOURCE_SUMMARY" \
  --checkpoint "scaled20000=$PROJECT_ROOT/outputs/G1WH-40_scaled_smolvla_20000step/checkpoints/020000/pretrained_model" \
  --experiment-id G1MPC-04-rank-filtered-independent-heldout \
  --train-repo-id local/g1_scaled_phase_recovery_train \
  --train-root "$LEROBOT_ROOT/datasets/local/g1_scaled_phase_recovery_train" \
  --validation-start "$VALIDATION_START" \
  --episodes "$EPISODES" \
  --control-fps 15 \
  --replan-steps 5 \
  --duration-s "$DURATION_S" \
  --phase-language-scheduler \
  --world-model-checkpoint "$PROJECT_ROOT/outputs/G1WM-03_hybrid_constraint_world_model/best_model.pt" \
  --world-model-ensemble-checkpoint "$PROJECT_ROOT/outputs/G1WM-04_deep_ensemble_uncertainty/member_seed4304/best_model.pt" \
  --world-model-ensemble-checkpoint "$PROJECT_ROOT/outputs/G1WM-04_deep_ensemble_uncertainty/member_seed4305/best_model.pt" \
  --world-model-uncertainty-keep-fraction 0.5 \
  --world-model-candidates 4 \
  --world-model-horizon 5 \
  --world-model-batched-candidates \
  --planner-body-safety-margin-fraction 0.03 \
  --planner-hand-safety-margin-fraction 0.08 \
  --compare-world-model-planner \
  --compare-unfiltered-ensemble-planner \
  --device cuda \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_PATH"
