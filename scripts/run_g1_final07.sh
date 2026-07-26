#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
ROUTER_CHECKPOINT="${ROUTER_CHECKPOINT:-$PROJECT_ROOT/outputs/G1LANG-34_context_target_decoder_44000step/checkpoints/044000/pretrained_model}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/G1FINAL-07_router_classical_3ep}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export LEROBOT_SITE_PACKAGES="${LEROBOT_SITE_PACKAGES:-/home/ubuntu/miniconda3/envs/lerobot/lib/python3.10/site-packages}"
export PYTHONPATH="$LEROBOT_ROOT/src:$PROJECT_ROOT/scripts"

if [[ -e "$OUTPUT_DIR/summary.json" ]]; then
  echo "Refusing to overwrite existing result: $OUTPUT_DIR/summary.json" >&2
  exit 1
fi

exec "$PYTHON" "$PROJECT_ROOT/scripts/evaluate_g1_language_router_classical.py" \
  --asset "$PROJECT_ROOT/outputs/G1WH-15_continuous_pregrasp_trajectory/x055_simultaneous/g1_warehouse_pregrasp.xml" \
  --router-checkpoint "$ROUTER_CHECKPOINT" \
  --train-root "$LEROBOT_ROOT/datasets/local/g1_language_paired_train" \
  --output-dir "$OUTPUT_DIR" \
  --episodes "${EPISODES:-3}" \
  --device "${DEVICE:-cuda}"
