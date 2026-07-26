#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-/home/ubuntu/miniconda3/envs/lerobot/bin/python}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/outputs/G1LANG-34_context_target_decoder_44000step/checkpoints/044000/pretrained_model}"
GATE_ROOT="${GATE_ROOT:-$PROJECT_ROOT/outputs/G1LANG-34_context_target_gate_044000}"
LOG_PATH="${LOG_PATH:-$PROJECT_ROOT/outputs/G1LANG-34_context_target_gate.log}"

export PYTHONPATH="$LEROBOT_ROOT/src:$PROJECT_ROOT/scripts"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/g1_hf_datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -e "$GATE_ROOT/summary.json" ]]; then
  echo "Refusing to overwrite existing gate: $GATE_ROOT/summary.json" >&2
  exit 1
fi
mkdir -p "$GATE_ROOT"
exec > >(tee -a "$LOG_PATH") 2>&1

summaries=()
for seed in 20260727 20260728 20260729 20260730; do
  seed_dir="$GATE_ROOT/seed_${seed}"
  "$LEROBOT_PYTHON" "$PROJECT_ROOT/scripts/evaluate_g1_language_counterfactual.py" \
    --checkpoint "$CHECKPOINT" \
    --train-root /home/ubuntu/lerobot/datasets/local/g1_language_paired_train \
    --val-root /home/ubuntu/lerobot/datasets/local/g1_language_paired_val \
    --train-repo-id local/g1_language_paired_train \
    --val-repo-id local/g1_language_paired_val \
    --scenes 10 \
    --seed "$seed" \
    --output-dir "$seed_dir"
  summaries+=("$seed_dir/summary.json")
done

"$LEROBOT_PYTHON" "$PROJECT_ROOT/scripts/summarize_g1_language_gate.py" \
  --summaries "${summaries[@]}" \
  --output "$GATE_ROOT/summary.json" \
  --threshold 0.8

echo "G1_LANGUAGE_CONTEXT_TARGET_GATE_COMPLETE=1"
