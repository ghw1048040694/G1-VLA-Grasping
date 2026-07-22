#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
PYTHON="${LEROBOT_PYTHON:-/home/ubuntu/miniconda3/envs/lerobot/bin/python}"
TRAIN_ROOT="$LEROBOT_ROOT/datasets/local/g1_assisted_lift_phase_recovery_train"
ADAPTED_POLICY="$PROJECT_ROOT/outputs/G1WH-34R_adapted_recovery_smolvla_policy"
TRAIN_OUTPUT="$PROJECT_ROOT/outputs/G1WH-34R_recovery_smolvla_3000step"

export PYTHONPATH="$LEROBOT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$PROJECT_ROOT/scripts/prepare_g1wh_smolvla_policy.py" \
  --base-policy "$PROJECT_ROOT/outputs/G1WH-29_phase_smolvla_1000step/checkpoints/000500/pretrained_model" \
  --dataset-repo-id local/g1_assisted_lift_phase_recovery_train \
  --dataset-root "$TRAIN_ROOT" \
  --output-dir "$ADAPTED_POLICY" \
  --experiment-id G1WH-34R-normalization-corrected-recovery-finetuning \
  --refresh-normalization-stats \
  --overwrite

cd "$LEROBOT_ROOT"
"$PYTHON" -m lerobot.scripts.train \
  --policy.path="$ADAPTED_POLICY" \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/g1_assisted_lift_phase_recovery_train \
  --dataset.root="$TRAIN_ROOT" \
  --dataset.video_backend=pyav \
  --batch_size=2 \
  --steps=3000 \
  --num_workers=0 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=1000 \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir="$TRAIN_OUTPUT"
