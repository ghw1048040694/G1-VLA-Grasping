#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
PYTHON="${LEROBOT_PYTHON:-/home/ubuntu/miniconda3/envs/lerobot/bin/python}"
DATASET_ROOT="$LEROBOT_ROOT/datasets/local/g1_assisted_lift_train"
ADAPTED_POLICY="$PROJECT_ROOT/outputs/G1WH-25_adapted_smolvla_policy"
TRAIN_OUTPUT="$PROJECT_ROOT/outputs/G1WH-25_smolvla_upper_body_1000step"

export PYTHONPATH="$LEROBOT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$PROJECT_ROOT/scripts/prepare_g1wh_smolvla_policy.py" \
  --base-policy "$LEROBOT_ROOT/models/smolvla_base" \
  --dataset-repo-id local/g1_assisted_lift_train \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$ADAPTED_POLICY"

cd "$LEROBOT_ROOT"
"$PYTHON" -m lerobot.scripts.train \
  --policy.path="$ADAPTED_POLICY" \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/g1_assisted_lift_train \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --batch_size=2 \
  --steps=1000 \
  --num_workers=0 \
  --log_freq=10 \
  --save_checkpoint=true \
  --save_freq=500 \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir="$TRAIN_OUTPUT"
