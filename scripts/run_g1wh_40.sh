#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
GENESIS_PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-/home/ubuntu/miniconda3/envs/lerobot/bin/python}"
ASSET="$PROJECT_ROOT/outputs/G1WH-15_continuous_pregrasp_trajectory/x055_simultaneous/g1_warehouse_pregrasp.xml"
SOURCE_ROOT="$PROJECT_ROOT/outputs/G1WH-40_scaled_expert_dataset"
TRAIN_ROOT="$LEROBOT_ROOT/datasets/local/g1_scaled_phase_recovery_train"
VAL_ROOT="$LEROBOT_ROOT/datasets/local/g1_scaled_phase_val"
ADAPTED_POLICY="$PROJECT_ROOT/outputs/G1WH-40_adapted_smolvla_policy"
TRAIN_OUTPUT="$PROJECT_ROOT/outputs/G1WH-40_scaled_smolvla_20000step"
EVAL_OUTPUT="$PROJECT_ROOT/outputs/G1WH-40_scaled_smolvla_closed_loop_20ep"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

"$GENESIS_PYTHON" "$PROJECT_ROOT/scripts/collect_g1_assisted_lift_dataset.py" \
  --python "$GENESIS_PYTHON" \
  --collector "$PROJECT_ROOT/scripts/run_g1_assisted_tote_lift.py" \
  --asset "$ASSET" \
  --output-dir "$SOURCE_ROOT" \
  --episodes 120 \
  --seed 4007 \
  --tote-x-min 0.43 \
  --tote-x-max 0.45 \
  --resume-successful \
  --experiment-id G1WH-40-scaled-expert-dataset

export PYTHONPATH="$LEROBOT_ROOT/src"

"$LEROBOT_PYTHON" "$PROJECT_ROOT/scripts/convert_g1_assisted_lift_to_lerobot.py" \
  --source-root "$SOURCE_ROOT" \
  --recovery-root "$PROJECT_ROOT/outputs/G1WH-33_policy_recovery_dataset" \
  --train-root "$TRAIN_ROOT" \
  --val-root "$VAL_ROOT" \
  --train-repo-id local/g1_scaled_phase_recovery_train \
  --val-repo-id local/g1_scaled_phase_val \
  --train-episodes 100 \
  --fps 15 \
  --phase-conditioned-language \
  --split-by-phase \
  --experiment-id G1WH-40-scaled-phase-recovery-dataset \
  --overwrite \
  --output-dir "$PROJECT_ROOT/outputs/G1WH-40_scaled_lerobot_dataset"

"$LEROBOT_PYTHON" "$PROJECT_ROOT/scripts/prepare_g1wh_smolvla_policy.py" \
  --base-policy "$LEROBOT_ROOT/models/smolvla_base" \
  --dataset-repo-id local/g1_scaled_phase_recovery_train \
  --dataset-root "$TRAIN_ROOT" \
  --output-dir "$ADAPTED_POLICY" \
  --experiment-id G1WH-40-scaled-smolvla-finetuning \
  --overwrite

cd "$LEROBOT_ROOT"
"$LEROBOT_PYTHON" -m lerobot.scripts.train \
  --policy.path="$ADAPTED_POLICY" \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/g1_scaled_phase_recovery_train \
  --dataset.root="$TRAIN_ROOT" \
  --dataset.video_backend=pyav \
  --batch_size=2 \
  --steps=20000 \
  --num_workers=0 \
  --log_freq=20 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --eval_freq=0 \
  --wandb.enable=false \
  --output_dir="$TRAIN_OUTPUT"

export LEROBOT_SITE_PACKAGES="${LEROBOT_SITE_PACKAGES:-/home/ubuntu/miniconda3/envs/lerobot/lib/python3.10/site-packages}"
export PYTHONPATH="$LEROBOT_ROOT/src:$PROJECT_ROOT/scripts"

"$GENESIS_PYTHON" "$PROJECT_ROOT/scripts/evaluate_g1wh_smolvla_closed_loop.py" \
  --asset "$ASSET" \
  --source-summary "$SOURCE_ROOT/assisted_lift_dataset_summary.json" \
  --checkpoint "scaled20000=$TRAIN_OUTPUT/checkpoints/020000/pretrained_model" \
  --experiment-id G1WH-40-scaled-smolvla-closed-loop-20ep \
  --train-repo-id local/g1_scaled_phase_recovery_train \
  --train-root "$TRAIN_ROOT" \
  --validation-start 100 \
  --episodes 20 \
  --control-fps 15 \
  --replan-steps 5 \
  --duration-s 10 \
  --phase-language-scheduler \
  --device cuda \
  --output-dir "$EVAL_OUTPUT"
