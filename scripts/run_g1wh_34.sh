#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_ARCHIVED_G1WH34:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
G1WH-34 is archived because it reused stale normalization statistics.
Run the corrected experiment instead:
  set -o pipefail; bash /home/ubuntu/G1-UpperBody/scripts/run_g1wh_34r.sh 2>&1 | tee /home/ubuntu/G1-UpperBody/outputs/G1WH-34R.log

For failure reproduction only, set ALLOW_ARCHIVED_G1WH34=1.
EOF
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
PYTHON="${LEROBOT_PYTHON:-/home/ubuntu/miniconda3/envs/lerobot/bin/python}"
TRAIN_ROOT="$LEROBOT_ROOT/datasets/local/g1_assisted_lift_phase_recovery_train"
VAL_ROOT="$LEROBOT_ROOT/datasets/local/g1_assisted_lift_phase_recovery_val"
ADAPTED_POLICY="$PROJECT_ROOT/outputs/G1WH-34_adapted_recovery_smolvla_policy"
TRAIN_OUTPUT="$PROJECT_ROOT/outputs/G1WH-34_recovery_smolvla_3000step"

export PYTHONPATH="$LEROBOT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$PROJECT_ROOT/scripts/convert_g1_assisted_lift_to_lerobot.py" \
  --source-root "$PROJECT_ROOT/outputs/G1WH-23R_bounded_assisted_lift_dataset" \
  --recovery-root "$PROJECT_ROOT/outputs/G1WH-33_policy_recovery_dataset" \
  --train-root "$TRAIN_ROOT" \
  --val-root "$VAL_ROOT" \
  --train-repo-id local/g1_assisted_lift_phase_recovery_train \
  --val-repo-id local/g1_assisted_lift_phase_recovery_val \
  --train-episodes 16 \
  --fps 15 \
  --phase-conditioned-language \
  --split-by-phase \
  --experiment-id G1WH-34-recovery-augmented-smolvla-finetuning \
  --overwrite \
  --output-dir "$PROJECT_ROOT/outputs/G1WH-34_recovery_augmented_dataset"

"$PYTHON" "$PROJECT_ROOT/scripts/prepare_g1wh_smolvla_policy.py" \
  --base-policy "$PROJECT_ROOT/outputs/G1WH-29_phase_smolvla_1000step/checkpoints/000500/pretrained_model" \
  --dataset-repo-id local/g1_assisted_lift_phase_recovery_train \
  --dataset-root "$TRAIN_ROOT" \
  --output-dir "$ADAPTED_POLICY" \
  --experiment-id G1WH-34-recovery-augmented-smolvla-finetuning \
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
