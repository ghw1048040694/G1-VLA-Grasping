#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
GENESIS_PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-/home/ubuntu/miniconda3/envs/lerobot/bin/python}"
ASSET="$PROJECT_ROOT/outputs/G1WH-15_continuous_pregrasp_trajectory/x055_simultaneous/g1_warehouse_pregrasp.xml"
SOURCE_ROOT="$PROJECT_ROOT/outputs/G1LANG-01_production_dataset"
TRAIN_ROOT="$LEROBOT_ROOT/datasets/local/g1_language_pick_place_train"
VAL_ROOT="$LEROBOT_ROOT/datasets/local/g1_language_pick_place_val"
CONVERSION_OUTPUT="$PROJECT_ROOT/outputs/G1LANG-01_lerobot_dataset"
SOURCE_POLICY="$PROJECT_ROOT/outputs/G1WH-40_scaled_smolvla_20000step/checkpoints/020000/pretrained_model"
ADAPTED_POLICY="$PROJECT_ROOT/outputs/G1LANG-02_adapted_smolvla_policy"
TRAIN_OUTPUT="$PROJECT_ROOT/outputs/G1LANG-02_smolvla_20000step"
EVAL_OUTPUT="$PROJECT_ROOT/outputs/G1LANG-02_closed_loop_30ep"
LOG_PATH="$PROJECT_ROOT/outputs/G1LANG-main.log"
START_STAGE="${START_STAGE:-1}"

if ! [[ "$START_STAGE" =~ ^[1-4]$ ]]; then
  echo "START_STAGE must be 1, 2, 3, or 4" >&2
  exit 2
fi
if [[ ! -f "$ASSET" || ! -f "$SOURCE_POLICY/model.safetensors" ]]; then
  echo "Missing source G1 asset or G1WH-40 policy checkpoint" >&2
  exit 2
fi

mkdir -p "$PROJECT_ROOT/outputs"
exec > >(tee -a "$LOG_PATH") 2>&1

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "G1 Language-Grounded Manipulation pipeline"
echo "START_STAGE=$START_STAGE"
echo "Log: $LOG_PATH"

if (( START_STAGE <= 1 )); then
  "$GENESIS_PYTHON" "$PROJECT_ROOT/scripts/collect_g1_language_pick_place_dataset.py" \
    --python "$GENESIS_PYTHON" \
    --collector "$PROJECT_ROOT/scripts/run_g1_language_pick_place.py" \
    --asset "$ASSET" \
    --output-dir "$SOURCE_ROOT" \
    --episodes 180 \
    --seed 20260724 \
    --position-jitter-m 0.018 \
    --workers "${DATA_WORKERS:-2}" \
    --resume-successful
fi

if (( START_STAGE <= 2 )); then
  export PYTHONPATH="$LEROBOT_ROOT/src"
  "$LEROBOT_PYTHON" "$PROJECT_ROOT/scripts/convert_g1_language_pick_place_to_lerobot.py" \
    --source-root "$SOURCE_ROOT" \
    --train-root "$TRAIN_ROOT" \
    --val-root "$VAL_ROOT" \
    --train-repo-id local/g1_language_pick_place_train \
    --val-repo-id local/g1_language_pick_place_val \
    --train-episodes 150 \
    --fps 15 \
    --overwrite \
    --output-dir "$CONVERSION_OUTPUT"

  "$LEROBOT_PYTHON" "$PROJECT_ROOT/scripts/prepare_g1wh_smolvla_policy.py" \
    --base-policy "$SOURCE_POLICY" \
    --dataset-repo-id local/g1_language_pick_place_train \
    --dataset-root "$TRAIN_ROOT" \
    --output-dir "$ADAPTED_POLICY" \
    --experiment-id G1LANG-02-language-grounded-smolvla-finetuning \
    --refresh-normalization-stats \
    --overwrite
fi

if (( START_STAGE <= 3 )); then
  export PYTHONPATH="$LEROBOT_ROOT/src"
  cd "$LEROBOT_ROOT"
  RESUME_CONFIG=""
  if [[ -d "$TRAIN_OUTPUT/checkpoints" ]]; then
    RESUME_CONFIG="$(find "$TRAIN_OUTPUT/checkpoints" -name train_config.json -print | sort | tail -n 1)"
  fi
  if [[ -n "$RESUME_CONFIG" ]]; then
    echo "Resuming SmolVLA training from $RESUME_CONFIG"
    "$LEROBOT_PYTHON" -m lerobot.scripts.train \
      --config_path="$RESUME_CONFIG" \
      --resume=true
  elif [[ -e "$TRAIN_OUTPUT" ]]; then
    echo "Training output exists but has no resumable checkpoint: $TRAIN_OUTPUT" >&2
    echo "Move that incomplete directory aside, then rerun with START_STAGE=3." >&2
    exit 2
  else
    "$LEROBOT_PYTHON" -m lerobot.scripts.train \
      --policy.path="$ADAPTED_POLICY" \
      --policy.device=cuda \
      --policy.use_amp=false \
      --policy.push_to_hub=false \
      --dataset.repo_id=local/g1_language_pick_place_train \
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
  fi
fi

if (( START_STAGE <= 4 )); then
  CHECKPOINT="$TRAIN_OUTPUT/checkpoints/020000/pretrained_model"
  if [[ ! -f "$CHECKPOINT/model.safetensors" ]]; then
    echo "Missing completed 20K checkpoint: $CHECKPOINT" >&2
    exit 2
  fi
  export LEROBOT_SITE_PACKAGES="${LEROBOT_SITE_PACKAGES:-/home/ubuntu/miniconda3/envs/lerobot/lib/python3.10/site-packages}"
  export PYTHONPATH="$LEROBOT_ROOT/src:$PROJECT_ROOT/scripts"
  cd "$PROJECT_ROOT"
  "$GENESIS_PYTHON" "$PROJECT_ROOT/scripts/evaluate_g1_language_pick_place.py" \
    --asset "$ASSET" \
    --checkpoint "$CHECKPOINT" \
    --train-root "$TRAIN_ROOT" \
    --train-repo-id local/g1_language_pick_place_train \
    --output-dir "$EVAL_OUTPUT" \
    --episodes 30 \
    --seed 20260725 \
    --control-fps 15 \
    --duration-s 12 \
    --replan-steps 5 \
    --assist-distance-m 0.075 \
    --position-jitter-m 0.018 \
    --device cuda
fi

echo "G1_LANGUAGE_PROJECT_COMPLETE=1"
echo "Evaluation summary: $EVAL_OUTPUT/summary.json"
