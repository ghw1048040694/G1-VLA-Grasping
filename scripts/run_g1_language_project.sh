#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"
GENESIS_PYTHON="${G1UB_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-/home/ubuntu/miniconda3/envs/lerobot/bin/python}"
ASSET="$PROJECT_ROOT/outputs/G1WH-15_continuous_pregrasp_trajectory/x055_simultaneous/g1_warehouse_pregrasp.xml"
SOURCE_ROOT="${SOURCE_ROOT:-$PROJECT_ROOT/outputs/G1LANG-01_production_dataset}"
TRAIN_ROOT="${TRAIN_ROOT:-$LEROBOT_ROOT/datasets/local/g1_language_pick_place_train}"
VAL_ROOT="${VAL_ROOT:-$LEROBOT_ROOT/datasets/local/g1_language_pick_place_val}"
CONVERSION_OUTPUT="${CONVERSION_OUTPUT:-$PROJECT_ROOT/outputs/G1LANG-01_lerobot_dataset}"
SOURCE_POLICY="${SOURCE_POLICY:-$PROJECT_ROOT/outputs/G1WH-40_scaled_smolvla_20000step/checkpoints/020000/pretrained_model}"
ADAPTED_POLICY="${ADAPTED_POLICY:-$PROJECT_ROOT/outputs/G1LANG-02_adapted_smolvla_policy}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-$PROJECT_ROOT/outputs/G1LANG-02_smolvla_20000step}"
LOG_PATH="${LOG_PATH:-$PROJECT_ROOT/outputs/G1LANG-main.log}"
TRAIN_REPO_ID="${TRAIN_REPO_ID:-local/g1_language_pick_place_train}"
VAL_REPO_ID="${VAL_REPO_ID:-local/g1_language_pick_place_val}"
EXPERIMENT_ID="${EXPERIMENT_ID:-G1LANG-02-language-grounded-smolvla-finetuning}"
REQUIRED_DATASET_CONTRACT_VERSION="${REQUIRED_DATASET_CONTRACT_VERSION:-}"
COUNTERFACTUAL_OUTPUT_PREFIX="${COUNTERFACTUAL_OUTPUT_PREFIX:-G1LANG-08_counterfactual}"
PAIRED_OUTPUT_PREFIX="${PAIRED_OUTPUT_PREFIX:-G1LANG-09_paired_closed_loop}"
FORMAL_OUTPUT_PREFIX="${FORMAL_OUTPUT_PREFIX:-G1LANG-10_formal_closed_loop}"
START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-4}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
COUNTERFACTUAL_MIN_ACCURACY="${COUNTERFACTUAL_MIN_ACCURACY:-0.80}"

if ! [[ "$START_STAGE" =~ ^[1-4]$ && "$END_STAGE" =~ ^[1-4]$ ]]; then
  echo "START_STAGE and END_STAGE must be 1, 2, 3, or 4" >&2
  exit 2
fi
if (( START_STAGE > END_STAGE )); then
  echo "START_STAGE must be less than or equal to END_STAGE" >&2
  exit 2
fi
if ! [[ "$TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRAIN_STEPS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$COUNTERFACTUAL_MIN_ACCURACY" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]]; then
  echo "COUNTERFACTUAL_MIN_ACCURACY must be between 0 and 1" >&2
  exit 2
fi
if (( START_STAGE <= 1 && END_STAGE >= 1 )) \
  && [[ "$REQUIRED_DATASET_CONTRACT_VERSION" != "g1lang_dataset_v3_exact_scene_triplets" ]]; then
  echo "Stage 1 requires the paired-scene dataset contract." >&2
  echo "Use scripts/run_g1_language_paired_project.sh for new collection." >&2
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
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "G1 Language-Grounded Manipulation pipeline"
echo "START_STAGE=$START_STAGE"
echo "END_STAGE=$END_STAGE"
echo "TRAIN_STEPS=$TRAIN_STEPS"
echo "Log: $LOG_PATH"

if (( START_STAGE <= 1 && END_STAGE >= 1 )); then
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

if (( START_STAGE <= 2 && END_STAGE >= 2 )); then
  export PYTHONPATH="$LEROBOT_ROOT/src"
  CONTRACT_ARGS=()
  if [[ -n "$REQUIRED_DATASET_CONTRACT_VERSION" ]]; then
    CONTRACT_ARGS+=(
      --required-dataset-contract-version "$REQUIRED_DATASET_CONTRACT_VERSION"
    )
  fi
  "$LEROBOT_PYTHON" "$PROJECT_ROOT/scripts/convert_g1_language_pick_place_to_lerobot.py" \
    --source-root "$SOURCE_ROOT" \
    --train-root "$TRAIN_ROOT" \
    --val-root "$VAL_ROOT" \
    --train-repo-id "$TRAIN_REPO_ID" \
    --val-repo-id "$VAL_REPO_ID" \
    --train-episodes 150 \
    --fps 15 \
    --overwrite \
    "${CONTRACT_ARGS[@]}" \
    --output-dir "$CONVERSION_OUTPUT"

  "$LEROBOT_PYTHON" "$PROJECT_ROOT/scripts/prepare_g1wh_smolvla_policy.py" \
    --base-policy "$SOURCE_POLICY" \
    --dataset-repo-id "$TRAIN_REPO_ID" \
    --dataset-root "$TRAIN_ROOT" \
    --output-dir "$ADAPTED_POLICY" \
    --experiment-id "$EXPERIMENT_ID" \
    --refresh-normalization-stats \
    --overwrite
fi

if (( START_STAGE <= 3 && END_STAGE >= 3 )); then
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
      --resume=true \
      --steps="$TRAIN_STEPS"
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
      --dataset.repo_id="$TRAIN_REPO_ID" \
      --dataset.root="$TRAIN_ROOT" \
      --dataset.video_backend=pyav \
      --batch_size=2 \
      --steps="$TRAIN_STEPS" \
      --num_workers=0 \
      --log_freq=20 \
      --save_checkpoint=true \
      --save_freq=10000 \
      --eval_freq=0 \
      --wandb.enable=false \
      --output_dir="$TRAIN_OUTPUT"
  fi
fi

if (( START_STAGE <= 4 && END_STAGE >= 4 )); then
  printf -v CHECKPOINT_STEP "%06d" "$TRAIN_STEPS"
  CHECKPOINT="$TRAIN_OUTPUT/checkpoints/$CHECKPOINT_STEP/pretrained_model"
  COUNTERFACTUAL_OUTPUT="$PROJECT_ROOT/outputs/${COUNTERFACTUAL_OUTPUT_PREFIX}_${CHECKPOINT_STEP}"
  PAIRED_OUTPUT="$PROJECT_ROOT/outputs/${PAIRED_OUTPUT_PREFIX}_${CHECKPOINT_STEP}"
  EVAL_OUTPUT="$PROJECT_ROOT/outputs/${FORMAL_OUTPUT_PREFIX}_${CHECKPOINT_STEP}"
  if [[ ! -f "$CHECKPOINT/model.safetensors" ]]; then
    echo "Missing completed checkpoint: $CHECKPOINT" >&2
    exit 2
  fi
  export LEROBOT_SITE_PACKAGES="${LEROBOT_SITE_PACKAGES:-/home/ubuntu/miniconda3/envs/lerobot/lib/python3.10/site-packages}"
  export PYTHONPATH="$LEROBOT_ROOT/src:$PROJECT_ROOT/scripts"
  cd "$PROJECT_ROOT"

  "$LEROBOT_PYTHON" "$PROJECT_ROOT/scripts/evaluate_g1_language_counterfactual.py" \
    --checkpoint "$CHECKPOINT" \
    --train-root "$TRAIN_ROOT" \
    --val-root "$VAL_ROOT" \
    --train-repo-id "$TRAIN_REPO_ID" \
    --val-repo-id "$VAL_REPO_ID" \
    --scenes 10 \
    --seed 20260727 \
    --device cuda \
    --output-dir "$COUNTERFACTUAL_OUTPUT"
  if ! "$LEROBOT_PYTHON" - "$COUNTERFACTUAL_OUTPUT/summary.json" "$COUNTERFACTUAL_MIN_ACCURACY" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
accuracy = float(report["nearest_expert_classification_accuracy"])
threshold = float(sys.argv[2])
print(f"COUNTERFACTUAL_ACCURACY={accuracy:.6f}")
print(f"COUNTERFACTUAL_THRESHOLD={threshold:.6f}")
raise SystemExit(0 if accuracy >= threshold else 1)
PY
  then
    echo "Counterfactual language gate failed; skipping closed-loop evaluation." >&2
    exit 3
  fi

  "$GENESIS_PYTHON" "$PROJECT_ROOT/scripts/evaluate_g1_language_pick_place.py" \
    --asset "$ASSET" \
    --checkpoint "$CHECKPOINT" \
    --train-root "$TRAIN_ROOT" \
    --train-repo-id "$TRAIN_REPO_ID" \
    --output-dir "$PAIRED_OUTPUT" \
    --episodes 3 \
    --seed 20260725 \
    --control-fps 15 \
    --duration-s 12 \
    --replan-steps 5 \
    --assist-distance-m 0.075 \
    --position-jitter-m 0.018 \
    --device cuda
  if ! "$LEROBOT_PYTHON" - "$PAIRED_OUTPUT/summary.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
strict_success_rate = float(report["strict_success_rate"])
print(f"PAIRED_STRICT_SUCCESS_RATE={strict_success_rate:.6f}")
raise SystemExit(0 if strict_success_rate == 1.0 else 1)
PY
  then
    echo "Paired closed-loop gate failed; skipping the formal 30 episodes." >&2
    exit 3
  fi

  "$GENESIS_PYTHON" "$PROJECT_ROOT/scripts/evaluate_g1_language_pick_place.py" \
    --asset "$ASSET" \
    --checkpoint "$CHECKPOINT" \
    --train-root "$TRAIN_ROOT" \
    --train-repo-id "$TRAIN_REPO_ID" \
    --output-dir "$EVAL_OUTPUT" \
    --episodes 30 \
    --seed 20260725 \
    --control-fps 15 \
    --duration-s 12 \
    --replan-steps 5 \
    --assist-distance-m 0.075 \
    --position-jitter-m 0.018 \
    --device cuda
  if ! "$LEROBOT_PYTHON" - "$EVAL_OUTPUT/summary.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"FORMAL_EVALUATION_PASSED={int(bool(report['passed']))}")
raise SystemExit(0 if report["passed"] else 1)
PY
  then
    echo "Formal 30-episode acceptance gate failed." >&2
    exit 3
  fi
fi

echo "G1_LANGUAGE_STAGE_RANGE_COMPLETE=$START_STAGE-$END_STAGE"
if (( END_STAGE == 4 )); then
  echo "G1_LANGUAGE_PROJECT_COMPLETE=1"
  echo "Evaluation summary: $EVAL_OUTPUT/summary.json"
fi
