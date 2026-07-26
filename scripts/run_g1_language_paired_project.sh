#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/ubuntu/lerobot}"

export SOURCE_ROOT="$PROJECT_ROOT/outputs/G1LANG-11_paired_production_dataset"
export TRAIN_ROOT="$LEROBOT_ROOT/datasets/local/g1_language_paired_train"
export VAL_ROOT="$LEROBOT_ROOT/datasets/local/g1_language_paired_val"
export CONVERSION_OUTPUT="$PROJECT_ROOT/outputs/G1LANG-12_paired_lerobot_dataset"
export ADAPTED_POLICY="$PROJECT_ROOT/outputs/G1LANG-12_paired_adapted_smolvla_policy"
export TRAIN_OUTPUT="$PROJECT_ROOT/outputs/G1LANG-12_paired_smolvla_20000step"
export LOG_PATH="$PROJECT_ROOT/outputs/G1LANG-paired-main.log"
export TRAIN_REPO_ID="local/g1_language_paired_train"
export VAL_REPO_ID="local/g1_language_paired_val"
export EXPERIMENT_ID="G1LANG-12-paired-language-smolvla-finetuning"
export REQUIRED_DATASET_CONTRACT_VERSION="g1lang_dataset_v3_exact_scene_triplets"
export COUNTERFACTUAL_OUTPUT_PREFIX="G1LANG-13_paired_counterfactual"
export PAIRED_OUTPUT_PREFIX="G1LANG-14_paired_closed_loop"
export FORMAL_OUTPUT_PREFIX="G1LANG-15_paired_formal_closed_loop"

exec bash "$PROJECT_ROOT/scripts/run_g1_language_project.sh"
