#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASAP_ROOT="${ASAP_ROOT:-$PROJECT_ROOT/third_party/ASAP}"
CONDA_ROOT="${CONDA_ROOT:-/home/ubuntu/miniconda3}"
ENV_NAME="${G1UB_GENESIS_ENV_NAME:-g1ub_genesis}"
ENV_DIR="$CONDA_ROOT/envs/$ENV_NAME"
PYTHON="$ENV_DIR/bin/python"
OUTPUT_DIR="$PROJECT_ROOT/outputs/G1UB-02_Genesis_CR7_baseline"
MOTION_FILE="humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl"

bash "$PROJECT_ROOT/scripts/setup_g1ub_genesis.sh"
mkdir -p "$OUTPUT_DIR"

cd "$ASAP_ROOT"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
export HYDRA_FULL_ERROR=1

"$PYTHON" humanoidverse/train_agent.py \
  +simulator=genesis \
  +exp=motion_tracking \
  +domain_rand=NO_domain_rand \
  +rewards=motion_tracking/reward_motion_tracking_dm_2real \
  +robot=g1/g1_29dof_anneal_23dof \
  +terrain=terrain_locomotion_plane \
  +obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
  num_envs=32 \
  seed=0 \
  headless=True \
  use_wandb=False \
  base_dir="$OUTPUT_DIR/training_logs" \
  project_name=G1UB-02 \
  experiment_name=Genesis_CR7_baseline \
  robot.motion.motion_file="$MOTION_FILE" \
  rewards.reward_penalty_curriculum=True \
  rewards.reward_penalty_degree=0.00001 \
  env.config.resample_motion_when_training=False \
  env.config.termination.terminate_when_motion_far=True \
  env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
  env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
  env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
  robot.asset.self_collisions=0 \
  algo.config.num_learning_iterations=200 \
  algo.config.save_interval=50
