#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/home/ubuntu/miniconda3}"
ENV_NAME="${G1UB_ENV_NAME:-g1ub_isaacgym}"
ENV_DIR="$CONDA_ROOT/envs/$ENV_NAME"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_ARCHIVE="${ISAAC_ARCHIVE:-/home/ubuntu/.cache/g1ub/IsaacGym_Preview_4_Package.tar.gz}"
ISAAC_PARENT="${ISAAC_PARENT:-/home/ubuntu/.cache/g1ub/extracted}"
ISAAC_ROOT="$ISAAC_PARENT/isaacgym"
ASAP_ROOT="${ASAP_ROOT:-$PROJECT_ROOT/third_party/ASAP}"
SENTINEL="$ENV_DIR/.g1ub_setup_complete"

if [[ -f "$SENTINEL" && -x "$ENV_DIR/bin/python" ]] && \
   PATH="$ENV_DIR/bin:$PATH" \
   LD_LIBRARY_PATH="/usr/lib/wsl/lib:$ENV_DIR/lib:${LD_LIBRARY_PATH:-}" \
   "$ENV_DIR/bin/python" -c 'from isaacgym import gymapi, gymtorch; import torch, cv2, open3d, stl; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  echo "Environment already ready: $ENV_NAME"
  exit 0
fi

mkdir -p "$(dirname "$ISAAC_ARCHIVE")" "$ISAAC_PARENT"

if [[ ! -f "$ISAAC_ARCHIVE" ]]; then
  echo "Downloading Isaac Gym Preview 4..."
  wget -c --tries=3 --timeout=30 \
    https://developer.nvidia.com/isaac-gym-preview-4 \
    -O "$ISAAC_ARCHIVE"
fi

if [[ ! -d "$ISAAC_ROOT/python/isaacgym" ]]; then
  echo "Extracting Isaac Gym Preview 4..."
  tar -xzf "$ISAAC_ARCHIVE" -C "$ISAAC_PARENT"
fi

if [[ ! -d "$ASAP_ROOT/humanoidverse" ]]; then
  echo "ASAP repository not found at $ASAP_ROOT" >&2
  exit 1
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$CONDA_ROOT/bin/conda" create -y -n "$ENV_NAME" \
    -c conda-forge --override-channels python=3.8 pip
fi

PYTHON="$ENV_DIR/bin/python"
PIP="$PYTHON -m pip"

$PIP install --upgrade "pip<25" "setuptools<70" wheel

if ! $PYTHON -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  $PIP install \
    torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118
fi

$PIP install -e "$ISAAC_ROOT/python" --no-deps
$PIP install \
  hydra-core==1.3.2 numpy==1.23.5 scipy pyyaml imageio ninja \
  rich ipdb matplotlib termcolor wandb plotly tqdm loguru \
  tensorboard joblib easydict lxml numpy-stl open3d==0.19.0 opencv-python
$PIP install -e "$ASAP_ROOT" --no-deps
$PIP install -e "$ASAP_ROOT/isaac_utils"

export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
$PYTHON - <<'PY'
from isaacgym import gymapi, gymtorch
import cv2
import open3d
import stl
import torch

print("Isaac Gym import: OK")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
PY

touch "$SENTINEL"
echo "Environment ready: $ENV_NAME"
