#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/home/ubuntu/miniconda3}"
ENV_NAME="${G1UB_GENESIS_ENV_NAME:-g1ub_genesis}"
ENV_DIR="$CONDA_ROOT/envs/$ENV_NAME"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASAP_ROOT="${ASAP_ROOT:-$PROJECT_ROOT/third_party/ASAP}"
SENTINEL="$ENV_DIR/.g1ub_genesis_setup_complete"

if [[ -f "$SENTINEL" && -x "$ENV_DIR/bin/python" ]] && \
   PATH="$ENV_DIR/bin:$PATH" \
   LD_LIBRARY_PATH="/usr/lib/wsl/lib:$ENV_DIR/lib:${LD_LIBRARY_PATH:-}" \
   "$ENV_DIR/bin/python" -c 'import genesis, torch; from importlib.metadata import version; assert torch.cuda.is_available(); assert version("libigl") == "2.5.1"' >/dev/null 2>&1; then
  echo "Environment already ready: $ENV_NAME"
  exit 0
fi

if [[ ! -d "$ASAP_ROOT/humanoidverse" ]]; then
  echo "ASAP repository not found at $ASAP_ROOT" >&2
  exit 1
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$CONDA_ROOT/bin/conda" create -y -n "$ENV_NAME" \
    -c conda-forge --override-channels python=3.10 pip
fi

PYTHON="$ENV_DIR/bin/python"
PIP="$PYTHON -m pip"

$PIP install --upgrade pip setuptools wheel

if ! $PYTHON -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  $PIP install \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
fi

$PIP install \
  genesis-world==0.2.1 hydra-core==1.3.2 numpy==1.26.4 scipy \
  rich ipdb matplotlib termcolor wandb plotly tqdm loguru \
  tensorboard joblib easydict lxml numpy-stl open3d opencv-python==4.10.0.84 \
  libigl==2.5.1
$PIP install -e "$ASAP_ROOT" --no-deps
$PIP install -e "$ASAP_ROOT/isaac_utils"

export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
$PYTHON - <<'PY'
import genesis as gs
import torch

gs.init(backend=gs.gpu)
print("Genesis GPU initialization: OK")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
PY

touch "$SENTINEL"
echo "Environment ready: $ENV_NAME"
