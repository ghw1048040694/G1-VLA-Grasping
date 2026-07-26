#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINUX_PYTHON="${LINUX_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
WINDOWS_ROOT="D:\G1-UpperBody-Sim2Sim\G1SIM-02"
OUTPUT_ROOT="/mnt/d/G1-UpperBody-Sim2Sim/G1SIM-02"

"$LINUX_PYTHON" "$PROJECT_ROOT/scripts/prepare_g1sim_01.py" \
  --output "$OUTPUT_ROOT" \
  --experiment-id G1SIM-02-isaacsim-assisted-grasp-task-replay \
  --activate-assisted-grasp

EXTRA_ARGS=""
if [[ "${HEADLESS:-1}" == "1" ]]; then
  EXTRA_ARGS="$EXTRA_ARGS --headless"
fi
if [[ "${INSPECT_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS="$EXTRA_ARGS --inspect-only"
fi

mkdir -p "$OUTPUT_ROOT"
(
  cd /mnt/c
  cmd.exe /d /c "cd /d D:\isaacsim && python.bat $WINDOWS_ROOT\scripts\g1sim_replay.py --root $WINDOWS_ROOT $EXTRA_ARGS"
) 2>&1 | tee "$OUTPUT_ROOT/g1sim_02.log"
