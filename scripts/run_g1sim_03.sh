#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINUX_PYTHON="${LINUX_PYTHON:-/home/ubuntu/miniconda3/envs/g1ub_genesis/bin/python}"
WINDOWS_ROOT="D:\G1-UpperBody-Sim2Sim\G1SIM-03"
OUTPUT_ROOT="/mnt/d/G1-UpperBody-Sim2Sim/G1SIM-03"
SOURCE_EPISODE="${SOURCE_EPISODE:?Set SOURCE_EPISODE to a strict-success language episode directory}"
SOURCE_EXPERIMENT="${SOURCE_EXPERIMENT:-G1FINAL-language-target-router}"

"$LINUX_PYTHON" "$PROJECT_ROOT/scripts/prepare_g1sim_03.py" \
  --episode "$SOURCE_EPISODE" \
  --output "$OUTPUT_ROOT" \
  --source-experiment "$SOURCE_EXPERIMENT" \
  ${TARGET_ONLY_ASSISTED_GRASP:+--target-only-assisted-grasp}

EXTRA_ARGS=""
if [[ "${HEADLESS:-1}" == "1" ]]; then
  EXTRA_ARGS="$EXTRA_ARGS --headless"
fi

(
  cd /mnt/c
  cmd.exe /d /c "cd /d D:\isaacsim && python.bat $WINDOWS_ROOT\scripts\g1sim_replay.py --root $WINDOWS_ROOT $EXTRA_ARGS"
) 2>&1 | tee "$OUTPUT_ROOT/g1sim_03.log"
