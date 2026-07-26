#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/G1FINAL-09_router_classical_9ep}"

if [[ -e "$OUTPUT_DIR/summary.json" ]]; then
  echo "Refusing to overwrite existing result: $OUTPUT_DIR/summary.json" >&2
  exit 1
fi

exec env \
  EPISODES="${EPISODES:-9}" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  bash "$PROJECT_ROOT/scripts/run_g1_final07.sh"
