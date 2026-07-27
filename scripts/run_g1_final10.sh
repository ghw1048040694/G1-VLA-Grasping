#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/G1FINAL-10_random_rgb_classical_3ep}"

if [[ -e "$OUTPUT_DIR/summary.json" ]]; then
  echo "Refusing to overwrite existing result: $OUTPUT_DIR/summary.json" >&2
  exit 1
fi

exec env \
  EPISODES="${EPISODES:-3}" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  LAYOUT_MODE=random \
  LOCALIZATION_SOURCE=head_rgb \
  EXPERIMENT_ID=G1FINAL-10-random-layout-rgb-localized-classical-controller \
  bash "$PROJECT_ROOT/scripts/run_g1_final07.sh"
