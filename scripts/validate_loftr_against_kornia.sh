#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 <left> <right> [weights] [output-dir]" >&2
  exit 2
fi

LEFT="$1"
RIGHT="$2"
WEIGHTS="${3:-artifacts/weights/loftr_outdoor_state_dict.safetensors}"
OUT_DIR="${4:-target/loftr_validation}"

mkdir -p "$OUT_DIR"

python3 ./scripts/dump_kornia_loftr_stages.py \
  --output "$OUT_DIR/kornia_stages.json" \
  "$LEFT" \
  "$RIGHT"

cargo run -p loftr --example dump_stages -- \
  "$WEIGHTS" \
  "$LEFT" \
  "$RIGHT" \
  "$OUT_DIR/rust_stages.json"

python3 ./scripts/compare_stage_stats.py \
  "$OUT_DIR/rust_stages.json" \
  "$OUT_DIR/kornia_stages.json"
