#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${REPO_DIR}/.venv-loftr/bin/python"
PYTHON_BIN="python3"

if [ -x "${VENV_PYTHON}" ]; then
  PYTHON_BIN="${VENV_PYTHON}"
fi

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 <left> <right> [weights] [output-dir]" >&2
  exit 2
fi

LEFT="$1"
RIGHT="$2"
WEIGHTS="${3:-artifacts/weights/loftr_outdoor_state_dict.safetensors}"
OUT_DIR="${4:-target/loftr_validation}"

mkdir -p "$OUT_DIR"

"${PYTHON_BIN}" "${SCRIPT_DIR}/dump_kornia_loftr_stages.py" \
  --output "$OUT_DIR/kornia_stages.json" \
  "$LEFT" \
  "$RIGHT"

cargo run -p loftr --example dump_stages --features download-libtorch -- \
  "$WEIGHTS" \
  "$LEFT" \
  "$RIGHT" \
  "$OUT_DIR/rust_stages.json"

"${PYTHON_BIN}" "${SCRIPT_DIR}/compare_stage_stats.py" \
  "$OUT_DIR/rust_stages.json" \
  "$OUT_DIR/kornia_stages.json"
