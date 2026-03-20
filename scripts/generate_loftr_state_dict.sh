#!/bin/bash
# Generate a Kornia LoFTR state dict in safetensors format.
#
# Run from the loftr-rs workspace root:
#
#   ./scripts/generate_loftr_state_dict.sh
#   ./scripts/generate_loftr_state_dict.sh --pretrained indoor_new --out artifacts/weights/loftr_indoor_new_state_dict.safetensors
#
# Prerequisites: python3, internet access (downloads Kornia weights on first run).
# The venv is cached at .venv-loftr and torch cache is kept under .cache/torch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_DIR}/.venv-loftr"
TORCH_HOME="${REPO_DIR}/.cache/torch"
PRETRAINED="outdoor"
OUT_FILE="${REPO_DIR}/artifacts/weights/loftr_outdoor_state_dict.safetensors"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrained)
            PRETRAINED="$2"
            shift 2
            ;;
        --out)
            OUT_FILE="$2"
            shift 2
            ;;
        *)
            error "unknown argument: $1"
            exit 2
            ;;
    esac
done

case "${PRETRAINED}" in
    outdoor|indoor|indoor_new)
        ;;
    *)
        error "unsupported preset: ${PRETRAINED}"
        exit 2
        ;;
esac

if ! command -v python3 &>/dev/null; then
    error "python3 not found. Please install Python 3."
    exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
    info "Creating Python venv at ${VENV_DIR} ..."
    python3 -m venv "${VENV_DIR}"
fi

PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

mkdir -p "${TORCH_HOME}"
export TORCH_HOME

if ! "${PYTHON}" -c "import kornia, numpy, safetensors, torch" &>/dev/null; then
    info "Upgrading pip ..."
    "${PIP}" install --quiet --upgrade pip

    info "Installing torch, kornia==0.8.2, and safetensors (this may take a few minutes on first run) ..."
    "${PIP}" install --quiet torch kornia==0.8.2 safetensors numpy
fi

mkdir -p "$(dirname "${OUT_FILE}")"

if [ -f "${OUT_FILE}" ]; then
    info "$(ls -lh "${OUT_FILE}" | awk '{print $5}') already exists at ${OUT_FILE}"
    info "Delete it and re-run to regenerate."
    exit 0
fi

info "Exporting LoFTR ${PRETRAINED} state dict ..."
"${PYTHON}" "${SCRIPT_DIR}/export_loftr_state_dict.py" \
    --pretrained "${PRETRAINED}" \
    --out "${OUT_FILE}"

info "Done. Model saved to ${OUT_FILE}"
info "File size: $(du -h "${OUT_FILE}" | cut -f1)"
