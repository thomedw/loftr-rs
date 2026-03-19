#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXTURE_DIR="${REPO_DIR}/crates/loftr/tests/data"
ARTIFACT_WEIGHTS="${REPO_DIR}/artifacts/weights/loftr_outdoor_state_dict.safetensors"
FIXTURE_WEIGHTS="${FIXTURE_DIR}/loftr_outdoor_state_dict.safetensors"
LEFT_IMAGE="${FIXTURE_DIR}/kn_church-2.jpg"
RIGHT_IMAGE="${FIXTURE_DIR}/kn_church-8.jpg"

info() { echo "[INFO] $*"; }
error() { echo "[ERROR] $*" >&2; }

download_if_missing() {
    local url="$1"
    local output="$2"
    if [ -f "${output}" ]; then
        info "Fixture already present: ${output}"
        return 0
    fi

    if ! command -v curl >/dev/null 2>&1; then
        error "curl not found. Please install curl or download ${url} manually."
        exit 1
    fi

    info "Downloading ${url} -> ${output}"
    curl -L --fail --retry 3 --silent --show-error -o "${output}" "${url}"
}

mkdir -p "${FIXTURE_DIR}"

if [ ! -f "${FIXTURE_WEIGHTS}" ]; then
    if [ ! -f "${ARTIFACT_WEIGHTS}" ]; then
        info "Generating LoFTR outdoor weights ..."
        "${SCRIPT_DIR}/generate_loftr_state_dict.sh"
    else
        info "Reusing generated weights from ${ARTIFACT_WEIGHTS}"
    fi

    cp "${ARTIFACT_WEIGHTS}" "${FIXTURE_WEIGHTS}"
    info "Copied test weights to ${FIXTURE_WEIGHTS}"
else
    info "Fixture already present: ${FIXTURE_WEIGHTS}"
fi

download_if_missing \
    "https://github.com/kornia/data/raw/main/matching/kn_church-2.jpg" \
    "${LEFT_IMAGE}"
download_if_missing \
    "https://github.com/kornia/data/raw/main/matching/kn_church-8.jpg" \
    "${RIGHT_IMAGE}"

info "Prepared test fixtures under ${FIXTURE_DIR}"
