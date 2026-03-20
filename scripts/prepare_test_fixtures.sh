#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXTURE_DIR="${REPO_DIR}/crates/loftr/tests/data"
ARTIFACT_OUTDOOR_WEIGHTS="${REPO_DIR}/artifacts/weights/loftr_outdoor_state_dict.safetensors"
ARTIFACT_INDOOR_WEIGHTS="${REPO_DIR}/artifacts/weights/loftr_indoor_state_dict.safetensors"
FIXTURE_OUTDOOR_WEIGHTS="${FIXTURE_DIR}/loftr_outdoor_state_dict.safetensors"
FIXTURE_INDOOR_WEIGHTS="${FIXTURE_DIR}/loftr_indoor_state_dict.safetensors"
OUTDOOR_REFERENCE_PT="${FIXTURE_DIR}/loftr_outdoor_and_homography_data.pt"
INDOOR_REFERENCE_PT="${FIXTURE_DIR}/loftr_indoor_and_fundamental_data.pt"
OUTDOOR_REFERENCE_SAFETENSORS="${FIXTURE_DIR}/loftr_outdoor_reference.safetensors"
OUTDOOR_REFERENCE_METADATA="${FIXTURE_DIR}/loftr_outdoor_reference.json"
INDOOR_REFERENCE_SAFETENSORS="${FIXTURE_DIR}/loftr_indoor_reference.safetensors"
INDOOR_REFERENCE_METADATA="${FIXTURE_DIR}/loftr_indoor_reference.json"
LEFT_IMAGE="${FIXTURE_DIR}/kn_church-2.jpg"
RIGHT_IMAGE="${FIXTURE_DIR}/kn_church-8.jpg"
TORCH_HOME="${REPO_DIR}/.cache/torch"

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

prepare_weights() {
    local preset="$1"
    local artifact="$2"
    local fixture="$3"

    "${SCRIPT_DIR}/generate_loftr_state_dict.sh" --pretrained "${preset}" --out "${artifact}"
    cp "${artifact}" "${fixture}"
    info "Copied ${preset} test weights to ${fixture}"
}

export_reference_fixture() {
    local input_pt="$1"
    local output_safetensors="$2"
    local output_metadata="$3"

    "${REPO_DIR}/.venv-loftr/bin/python" "${SCRIPT_DIR}/export_kornia_test_data.py" \
        --input "${input_pt}" \
        --output "${output_safetensors}" \
        --metadata "${output_metadata}"
}

mkdir -p "${FIXTURE_DIR}"
mkdir -p "${TORCH_HOME}"
export TORCH_HOME

prepare_weights outdoor "${ARTIFACT_OUTDOOR_WEIGHTS}" "${FIXTURE_OUTDOOR_WEIGHTS}"
prepare_weights indoor "${ARTIFACT_INDOOR_WEIGHTS}" "${FIXTURE_INDOOR_WEIGHTS}"

download_if_missing \
    "https://github.com/kornia/data/raw/main/matching/kn_church-2.jpg" \
    "${LEFT_IMAGE}"
download_if_missing \
    "https://github.com/kornia/data/raw/main/matching/kn_church-8.jpg" \
    "${RIGHT_IMAGE}"
download_if_missing \
    "https://raw.githubusercontent.com/kornia/data_test/main/loftr_outdoor_and_homography_data.pt" \
    "${OUTDOOR_REFERENCE_PT}"
download_if_missing \
    "https://raw.githubusercontent.com/kornia/data_test/main/loftr_indoor_and_fundamental_data.pt" \
    "${INDOOR_REFERENCE_PT}"

export_reference_fixture \
    "${OUTDOOR_REFERENCE_PT}" \
    "${OUTDOOR_REFERENCE_SAFETENSORS}" \
    "${OUTDOOR_REFERENCE_METADATA}"
export_reference_fixture \
    "${INDOOR_REFERENCE_PT}" \
    "${INDOOR_REFERENCE_SAFETENSORS}" \
    "${INDOOR_REFERENCE_METADATA}"

info "Prepared test fixtures under ${FIXTURE_DIR}"
