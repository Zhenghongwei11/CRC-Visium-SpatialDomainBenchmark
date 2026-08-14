#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ID="${1:-GSE285505}"
VENV_DIR="${ROOT_DIR}/.venv"

cd "${ROOT_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

if [[ -z "${PIP_INDEX_URL:-}" ]]; then
  export PIP_INDEX_URL="https://pypi.org/simple"
fi

python -m pip install -r scripts/requirements_smoketest.txt

python scripts/download_geo_from_manifest.py --dataset-id "${DATASET_ID}"
python scripts/run_crc_spatial_smoketest.py \
  --dataset-id "${DATASET_ID}" \
  --dataset-root "data/raw/${DATASET_ID}/extracted"

echo "Smoke test finished for ${DATASET_ID}"
