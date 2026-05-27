#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
SUMMARY_TSV="${ROOT_DIR}/results/benchmarks/stage3d_run_summary.tsv"
K_GRID="${K_GRID:-4,6}"
SEEDS="${SEEDS:-11,23,37}"

cd "${ROOT_DIR}"
mkdir -p "${ROOT_DIR}/results/benchmarks" "${ROOT_DIR}/results/figures"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
if [[ -z "${PIP_INDEX_URL:-}" ]]; then
  export PIP_INDEX_URL="https://pypi.org/simple"
fi
python -m pip install -r scripts/requirements_smoketest.txt >/dev/null

count_samples() {
  local dataset_id="$1"
  python3 - "$dataset_id" <<'PY'
import pathlib
import sys

dataset_id = sys.argv[1]
root = pathlib.Path(f"data/raw/{dataset_id}/extracted")
if not root.exists():
    print(0)
    raise SystemExit(0)

sample_ids = set()
for path in root.glob("*_filtered_feature_bc_matrix.h5"):
    sample_ids.add(path.name.replace("_filtered_feature_bc_matrix.h5", ""))
for path in root.rglob("*matrix.mtx*"):
    name = path.name
    if name.endswith("_matrix.mtx.gz"):
        sample_ids.add(name.replace("_matrix.mtx.gz", ""))
    elif name.endswith("_matrix.mtx"):
        sample_ids.add(name.replace("_matrix.mtx", ""))
    else:
        sample_ids.add(path.parent.parent.name)
print(len(sample_ids))
PY
}

append_summary() {
  local stage_id="$1"
  local dataset_id="$2"
  local run_scope="$3"
  local requested="$4"
  local discovered="$5"
  local processed="$6"
  local started="$7"
  local finished="$8"
  local status="$9"
  local notes="${10}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${stage_id}" "${dataset_id}" "${run_scope}" "${requested}" "${discovered}" \
    "${processed}" "${K_GRID}" "${SEEDS}" "${started}" "${finished}" "${status}" "${notes}" \
    >> "${SUMMARY_TSV}"
}

run_full_stage_m2() {
  local stage_id="$1"
  local dataset_id="$2"
  local note="$3"
  local started finished discovered status processed
  status="success"
  started="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  python scripts/download_geo_from_manifest.py --dataset-id "${dataset_id}"
  discovered="$(count_samples "${dataset_id}")"
  processed="${discovered}"

  if ! python scripts/run_crc_spatial_smoketest.py \
    --dataset-id "${dataset_id}" \
    --dataset-root "data/raw/${dataset_id}/extracted" \
    --max-samples 999 \
    --k-grid "${K_GRID}" \
    --seeds "${SEEDS}" \
    --methods "M2_spatial_ward" \
    --note "${note}"; then
    status="failed"
    processed="0"
  fi

  finished="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  append_summary "${stage_id}" "${dataset_id}" "full-replication-m2" "999" "${discovered}" "${processed}" "${started}" "${finished}" "${status}" "${note}"
  if [[ "${status}" != "success" ]]; then
    return 1
  fi
}

run_full_stage_m2 "stage3d" "GSE311294" "stage3a-full-replication-m2"
run_full_stage_m2 "stage3d" "GSE267401" "stage3b-full-replication-m2"
run_full_stage_m2 "stage3d" "GSE280318" "stage3g-full-replication-m2"

echo "Stage-3d runs completed (M2 spatial Ward baseline)."
