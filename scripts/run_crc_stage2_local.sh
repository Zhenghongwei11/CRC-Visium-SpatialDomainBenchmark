#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
SUMMARY_TSV="${ROOT_DIR}/results/benchmarks/stage2_run_summary.tsv"
K_GRID="${K_GRID:-4,6}"
SEEDS="${SEEDS:-11,23,37}"
METHODS="${METHODS:-M0_expr_kmeans,M1_spatial_concat_kmeans,M3_spatial_leiden}"
STAGES="${STAGES:-stage2a,stage2b,stage2c}"

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
  python - "$dataset_id" <<'PY'
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

run_stage() {
  local stage_id="$1"
  local dataset_id="$2"
  local scope="$3"
  local max_samples="$4"
  local note="$5"
  local started_utc
  local finished_utc
  local discovered
  local processed
  local status="success"

  started_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  python scripts/download_geo_from_manifest.py --dataset-id "${dataset_id}"
  discovered="$(count_samples "${dataset_id}")"
  processed="${max_samples}"
  if [[ "${max_samples}" =~ ^[0-9]+$ ]] && [[ "${discovered}" =~ ^[0-9]+$ ]]; then
    if (( discovered < max_samples )); then
      processed="${discovered}"
    fi
  fi

  if ! python scripts/run_crc_spatial_smoketest.py \
    --dataset-id "${dataset_id}" \
    --dataset-root "data/raw/${dataset_id}/extracted" \
    --max-samples "${max_samples}" \
    --k-grid "${K_GRID}" \
    --seeds "${SEEDS}" \
    --methods "${METHODS}" \
    --note "${note}"; then
    status="failed"
  fi
  finished_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${stage_id}" "${dataset_id}" "${scope}" "${max_samples}" "${discovered}" \
    "${processed}" "${K_GRID}" "${SEEDS}" "${started_utc}" "${finished_utc}" \
    "${status}" "${note}" >> "${SUMMARY_TSV}"

  if [[ "${status}" != "success" ]]; then
    return 1
  fi
}

run_if_selected() {
  local stage_id="$1"
  shift
  if [[ ",${STAGES}," == *",${stage_id},"* ]]; then
    run_stage "${stage_id}" "$@"
  fi
}

run_if_selected "stage2a" "GSE280318" "all-samples-local-pilot" "99" "stage2a-local-full"
run_if_selected "stage2b" "GSE311294" "replication-smoke" "1" "stage2b-replication-smoke"
run_if_selected "stage2c" "GSE267401" "replication-smoke" "1" "stage2c-replication-smoke"

echo "Stage-2 runs completed."
