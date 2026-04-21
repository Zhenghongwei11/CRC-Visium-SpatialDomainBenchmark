#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv_spagcn"
SUMMARY_TSV="${ROOT_DIR}/results/benchmarks/stage3e_run_summary.tsv"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
K_GRID="${K_GRID:-4,6}"
SEEDS="${SEEDS:-11,23}"
MAX_SAMPLES="${MAX_SAMPLES:-999}"

cd "${ROOT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Missing ${PYTHON_BIN}. Install Python 3.11 or set PYTHON_BIN to a working interpreter."
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
# SpaGCN (torch) relies on large binary wheels. Prefer the upstream index unless overridden.
export PIP_INDEX_URL="${PIP_INDEX_URL_SPAGCN:-https://pypi.org/simple}"

echo "[stage3e] Installing baseline requirements..."
python -m pip install --progress-bar off -r scripts/requirements_spagcn_env.txt
echo "[stage3e] Installing SpaGCN torch requirements..."
python -m pip install --progress-bar off -r scripts/requirements_spagcn.txt

count_spagcn_rows() {
  if [[ ! -f results/benchmarks/method_benchmark.tsv ]]; then
    echo "0"
    return
  fi
  tail -n +2 results/benchmarks/method_benchmark.tsv | awk -F'\t' '$3=="M4_spagcn"{n++} END{print n+0}'
}

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

run_full_stage_m4() {
  local stage_id="$1"
  local dataset_id="$2"
  local note="$3"
  local started finished discovered status processed
  status="success"
  started="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  python scripts/download_geo_from_manifest.py --dataset-id "${dataset_id}"
  discovered="$(count_samples "${dataset_id}")"
  processed="${discovered}"

  local before_count after_count added_count
  before_count="$(count_spagcn_rows)"
  if ! python scripts/run_crc_spatial_smoketest.py \
    --dataset-id "${dataset_id}" \
    --dataset-root "data/raw/${dataset_id}/extracted" \
    --max-samples "${MAX_SAMPLES}" \
    --k-grid "${K_GRID}" \
    --seeds "${SEEDS}" \
    --methods "M4_spagcn" \
    --note "${note}"; then
    status="failed"
    processed="0"
  fi
  after_count="$(count_spagcn_rows)"
  added_count=$((after_count - before_count))
  if [[ "${added_count}" -le 0 ]]; then
    status="failed"
  fi

  finished="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  append_summary "${stage_id}" "${dataset_id}" "full-replication-m4" "999" "${discovered}" "${processed}" "${started}" "${finished}" "${status}" "${note};m4_rows_added=${added_count}"
  if [[ "${status}" != "success" ]]; then
    return 1
  fi
}

run_full_stage_m4 "stage3e" "GSE311294" "stage3a-full-replication-m4"
run_full_stage_m4 "stage3e" "GSE267401" "stage3b-full-replication-m4"

echo "Stage-3e runs completed (M4 SpaGCN baseline)."
