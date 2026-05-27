#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv_stagate"
SUMMARY_TSV="${ROOT_DIR}/results/benchmarks/stage3f_run_summary.tsv"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
K_GRID="${K_GRID:-4,6}"
SEEDS="${SEEDS:-11,23}"
MAX_SAMPLES="${MAX_SAMPLES:-999}"
STAGATE_MAX_EPOCHS="${STAGATE_MAX_EPOCHS:-120}"
STAGATE_NUM_PCS="${STAGATE_NUM_PCS:-50}"
STAGATE_HIDDEN_DIM="${STAGATE_HIDDEN_DIM:-64}"
STAGATE_LATENT_DIM="${STAGATE_LATENT_DIM:-30}"

cd "${ROOT_DIR}"
mkdir -p "${ROOT_DIR}/results/benchmarks" "${ROOT_DIR}/results/figures"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Missing ${PYTHON_BIN}. Install Python 3.11 or set PYTHON_BIN to a working interpreter."
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
export PIP_INDEX_URL="${PIP_INDEX_URL_STAGATE:-https://pypi.org/simple}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "[stage3f] Installing baseline requirements..."
python -m pip install --progress-bar off -r scripts/requirements_spagcn_env.txt
echo "[stage3f] Installing torch requirements..."
python -m pip install --progress-bar off -r scripts/requirements_stagate.txt

count_stagate_rows() {
  if [[ ! -f results/benchmarks/method_benchmark.tsv ]]; then
    echo "0"
    return
  fi
  tail -n +2 results/benchmarks/method_benchmark.tsv | awk -F'\t' '$3=="M5_stagate"{n++} END{print n+0}'
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

run_full_stage_m5() {
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
  before_count="$(count_stagate_rows)"
  cmd=(python scripts/run_crc_spatial_smoketest.py \
    --dataset-id "${dataset_id}" \
    --dataset-root "data/raw/${dataset_id}/extracted" \
    --max-samples "${MAX_SAMPLES}" \
    --k-grid "${K_GRID}" \
    --seeds "${SEEDS}" \
    --methods "M5_stagate" \
    --stagate-num-pcs "${STAGATE_NUM_PCS}" \
    --stagate-hidden-dim "${STAGATE_HIDDEN_DIM}" \
    --stagate-latent-dim "${STAGATE_LATENT_DIM}" \
    --stagate-max-epochs "${STAGATE_MAX_EPOCHS}" \
    --note "${note}")
  if ! "${cmd[@]}"; then
    status="failed"
    processed="0"
  fi
  after_count="$(count_stagate_rows)"
  added_count=$((after_count - before_count))
  if [[ "${added_count}" -le 0 ]]; then
    status="failed"
  fi

  finished="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  append_summary "${stage_id}" "${dataset_id}" "full-replication-m5" "999" "${discovered}" "${processed}" "${started}" "${finished}" "${status}" "${note};m5_rows_added=${added_count}"
  if [[ "${status}" != "success" ]]; then
    return 1
  fi
}

run_full_stage_m5 "stage3f" "GSE311294" "stage3a-full-replication-m5"
run_full_stage_m5 "stage3f" "GSE267401" "stage3b-full-replication-m5"
run_full_stage_m5 "stage3f" "GSE280318" "stage3g-full-replication-m5"

echo "Stage-3f runs completed (M5 STAGATE baseline)."
