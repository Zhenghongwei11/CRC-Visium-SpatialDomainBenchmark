#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUMMARY_TSV="${ROOT_DIR}/results/benchmarks/stage4_run_summary.tsv"
VENV_DIR="${ROOT_DIR}/.venv"

# Controls
DATASETS="${DATASETS:-GSE311294,GSE267401,GSE280318}"
K_GRID="${K_GRID:-4,6}"
SEEDS="${SEEDS:-11,23}"
NOTE_PREFIX="${NOTE_PREFIX:-rigor-backfill-v3}"
BAYES_INSTALL="${BAYES_INSTALL:-1}"
BAYES_NREP="${BAYES_NREP:-100}"
BAYES_GAMMA="${BAYES_GAMMA:-3}"

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

if [[ "${BAYES_INSTALL}" == "1" ]]; then
  R_CHECK_CMD=(Rscript scripts/check_or_install_bayesspace.R)
else
  R_CHECK_CMD=(Rscript scripts/check_or_install_bayesspace.R --no-install)
fi

started_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
status="success"

if ! "${R_CHECK_CMD[@]}" >/dev/null 2>&1; then
  status="failed"
fi

list_samples() {
  local dataset_id="$1"
  python - "$dataset_id" <<'PY'
import pathlib
import sys

dataset_id = sys.argv[1]
root = pathlib.Path(f"data/raw/{dataset_id}/extracted")
ids = set()
for p in root.glob("*_matrix.mtx.gz"):
  ids.add(p.name.replace("_matrix.mtx.gz", ""))
for p in root.glob("*_filtered_feature_bc_matrix.h5"):
  ids.add(p.name.replace("_filtered_feature_bc_matrix.h5", ""))
for p in root.glob("*_barcodes.tsv.gz"):
  ids.add(p.name.replace("_barcodes.tsv.gz", ""))
for sid in sorted(ids):
  print(sid)
PY
}

count_bayes_rows() {
  if [[ ! -f results/benchmarks/method_benchmark.tsv ]]; then
    echo "0"
    return
  fi
  tail -n +2 results/benchmarks/method_benchmark.tsv | awk -F'\t' '$3=="BayesSpace"{n++} END{print n+0}'
}

append_failure_log() {
  local dataset_id="$1"
  local sample_id="$2"
  local err_msg="$3"
  local now
  now="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  err_msg="$(echo "${err_msg}" | tr '\t' ' ' | tr '\n' ' ' | cut -c1-500)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${dataset_id}" "${sample_id}" "BayesSpace" "bayesspace_default" "K${K_GRID}" "${K_GRID}" "${SEEDS}" \
    "bayesspace_error" "${err_msg}" "" "" "" "" "${now}" "${NOTE_PREFIX}" \
    >> "${ROOT_DIR}/results/benchmarks/failure_log.tsv"
}

finished_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
if [[ "${status}" != "success" ]]; then
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "stage4-bayesspace" "${started_utc}" "${finished_utc}" "${BAYES_INSTALL}" \
    "${status}" "0" "0" "0" "bayesspace-install-check-failed" >> "${SUMMARY_TSV}"
  echo "BayesSpace dependencies not available. Re-run with BAYES_INSTALL=1 or install BayesSpace in R."
  exit 1
fi

before_count="$(count_bayes_rows)"

IFS=',' read -r -a dataset_list <<< "${DATASETS}"
overall_status="success"
total_added=0

for dataset_id in "${dataset_list[@]}"; do
  dataset_id="$(echo "${dataset_id}" | xargs)"
  if [[ -z "${dataset_id}" ]]; then
    continue
  fi
  python scripts/download_geo_from_manifest.py --dataset-id "${dataset_id}"

  while IFS= read -r sample_id; do
    if [[ -z "${sample_id}" ]]; then
      continue
    fi
    dataset_root="data/raw/${dataset_id}/extracted"
    note="${NOTE_PREFIX}-${dataset_id}-${sample_id}"
    tmp_tsv="$(mktemp)"
    tmp_log="$(mktemp)"
    started_sample="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    before_sample="$(count_bayes_rows)"
    sample_status="success"

    echo "[stage4] BayesSpace ${dataset_id}/${sample_id} K=${K_GRID} seeds=${SEEDS}"
    if ! Rscript scripts/run_bayesspace_baseline.R \
      --dataset-id "${dataset_id}" \
      --dataset-root "${dataset_root}" \
      --sample-id "${sample_id}" \
      --k-grid "${K_GRID}" \
      --seeds "${SEEDS}" \
      --nrep "${BAYES_NREP}" \
      --gamma "${BAYES_GAMMA}" \
      --note "${note}" \
      --output-tsv "${tmp_tsv}" >"${tmp_log}" 2>&1; then
      sample_status="failed"
      overall_status="failed"
      append_failure_log "${dataset_id}" "${sample_id}" "$(tail -n 40 "${tmp_log}")"
      echo "[stage4] FAILED ${dataset_id}/${sample_id}"
    else
      tail -n +2 "${tmp_tsv}" >> "${ROOT_DIR}/results/benchmarks/method_benchmark.tsv"
      tail -n +2 "${tmp_tsv}" | sed 's/^/Fig2R\t/' >> "${ROOT_DIR}/results/figures/fig2_benchmark_summary.tsv"
      while IFS=$'\t' read -r ds sid mid mf prep pset kval seed_cnt s_ari s_iqr sp sp_iqr mk mk_iqr wt mem fail notes; do
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "Fig4R" "${ds}" "${sid}" "${mid}" "${kval}" "success" "${wt}" "${mem}" "1" "0" "local-first" "${notes}" \
          >> "${ROOT_DIR}/results/figures/fig4_compute_and_guidance.tsv"
      done < <(tail -n +2 "${tmp_tsv}")
    fi

    after_sample="$(count_bayes_rows)"
    added_sample=$((after_sample - before_sample))
    total_added=$((total_added + added_sample))
    finished_sample="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "stage4-bayesspace" "${started_sample}" "${finished_sample}" "${BAYES_INSTALL}" \
      "${dataset_id}" "${sample_id}" "${added_sample}" "${sample_status}" "${note}" >> "${SUMMARY_TSV}"
  done < <(list_samples "${dataset_id}")
done

after_count="$(count_bayes_rows)"
added_count=$((after_count - before_count))
finished_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "stage4-bayesspace-total" "${started_utc}" "${finished_utc}" "${BAYES_INSTALL}" \
  "${before_count}" "${after_count}" "${added_count}" "${overall_status}" \
  "datasets=${DATASETS};K_GRID=${K_GRID};SEEDS=${SEEDS};NOTE_PREFIX=${NOTE_PREFIX}" >> "${SUMMARY_TSV}"

if [[ "${overall_status}" != "success" || "${added_count}" -le 0 ]]; then
  echo "Stage-4 completed with BayesSpace failures or no rows added. Check results/benchmarks/failure_log.tsv."
  exit 1
fi

echo "Stage-4 completed with BayesSpace rows added: ${added_count}"
