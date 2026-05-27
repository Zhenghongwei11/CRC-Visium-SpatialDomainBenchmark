#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
SUMMARY_TSV="${ROOT_DIR}/results/benchmarks/stage3_run_summary.tsv"
K_GRID="${K_GRID:-4,6}"
SEEDS="${SEEDS:-11,23,37}"
METHODS="${METHODS:-M0_expr_kmeans,M1_spatial_concat_kmeans,M3_spatial_leiden}"
BAYES_INSTALL="${BAYES_INSTALL:-0}"
STAGES="${STAGES:-stage3a,stage3b,stage3g}"

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

run_full_stage() {
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
    --methods "${METHODS}" \
    --note "${note}"; then
    status="failed"
    processed="0"
  fi
  finished="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  append_summary "${stage_id}" "${dataset_id}" "full-replication" "999" "${discovered}" "${processed}" "${started}" "${finished}" "${status}" "${note}"
  if [[ "${status}" != "success" ]]; then
    return 1
  fi
}

append_bayesspace_failure() {
  local dataset_id="$1"
  local sample_id="$2"
  local err_msg="$3"
  local now
  now="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  err_msg="$(echo "${err_msg}" | tr '\t' ' ' | tr '\n' ' ' | cut -c1-500)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${dataset_id}" "${sample_id}" "BayesSpace" "bayesspace_default" "K4" "4" "11" \
    "bayesspace_error" "${err_msg}" "" "" "" "" "${now}" "stage3-bayesspace-attempt" \
    >> "${ROOT_DIR}/results/benchmarks/failure_log.tsv"
}

run_bayesspace_stage() {
  local stage_id dataset_id started finished status tmp_tsv tmp_log elapsed sample_id
  stage_id="stage3c"
  dataset_id="GSE311294"
  status="success"
  started="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  tmp_tsv="$(mktemp)"
  tmp_log="$(mktemp)"

  if [[ "${BAYES_INSTALL}" == "1" ]]; then
    R_CHECK_CMD=(Rscript scripts/check_or_install_bayesspace.R)
  else
    R_CHECK_CMD=(Rscript scripts/check_or_install_bayesspace.R --no-install)
  fi

  if ! "${R_CHECK_CMD[@]}" >"${tmp_log}" 2>&1; then
    status="failed"
    append_bayesspace_failure "${dataset_id}" "NA" "$(tail -n 40 "${tmp_log}")"
  else
    if ! Rscript scripts/run_bayesspace_baseline.R \
      --dataset-id "${dataset_id}" \
      --dataset-root "data/raw/${dataset_id}/extracted" \
      --k-grid "4" \
      --seed "11" \
      --note "stage3-bayesspace" \
      --output-tsv "${tmp_tsv}" >"${tmp_log}" 2>&1; then
      status="failed"
      append_bayesspace_failure "${dataset_id}" "NA" "$(tail -n 40 "${tmp_log}")"
    else
      tail -n +2 "${tmp_tsv}" >> "${ROOT_DIR}/results/benchmarks/method_benchmark.tsv"
      tail -n +2 "${tmp_tsv}" | sed 's/^/Fig2A\t/' >> "${ROOT_DIR}/results/figures/fig2_benchmark_summary.tsv"
      while IFS=$'\t' read -r ds sid mid mf prep pset kval seed_cnt s_ari s_iqr sp sp_iqr mk mk_iqr wt mem fail notes; do
        sample_id="${sid}"
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "Fig4A" "${ds}" "${sid}" "${mid}" "${kval}" "success" "${wt}" "${mem}" "1" "0" "local-first" "${notes}" \
          >> "${ROOT_DIR}/results/figures/fig4_compute_and_guidance.tsv"
      done < <(tail -n +2 "${tmp_tsv}")
    fi
  fi

  finished="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  append_summary "${stage_id}" "${dataset_id}" "bayesspace-attempt" "1" "1" "1" "${started}" "${finished}" "${status}" "stage3-bayesspace-attempt"
}

run_if_selected() {
  local stage_id="$1"
  shift
  if [[ ",${STAGES}," == *",${stage_id},"* ]]; then
    "$@"
  fi
}

run_if_selected "stage3a" run_full_stage "stage3a" "GSE311294" "stage3a-full-replication"
run_if_selected "stage3b" run_full_stage "stage3b" "GSE267401" "stage3b-full-replication"
run_if_selected "stage3g" run_full_stage "stage3g" "GSE280318" "stage3g-full-replication"
run_if_selected "stage3c" run_bayesspace_stage

echo "Stage-3 runs completed."
