#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/results/r3_cross_method/domain_maps"
SUMMARY_DIR="${ROOT_DIR}/results/r3_cross_method/runs/bayesspace"
SEEDS="${SEEDS:-11,23,37}"
K_GRID="${K_GRID:-4,6}"
NREP="${NREP:-100}"
DATASET_FILTER="${DATASET_FILTER:-all}"

mkdir -p "${OUT_DIR}" "${SUMMARY_DIR}"
cd "${ROOT_DIR}"

run_sample() {
  local dataset_id="$1"
  local dataset_root="$2"
  local sample_id="$3"
  local final_map="${OUT_DIR}/${dataset_id}_bayes_${sample_id}.tsv"
  local final_summary="${SUMMARY_DIR}/${dataset_id}_${sample_id}.tsv"

  if [[ -s "${final_map}" && -s "${final_summary}" ]]; then
    echo "[bayes] skip complete ${dataset_id} ${sample_id}"
    return 0
  fi

  local tmp_map
  local tmp_summary
  tmp_map="$(mktemp "${OUT_DIR}/.${dataset_id}_${sample_id}.map.XXXXXX")"
  tmp_summary="$(mktemp "${SUMMARY_DIR}/.${dataset_id}_${sample_id}.summary.XXXXXX")"

  echo "[bayes] run ${dataset_id} ${sample_id} K=${K_GRID} seeds=${SEEDS} nrep=${NREP}"
  if Rscript scripts/run_bayesspace_baseline.R \
    --dataset-id "${dataset_id}" \
    --dataset-root "${dataset_root}" \
    --sample-id "${sample_id}" \
    --k-grid "${K_GRID}" \
    --seeds "${SEEDS}" \
    --nrep "${NREP}" \
    --output-domain-map-tsv "${tmp_map}" \
    --output-tsv "${tmp_summary}" \
    --note "r3-cross-method"; then
    mv "${tmp_map}" "${final_map}"
    mv "${tmp_summary}" "${final_summary}"
    echo "[bayes] complete ${dataset_id} ${sample_id}"
  else
    mv "${tmp_map}" "${tmp_map}.failed"
    mv "${tmp_summary}" "${tmp_summary}.failed"
    echo "[bayes] failed ${dataset_id} ${sample_id}" >&2
    return 1
  fi
}

run_dataset() {
  local dataset_id="$1"
  local dataset_root="$2"
  shift 2
  if [[ "${DATASET_FILTER}" != "all" && "${DATASET_FILTER}" != "${dataset_id}" ]]; then
    return 0
  fi
  local sample_id
  for sample_id in "$@"; do
    run_sample "${dataset_id}" "${dataset_root}" "${sample_id}"
  done
}

run_dataset "GSE267401" "data/raw/GSE267401/extracted" \
  "GSM8265211_CTC21P" "GSM8265212_CTC21M" "GSM8265213_CTC17P" "GSM8265214_CTC17M"

run_dataset "GSE311294" "data/raw/GSE311294/extracted" \
  "GSM9322957_TR11_206" "GSM9322958_TR11_16184" "GSM9322959_TR11_18105" \
  "GSM9322960_TR11_21723" "GSM9322961_TR16_23542"

run_dataset "GSE285505" "data/raw/GSE285505/extracted" \
  "GSM8703563_Tumor19" "GSM8703564_Tumor20" "GSM8703565_Tumor24" "GSM8703566_Tumor26"

echo "R3 BayesSpace partition export complete."
