#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv_stagate"

K_GRID="${K_GRID:-4,6}"
SEEDS="${SEEDS:-11,23}"

# Two representative sections (one boundary-ambiguous, one external cohort section).
SAMPLE_GSE311294="${SAMPLE_GSE311294:-GSM9322957_TR11_206}"
SAMPLE_GSE280318="${SAMPLE_GSE280318:-GSM8703563_Tumor19}"

OUT_ROOT_DIR="${OUT_ROOT_DIR:-${ROOT_DIR}/results/sensitivity_runs}"
RESET_OUT="${RESET_OUT:-1}"

cd "${ROOT_DIR}"
mkdir -p "${OUT_ROOT_DIR}"

if [[ "${RESET_OUT}" == "1" ]]; then
  rm -rf "${OUT_ROOT_DIR}" || true
  mkdir -p "${OUT_ROOT_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing ${VENV_DIR}. Run scripts/run_crc_stage3f_stagate_baseline.sh once to create the STAGATE/SpaGCN environment."
  exit 1
fi

source "${VENV_DIR}/bin/activate"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
# Reduce risk of OpenMP mixed-runtime deadlocks observed on some macOS Python stacks.
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

run_one() {
  local dataset_id="$1"
  local dataset_root="$2"
  local sample_id="$3"
  local method_id="$4"
  local config_id="$5"
  shift 5
  local bench_dir="${OUT_ROOT_DIR}/${config_id}"
  mkdir -p "${bench_dir}"
  python scripts/run_crc_spatial_smoketest.py \
    --dataset-id "${dataset_id}" \
    --dataset-root "${dataset_root}" \
    --sample-ids "${sample_id}" \
    --max-samples 1 \
    --k-grid "${K_GRID}" \
    --seeds "${SEEDS}" \
    --methods "${method_id}" \
    --results-bench "${bench_dir}" \
    --results-fig "${bench_dir}/figures" \
    --hardware-id "sensitivity-run" \
    --note "${config_id}" \
    "$@"
}

echo "[stage10] Downloading datasets (if needed)…"
python scripts/download_geo_from_manifest.py --dataset-id "GSE311294" >/dev/null
python scripts/download_geo_from_manifest.py --dataset-id "GSE280318" >/dev/null

echo "[stage10] STAGATE hyperparameter sweep (2 sections)…"
for sample in "${SAMPLE_GSE311294}" "${SAMPLE_GSE280318}"; do
  if [[ "${sample}" == "${SAMPLE_GSE311294}" ]]; then
    ds="GSE311294"; root="data/raw/GSE311294/extracted"
  else
    ds="GSE280318"; root="data/raw/GSE280318/extracted"
  fi
  for latent in 15 30 50; do
    for epochs in 120 200; do
      config_id="sensitivity_stagate_latent${latent}_epochs${epochs}"
      run_one "${ds}" "${root}" "${sample}" "M5_stagate" "${config_id}" \
        --stagate-num-pcs 50 \
        --stagate-hidden-dim 64 \
        --stagate-latent-dim "${latent}" \
        --stagate-max-epochs "${epochs}"
    done
  done
done

echo "[stage10] SpaGCN-style hyperparameter sweep (2 sections)…"
for sample in "${SAMPLE_GSE311294}" "${SAMPLE_GSE280318}"; do
  if [[ "${sample}" == "${SAMPLE_GSE311294}" ]]; then
    ds="GSE311294"; root="data/raw/GSE311294/extracted"
  else
    ds="GSE280318"; root="data/raw/GSE280318/extracted"
  fi
  for p in 0.35 0.50 0.65; do
    config_id="sensitivity_spagcn_p${p}"
    run_one "${ds}" "${root}" "${sample}" "M4_spagcn" "${config_id}" \
      --spagcn-p "${p}" \
      --spagcn-max-epochs 200
  done
done

echo "Wrote: ${OUT_ROOT_DIR}/*/method_benchmark.tsv"
