#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

# Targeted BayesSpace nrep sensitivity (nrep=100 vs 1000 is summarized in S17 Data).
# This stage re-runs BayesSpace on a small, fixed subset of sections with nrep=1000.
DATASET_ID="${DATASET_ID:-GSE311294}"
SAMPLES="${SAMPLES:-GSM9322957_TR11_206,GSM9322960_TR11_21723}"
K_GRID="${K_GRID:-4,6}"
SEEDS="${SEEDS:-11,23}"
NREP="${NREP:-1000}"
GAMMA="${GAMMA:-3}"
NOTE_PREFIX="${NOTE_PREFIX:-nrep-sensitivity-nrep1000}"

OUT_TSV="${OUT_TSV:-${ROOT_DIR}/results/benchmarks/bayesspace_nrep1000_sensitivity.tsv}"
TMP_DIR="${TMP_DIR:-${ROOT_DIR}/results/_tmp_nrep_sensitivity}"

cd "${ROOT_DIR}"
mkdir -p "$(dirname "${OUT_TSV}")" "${TMP_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
if [[ -z "${PIP_INDEX_URL:-}" ]]; then
  export PIP_INDEX_URL="https://pypi.org/simple"
fi
python -m pip install -r scripts/requirements_smoketest.txt >/dev/null

# Overwrite to keep this stage deterministic and avoid accidental duplication.
printf "dataset_id\tsample_id\tmethod_id\tmethod_family\tpreprocessing_id\tparam_set_id\tK\tseed_count\tstability_ari_median\tstability_ari_iqr\tspatial_coherence_median\tspatial_coherence_iqr\tmarker_coherence_median\tmarker_coherence_iqr\twall_time_sec_median\tpeak_rss_mb_median\tfailure_rate\tnotes\n" > "${OUT_TSV}"

IFS=',' read -r -a sample_list <<< "${SAMPLES}"
for sample_id in "${sample_list[@]}"; do
  sample_id="$(echo "${sample_id}" | xargs)"
  [[ -z "${sample_id}" ]] && continue

  tmp_tsv="${TMP_DIR}/tmp_${sample_id}.tsv"
  tmp_log="${TMP_DIR}/${sample_id}.log"

  echo "[stage3g] BayesSpace nrep=${NREP} ${DATASET_ID}/${sample_id} K=${K_GRID} seeds=${SEEDS}"
  Rscript scripts/run_bayesspace_baseline.R \
    --dataset-id "${DATASET_ID}" \
    --dataset-root "data/raw/${DATASET_ID}/extracted" \
    --sample-id "${sample_id}" \
    --k-grid "${K_GRID}" \
    --seeds "${SEEDS}" \
    --nrep "${NREP}" \
    --gamma "${GAMMA}" \
    --note "${NOTE_PREFIX}-${DATASET_ID}-${sample_id}" \
    --output-tsv "${tmp_tsv}" >"${tmp_log}" 2>&1

  tail -n +2 "${tmp_tsv}" >> "${OUT_TSV}"
done

echo "[stage3g] Wrote $(($(wc -l < "${OUT_TSV}") - 1)) rows to ${OUT_TSV}"
