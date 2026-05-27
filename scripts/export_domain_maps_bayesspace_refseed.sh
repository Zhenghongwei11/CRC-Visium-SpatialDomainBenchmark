#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASETS="${DATASETS:-GSE311294,GSE267401,GSE280318}"
K_GRID="${K_GRID:-4,6}"
SEED="${SEED:-11}"
BAYES_NREP="${BAYES_NREP:-100}"
BAYES_GAMMA="${BAYES_GAMMA:-3}"
NOTE_PREFIX="${NOTE_PREFIX:-bayesspace-domain-map-refseed}"
OUT_TSV="${OUT_TSV:-${ROOT_DIR}/results/domain_maps/domain_maps_bayesspace_refseed.tsv}"
RESET_OUT="${RESET_OUT:-1}"
BAYES_INSTALL="${BAYES_INSTALL:-0}"

cd "${ROOT_DIR}"
mkdir -p results/domain_maps

if [[ "${RESET_OUT}" == "1" ]]; then
  rm -f "${OUT_TSV}"
fi

if [[ "${BAYES_INSTALL}" == "1" ]]; then
  Rscript scripts/check_or_install_bayesspace.R >/dev/null
else
  Rscript scripts/check_or_install_bayesspace.R --no-install >/dev/null
fi

list_samples() {
  local dataset_id="$1"
  "${PYTHON_BIN}" - "$dataset_id" <<'PY'
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

IFS=',' read -r -a dataset_list <<< "${DATASETS}"
for dataset_id in "${dataset_list[@]}"; do
  dataset_id="$(echo "${dataset_id}" | xargs)"
  [[ -z "${dataset_id}" ]] && continue

  "${PYTHON_BIN}" scripts/download_geo_from_manifest.py --dataset-id "${dataset_id}"

  dataset_root="data/raw/${dataset_id}/extracted"
  while IFS= read -r sample_id; do
    [[ -z "${sample_id}" ]] && continue
    note="${NOTE_PREFIX}-${dataset_id}-${sample_id}"
    tmp_tsv="$(mktemp)"
    echo "[export] BayesSpace domain map ${dataset_id}/${sample_id} K=${K_GRID} seed=${SEED}"
    Rscript scripts/run_bayesspace_baseline.R \
      --dataset-id "${dataset_id}" \
      --dataset-root "${dataset_root}" \
      --sample-id "${sample_id}" \
      --k-grid "${K_GRID}" \
      --seed "${SEED}" \
      --nrep "${BAYES_NREP}" \
      --gamma "${BAYES_GAMMA}" \
      --note "${note}" \
      --output-domain-map-tsv "${OUT_TSV}" \
      --domain-map-seed "${SEED}" \
      --output-tsv "${tmp_tsv}" >/dev/null
  done < <(list_samples "${dataset_id}")
done

echo "Wrote ${OUT_TSV}"
