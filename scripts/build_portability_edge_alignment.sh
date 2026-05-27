#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_ID="${DATASET_ID:-GSE289934}"
K_GRID="${K_GRID:-4,6}"
SEED="${SEED:-11}"
BAYES_NREP="${BAYES_NREP:-100}"
BAYES_GAMMA="${BAYES_GAMMA:-3}"

OUT_BAYES="${OUT_BAYES:-${ROOT_DIR}/results/domain_maps/domain_maps_bayesspace_portability_refseed.tsv}"
OUT_M5="${OUT_M5:-${ROOT_DIR}/results/domain_maps/domain_maps_m5_portability_refseed.tsv}"
OUT_EDGE="${OUT_EDGE:-${ROOT_DIR}/results/benchmarks/portability_histology_edge_alignment.tsv}"

cd "${ROOT_DIR}"
mkdir -p results/domain_maps results/benchmarks

# 1) Export reference-seed domain maps for STAGATE (portable implementation).
if [[ -x ".venv_stagate/bin/python" ]]; then
  .venv_stagate/bin/python scripts/export_domain_maps_m4_m5_refseed.py \
    --dataset-id "${DATASET_ID}" \
    --dataset-root "data/raw/${DATASET_ID}/extracted" \
    --k-grid "${K_GRID}" \
    --seed "${SEED}" \
    --methods "M5_stagate" \
    --output-tsv "${OUT_M5}" \
    --reset
else
  echo "[warn] Missing .venv_stagate; cannot export STAGATE refseed domain maps for portability edge-alignment." >&2
  exit 0
fi

# 2) Export reference-seed domain maps for BayesSpace.
DATASETS="${DATASET_ID}" \
OUT_TSV="${OUT_BAYES}" \
RESET_OUT=1 \
SEED="${SEED}" \
K_GRID="${K_GRID}" \
BAYES_NREP="${BAYES_NREP}" \
BAYES_GAMMA="${BAYES_GAMMA}" \
BAYES_INSTALL=0 \
bash scripts/export_domain_maps_bayesspace_refseed.sh

# 3) Compute weak external anchor: histology edge alignment.
python3 scripts/build_histology_edge_alignment.py \
  --domain-maps "${OUT_BAYES}" \
  --domain-maps "${OUT_M5}" \
  --output-tsv "${OUT_EDGE}" \
  --data-root "data/raw"

echo "Wrote ${OUT_EDGE}"

