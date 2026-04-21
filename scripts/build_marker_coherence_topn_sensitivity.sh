#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_TSV="${ROOT_DIR}/results/benchmarks/marker_coherence_topn_sensitivity.tsv"

TOPN_GRID="${TOPN_GRID:-10,20,50}"
K_GRID="${K_GRID:-4,6}"
SEED="${SEED:-11}"

TMP_DIR="${ROOT_DIR}/results/_tmp_topn_sensitivity"
mkdir -p "${TMP_DIR}"

tmp_bs="${TMP_DIR}/bayesspace.tsv"
tmp_m5="${TMP_DIR}/stagate.tsv"

Rscript "${ROOT_DIR}/scripts/marker_coherence_topn_sensitivity_bayesspace.R" \
  --dataset-id "GSE311294" \
  --dataset-root "${ROOT_DIR}/data/raw/GSE311294/extracted" \
  --sample-ids "GSM9322957_TR11_206,GSM9322958_TR11_16184,GSM9322959_TR11_18105" \
  --k-grid "${K_GRID}" \
  --seed "${SEED}" \
  --top-n-grid "${TOPN_GRID}" \
  --output-tsv "${tmp_bs}" \
  --note "topn-sensitivity-bayesspace"

Rscript "${ROOT_DIR}/scripts/marker_coherence_topn_sensitivity_bayesspace.R" \
  --dataset-id "GSE267401" \
  --dataset-root "${ROOT_DIR}/data/raw/GSE267401/extracted" \
  --sample-ids "GSM8265211_CTC21P,GSM8265212_CTC21M,GSM8265213_CTC17P" \
  --k-grid "${K_GRID}" \
  --seed "${SEED}" \
  --top-n-grid "${TOPN_GRID}" \
  --output-tsv "${TMP_DIR}/bayesspace2.tsv" \
  --note "topn-sensitivity-bayesspace"

cat "${tmp_bs}" > "${TMP_DIR}/bayesspace_all.tsv"
tail -n +2 "${TMP_DIR}/bayesspace2.tsv" >> "${TMP_DIR}/bayesspace_all.tsv"

PY_BIN="${PY_BIN:-${ROOT_DIR}/.venv_stagate/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="${PY_BIN_FALLBACK:-python3}"
fi

"${PY_BIN}" "${ROOT_DIR}/scripts/marker_coherence_topn_sensitivity_stagate.py" \
  --dataset-id "GSE311294" \
  --dataset-root "${ROOT_DIR}/data/raw/GSE311294/extracted" \
  --sample-ids "GSM9322957_TR11_206,GSM9322958_TR11_16184,GSM9322959_TR11_18105" \
  --k-grid "${K_GRID}" \
  --seed "${SEED}" \
  --top-n-grid "${TOPN_GRID}" \
  --stagate-num-pcs 50 \
  --stagate-hidden-dim 64 \
  --stagate-latent-dim 30 \
  --stagate-max-epochs 120 \
  --output-tsv "${tmp_m5}" \
  --note "topn-sensitivity-stagate"

"${PY_BIN}" "${ROOT_DIR}/scripts/marker_coherence_topn_sensitivity_stagate.py" \
  --dataset-id "GSE267401" \
  --dataset-root "${ROOT_DIR}/data/raw/GSE267401/extracted" \
  --sample-ids "GSM8265211_CTC21P,GSM8265212_CTC21M,GSM8265213_CTC17P" \
  --k-grid "${K_GRID}" \
  --seed "${SEED}" \
  --top-n-grid "${TOPN_GRID}" \
  --stagate-num-pcs 50 \
  --stagate-hidden-dim 64 \
  --stagate-latent-dim 30 \
  --stagate-max-epochs 120 \
  --output-tsv "${TMP_DIR}/stagate2.tsv" \
  --note "topn-sensitivity-stagate"

cat "${tmp_m5}" > "${TMP_DIR}/stagate_all.tsv"
tail -n +2 "${TMP_DIR}/stagate2.tsv" >> "${TMP_DIR}/stagate_all.tsv"

cat "${TMP_DIR}/bayesspace_all.tsv" > "${OUT_TSV}"
tail -n +2 "${TMP_DIR}/stagate_all.tsv" >> "${OUT_TSV}"

echo "Wrote ${OUT_TSV}"

