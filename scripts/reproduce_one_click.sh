#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Minimal end-to-end reproduction (tables first; figures optional).
#
# Controls:
# - PYTHON_BIN: python interpreter for optional baselines (default: python3.11)
# - BAYES_INSTALL: set to 1 to auto-install BayesSpace deps when needed (default: 1)
# - RUN_SPAGCN / RUN_STAGATE: set to 0 to skip those optional baselines (default: 1)
# - SKIP_FIGURES: set to 1 to skip figure regeneration (default: 0)

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
BAYES_INSTALL="${BAYES_INSTALL:-1}"
RUN_SPAGCN="${RUN_SPAGCN:-1}"
RUN_STAGATE="${RUN_STAGATE:-1}"
SKIP_FIGURES="${SKIP_FIGURES:-0}"

bash scripts/run_crc_stage2_local.sh
bash scripts/run_crc_stage3_full_replication.sh

if [[ "${RUN_SPAGCN}" == "1" ]]; then
  PYTHON_BIN="${PYTHON_BIN}" bash scripts/run_crc_stage3e_spagcn_baseline.sh
fi

if [[ "${RUN_STAGATE}" == "1" ]]; then
  PYTHON_BIN="${PYTHON_BIN}" bash scripts/run_crc_stage3f_stagate_baseline.sh
fi

BAYES_INSTALL="${BAYES_INSTALL}" bash scripts/run_crc_stage4_bayesspace.sh
bash scripts/run_crc_stage10_bayesspace_rigor_backfill.sh

Rscript scripts/build_statistical_gate_summary.R
python3 scripts/build_required_artifacts.py

if [[ "${SKIP_FIGURES}" != "1" ]]; then
  python3 scripts/make_publication_figures_v2.py
  python3 scripts/make_supplementary_figures.py
fi

echo "[ok] reproduction complete"
