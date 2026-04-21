#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-$(date -u +"%Y-%m-%d_run%H%M%SZ_local")}"
OUT_DIR="${ROOT_DIR}/docs/audit_runs/${RUN_ID}"

mkdir -p "${OUT_DIR}"

date -u +"%Y-%m-%dT%H:%M:%SZ" > "${OUT_DIR}/created_utc.txt"

{
  echo "# Audit run bundle"
  echo
  echo "- run_id: ${RUN_ID}"
  echo "- created_utc: $(cat "${OUT_DIR}/created_utc.txt")"
  echo
  echo "## What this is"
  echo "A lightweight, internal provenance bundle capturing environment info and checksums for key outputs."
  echo
  echo "## What this is not"
  echo "- Not a submission artifact."
  echo "- Not a guarantee of exact numerical reproducibility across platforms (Bayesian MCMC and numeric libs can differ)."
  echo
  echo "## Reproduction entrypoints"
  echo "- Baseline pipeline: scripts/run_crc_stage2_local.sh, scripts/run_crc_stage3_full_replication.sh"
  echo "- BayesSpace stages: scripts/run_crc_stage4_bayesspace.sh ... scripts/run_crc_stage10_bayesspace_rigor_backfill.sh"
  echo "- Claim gates: scripts/build_statistical_gate_summary.R"
  echo "- Derived tables: scripts/build_required_artifacts.py"
} > "${OUT_DIR}/README.md"

{
  echo "uname: $(uname -a)"
  echo "date_utc: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo
  echo "python: $(python3 --version 2>&1 || true)"
  echo "pip: $(python3 -m pip --version 2>&1 || true)"
  echo
  echo "R: $(R --version 2>&1 | head -n 1 || true)"
  echo
  if [[ -d "${ROOT_DIR}/.venv" ]]; then
    echo "venv: ${ROOT_DIR}/.venv"
    "${ROOT_DIR}/.venv/bin/python" --version 2>&1 || true
    "${ROOT_DIR}/.venv/bin/python" -m pip freeze 2>/dev/null | sed 's/^/pip_freeze: /' || true
  else
    echo "venv: none"
  fi
} > "${OUT_DIR}/environment.txt"

FILES_LIST="${OUT_DIR}/files.txt"
CHECKSUMS="${OUT_DIR}/checksums.sha256"

(
  cd "${ROOT_DIR}"
  {
    find results -type f -name "*.tsv" 2>/dev/null || true
    find results -type f -name "*.md" 2>/dev/null || true
    find scripts -maxdepth 1 -type f -name "*.sh" 2>/dev/null || true
    find scripts -maxdepth 1 -type f -name "*.py" 2>/dev/null || true
    echo "docs/STATISTICAL_DECISION_RULES.md"
    echo "docs/CLAIMS.tsv"
    echo "docs/SOURCE_DATA_MAP.tsv"
    echo "docs/HARMONIZATION_NOTES.md"
  } | awk 'NF>0' | sort -u > "${FILES_LIST}"

  : > "${CHECKSUMS}"
  while IFS= read -r path; do
    if [[ -f "${path}" ]]; then
      shasum -a 256 "${path}" >> "${CHECKSUMS}"
    fi
  done < "${FILES_LIST}"
)

echo "Wrote audit bundle to ${OUT_DIR}"
