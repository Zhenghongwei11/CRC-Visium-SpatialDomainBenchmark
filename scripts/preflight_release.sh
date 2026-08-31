#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[preflight] repo_root=${ROOT_DIR}"

required_paths=(
  "README.md"
  "LICENSE"
  "requirements.txt"
  "scripts/reproduce_one_click.sh"
  "docs/DATA_MANIFEST.tsv"
  "docs/FIGURE_PROVENANCE.tsv"
  "docs/STATISTICAL_DECISION_RULES.md"
  "results/benchmarks/statistical_gate_summary.tsv"
)

missing=0
for p in "${required_paths[@]}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[preflight] MISSING: ${p}"
    missing=1
  fi
done
if [[ "${missing}" == "1" ]]; then
  echo "[preflight] FAILED: required files missing"
  exit 2
fi

echo "[preflight] checking for forbidden paths..."
planning_dir="open""spec"
workflow_dir="con""ductor"
staging_dir="release""_staging"
submission_dir="sub""missions"
forbidden_rg="(^|/)(${planning_dir}|${workflow_dir}|${staging_dir})/|(^|/)docs/${submission_dir}/|\\.(docx|pdf)$|(^|/)\\.env|token|id_rsa|BEGIN (RSA|OPENSSH) PRIVATE KEY"
if rg -n --glob '!scripts/preflight_release.sh' "${forbidden_rg}" . >/dev/null 2>&1; then
  echo "[preflight] FAILED: forbidden patterns found (showing matches)"
  rg -n --glob '!scripts/preflight_release.sh' "${forbidden_rg}" . | head -n 200
  exit 3
fi

echo "[preflight] checking for non-research placeholder terms..."
placeholder_rg='placeholder|todo|tbd'
if rg -n -i --glob '!scripts/preflight_release.sh' "${placeholder_rg}" docs README.md scripts >/dev/null 2>&1; then
  echo "[preflight] FAILED: placeholder terms found (showing matches)"
  rg -n -i --glob '!scripts/preflight_release.sh' "${placeholder_rg}" docs README.md scripts | head -n 200
  exit 4
fi

echo "[preflight] OK"
