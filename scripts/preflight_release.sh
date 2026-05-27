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

echo "[preflight] checking for forbidden paths…"
forbidden_rg='(^|/)(openspec|conductor|release_staging)/|(^|/)docs/submissions/|\\.(docx|pdf)$|(^|/)\\.env|token|id_rsa|BEGIN (RSA|OPENSSH) PRIVATE KEY'
if rg -n --glob '!scripts/preflight_release.sh' "${forbidden_rg}" . >/dev/null 2>&1; then
  echo "[preflight] FAILED: forbidden patterns found (showing matches)"
  rg -n --glob '!scripts/preflight_release.sh' "${forbidden_rg}" . | head -n 200
  exit 3
fi

echo "[preflight] checking for internal/LLM scaffolding phrases…"
scaffold_rg='\\bassistant\\b|\\bagent\\b|\\bprompt\\b|\\bllm\\b|rewrite to sound human|AI味|去AI|placeholder'
if rg -n -i --glob '!scripts/preflight_release.sh' "${scaffold_rg}" docs README.md scripts >/dev/null 2>&1; then
  echo "[preflight] WARN: internal/scaffolding phrases found (review before release)"
  rg -n -i --glob '!scripts/preflight_release.sh' "${scaffold_rg}" docs README.md scripts | head -n 200
fi

echo "[preflight] OK"
