#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ts="$(date -u +\"%Y%m%dT%H%M%SZ\")"
archive_dir="results/_archive/${ts}"
plots_archive_dir="plots/_archive/${ts}"

mkdir -p "${archive_dir}" "${plots_archive_dir}"

move_if_exists() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    local base
    base="$(basename "${path}")"
    mv "${path}" "${archive_dir}/${base}"
  fi
}

move_plots_if_exists() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    local base
    base="$(basename "${path}")"
    mv "${path}" "${plots_archive_dir}/${base}"
  fi
}

move_if_exists "results/benchmarks"
move_if_exists "results/figures"
move_if_exists "results/replication"
move_if_exists "results/effect_sizes"
move_if_exists "results/dataset_summary.tsv"

move_plots_if_exists "plots/publication"

echo "[reset] archived previous outputs to:"
echo "  - ${archive_dir}"
echo "  - ${plots_archive_dir}"
echo "[reset] ready for rerun"

