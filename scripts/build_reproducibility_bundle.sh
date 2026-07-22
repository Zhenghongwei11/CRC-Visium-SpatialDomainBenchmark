#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/docs/reproducibility_bundle"
ZIP_NAME="${ZIP_NAME:-crc_spatial_benchmark_reproducibility_bundle.zip}"
export LC_ALL=C
export LANG=C

mkdir -p "${OUT_DIR}"

cat > "${OUT_DIR}/POLICY.md" <<'MD'
# Public reproducibility bundle policy

## Purpose
Provide a clean package for reproducing the analysis tables and figures from public data.

## Canonical bundle rule
- There is exactly one canonical ZIP in this folder.
- The GitHub release asset and archived reproducibility bundle should be identical (verified by checksums).

## Included (high level)
- `scripts/`: analysis entrypoints and helper scripts
- `results/`: benchmark tables and derived summary tables
- `docs/`: data manifest, figure provenance, source-data map, and statistical decision rules
- `supplementary_tables/`: consolidated supplementary-table workbook

## Excluded (by default)
- Writing and administrative materials: drafts, letters, forms, and checklist files
- Development-only tooling and local configuration: editor metadata, local planning notes, temporary files, and workflow notes
- Raw data and large intermediates: `data/` (reviewers can download public data separately)
- Local environments/caches: `.venv/`, `__pycache__/`, OS/editor metadata

## Rationale
Readers should see the data-processing and figure/table reproduction materials, not internal writing or project-management scaffolding.
MD

cat > "${OUT_DIR}/REVIEWER_GUIDE.md" <<'MD'
# Reproduction guide

## Minimal reproduction
1. Ensure Python and R are available.
2. Run the one-click entrypoint:
   - `bash scripts/reproduce_one_click.sh`
3. To skip figure regeneration and rebuild tables only:
   - `SKIP_FIGURES=1 bash scripts/reproduce_one_click.sh`

## What to check
- Summary statistics: `results/benchmarks/statistical_gate_summary.tsv`
- Effect sizes with CIs: `results/effect_sizes/claim_effects.tsv`
- Dataset coverage: `results/dataset_summary.tsv`
- Replication tables: `results/replication/*.tsv`
- Figure source tables: `results/figures/*.tsv`
- Consolidated supplementary table workbook: `supplementary_tables/SUPPLEMENTARY_TABLES.xlsx`

## Notes
- Bayesian MCMC methods can be slow; long runtimes are expected and should be recorded in the benchmark tables.
- The optional SpaGCN baseline (stage3e) uses an isolated Python 3.11 virtual environment (`.venv_spagcn`) and a torch dependency stack (installed automatically by `scripts/run_crc_stage3e_spagcn_baseline.sh`).
- The optional STAGATE baseline (stage3f) uses an isolated Python 3.11 virtual environment (`.venv_stagate`) and a torch dependency stack (installed automatically by `scripts/run_crc_stage3f_stagate_baseline.sh`).
- Raw data are not redistributed; all datasets are public and can be downloaded from GEO.
MD

python3 - <<'PY' "${ROOT_DIR}" "${OUT_DIR}"
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
out_dir = Path(sys.argv[2]).resolve()

include_roots = [
    root / "scripts",
    root / "results",
    root / "supplementary_tables",
]

include_docs_files = [
    root / "docs" / "DATA_MANIFEST.tsv",
    root / "docs" / "FIGURE_PROVENANCE.tsv",
    root / "docs" / "STATISTICAL_DECISION_RULES.md",
    root / "docs" / "SOURCE_DATA_MAP.tsv",
]

exclude_dir_prefixes = [
    root / "data",
    root / ".venv",
    root / "docs" / "reproducibility_bundle",
]

def is_excluded(path: Path) -> bool:
    for prefix in exclude_dir_prefixes:
        try:
            path.resolve().relative_to(prefix.resolve())
            return True
        except Exception:
            continue
    return False

def collect_files() -> list[Path]:
    files: list[Path] = []
    for base in include_roots:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if is_excluded(p):
                continue
            if "__pycache__" in p.parts:
                continue
            if p.name in [".DS_Store"]:
                continue
            if p.name in ["build_references.py", "reorder_refs.py"] or (p.name.startswith("lint_") and p.name.endswith("_style.sh")):
                # Local-only writing/reference utilities are excluded from public bundles.
                continue
            private_name_terms = ["sub" + "mission", "jb" + "cb", "bm" + "c"]
            if any(term in p.name.lower() for term in private_name_terms):
                continue
            if p.suffix in [".pyc"]:
                continue
            if p.suffix in [".zip"]:
                continue
            files.append(p)
    for p in include_docs_files:
        if p.exists() and p.is_file():
            if not is_excluded(p):
                files.append(p)
    # De-dup and sort by repo-relative path
    uniq = {}
    for p in files:
        rel = p.resolve().relative_to(root)
        uniq[str(rel)] = rel
    return [root / uniq[k] for k in sorted(uniq.keys())]

files = collect_files()
filelist_path = out_dir / "FILELIST.txt"
with filelist_path.open("w", encoding="utf-8") as h:
    for p in files:
        h.write(str(p.resolve().relative_to(root)) + "\n")

print(f"file_count={len(files)}")
PY

FILELIST="${OUT_DIR}/FILELIST.txt"
ZIP_PATH="${OUT_DIR}/${ZIP_NAME}"

rm -f "${ZIP_PATH}"

(cd "${ROOT_DIR}" && zip -q -@ "${ZIP_PATH}" < "${FILELIST}")

(
  cd "${OUT_DIR}"
  shasum -a 256 "${ZIP_NAME}" > CHECKSUMS.sha256
)

echo "Wrote ${ZIP_PATH}"
