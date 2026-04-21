#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/docs/review_bundle"
ZIP_NAME="${ZIP_NAME:-crc_spatial_benchmark_review_bundle.zip}"

mkdir -p "${OUT_DIR}"

cat > "${OUT_DIR}/POLICY.md" <<'MD'
# Public review bundle policy (canonical zip)

## Purpose
Provide a single, clean reproducibility bundle suitable for peer review and public release.

## Canonical bundle rule
- There is exactly one canonical ZIP in this folder.
- The journal ZIP and GitHub release ZIP should be identical (verified by checksums).

## Included (high level)
- `scripts/`: analysis entrypoints and helper scripts
- `results/`: benchmark tables and derived summary tables
- `docs/`: protocol and reporting artifacts needed to interpret/reproduce results (excluding submission manuscripts)
- `docs/audit_runs/`: lightweight run provenance (environment + checksums), when safe

## Excluded (by default)
- Submission-only materials: `docs/submissions/`, `docs/manuscript/`, cover letters, checklists tied to a specific submission UI
- Development-only tooling and local configuration: `openspec/` and any editor/IDE metadata directories
- Raw data and large intermediates: `data/` (reviewers can download public data separately)
- Local environments/caches: `.venv/`, `__pycache__/`, OS/editor metadata

## Rationale
Reviewers should see the science and the reproducibility artifacts, not internal spec tooling or submission packaging.
MD

cat > "${OUT_DIR}/REVIEWER_GUIDE.md" <<'MD'
# Reviewer guide (how to reproduce key numbers)

## Minimal reproduction (tables)
1. Ensure Python and R are available.
2. Run the local stages (baselines + BayesSpace where applicable):
   - `bash scripts/run_crc_stage2_local.sh`
   - `bash scripts/run_crc_stage3_full_replication.sh`
   - `PYTHON_BIN=python3.11 bash scripts/run_crc_stage3e_spagcn_baseline.sh`
   - `PYTHON_BIN=python3.11 bash scripts/run_crc_stage3f_stagate_baseline.sh`
   - `BAYES_INSTALL=1 bash scripts/run_crc_stage4_bayesspace.sh`
   - `bash scripts/run_crc_stage10_bayesspace_rigor_backfill.sh`
3. Rebuild the claim-gate table:
   - `Rscript scripts/build_statistical_gate_summary.R`
4. Rebuild derived artifacts:
   - `python3 scripts/build_required_artifacts.py`

## What to check
- Claim gates: `results/benchmarks/statistical_gate_summary.tsv`
- Effect sizes with CIs: `results/effect_sizes/claim_effects.tsv`
- Dataset coverage: `results/dataset_summary.tsv`
- Replication tables: `results/replication/*.tsv`

## Notes
- Bayesian MCMC methods can be slow; long runtimes are expected and should be recorded in the benchmark tables.
- The optional SpaGCN baseline (stage3e) uses an isolated Python 3.11 virtual environment (`.venv_spagcn`) and a torch dependency stack (installed automatically by `scripts/run_crc_stage3e_spagcn_baseline.sh`).
- The optional STAGATE baseline (stage3f) uses an isolated Python 3.11 virtual environment (`.venv_stagate`) and a torch dependency stack (installed automatically by `scripts/run_crc_stage3f_stagate_baseline.sh`).
- The bundle intentionally excludes raw data; all datasets are public and can be downloaded from GEO.
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
]

include_docs_files = [
    root / "docs" / "DATA_MANIFEST.tsv",
    root / "docs" / "FIGURE_PROVENANCE.tsv",
    root / "docs" / "CITATION_VERIFICATION.tsv",
    root / "docs" / "STATISTICAL_DECISION_RULES.md",
    root / "docs" / "CLAIMS.tsv",
    root / "docs" / "SOURCE_DATA_MAP.tsv",
    root / "docs" / "PROJECT_HYPOTHESIS.md",
    root / "docs" / "ALGORITHM_SELECTION.md",
    root / "docs" / "HARMONIZATION_NOTES.md",
    root / "docs" / "DATASET_LANDSCAPE.tsv",
    root / "docs" / "LITERATURE_BENCHMARK.md",
    root / "docs" / "REPORTING_GUIDELINES.md",
    root / "docs" / "AUDIT_REPORT.md",
    root / "docs" / "references" / "doi_list.csv",
    root / "docs" / "references" / "references.bib",
    root / "docs" / "references" / "vancouver.csl",
]

optional_dirs = [
    root / "docs" / "audit_runs",
]

exclude_dir_prefixes = [
    root / "openspec",
    root / "data",
    root / ".venv",
    root / "docs" / "submissions",
    root / "docs" / "manuscript",
    root / "docs" / "review_bundle",
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
            if p.name in ["lint_manuscript_style.sh"]:
                # Local-only writing lint (explicitly excluded from public review bundles)
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
    for d in optional_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file() and not is_excluded(p):
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
