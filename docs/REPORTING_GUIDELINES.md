# Reporting guideline map (PLOS ONE; benchmark study)

This file maps journal-facing requirements to where we will satisfy them in this repo.

## Target journal
- PLOS ONE

## Study type
- Computational benchmark / best-practice paper using public data (spatial transcriptomics).

## Applicable “checklist” concepts
- PLOS ONE: data availability + methods transparency + reproducibility expectations.
- STROBE‑MR: **Not applicable** unless we add MR/coloc as an evidence module.

## Requirement → artifact mapping (planned)

### Data availability
- Where: `data/manifest.tsv`, `docs/DATA_PLAN.md`, and (submission layer later) `docs/submissions/PLOS_ONE/DATA_AVAILABILITY_STATEMENT.md`
- What: accession IDs, download URLs, and how to reproduce processed artifacts from raw.

### Code availability and reproducibility
- Where: `scripts/` (pipeline entrypoints later), `docs/audit_runs/` (run bundles later)
- What: pinned versions, seeds, environment export, and end-to-end reproduction instructions.

### Method transparency (protocol lock)
- Where: `docs/ALGORITHM_SELECTION.md`, `docs/CLAIMS.tsv`, `docs/STATISTICAL_DECISION_RULES.md`
- What: predeclared metrics, parameter grids, fairness rules, overclaim boundaries, alpha/multiplicity/effect-size/direction-consistency gates.

### Statistical decision-gate traceability
- Where: `results/benchmarks/statistical_gate_summary.tsv`
- What: claim-level test family, multiplicity correction, effect-size gate, direction-consistency gate, and final support tier (`supported/suggestive/not supported`).

### Results traceability (tables-first)
- Where: `results/benchmarks/*.tsv`, `results/figures/*.tsv`, `docs/FIGURE_STORYBOARD.tsv`
- What: every figure number originates from an explicit table.

### Figure quality and accessibility
- Where: `docs/FIGURE_STYLE_BENCHMARK.md`, `docs/PLOT_STYLE_GUIDE.md`, `plots/publication/`
- What: exemplar benchmarking, consistent styling, color-blind safe palette, and publication exports.

### Ethics / human subjects
- Where: manuscript Methods (later)
- What: statement that only de-identified public datasets were used; no new human subject recruitment.

### Submission-layer statistical checklist
- Where: `docs/submissions/PLOS_ONE/STATISTICAL_ANALYSIS_CHECKLIST.md`
- What: explicit mapping from predeclared gates to manuscript claim wording and support tier; document any downgraded claims.
