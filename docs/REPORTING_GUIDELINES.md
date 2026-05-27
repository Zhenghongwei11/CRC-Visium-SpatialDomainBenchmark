# Reporting guideline map (benchmark study; journal-agnostic)

This file maps journal-facing requirements to where we will satisfy them in this repo.

## Target journal
- Current draft target: BMC Bioinformatics (but artifacts are designed to be journal-agnostic).

## Study type
- Computational benchmark / best-practice paper using public data (spatial transcriptomics).

## Applicable “checklist” concepts
- General computational-biology journals: data availability + methods transparency + reproducibility expectations.
- STROBE‑MR: **Not applicable** unless we add MR/coloc as an evidence module.

## Requirement → artifact mapping (planned)

### Data availability
- Where: `docs/DATA_MANIFEST.tsv`
- What: accession IDs, download URLs, expected bundle sizes, and how to reproduce derived artifacts from raw.

### Code availability and reproducibility
- Where: `scripts/`, `docs/audit_runs/`
- What: staged entrypoints, fixed seeds where applicable, and end-to-end reproduction instructions (`README.md`).

### Method transparency (protocol lock)
- Where: `docs/ALGORITHM_SELECTION.md`, `docs/CLAIMS.tsv`, `docs/STATISTICAL_DECISION_RULES.md`
- What: predeclared metrics, parameter grids, fairness rules, overclaim boundaries, alpha/multiplicity/effect-size/direction-consistency gates.

### Statistical decision-gate traceability
- Where: `results/benchmarks/statistical_gate_summary.tsv`
- What: claim-level test family, multiplicity correction, effect-size gate, direction-consistency gate, and final support tier (`supported/suggestive/not supported`).

### Results traceability (tables-first)
- Where: `results/benchmarks/*.tsv`, `results/figures/*.tsv`, `docs/FIGURE_PROVENANCE.tsv`, `docs/SOURCE_DATA_MAP.tsv`
- What: every figure number originates from an explicit table.

### Figure quality and accessibility
- Where: `docs/FIGURE_STYLE_BENCHMARK.md`, `docs/PLOT_STYLE_GUIDE.md`, `plots/publication/`
- What: exemplar benchmarking, consistent styling, color-blind safe palette, and publication exports.

### Ethics / human subjects
- Where: manuscript Methods (later)
- What: statement that only de-identified public datasets were used; no new human subject recruitment.

### Submission-layer statistical checklist
- Where: `docs/CLAIMS.tsv` + `results/benchmarks/statistical_gate_summary.tsv`
- What: explicit mapping from prespecified gates to claim wording and support tier; document any downgraded claims.
