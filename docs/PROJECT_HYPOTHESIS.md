# Project Hypothesis: CRC Visium spatial-domain identification benchmark

## Primary hypothesis
Spatially explicit domain identification methods (e.g., BayesSpace and spatial-graph clustering) produce domain maps that are:
- more **spatially coherent**, and
- more **reproducible** (seed/subsample stability + cross-sample marker coherence),
than expression-only clustering baselines, across multiple independent CRC Visium datasets, under a **predeclared** protocol.

## Scope
- Study type: benchmark / best-practice paper (pure computational biology; no wet lab).
- Task: **spatial domain identification / clustering** (NOT deconvolution as the mainline claim driver).
- Disease context: colorectal cancer (CRC) Visium spatial transcriptomics (human primary; mouse optional as stress-test).
- Target journal class: computational methods / bioinformatics benchmark venue (current draft target: **BMC Bioinformatics**).

## Non-goals (explicit)
- No new CRC “mechanism” claims beyond what is required to interpret domain marker coherence.
- No causal inference (MR/coloc) in the pilot.
- No clinical prediction claims (prognosis / treatment response) in the pilot.
- No requirement to include Visium HD / ultra-high-resolution datasets in the laptop pilot (may be an extension).
- No GPU-only methods as a dependency for the core results.

## Evidence-tier convention (project-local)
- Tier 0: plan only (no results).
- Tier 1: single-dataset computational evidence.
- Tier 2: multi-dataset replication + negative controls + sensitivity/stability reporting.
- Tier 3: Tier 2 plus orthogonal validation (e.g., histology alignment and/or external cell-type references).

## Success criteria (what “done” looks like)
- A locked evaluation protocol (`docs/ALGORITHM_SELECTION.md`) with fairness rules and negative controls.
- Standardized benchmark tables under `results/benchmarks/` that can regenerate all numbers in the paper.
- Figure lineage metadata (`docs/FIGURE_PROVENANCE.tsv` + `docs/SOURCE_DATA_MAP.tsv`) so that each main figure maps to one anchor table.
- Runtime/memory feasibility demonstrated on an **8GB laptop** for the core method set, with a clear cloud scale-up path.
