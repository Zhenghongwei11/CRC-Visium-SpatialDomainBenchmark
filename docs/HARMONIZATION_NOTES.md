# Harmonization notes (CRC Visium benchmark)

## Purpose
Record dataset-to-dataset differences and the harmonization decisions used to make the benchmark comparable and auditable.

## Datasets (current)
- `GSE311294`: CRC Visium, flat `matrix.mtx.gz`-style inputs.
- `GSE267401`: CRC Visium, flat `matrix.mtx.gz`-style inputs.
- `GSE285505`: CRC Visium, 10x `filtered_feature_bc_matrix.h5`-style inputs (used for baseline-only local pilot in this version).

## Input-format harmonization
- The core pipeline currently operates on flat `matrix.mtx.gz` + barcodes + features + tissue-position CSV inputs.
- `GSE285505` is available locally and is included in baseline smoke tests, but BayesSpace ingestion in this repository currently expects the flat matrix format. As a result, BayesSpace benchmarking and claim-gated comparisons are restricted to datasets with the supported flat format.

## Preprocessing harmonization (principles)
- Use the same high-level preprocessing steps across datasets/samples: log normalization, HVG selection, PCA embedding, and fixed cluster-number grids.
- Avoid post hoc parameter tuning against the evaluation metrics (no "optimize K after seeing coherence").
- Report failures and incomplete coverage explicitly rather than silently excluding samples/methods.

## Evaluation harmonization (principles)
- Use the same metric definitions across datasets: spatial coherence, marker coherence, and stability summaries.
- Treat compute feasibility as environment-dependent and explicitly scope it to the measured hardware/software.

## Known limitations (tracked)
- Cross-dataset harmonization is constrained by input file formats; extending BayesSpace coverage to `GSE285505` requires a validated ingestion path that preserves comparability.
- Biological "ground truth" spatial domains are not assumed; the benchmark focuses on coherence and stability, not correctness against an unavailable truth label.

