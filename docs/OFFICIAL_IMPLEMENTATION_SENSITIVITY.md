# Full-cohort official-implementation sensitivity

This release includes full-13 sensitivity checks for official SpaGCN, official-linked STAGATE_pyG, and higher-nrep BayesSpace across the colorectal cancer Visium sections used in the manuscript.

## Scope

- Official SpaGCN 1.2.7 was run through its `SpaGCN` class with the histology graph and fixed-K k-means initialization.
- STAGATE was run through the STAGATE-linked PyG implementation with a KNN spatial graph, 1000 epochs, and KMeans fixed-K clustering of the learned embedding.
- BayesSpace was run through Bioconductor/Bioconda with `nrep=1000`.
- All checks used K=4 and K=6 with seeds 11, 23, and 37.
- Heavy SpaGCN, STAGATE, and high-nrep BayesSpace runs were executed on Colab sessions instead of the local laptop.

## Released Outputs

- `results/official_sensitivity/domain_maps_all/official_spagcn_stagate_bayesspace_maps.tsv`
- `results/official_sensitivity/evidence_all/`
- `results/official_sensitivity/comparison_all/`
- `results/official_sensitivity/run_logs/`
- `results/cross_method_boundary/full13_official_sensitivity.tsv`

## Interpretation

These checks calibrate implementation scope. They do not replace the full 13-section matched-input analysis. BayesSpace nrep=1,000 produced downstream opposite-sign point estimates at a frequency close to the matched-input BayesSpace analysis. Official SpaGCN retained seed sensitivity. Official-linked STAGATE_pyG was more stable and produced fewer downstream opposite-sign point estimates, showing that the matched-input STAGATE-style result should not be read as an official STAGATE instability claim.

## STAGATE implementation contrast

The matched-input STAGATE-style analysis used the common PCA matrix, a six-neighbor spatial graph, compact 200-epoch graph-attention training, and downstream clustering in the shared benchmark framework. The official-linked STAGATE_pyG check used the package workflow, including `Cal_Spatial_Net`, 1,000 training epochs, and fixed-K KMeans on the learned embedding. The two implementations differ in graph construction, optimization length, embedding formation, and clustering handoff; the official-linked result narrows the STAGATE conclusion to the tested implementation path.
