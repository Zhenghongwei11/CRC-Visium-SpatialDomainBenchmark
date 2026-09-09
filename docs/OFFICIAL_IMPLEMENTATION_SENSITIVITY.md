# Targeted official-implementation sensitivity

This release includes a targeted sensitivity check for three representative colorectal cancer Visium sections: `GSM9322957_TR11_206`, `GSM8265211_CTC21P`, and `GSM9322959_TR11_18105`.

## Scope

- Official SpaGCN 1.2.7 was run through its `SpaGCN` class with the histology graph and fixed-K k-means initialization.
- STAGATE was run through the STAGATE-linked PyG implementation with a KNN spatial graph, 1000 epochs, and KMeans fixed-K clustering of the learned embedding.
- BayesSpace was run through Bioconductor/Bioconda with `nrep=1000`.
- All checks used K=4 and K=6 with seeds 11, 23, and 37.
- Heavy SpaGCN, STAGATE, and high-nrep BayesSpace runs were executed on Colab T4 sessions.

## Released Outputs

- `results/official_sensitivity/domain_maps_all/official_spagcn_stagate_bayesspace_maps.tsv`
- `results/official_sensitivity/evidence_all/`
- `results/official_sensitivity/comparison_all/`
- `results/official_sensitivity/run_logs/`
- `results/cross_method_boundary/targeted_official_sensitivity.tsv`

## Interpretation

These checks calibrate implementation scope. They do not replace the full 13-section matched-input analysis. BayesSpace nrep=1,000 preserved the current BayesSpace reference-delta direction in the targeted set. Official SpaGCN retained seed sensitivity in representative sections. STAGATE_pyG was more stable in the same targeted design and produced no downstream opposite-sign point estimates in that subset.
