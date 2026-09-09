# v1.0.17: Official sensitivity and stability-stratified downstream evidence

This release updates the public reproducibility package for the final JBCB R3 revision.

## Included

- Targeted official-implementation sensitivity results for SpaGCN 1.2.7, STAGATE-linked PyG, and BayesSpace nrep=1,000.
- Colab scripts used for the targeted official SpaGCN/STAGATE/BayesSpace runs.
- Sample-K-method S14 sensitivity table with row-level stability, switching, boundary-enrichment, and downstream sign-sensitivity metrics.
- S15 stability-stratified downstream summary separating low-, moderate-, and high-ARI settings.
- Updated consolidated supplementary workbook containing S1-S15 Data.
- Updated protocol, source-data map, citation metadata, and release notes.

## Interpretation

The targeted official checks are scope calibration rather than a replacement for the full 13-section matched-input analysis. They show that downstream boundary sensitivity persists for BayesSpace and SpaGCN in the targeted design, while official-linked STAGATE_pyG is more stable and produces no downstream opposite-sign point estimates in this subset.

The stability-stratified summary shows that interval-supported opposite-sign estimates are concentrated in low- or moderate-stability settings and were not observed above median ARI 0.80.
