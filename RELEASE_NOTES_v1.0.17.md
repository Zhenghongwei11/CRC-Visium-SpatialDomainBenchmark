# v1.0.17: Official sensitivity and stability-stratified downstream evidence

This release updates the public reproducibility package for the final JBCB R3 revision.

## Included

- Full-13 official-implementation sensitivity results for SpaGCN 1.2.7, STAGATE-linked PyG, and BayesSpace nrep=1,000.
- Colab scripts used for the targeted official SpaGCN/STAGATE/BayesSpace runs.
- Sample-K-method S14 sensitivity table with row-level stability, switching, boundary-enrichment, and downstream sign-sensitivity metrics.
- S15 stability-stratified downstream summary separating low-, moderate-, and high-ARI settings.
- Molecular tumor-stroma proxy interface results using EPCAM-high and COL1A1/FAP-high inferred domains.
- Updated consolidated supplementary workbook containing S1-S17 Data.
- Updated protocol, source-data map, citation metadata, and release notes.

## Interpretation

The official checks calibrate implementation scope and do not replace the full 13-section matched-input analysis. They show that downstream boundary sensitivity persists for BayesSpace and SpaGCN in the tested design, while official-linked STAGATE_pyG is more stable and produces fewer downstream opposite-sign point estimates.

The stability-stratified summary shows that interval-supported opposite-sign estimates are concentrated in low- or moderate-stability settings and were not observed above median ARI 0.80.

The molecular proxy-interface analysis moves the downstream test closer to the requested tumor-stroma consequence question. It remains expression-derived and includes a disclosed overlap because COL1A1/FAP helps define the proxy stromal side and is also reported as a readout.
