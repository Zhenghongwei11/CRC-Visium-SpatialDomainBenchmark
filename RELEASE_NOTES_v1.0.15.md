# v1.0.15: Cross-method boundary reliability analysis

This release extends the CRC Visium analysis from global domain-map characteristics to a common boundary-reliability sequence across seven approaches.

## Included

- Coverage and alternative-partition stability across 13 CRC sections at `K=4` and `K=6`.
- Switching-spot localization relative to inferred domain boundaries and image-gradient context.
- Boundary-versus-interior expression and program-score robustness across alternative partitions.
- Method-relevant computational sensitivity summaries.
- Main and supplementary figures in PNG and PDF formats.
- Machine-readable TSV source tables, compressed spot-level switching data, and supplementary tables S1-S37.
- Scripts and a fixed protocol for regenerating intermediate maps, evidence tables, and figures from public GEO data.

## Interpretation

Sensitivity was heterogeneous across methods. Expression-only and coordinate-augmented k-means were stable under the tested seeds, whereas several spatial methods showed appreciable assignment changes under seed or graph perturbations. Switching spots were consistently enriched near inferred domain boundaries for the spatial methods with evaluable comparisons. Image-gradient evidence was mixed, so inferred boundaries should not be interpreted as pathology-validated interfaces. Selected downstream boundary summaries attenuated, disappeared, or changed direction across plausible partitions.
