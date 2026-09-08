# Cross-Method Boundary Reliability Protocol

Date locked: 2026-09-07

## Scope

This protocol defines a common evidence sequence for evaluating boundary reliability across seven spatial-domain approaches. It separates internal map properties, partition stability, spatial localization, downstream robustness, and computational sensitivity.

## Analysis set

- Human colorectal cancer Visium sections: 13 sections from GSE267401, GSE311294, and GSE285505.
- Domain resolutions: `K={4,6}`.
- Methods: BayesSpace, STAGATE-style, SpaGCN-style without histology input, spatial Leiden, spatial Ward, expression-only k-means, and coordinate-augmented k-means.
- Shared stochastic seeds: 11, 23, and 37.
- Spatial Ward perturbations: 4-, 6-, and 8-nearest-neighbor connectivity graphs, with 6 neighbors as the reference.
- Spatial Leiden computational sensitivity: 4-, 6-, and 8-nearest-neighbor graphs at seed 11, with the resolution recalibrated to the target `K` for each graph and 6 neighbors as the reference.

## Why K=4 and K=6

The two values provide a coarse and a finer practical partition while maintaining sufficiently populated domains across the 13 sections. They are used to examine resolution dependence under a compact fixed grid. They are not estimates of the true number of biological compartments and will not be described as optimal values.

## Operational definitions

- **Reference partition:** seed 11 for stochastic methods; the 6-neighbor connectivity graph for spatial Ward.
- **Plausible alternative partition:** a successfully completed partition from seed 23 or 37 at otherwise fixed settings, or a spatial Ward partition from the 4- or 8-neighbor graph at otherwise fixed settings.
- **Label alignment:** maximum-overlap assignment of alternative labels to reference labels at the same method, section, and `K`.
- **Switching spot:** a spot whose aligned label differs between an alternative partition and its reference partition.
- **Boundary spot:** a spot with at least one differently labelled neighbor in the six-nearest-neighbor spatial graph for the partition being analyzed.
- **Unstable analysis setting:** a method-section-`K` unit with median pairwise ARI below 0.60. This remains a descriptive screen for detailed localization, not a universal validity threshold.
- **Clear interior:** a spot that is not a boundary spot in the partition and does not switch across the declared alternatives.
- **Uncertain transition set:** switching spots and the partition-specific boundary neighborhoods used to test whether boundary placement affects downstream summaries. This computational set is not a pathology annotation.

## Evidence sequence

1. **Global characterization:** report spatial coherence and marker coherence separately for each method across all section-`K` units. These are internal map properties, not external accuracy measures.
2. **Alternative-partition stability:** report pairwise ARI and coverage for the declared seeds or Ward graph perturbations.
3. **Spatial localization:** map switching spots and quantify their relation to inferred boundary sets and image-gradient context.
4. **Downstream robustness:** recompute the same boundary-minus-interior biological features for every partition and report ranges, uncertainty, attenuation, disappearance, and direction changes.
5. **Computational sensitivity:** report available method-relevant settings without using them to select a winner.

## Switching-boundary relationship

For each reference-alternative comparison, report:

- number and fraction of switching spots;
- reference, alternative, and union boundary counts;
- fraction of switching and non-switching spots in the union boundary set;
- risk ratio and odds ratio with a 95% confidence interval where estimable;
- Jaccard overlap between switching and union-boundary sets;
- median distance from switching and non-switching spots to the nearest reference-boundary spot.

## Histology context

Image gradient will be sampled from GEO-provided detected-tissue images at Visium spot coordinates. Switching-versus-stable gradient differences will be reported as contextual evidence. No result will be described as a pathology-validated tumor-stroma interface without independent annotation.

## Downstream features

The same declared feature set will be used across methods and partitions where genes are available: EPCAM, COL1A1, FAP, SPP1, PTPRC, CAF/FAP-associated, SPP1-myeloid-associated, T-cell-associated, TGF-beta/CXCL12 myofibroblast-barrier, cytotoxic-lymphocyte, and CAF-plus-SPP1-myeloid-minus-T-cell contrast scores.

Primary reporting will use boundary-minus-interior median differences with bootstrap 95% intervals. A spatial-block analysis based on a fixed 6x6 array-coordinate grid will be reported where at least five spots from each comparison group occur in enough mixed blocks to estimate a contrast.

## Statistical interpretation

- ARI is continuous; 0.60 is a descriptive screen only.
- Localization and downstream results will be summarized per method and section before any cross-method statement.
- Direction change is defined relative to the reference partition and will be reported for every declared alternative, not selected examples alone.
- Spot-level P values will not be treated as independent-sample confirmation because neighboring Visium spots are spatially correlated.
- A cross-method conclusion requires recurrence across substantially different algorithmic families and adequate coverage. Otherwise, the conclusion will remain method- or setting-specific.

## Failure and missingness handling

- Failed runs remain in the coverage manifest with the error category and configuration.
- No section or method will be removed because its result weakens the proposed narrative.
- Units with fewer than two successful alternatives cannot support switching or downstream robustness and will be labeled not evaluable.
- Derived tables are written under `results/cross_method_boundary/`; intermediate domain maps and run logs are generated locally and are not required to inspect the reported results.
