# Tissue-anchored regional estimand: revision-stage protocol

Drafted: 2026-09-28. Status: **design freeze pending simulation-manifest and input gates**. This protocol was written after the historical R3/R4 outcomes were inspected. It governs new revision analyses prospectively from its freeze timestamp; it does not convert historical analyses into preregistered validation. The prior protocol in `docs/interface_validation/protocol.md` remains the record of those analyses.

## Question and units

The methodological question is whether a region selected by a computational spatial domain preserves the regional expression contrast defined by morphology-based tissue labels. The primary analysis unit in a real resource is a patient. Sections are nested within patients; maps, methods, seeds, scores, and resampling blocks are repeated measurements within sections.

The primary section-level estimand is a continuous near-minus-far contrast on a fixed tissue anchor. For a section `s`, let `A_near(s)` and `A_far(s)` be the morphology-labelled stromal spot sets defined below, and let `Y(s,i)` be the fixed score for spot `i`:

`D_anchor(s) = median_{i in A_near(s)} Y(s,i) - median_{i in A_far(s)} Y(s,i)`.

For a computational map `m`, use the outcome-blind tumor-like and stromal-like domain mapping from the historical protocol to retain eligible spots in those same anchor bands. The resulting contrast is `D_map(s,m)`. For a declared reference map `m0`, report both `D_map(s,m0) - D_anchor(s)` (reference map-to-anchor discrepancy) and `D_map(s,m) - D_map(s,m0)` (paired alternative-map change). These are changes in an **estimate** caused by selecting different spots; they are not causal effects on tissue biology.

## Tissue anchor, score, and eligibility

The real-resource tissue anchor is supplied barcode-keyed morphology/pathology-based labels, not a continuous histological contour. Pure tumor and pure stromal labels are mapped without the primary score; mixed, normal, unresolved, and excluded labels are not reassigned. Annotation provenance and expression blinding are recorded per resource. Both Valdeolivas and GSE294385 currently have morphology-based labels with expression blinding undocumented.

The fixed physical bands retain the historical definitions so that revision analyses can be compared with prior outputs: pure stromal spots with centers within 100 micrometres of a pure tumor spot are near; those 200–400 micrometres away in the same undirected stromal component are far. These are operational measurement windows at Visium resolution, not cytokine diffusion distances or cell-scale invasive-front widths. The full-resolution scalefactor must be finite. Each band requires at least 20 eligible spots and at least three nonempty spatial blocks. The existing code partitions the full array-coordinate extent into a 6-by-6 grid; these are not blocks containing 6-by-6 spots. Every failure is reported by reason.

The real-cohort primary readout is the historical arithmetic mean of log-normalized `TGFB1`, `CXCL12`, `ACTA2`, and `TAGLN`, requiring all four genes and the same normalization for every map within a section. It is called a **four-gene stromal-associated score**. This revision-stage reuse was chosen after historical outcomes were seen and will be reported as such. It does not measure fibroblast-specific function, a physical barrier, or immune exclusion. The score is never used to choose domains or eligible spots.

Simulation uses a generic generated regional signal with known truth, not a simulated claim that those four genes reproduce CRC pathology. Its score distribution and truth-generating model will be fixed in a separate simulation manifest before execution.

## Map eligibility and decomposition

The real-cohort map set is the existing saved, source-compatible partition set. New maps are not introduced because a result is favorable. The reference remains seed 11 for stochastic methods and the six-neighbor graph for spatial Ward; alternative seeds and graph neighborhoods remain those declared in the historical protocol. The analysis records method, K, seed/graph, map provenance, model diagnostics where available, and membership overlap. A map with ambiguous domain mapping, absent tumor/stromal pair, inadequate bands, failed run, or missing source registration is non-evaluable.

Every eligible map yields four separate outputs:

1. **Reference mismatch:** `D_map(s,m0) - D_anchor(s)` and whether a reference map reproduces an anchor-supported direction under the historical classification rule.
2. **Additional loss:** whether an alternative map loses a contrast supported by both anchor and reference, with denominator restricted to eligible reference-supported comparisons.
3. **Paired effect change:** `D_map(s,m) - D_map(s,m0)` with a paired interval and standardized magnitude.
4. **Direction reversal:** a separately labeled category requiring supported opposite directions under the same prespecified rule; opposite-sign point estimates alone are not reversals.

The strict computational-interface and anchor-retention estimators remain distinct. Local interface change and whole-compartment replacement are reported separately by membership overlap and spatial displacement; neither is inferred solely from a contrast change. `No supported difference` means the fixed support rule was not met, not that the true effect is zero. Equivalence or negligible-change claims require an interval contained within a prespecified margin.

## Uncertainty and decision rules

Continuous `D_anchor`, `D_map`, reference mismatch, and paired change are primary outputs. Within-section intervals use 1,000 spatial-block resamples with near and far blocks stratified and reference/alternative selections paired on the same sampled block IDs. The resampling code must record rejected draws and effective draws; a simulation calibration gate must assess coverage before intervals are used for strong real-resource claims. If conditional rejection creates poor coverage, the interval method must be revised under a documented post-freeze amendment and the original output preserved.

The historical 0.50×section stromal-score MAD support margin and 0.25×MAD paired-change flag are retained **only for comparability and sensitivity reporting**. They have no proven clinical or biological threshold interpretation. The main real-cohort summary will present continuous standardized effects (`D / MAD`), intervals, and distributions across patients. A threshold sweep fixed in the simulation manifest will show how classification changes near 0.50 and 0.25 rather than promoting one cutoff as a natural law. No new cutoff will be chosen based on real-cohort significance.

Patient-level aggregation first summarizes repeated sections within each patient, then reports patient distributions separately for Valdeolivas and GSE294385. Repeated seeds and methods do not increase patient `n`. A cross-patient recurrence statement requires eligible results from at least two independent patients with the same direction under the locked rule; a one-patient result is a case study. Report all eligible, failed, and non-evaluable settings, and do not pool cohorts into a single prevalence estimate without a declared sampling model.

## Smooth-muscle sensitivity gate

Before calculating any alternative score outcome, inventory raw or compatible spot-level expression matrices, gene identifiers, normalization, barcode-to-anchor registration, and spot membership for both resources. The first planned sensitivity readout is the two-gene mean of log-normalized `TGFB1` and `CXCL12`, using the identical anchor bands and maps. This removes `ACTA2` and `TAGLN` but changes the construct; compare direction, patient-level pattern, and score-MAD-standardized effects rather than raw effect magnitudes across scores. Its interpretation is **score-component dependence**, not proof of smooth-muscle contamination. A second marker set may be added only by a timestamped, literature-based amendment before its outcomes are viewed. If any required input fails, export a non-evaluable record and retain the limitation.

## Known-truth simulation gate

The simulation manifest must be finalized before the full run and must specify geometry, known boundary, true near/far contrast, spot mixing, spatial correlation, signal strength, boundary width, smoothing or graph perturbation, replications/seeds, and numeric tolerances. It must also specify interval-coverage target, bias summary, false-loss and false-reversal definitions, grid size, runtime ceiling, and a fixed threshold-sensitivity grid. Smoke tests may inspect generator validity and deterministic replay but not real-cohort outcome favorability. Parameter extensions after the freeze are labeled exploratory.

The simulation answers where the estimator works and fails under known conditions. It cannot certify the pathology accuracy of the public labels or establish CRC prevalence. If calibration is poor in relevant regimes, the real-resource inferential language must be downgraded accordingly.

## Amendments and release gate

At freeze, record UTC timestamp, SHA-256 of this file and the simulation manifest, code commit or file hashes, and the set of outcome tables already viewed. Every amendment records time, rationale, affected outputs, whether outcomes were visible, and a rerun identifier. Technical repairs discovered after outcomes remain labeled post-outcome implementation corrections.

No main-text conclusion, figure, or reviewer response may be finalized before simulation calibration, patient-level real-resource outputs, score-sensitivity eligibility, and claim registry review. The authors must approve the scientific claims and declarations before the package is called submission-ready.
