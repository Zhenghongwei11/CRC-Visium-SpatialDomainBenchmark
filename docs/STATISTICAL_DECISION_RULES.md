# Statistical decision rules (predeclared)

## Scope
This rulebook defines claim-eligibility gates for the CRC spatial-domain benchmark project and must be locked before claim-upgrading analyses.

## Claims covered
- `C1_domain_quality`
- `C2_sensitivity`
- `C3_compute_feasibility`

## Analysis units and comparators
- Primary unit: sample-level paired comparison.
- Main method comparisons: `BayesSpace` vs `M0` and `BayesSpace` vs `M1` (fixed preprocessing and fixed `K`).
- Primary `K` set for main claims: `K in {4, 6}`.

## Primary metrics
- Domain quality metrics:
  - `spatial_coherence_median` (higher is better)
  - `marker_coherence_median` (higher is better)
- Stability metric:
  - `stability_ari_median` (higher is better)
  - `stability_ari_iqr` (lower is better)
- Compute metric:
  - `wall_time_sec_median` (lower is better)

## Statistical tests and intervals
- Paired significance test (primary): two-sided Wilcoxon signed-rank test at sample level.
- Effect-size summary (primary): paired median difference (`delta`) with bootstrap 95% CI.
- Direction-consistency summary: fraction of samples with effect in expected direction.

## Multiplicity control
- Family `F1` (C1 primary): BayesSpace vs the two k-means baselines (`M0`, `M1`) across the primary metrics; BH-FDR controlled at `q < 0.05`.
- Family `F1_ext` (C1 extension; exploratory): BayesSpace vs an additional spatially constrained baseline (`M2_spatial_ward`) reported as a sensitivity analysis; BH-FDR controlled at `q < 0.05` and interpreted as supplementary.
- Family `F2` (C2 sensitivity): all C2 sensitivity comparisons in one family, BH-FDR controlled at `q < 0.10`.
- Family `F3` (C3 compute): descriptive-first family; inferential claims require predeclared test and BH-FDR at `q < 0.05`.

## Practical-effect and consistency gates
- Effect-size gate:
  - Domain metrics (`C1`): `delta >= 0.02`.
  - Stability (`C2`): `stability_ari_median >= 0.60` and `stability_ari_iqr <= 0.15`.
  - Compute (`C3`): scope-limited local-first gate requires `wall_time_sec_median <= 1800` seconds (30 minutes) in the declared hardware/software context, and must be supported by explicit runtime/memory measurements (not single-run outliers).
- Direction-consistency gate:
  - Primary claims require consistency in expected direction for `>= 70%` evaluable samples.
  - Compute (`C3`) uses a local-feasibility fraction gate: `>= 80%` of evaluated sample×K runs must meet the `<= 1800` seconds threshold.

## Coverage and failure handling
- No selective sample dropping after seeing outcomes.
- Methods with runtime failures must log failures in `results/benchmarks/failure_log.tsv`.
- A claim is not eligible for `supported` if evaluable sample coverage is `< 80%` of predeclared target samples.

## Decision tiers
- `supported`: multiplicity-adjusted significance gate + effect-size gate + direction-consistency gate + coverage gate all pass.
- `suggestive`: effect-size and direction-consistency pass, but adjusted significance or coverage gate fails.
- `not supported`: effect-size or direction-consistency fails, or major protocol violations occur.

## Protocol amendment policy
Any change to thresholds, families, or decision logic after this lock requires:
1) timestamped amendment note in this file,
2) rationale for change,
3) full rerun/re-evaluation of affected comparisons.

## Amendment log
- 2026-02-12: Clarified that claim-gated BayesSpace comparisons currently use the BayesSpace-compatible cohorts with flat matrix inputs (no change to thresholds/tests/families).
- 2026-02-13: Clarified that `C3_compute_feasibility` uses a scope-limited 30-minute practical gate (`<= 1800` seconds) and a local-feasibility fraction threshold (`>= 0.80`) consistent with the locked gate table outputs (no change to executed analyses).
- 2026-02-21: Added an exploratory extension comparator (`M2_spatial_ward`) to contextualize the primary C1 results; extension comparisons are reported under a separate multiplicity family (`F1_ext`) and do not upgrade/downgrade the primary C1 claim family.
