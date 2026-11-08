# Sensitivity to fitted spatial domains

The primary analyses use archived domain maps. A separate, outcome-informed sensitivity refitted the requested spatial-domain methods to the same 22 tissue sections, then compared the resulting partitions and regional contrasts with the archived maps. The two fits use the same tissue data and do not constitute independent biological validation. The complete software environment used for the original archived maps was not recorded.

`data/refit_domains/` contains the 834 available refitted configurations as one barcode-indexed table per section. `manifest.tsv` records row counts, configuration counts and SHA-256 digests. Six requested Leiden configurations were unavailable in both fits; one additional Leiden configuration was available only in the refit and did not meet the primary regional eligibility criteria. The recovered BayesSpace results for two sections were generated in a reconstructed R environment. A new fit may produce different partitions.

`scripts/source_pipeline/source_inputs.py` excludes raw 10x feature matrices during ZIP extraction, retaining the filtered matrices used for registered tissue spots. The original archives and their source identities are specified under `config/source_pipeline/`.

## Partition comparison

```bash
mkdir -p results/refit
python scripts/refit_sensitivity/compare_partitions.py \
  --output results/refit/map_comparisons.tsv
```

This computes barcode-matched adjusted Rand indices and selected-stromal-domain Jaccard overlaps, including unavailable configurations. The expected comparison is `expected/refit_sensitivity/map_comparisons.tsv`.

## Matched regional references

```bash
python scripts/refit_sensitivity/compare_sampling_reference.py \
  --output results/refit/baseline_map_results.jsonl
python scripts/refit_sensitivity/compare_wide_near175.py \
  --output results/refit/wide_band_map_results.jsonl
python scripts/refit_sensitivity/summarize_comparisons.py \
  --baseline results/refit/baseline_map_results.jsonl \
  --wide results/refit/wide_band_map_results.jsonl \
  --output-dir results/refit/summaries
```

Both comparisons hold the tissue scores and selection rules fixed while changing domain labels. The baseline comparison uses 1,000 size-only and block-matched sampling draws for each eligible selected domain; the 175-micrometer comparison uses the matched-location reference. Archived-map reference values are read from `expected/tables/`, which are generated separately by the primary archived-map pipeline. If the selected membership is identical, the matched reference can be reused; otherwise it is resampled using the applicable fixed seed rule. An eligible case requires at least 20 spots in at least three blocks per band. The four-way wide-band summary retains configurations evaluable in both fits under both band definitions. Patient summaries take medians across partitions, method/K settings and sections before taking resource medians across patients.

The two comparison commands accept `--sample` for a single-section run. `compare_spatial_settings.py` additionally evaluates both graph definitions and the five configured grid/band settings; select settings with `--settings` and graphs with `--graphs`. Full runs require computational time. Existing output files are resumed by configuration key; use a new output path for a fresh run.

For an immediate aggregation from the supplied map-level results, without recalculating sampling draws:

```bash
python scripts/refit_sensitivity/summarize_comparisons.py \
  --baseline expected/refit_sensitivity/baseline_map_results.tsv.gz \
  --wide expected/refit_sensitivity/wide_band_map_results.tsv.gz \
  --output-dir results/refit/summaries_from_supplied_results
```

Reference patient/resource summaries and the 394-map four-way comparison are in `expected/refit_sensitivity/`. The complete 840-configuration partition comparison and aggregation from the supplied map-level outputs were checked against those reference tables; the sampling-reference scripts were separately exercised on one section from each resource. A complete fresh, combined execution from downloaded original files through every figure and table has not been verified. For refitting original domains, use `scripts/source_pipeline/fit_source_maps.py` and the method-specific environments in `environments/README.md`; model fits and software versions may affect labels.
