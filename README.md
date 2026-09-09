# Boundary sensitivity of spatial-domain maps in CRC Visium data

This repository contains code, derived tables, and publication figures for evaluating how spatial-domain assignments respond to plausible analytic perturbations in colorectal cancer (CRC) 10x Genomics Visium data.

The analysis compares seven approaches across 13 sections from three public cohorts at `K=4` and `K=6`. Expression-only and coordinate-augmented k-means were stable under the tested seeds. BayesSpace, spatial Ward, spatial Leiden, SpaGCN-style, and STAGATE-style maps showed varying degrees of assignment sensitivity. Switching spots were repeatedly enriched near inferred domain boundaries, but their image-gradient context was mixed. Boundary-focused expression summaries produced opposite-sign point estimates for selected method-setting combinations; the smaller subset in which both intervals excluded zero is reported separately. The `v1.0.17` release adds targeted official-implementation sensitivity results and a stability-stratified downstream summary.

These results describe reliability under the specified perturbations. Spatial coherence and marker coherence are internal map properties rather than external accuracy measures, and inferred boundaries are not pathology annotations.

## Repository contents

- `results/cross_method_boundary/`: analysis-ready evidence tables, coverage records, compressed switching-spot data, targeted official-sensitivity summaries, and stability-stratified downstream summaries.
- `results/official_sensitivity/`: Colab-generated targeted official SpaGCN, STAGATE_pyG, and BayesSpace nrep=1,000 result tables.
- `figures/cross_method_boundary/`: main and supplementary figures in PNG and PDF formats.
- `scripts/`: data preparation, domain analysis, sensitivity analysis, Colab official-implementation checks, evidence-table, and figure-generation code.
- `supplementary_tables/SUPPLEMENTARY_TABLES.xlsx`: focused workbook containing the current supplementary tables S1-S15.
- `docs/BOUNDARY_RELIABILITY_PROTOCOL.md`: fixed analysis definitions and interpretation rules.
- `docs/DATA_MANIFEST.tsv`: public data sources and download information.
- `docs/FIGURE_PROVENANCE.tsv` and `docs/SOURCE_DATA_MAP.tsv`: links from figures and reported results to machine-readable tables.

Large raw data and intermediate domain maps are not redistributed. They can be regenerated from the public GEO records and the scripts in this repository.

## Data sources

CRC Visium data:

- GSE267401
- GSE311294
- GSE285505

## Reproduce the analysis

Prerequisites are Python 3 and R with `Rscript` available on `PATH`. Individual method environments and public GEO downloads are prepared by the stage scripts where needed.

Run the established end-to-end benchmark pipeline:

```bash
bash scripts/reproduce_one_click.sh
```

Generate the cross-method figures directly from the released evidence tables:

```bash
python3 scripts/make_cross_method_boundary_figures.py
```

The targeted official-implementation sensitivity check can be run on Google Colab with:

```bash
colab new --gpu T4 --session jbcb-official-sensitivity
colab exec --session jbcb-official-sensitivity --file scripts/colab_official_sensitivity.py
colab exec --session jbcb-official-sensitivity --file scripts/colab_bayesspace_sensitivity.py
colab stop --session jbcb-official-sensitivity
```

The released results from those runs are under `results/official_sensitivity/`.

Intermediate partitions for the extended boundary analysis can be generated with:

```bash
bash scripts/run_bayesspace_partition_replicates.sh
python3 scripts/run_spatial_ward_perturbations.py --help
python3 scripts/run_spatial_leiden_perturbations.py --help
python3 scripts/run_crc_spatial_smoketest.py --help
```

After the intermediate maps are present under `results/cross_method_boundary/domain_maps/`, rebuild the evidence tables with:

```bash
python3 scripts/build_cross_method_boundary_evidence.py
python3 scripts/build_cross_method_computational_sensitivity.py
python3 scripts/build_cross_method_opposite_sign_evidence.py
python3 scripts/make_cross_method_boundary_figures.py
```

## Citation

The archived `v1.0.17` code and data-derived tables are available through the Zenodo record family [10.5281/zenodo.19682586](https://doi.org/10.5281/zenodo.19682586). Citation metadata are provided in `CITATION.cff`.

## License

Code is released under the MIT License. Source datasets remain subject to their original repository terms.
