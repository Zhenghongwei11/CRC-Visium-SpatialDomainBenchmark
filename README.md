# Regional measurements in colorectal cancer spatial transcriptomics

This repository contains code and scientific inputs for examining how spatial domain selection affects regional expression comparisons in colorectal cancer Visium data. The analyses use 22 tissue sections from 12 patients and compare annotated stromal tissue near and farther from tumor. The exploratory four-gene readout comprises TGFB1, CXCL12, ACTA2 and TAGLN.

The primary comparisons use archived publication domain maps and the corresponding tissue annotations. The original count matrices are obtained from the public resources below. The repository also supplies separate domain-refitting code; the original fitting environment was not completely version locked, so a new fit need not reproduce the archived maps exactly.

## Environment and input data

The regional and table analyses use Python 3.13.7 and the versions in `environments/analysis-python313.txt`. The combined entrypoint checks these versions before starting because graph-clustering results can change between package versions. The reference-mixture analysis additionally requires R 4.4.3, `Matrix`, `jsonlite` and the official `spacexr` 2.2.1 package at commit `9f5dc33c8060f946c6072a138b70e189636e1435`, including its Q-matrix data files. `environments/rctd-R.txt` records the versions used; see `third_party/spacexr/` for the corresponding license and the source functions checked at runtime. The scientific pipeline requires substantial storage and CPU time; model fitting and repeated sampling can take several hours.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/source_pipeline/source_inputs.py --cache raw_cache --output original_inputs
```

The input preparation downloads and verifies public archives. To use original files already available locally, provide a directory organized as `RESOURCE/SAMPLE/members/`; checksums are verified before analysis. The real-cell reference is a selected subset of GSE132465; `scripts/select_reference_cells.py`, `scripts/fetch_reference_umi.py` and `scripts/build_reference_subset.py` describe its reconstruction from public annotations and UMI counts. The included frozen reference preserves the exact input serialization used for the reported model fits.

## Reproduce the reported comparisons

```bash
python scripts/reproduce_current.py --counts original_inputs/counts --output results --workers 2
```

The command uses the archived publication maps, recomputes the measurements and sensitivities from original count data, and produces the main and supplementary figure and table outputs under `results/figures/` and `results/tables/`. The model-conditional summaries use inferred RNA mixtures and do not measure physical cell fractions. Table 3 is a qualitative interpretation table supplied as text; it is not a numerical statistic calculated from count matrices. Floating-point roundoff and Excel storage conventions can produce differences in the final digits of regenerated tables.

Supplementary figure and table labels follow `config/supplementary_numbering.json`; the figure exports, source filenames, workbook worksheets, field dictionary and figure index use the mapped numbers. The marker-context stage also records the fixed 20-gene mesenchymal panel's membership in each section's native full-mode RCTD bulk and regression lists at `results/marker_context/rctd_mesenchymal_marker_fit_membership.tsv`; this is same-RNA input-list membership, not independent cell-identity validation. The table is included with the composition sensitivity source tables.

The individual analysis stages and figure generation were run and compared with the reported results. The complete combined command has not yet been tested as one uninterrupted run. The archived domain maps are required for reproducing the reported comparisons; a complete refit of the original maps has not been verified.

The conditional-reference analyses use the numerical random seeds in `data/reference_inputs/random_seeds/`. These fixed seeds reproduce the reported sample, setting, control and replicate combinations. Analyses of additional combinations require a separately specified seed plan.

In the simulation tables, `freeze_sha256` identifies the original study-design freeze associated with the published simulation, not a freeze of a new execution or of the current implementation. The historical identifier is retained from the archived table inputs; the simulation's own `started_inputs.json` separately records the manifest and model code hashes used for a new run.

The 14 generated PNGs can be compared with the archived reference renders using `python scripts/figures/check_png_exports.py --figures results/figures`. PNG byte differences can arise from rendering software or platform changes.

The fitting and analysis configurations, gene panels, tissue masks, source identities and scientific tables are in `config/` and `data/`. Intermediate results and diagnostics are written under the selected output directory. The output directory must be empty at the start of the combined run.

## Refit the spatial domains

`scripts/source_pipeline/fit_source_maps.py` refits the configured domain methods from original counts. The neural methods and BayesSpace require additional environments described in `environments/README.md`. Saved refits from 22 sections, comparisons with archived maps and commands for the matched-reference sensitivity are supplied in [the domain-refit sensitivity guide](scripts/refit_sensitivity/README.md). Six requested Leiden configurations were unavailable in both sets, and one was available only in the refit. Refits are scientifically distinct runs and are not assumed to be identical to the archived publication maps. The full combined command from source inputs through all outputs has not been validated as a single uninterrupted run.

## Data and software sources

- Valdeolivas colorectal cancer spatial transcriptomics data and deconvolution outputs: [Zenodo record 7760264](https://zenodo.org/records/7760264), CC BY 4.0.
- Additional spatial transcriptomics data: [GEO GSE294385](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE294385) and its [deposited annotation metadata](https://github.com/yliuup/CRC_micromets_ST/tree/main/Meta_data).
- Single-cell reference data: [GEO GSE132465](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132465).
- RCTD implementation: [spacexr](https://github.com/dmcable/spacexr), GPL version 3; the included third-party source files retain that license.

See `config/source_pipeline/sources.json` and `config/source_pipeline/source_counts_manifest.json` for the source files and identities. Third-party data and software retain their own terms; attribution and reuse should follow the original source records. Cite this repository using `CITATION.cff` and also cite the original datasets and methods.
