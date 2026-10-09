# Regional measurements in colorectal cancer spatial transcriptomics

This repository contains the analyses of how computational domain selection changes a regional expression comparison. The comparison uses pathologist-annotated stroma near and farther from tumor in 22 Visium sections from 12 patients. The exploratory readout is the mean log-normalized expression of TGFB1, CXCL12, ACTA2 and TAGLN.

The main command recalculates the regional statistics, spatial sensitivity analyses, simulations, additional muscle-marker summaries and readout-gene withholding experiment. It generates Figures 1–3 and S1–S8, Table 1 and the files underlying Tables S1–S16.

## Run the analyses

Use Python 3.13.7 and R with `Rscript` available on the command line. The main analysis does not require a GPU.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce.py --output results --workers 2
```

The command downloads the public count packages into `raw_cache/`, verifies them against the recorded source identities, and writes all generated files to the chosen output directory. Downloads can be slow; retain the cache. Use `--resume` to continue an interrupted run. When the original count members are already available, supply `--counts /path/to/counts`, organized as `RESOURCE/SAMPLE/members/`. Each count member is checked before use.

The Python versions used for the regional analyses and withholding experiment are listed in [environments/analysis-python313.txt](environments/analysis-python313.txt). Runtime depends strongly on processor availability and download speed. The full calculation includes 5,200 independent simulated tissues and repeated spatial resampling; allow several hours.

## Inputs and outputs

The main run uses the saved publication domain maps in `data/reference_inputs/`. All regional contrasts, resampling draws, patient summaries and simulation results are calculated again. This preserves the specific fitted maps examined in the paper. `expected/` contains the submitted tables and figures for comparison and is never used as a calculation input.

The regional calculations have been recomputed for all 22 sections. The generated scientific tables agree with the current submission, and all eleven generated PNG figures match its figure pixels. Original-count recovery, muscle-marker measurements and the paired readout-gene withholding analysis were also run across all sections.

Generated tables are in `results/tables/`; figures are in `results/figures/` as PNG, PDF and TIFF. Per-section results remain in `results/workspace/analysis/`. Figure source data are written beside the tables. The principal reporting files are:

| Analysis | Files |
|---|---|
| Availability and patient comparisons | `current_table1.tsv`, `primary_patient_summary.tsv` |
| Regional selection and matched references | `regional_map_sensitivity.tsv`, `method_selection_comparison.tsv` |
| Known-boundary simulation | `simulation_cell_metrics.tsv`, `simulation_summary.tsv`, `simulation_threshold_sensitivity.tsv` |
| Original expression and annotations | `component_source_recovery.tsv`, `source_label_coverage.tsv` |
| Spatial choices and common-block intervals | `regional_patient_sensitivity.tsv`, `common_block_intervals.tsv`, `near_band_common_map_summary.tsv` |
| Additional muscle markers | `contractile_marker_section_summary.tsv`, `contractile_marker_block_summary.tsv` |
| Reference precision and withheld genes | `control_reference_precision/`, `readout_withholding/` |

After completing the analyses, tables and figures can be generated separately:

```bash
python scripts/reproduce.py --output results --stage tables
python scripts/reproduce.py --output results --stage figures
```

The complete Table 1 and Tables S1–S16 file index is in [TABLES.md](TABLES.md). `TABLE_FIELD_DICTIONARY.tsv` defines the measurements. `TABLE_FIELD_NAME_MAP.tsv` links the names in the published tables to the internal numerical variable names.

## Refit domains from original counts

The original-count fitting scripts are supplied separately so that new fits can be examined alongside the saved publication fits. SpaGCN, STAGATE and BayesSpace need their additional scientific environments. Python 3.11 dependency specifications are provided in `environments/source-python311.txt` and `environments/neural-python311.txt`; see [environments/README.md](environments/README.md) for setup and the extent of local validation.

The original model fits were run in Colab. [notebooks/fit_domains_colab.ipynb](notebooks/fit_domains_colab.ipynb) provides a Colab entrypoint for the current source scripts, with persistent inputs and per-section fits in Google Drive. It also connects completed fits to the regional analysis. The notebook's full neural/BayesSpace execution remains untested; it is separate from the locally verified statistical recalculation using the publication maps.

```bash
python scripts/source_pipeline/source_inputs.py --cache raw_cache --output original_inputs
python scripts/source_pipeline/fit_source_maps.py --inputs original_inputs --output source_fits --methods all
python scripts/reproduce.py --source-fits source_fits --counts original_inputs/counts --output results_refitted
```

Use `--resume` with `fit_source_maps.py` to continue the same fitting run. Completed sections are checked against their input, software and file identities.

The final command requires all 22 sections and all seven requested map families. It preserves the paper's configuration grid, including unavailable fits: six families for Valdeolivas and seven for GSE294385. BayesSpace is included only in the latter comparison. It checks the original barcode and coordinate registration before using the same downstream analysis. New numerical results may differ when fitting software changes. The original Colab fitting environment was not completely version locked; the saved maps therefore define the exact publication comparison.

For a smaller original-count example in the main Python environment:

```bash
python scripts/source_pipeline/fit_source_maps.py --inputs original_inputs --output baseline_example --methods baseline --loader native --samples M-ST-13
```

## Data and citation

The original resources are the [Valdeolivas colorectal cancer Visium dataset](https://zenodo.org/records/7760264), [GEO GSE294385](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE294385), and its [deposited annotation metadata](https://github.com/yliuup/CRC_micromets_ST/tree/main/Meta_data). `config/source_pipeline/sources.json` records the exact downloaded files and their identities. Tissue images and annotation classes retain their original meanings; annotations are not replaced by a new pathology assessment.

Please cite the study using `CITATION.cff` and cite the original datasets and domain methods when using them. The repository license covers our code and derived materials; third-party data and software retain their original terms.
