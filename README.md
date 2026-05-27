# Benchmarking spatial domain identification in CRC Visium data

We benchmark spatial domain identification methods in colorectal cancer (CRC) 10x Genomics Visium spatial transcriptomics datasets and provide scripts plus derived tables to reproduce the key results. The benchmark emphasizes fixed-configuration comparisons, explicit statistical decision criteria, stability summaries across random seeds, and transparent reporting of trade-offs.

## Why this study matters
Spatial transcriptomics makes it possible to see how tumor cells, stroma, and immune compartments are organized in situ, but many downstream analyses depend on an upstream “spatial domain” map that is often chosen by eye. In colorectal cancer, domain boundaries can be gradual and mixed, so small analytic choices can change the apparent tissue structure. This project benchmarks a commonly used Bayesian spatial clustering method (BayesSpace) against simple baselines under fixed settings and reports quantitative evidence for domain quality and stability, with a focus on transparent, reproducible decision-making.

Zenodo DOIs:
- After a GitHub Release is published, Zenodo will automatically archive that version and mint (i) a **version DOI** and (ii) a **concept DOI** for the record family. Manuscripts should cite the **version DOI** corresponding to the exact release tag used.

## Quick start (reproduce key tables)
Prerequisites: Python (3.x) and R (with `Rscript`) available on PATH.

Run the minimal stages (these scripts will download public GEO data and create isolated environments automatically where needed):
- `bash scripts/run_crc_stage2_local.sh`
- `bash scripts/run_crc_stage3_full_replication.sh`
- `PYTHON_BIN=python3.11 bash scripts/run_crc_stage3e_spagcn_baseline.sh` (optional; SpaGCN baseline)
- `PYTHON_BIN=python3.11 bash scripts/run_crc_stage3f_stagate_baseline.sh` (optional; STAGATE-style baseline)
- `BAYES_INSTALL=1 bash scripts/run_crc_stage4_bayesspace.sh`
- `bash scripts/run_crc_stage10_bayesspace_rigor_backfill.sh`

Then rebuild the claim-gate table and derived artifacts:
- `Rscript scripts/build_statistical_gate_summary.R`
- `python3 scripts/build_required_artifacts.py`

Optional targeted sensitivity (BayesSpace MCMC depth):
- `bash scripts/run_crc_stage3g_bayesspace_nrep_sensitivity.sh` (nrep=1000 on two representative sections)
- `python3 scripts/build_bayesspace_nrep_sensitivity.py` (writes `results/benchmarks/bayesspace_nrep_sensitivity_summary.tsv`)

### One-click (end-to-end)
To run the full pipeline (tables + figures) with a single command:
- `bash scripts/reproduce_one_click.sh`

## Regenerating figures (optional)
If you want to regenerate publication figures locally, install the Python dependencies and rerun figure scripts:
- `python3 -m venv .venv && source .venv/bin/activate`
- `python -m pip install -r requirements.txt`
- `python scripts/make_publication_figures_v2.py`
- `python scripts/make_supplementary_figures.py`

## Data sources
Public GEO accessions used in this benchmark:
- GSE267401
- GSE311294
- GSE280318
- GSE289934 (optional portability demo; mouse brain)

The download URLs and file sizes are recorded in `docs/DATA_MANIFEST.tsv`.

## Optional portability demo (non-CRC)
To demonstrate that the evaluation framework can be applied outside CRC, we include a small non-CRC Visium dataset (GSE289934; mouse brain; 2 sections). If the dataset is downloaded and the STAGATE environment is available, you can build a descriptive portability summary and an image-based weak anchor (edge-alignment) table:

- `bash scripts/build_portability_edge_alignment.sh` (writes `results/benchmarks/portability_histology_edge_alignment.tsv`)
- `python3 scripts/build_portability_demo_summary.py` (writes `results/benchmarks/portability_demo_noncrc_summary.tsv`)
