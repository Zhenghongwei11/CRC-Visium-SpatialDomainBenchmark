# Tissue-anchored regional expression contrasts in colorectal cancer Visium data

This repository contains analysis code and derived evidence for a near-versus-far stromal expression comparison in colorectal cancer Visium sections. The current release supplies the paper's two figures, Table 1 and nine supporting tables with scripts and their retained inputs. The earlier cross-method boundary analysis remains available in its original directories and prior tagged releases.

## Main result

Four of 17 eligible sections supported a morphology-defined four-gene contrast under the historical 0.50-MAD rule. Of 117 evaluable stromal-retention maps in those sections, 41 retained support: 15/41 references and 26/76 alternatives. The maps select subsets of fixed morphology-defined bands; these repeated map comparisons are not independent patients. No eligible primary comparison showed an interval-supported reversal. The simulation applies deterministic displacements to a known segmentation and describes conditional support-rule behavior; it does not test clustering algorithms or a colorectal cancer pathology mechanism.

## Current release contents

- `docs/interface_validation/reframed_protocol.md`: tissue anchor, fixed score, eligibility and interpretation rules.
- `docs/interface_validation/boundary_simulation_manifest.json` and `boundary_simulation_freeze_v1.json`: frozen 52-condition simulation and source fingerprints.
- `scripts/tissue_anchor_simulation.py`: known-boundary generator, estimation and summary export.
- `scripts/tissue_anchor_real_application.py`: application to registered spot inputs and saved maps.
- `scripts/make_tissue_anchor_figures.py`: regeneration of the four-panel figure from released derived tables.
- `results/interface_validation/reframed/simulation_full_v1/`: aggregate cell metrics, calibration summaries, threshold sensitivity and run status.
- `results/interface_validation/reframed/real_application_v1/`: section, patient and cohort results, per-map effects and component-score eligibility.
- `results/interface_validation/reframed/figures/`: figure PNG and PDF.
- `results/interface_validation/reframed/submission_v1/`: manuscript Figure 1, Figure 2, Table 1 and nine submission-facing supporting tables.
- `data/figure2_source/`: registered H&E image, spot coordinates and one section's saved membership used for Figure 2. The image and original spot annotation derive from Valdeolivas et al.'s CRC Visium release (Zenodo record 7760264, CC BY 4.0); the membership was computed in this analysis.
- `scripts/build_jbcb_reframed_evidence.py` and `scripts/build_jbcb_tissue_figure.py`: regenerate the current figures and derived tables.

The release does not redistribute raw expression matrices, all per-section replay inputs or 5,200 simulation checkpoint pairs. The aggregate simulation tables and derived real-resource effects are included. Replaying the full real-resource stage requires source-compatible registered spot inputs and saved maps at the paths described in `docs/interface_validation/real_resource_application_v1.json`. Those inputs were derived from the public Valdeolivas release and GSE294385; source accessions and acquisition information are in `docs/DATA_MANIFEST.tsv`. The current manuscript's figures and tables can be regenerated from the released derived effects, simulation tables, and Figure 2 image and membership. This is a derived-data reproduction, not a raw-matrix-to-result replay.

## Reproduce the figures and simulation

Use Python 3.13 and install the core packages in `requirements.txt` (the simulation run recorded Python 3.13.7 and the exact numerical-library versions in `RUN.json`). The legacy workflows also use `psutil` and `h5py`, for which the requirements file specifies compatible major-version ranges. Then run:

```bash
python3 scripts/build_jbcb_reframed_evidence.py \
  --output-dir results/interface_validation/reframed/submission_v1
python3 scripts/build_jbcb_tissue_figure.py \
  --output-dir results/interface_validation/reframed/submission_v1/FIGURES
```

The first command regenerates Figure 1, Table 1 and nine submission-facing tables from the released full real-resource and simulation tables. The second regenerates Figure 2 from `data/figure2_source/`. The earlier four-panel figure can still be regenerated with `python3 scripts/make_tissue_anchor_figures.py`.

For a full simulation replay, use the frozen manifest and source files in this version:

```bash
python3 scripts/tissue_anchor_simulation.py full \
  --manifest docs/interface_validation/boundary_simulation_manifest.json \
  --freeze docs/interface_validation/boundary_simulation_freeze_v1.json \
  --output-dir results/interface_validation/reframed/simulation_replay
```

The full simulation comprises 5,200 realizations and may require multiple bounded invocations to finish. The original aggregate output hashes are recorded in `results/interface_validation/reframed/simulation_full_v1/STATUS.json`.

The prior 13-section benchmark can still be run with `bash scripts/reproduce_one_click.sh`; its intermediate maps and raw GEO data are downloaded or generated by its stage scripts. It is distinct from the tissue-anchored revision analysis. Full real-resource replay requires the source-compatible registered inputs and historical saved maps described above; the archive provides the derived patient, section and map tables needed to inspect the reported results.

## Data and interpretation

The original computational screen used GSE267401, GSE311294 and GSE285505. The tissue-anchored application used the public Valdeolivas CRC Visium release and eight GSE294385 Primary Colon sections. The fixed real-resource score is the mean log-normalized expression of `TGFB1`, `CXCL12`, `ACTA2` and `TAGLN`. It is a regional composite, not a fibroblast-specific or functional assay. Compatible component-gene inputs for the prespecified score sensitivity were unavailable in the retained replay inputs, so that analysis is recorded as non-evaluable.

Repeated methods, maps, seeds and serial sections are nested within patients. The released map counts are not patient prevalence estimates. The real-resource intervals are descriptive because the fixed-map simulation does not calibrate maps learned from the same expression matrix.

## Citation and license

Cite the version DOI for the exact archived release. The family concept DOI is [10.5281/zenodo.19682586](https://doi.org/10.5281/zenodo.19682586). Version and authorship metadata are in `CITATION.cff`.

Code is released under the MIT License. Source data remain under their original repository terms.
