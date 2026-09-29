# v1.0.19: Tissue-anchored regional expression contrasts

This release adds a revision-stage analysis of morphology-defined near-versus-far stromal expression in two public colorectal cancer Visium resources. It also includes a 52-condition, 5,200-realization known-boundary simulation that tests the regional measurement under specified conditions.

## Included

- Frozen simulation manifest, source fingerprints, analysis code, aggregate calibration tables and a preselected known-boundary example.
- Section-, patient- and cohort-level descriptive results for the Valdeolivas and GSE294385 Primary Colon resources, including non-evaluable settings and a four-gene score-component eligibility record.
- A four-panel figure with its generating script and panel-level source-table map.
- Revised source manifest, decision-rule cross-reference and reproduction instructions.

## Interpretation

One of nine eligible Valdeolivas sections and three of eight GSE294385 sections met the previously specified morphology-anchor support criterion. Some computational selections did not retain that support. No eligible primary comparison showed a supported direction reversal. The two qualifying paired additional-loss runs came from one patient section and involved broad selected-domain changes. These results do not establish local histological-boundary movement or a change in a cell-specific biological program. Simulation coverage varied across conditions; intervals for learned real-resource maps remain descriptive.

The release retains prior benchmark outputs for provenance. Raw expression matrices, registered replay inputs and large simulation checkpoints are not redistributed; acquisition and replay requirements are described in `README.md` and `docs/DATA_MANIFEST.tsv`.
