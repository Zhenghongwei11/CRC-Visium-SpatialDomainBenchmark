# Audit run bundle

- run_id: 2026-02-12_run155339Z_local
- created_utc: 2026-02-12T15:53:39Z

## What this is
A lightweight, internal provenance bundle capturing environment info and checksums for key outputs.

## What this is not
- Not a submission artifact.
- Not a guarantee of exact numerical reproducibility across platforms (Bayesian MCMC and numeric libs can differ).

## Reproduction entrypoints
- Baseline pipeline: scripts/run_crc_stage2_local.sh, scripts/run_crc_stage3_full_replication.sh
- BayesSpace stages: scripts/run_crc_stage4_bayesspace.sh ... scripts/run_crc_stage10_bayesspace_rigor_backfill.sh
- Claim gates: scripts/build_statistical_gate_summary.R
- Derived tables: scripts/build_required_artifacts.py
