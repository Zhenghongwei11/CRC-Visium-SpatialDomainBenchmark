#!/usr/bin/env python3
"""
Backfill missing `notes` fields for stage-3 baseline rows.

Why:
Some earlier benchmark runs produced summary TSVs with empty `notes` values,
which breaks downstream filters that rely on stage tags (e.g., statistical gate
summary). This script deterministically assigns stage-3 note strings based on
`dataset_id` and `method_id`, only when the current `notes` field is empty.

It does NOT modify BayesSpace (stage-4) rows and does NOT touch non-summary
tables (e.g., per-seed stability tables).
"""

from __future__ import annotations

import csv
from pathlib import Path


BASELINE_METHODS = {
    "M0_expr_kmeans": "",
    "M1_spatial_concat_kmeans": "",
    "M2_spatial_ward": "-m2",
    "M3_spatial_leiden": "",
    "M4_spagcn": "-m4",
    "M5_stagate": "-m5",
}

STAGE_BY_DATASET = {
    "GSE311294": "stage3a",
    "GSE267401": "stage3b",
    "GSE280318": "stage3g",
}


def backfill_file(path: Path) -> int:
    if not path.exists():
        return 0

    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            return 0
        fieldnames = list(reader.fieldnames)
        if "notes" not in fieldnames:
            return 0
        for row in reader:
            rows.append(row)

    changed = 0
    for row in rows:
        notes = (row.get("notes") or "").strip()
        if notes:
            continue
        method_id = (row.get("method_id") or "").strip()
        if method_id not in BASELINE_METHODS:
            continue
        dataset_id = (row.get("dataset_id") or "").strip()
        stage = STAGE_BY_DATASET.get(dataset_id)
        if not stage:
            continue
        suffix = BASELINE_METHODS[method_id]
        row["notes"] = f"{stage}-full-replication{suffix}"
        changed += 1

    if changed == 0:
        return 0

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)
    return changed


def main() -> int:
    targets = [
        Path("results/benchmarks/method_benchmark.tsv"),
        Path("results/benchmarks/sensitivity_summary.tsv"),
        Path("results/figures/fig2_benchmark_summary.tsv"),
        Path("results/figures/fig3_sensitivity.tsv"),
        Path("results/figures/fig4_compute_and_guidance.tsv"),
    ]

    total = 0
    for t in targets:
        n = backfill_file(t)
        if n:
            print(f"[backfill] {t}: {n} rows updated")
        total += n

    if total == 0:
        print("[backfill] no changes needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

