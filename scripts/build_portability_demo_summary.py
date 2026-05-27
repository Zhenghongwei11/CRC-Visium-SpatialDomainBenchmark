#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    inp = repo_root / "results" / "benchmarks" / "method_benchmark.tsv"
    out = repo_root / "results" / "benchmarks" / "portability_demo_noncrc_summary.tsv"
    if not inp.exists():
        raise FileNotFoundError(inp)

    df = pd.read_csv(inp, sep="\t", dtype=str).fillna("")
    df = df[(df["dataset_id"] == "GSE289934") & (df["method_id"].isin(["BayesSpace", "M5_stagate"]))].copy()
    if df.empty:
        raise SystemExit("No GSE289934 BayesSpace/STAGATE rows found in results/benchmarks/method_benchmark.tsv")

    # Remove internal run tags; the portability table is descriptive and should stay
    # neutral/readable for a journal submission.
    df["notes"] = ""

    # Optional weak external anchor for the portability demo: histology edge alignment
    # computed from reference-seed domain maps (if present).
    edge_path = repo_root / "results" / "benchmarks" / "portability_histology_edge_alignment.tsv"
    if edge_path.exists():
        edge = pd.read_csv(edge_path, sep="\t", dtype=str).fillna("")
        edge = edge[(edge["dataset_id"] == "GSE289934") & (edge["method_id"].isin(["BayesSpace", "M5_stagate"]))].copy()
        edge = edge.rename(
            columns={
                "delta_grad_boundary_minus_within": "edge_align_delta_grad_boundary_minus_within_refseed",
                "delta_grad_boundary_minus_interior": "edge_align_delta_grad_boundary_minus_interior_refseed",
                "seed": "edge_align_refseed",
            }
        )
        df = df.merge(
            edge[
                [
                    "dataset_id",
                    "sample_id",
                    "method_id",
                    "K",
                    "edge_align_refseed",
                    "edge_align_delta_grad_boundary_minus_within_refseed",
                    "edge_align_delta_grad_boundary_minus_interior_refseed",
                    "status",
                ]
            ].rename(columns={"status": "edge_align_status"}),
            on=["dataset_id", "sample_id", "method_id", "K"],
            how="left",
        )

    # Add reviewer-friendly, sign-agnostic summaries for the weak anchor.
    for col in [
        "edge_align_delta_grad_boundary_minus_within_refseed",
        "edge_align_delta_grad_boundary_minus_interior_refseed",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "edge_align_delta_grad_boundary_minus_within_refseed" in df.columns:
        df["edge_align_abs_boundary_minus_within_refseed"] = df[
            "edge_align_delta_grad_boundary_minus_within_refseed"
        ].abs()
        denom = df.groupby(["sample_id", "K"])["edge_align_abs_boundary_minus_within_refseed"].transform("median")
        df["edge_align_abs_boundary_minus_within_refseed_rel"] = df["edge_align_abs_boundary_minus_within_refseed"] / denom

    if "edge_align_delta_grad_boundary_minus_interior_refseed" in df.columns:
        df["edge_align_abs_boundary_minus_interior_refseed"] = df[
            "edge_align_delta_grad_boundary_minus_interior_refseed"
        ].abs()
        denom = df.groupby(["sample_id", "K"])["edge_align_abs_boundary_minus_interior_refseed"].transform("median")
        df["edge_align_abs_boundary_minus_interior_refseed_rel"] = df["edge_align_abs_boundary_minus_interior_refseed"] / denom

    keep = [
        "dataset_id",
        "sample_id",
        "method_id",
        "K",
        "seed_count",
        "spatial_coherence_median",
        "marker_coherence_median",
        "stability_ari_median",
        "wall_time_sec_median",
        "edge_align_refseed",
        "edge_align_delta_grad_boundary_minus_within_refseed",
        "edge_align_delta_grad_boundary_minus_interior_refseed",
        "edge_align_abs_boundary_minus_within_refseed",
        "edge_align_abs_boundary_minus_within_refseed_rel",
        "edge_align_abs_boundary_minus_interior_refseed",
        "edge_align_abs_boundary_minus_interior_refseed_rel",
        "edge_align_status",
        "notes",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = ""

    df = df[keep].sort_values(["sample_id", "method_id", "K"], kind="mergesort")
    rows = df.to_dict(orient="records")
    write_tsv(out, rows, fieldnames=keep)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
