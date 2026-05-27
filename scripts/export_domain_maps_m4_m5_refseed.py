#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import run_crc_spatial_smoketest as smoketest  # type: ignore


def append_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export reference-seed domain maps for M4 (SpaGCN) and M5 (STAGATE).")
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--sample-ids", default="", help="Optional comma-separated allowlist.")
    ap.add_argument("--k-grid", default="4,6")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-genes", type=int, default=2000)
    ap.add_argument("--spagcn-p", type=float, default=0.50)
    ap.add_argument("--spagcn-max-epochs", type=int, default=200)
    ap.add_argument("--spagcn-max-spots", type=int, default=6500)
    ap.add_argument("--stagate-num-pcs", type=int, default=50)
    ap.add_argument("--stagate-hidden-dim", type=int, default=64)
    ap.add_argument("--stagate-latent-dim", type=int, default=30)
    ap.add_argument("--stagate-max-epochs", type=int, default=200)
    ap.add_argument("--stagate-max-spots", type=int, default=6500)
    ap.add_argument(
        "--methods",
        default="M5_stagate",
        help="Comma-separated subset of methods to export: M4_spagcn,M5_stagate",
    )
    ap.add_argument(
        "--output-tsv",
        default="results/domain_maps/domain_maps_m4_m5_refseed.tsv",
        help="Append-only TSV output path.",
    )
    ap.add_argument("--reset", action="store_true", help="If set, delete output before writing.")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    sample_entries = smoketest.find_sample_entries(dataset_root)
    allow = [s.strip() for s in str(args.sample_ids).split(",") if s.strip()]
    if allow:
        by_id = {e["sample_id"]: e for e in sample_entries}
        sample_entries = [by_id[s] for s in allow]

    k_grid = [int(x) for x in str(args.k_grid).split(",") if x.strip()]
    methods = {m.strip() for m in str(args.methods).split(",") if m.strip()}
    out_path = Path(args.output_tsv)
    if args.reset and out_path.exists():
        out_path.unlink()

    fieldnames = [
        "dataset_id",
        "sample_id",
        "method_id",
        "K",
        "seed",
        "barcode",
        "x",
        "y",
        "domain_label",
        "notes",
    ]

    for entry in sample_entries:
        sample_id, matrix, coords, barcodes = smoketest.load_sample(entry)
        x = smoketest.normalize_and_select(matrix, max_genes=int(args.max_genes))
        spatial_graph = smoketest.spatial_connectivity_graph(coords, neighbors=6)

        for k in k_grid:
            if "M4_spagcn" in methods:
                # SpaGCN (histology-free).
                labels4, note4 = smoketest._spagcn_membership(  # type: ignore[attr-defined]
                    x,
                    coords,
                    target_k=int(k),
                    seed=int(args.seed),
                    p_target=float(args.spagcn_p),
                    max_epochs=int(args.spagcn_max_epochs),
                    max_spots=int(args.spagcn_max_spots),
                )
                rows4: list[dict[str, object]] = []
                for bc, (xv, yv), lab in zip(barcodes.tolist(), coords.tolist(), labels4.tolist(), strict=True):
                    rows4.append(
                        {
                            "dataset_id": str(args.dataset_id),
                            "sample_id": str(sample_id),
                            "method_id": "M4_spagcn",
                            "K": int(k),
                            "seed": int(args.seed),
                            "barcode": str(bc),
                            "x": float(xv),
                            "y": float(yv),
                            "domain_label": int(lab) + 1,
                            "notes": str(note4),
                        }
                    )
                append_rows(out_path, rows4, fieldnames)

            if "M5_stagate" in methods:
                # STAGATE (portable implementation).
                labels5, note5 = smoketest._stagate_membership(  # type: ignore[attr-defined]
                    x,
                    spatial_graph,
                    target_k=int(k),
                    seed=int(args.seed),
                    num_pcs=int(args.stagate_num_pcs),
                    hidden_dim=int(args.stagate_hidden_dim),
                    latent_dim=int(args.stagate_latent_dim),
                    max_epochs=int(args.stagate_max_epochs),
                    max_spots=int(args.stagate_max_spots),
                )
                rows5: list[dict[str, object]] = []
                for bc, (xv, yv), lab in zip(barcodes.tolist(), coords.tolist(), labels5.tolist(), strict=True):
                    rows5.append(
                        {
                            "dataset_id": str(args.dataset_id),
                            "sample_id": str(sample_id),
                            "method_id": "M5_stagate",
                            "K": int(k),
                            "seed": int(args.seed),
                            "barcode": str(bc),
                            "x": float(xv),
                            "y": float(yv),
                            "domain_label": int(lab) + 1,
                            "notes": str(note5),
                        }
                    )
                append_rows(out_path, rows5, fieldnames)

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
