#!/usr/bin/env python3
"""Export spatial Leiden partitions under declared graph perturbations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import run_crc_spatial_smoketest as benchmark


FIELDNAMES = [
    "dataset_id",
    "sample_id",
    "method_id",
    "K",
    "seed",
    "replicate_type",
    "replicate_id",
    "config_id",
    "barcode",
    "x",
    "y",
    "domain_label",
    "status",
    "notes",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--max-samples", type=int, default=999)
    parser.add_argument("--max-genes", type=int, default=2000)
    parser.add_argument("--k-grid", default="4,6")
    parser.add_argument("--neighbor-grid", default="4,6,8")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    entries = benchmark.find_sample_entries(dataset_root)
    allowlist = [value.strip() for value in str(args.sample_ids).split(",") if value.strip()]
    if allowlist:
        by_id = {entry["sample_id"]: entry for entry in entries}
        missing = [sample_id for sample_id in allowlist if sample_id not in by_id]
        if missing:
            raise FileNotFoundError(f"Requested samples not found: {', '.join(missing)}")
        entries = [by_id[sample_id] for sample_id in allowlist]
    else:
        entries = entries[: int(args.max_samples)]

    k_values = [int(value) for value in str(args.k_grid).split(",") if value.strip()]
    neighbor_values = [int(value) for value in str(args.neighbor_grid).split(",") if value.strip()]
    output_path = Path(args.output_tsv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.reset and output_path.exists():
        output_path.unlink()

    write_header = not output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        if write_header:
            writer.writeheader()

        for entry in entries:
            sample_id, matrix, coords, barcodes = benchmark.load_sample(entry)
            expression = benchmark.normalize_and_select(matrix, max_genes=int(args.max_genes))
            pcs = benchmark.build_pcs(expression)

            for neighbor_count in neighbor_values:
                spatial_graph = benchmark.spatial_connectivity_graph(
                    coords, neighbors=int(neighbor_count)
                )
                graph = benchmark._build_weighted_spatial_igraph(pcs, spatial_graph)
                for k_value in k_values:
                    resolution, observed, exact = benchmark._select_resolution_for_exact_k(
                        graph, target_k=int(k_value), seed=int(args.seed)
                    )
                    labels = benchmark._leiden_membership(
                        graph, resolution=float(resolution), seed=int(args.seed)
                    )
                    observed = int(np.unique(labels).size)
                    status = "success" if observed == int(k_value) else "nonexact_k"
                    note = (
                        "spatial Leiden graph-specification sensitivity;"
                        f"resolution={resolution:.8g};observed_k={observed};exact_k={int(exact)}"
                    )

                    for barcode, (x_coord, y_coord), label in zip(
                        barcodes.tolist(), coords.tolist(), labels.tolist(), strict=True
                    ):
                        writer.writerow(
                            {
                                "dataset_id": str(args.dataset_id),
                                "sample_id": str(sample_id),
                                "method_id": "M3_spatial_leiden",
                                "K": int(k_value),
                                "seed": int(args.seed),
                                "replicate_type": "graph_neighbors",
                                "replicate_id": f"neighbors_{int(neighbor_count)}",
                                "config_id": f"K{int(k_value)}_neighbors{int(neighbor_count)}",
                                "barcode": str(barcode),
                                "x": float(x_coord),
                                "y": float(y_coord),
                                "domain_label": int(label) + 1,
                                "status": status,
                                "notes": note,
                            }
                        )
                    handle.flush()
                    print(
                        f"[leiden] dataset={args.dataset_id} sample={sample_id} "
                        f"K={k_value} neighbors={neighbor_count} observed={observed}",
                        flush=True,
                    )

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
