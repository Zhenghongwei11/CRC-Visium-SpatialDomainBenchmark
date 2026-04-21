#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from sklearn.neighbors import NearestNeighbors


def _read_lines(path: Path) -> list[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [line.strip() for line in handle]
    return path.read_text(encoding="utf-8").splitlines()


def load_flat_sample(dataset_root: Path, sample_id: str) -> tuple[sparse.csr_matrix, np.ndarray]:
    matrix_path = dataset_root / f"{sample_id}_matrix.mtx.gz"
    barcodes_path = dataset_root / f"{sample_id}_barcodes.tsv.gz"
    coords_path = dataset_root / f"{sample_id}_tissue_positions.csv.gz"
    if not coords_path.exists():
        coords_path = dataset_root / f"{sample_id}_tissue_positions_list.csv.gz"

    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)
    if not barcodes_path.exists():
        raise FileNotFoundError(barcodes_path)
    if not coords_path.exists():
        raise FileNotFoundError(coords_path)

    counts = mmread(matrix_path).tocsr()
    barcodes = _read_lines(barcodes_path)
    coords = pd.read_csv(coords_path, header=None, compression="infer")
    if isinstance(coords.iloc[0, 0], str) and coords.iloc[0, 0] == "barcode":
        coords = pd.read_csv(coords_path, compression="infer")
    else:
        coords.columns = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ]

    if counts.shape[1] != len(barcodes):
        raise ValueError(f"Barcode mismatch: matrix spots={counts.shape[1]} barcodes={len(barcodes)}")

    coords = coords[coords["barcode"].isin(barcodes)].copy()
    coords["barcode"] = pd.Categorical(coords["barcode"], categories=barcodes, ordered=True)
    coords = coords.sort_values("barcode")
    in_tissue = coords["in_tissue"].to_numpy().astype(int) == 1
    if in_tissue.sum() < 50:
        raise ValueError("Too few in-tissue spots")

    # counts is genes x spots; transpose to spots x genes
    counts = counts.transpose().tocsr()
    counts = counts[in_tissue, :]
    xy = coords.loc[in_tissue, ["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()
    return counts, xy


def normalize_and_hvg(counts: sparse.csr_matrix, n_hvg: int = 2000) -> np.ndarray:
    counts_per_spot = np.asarray(counts.sum(axis=1)).ravel()
    counts_per_spot[counts_per_spot == 0] = 1.0
    scale = 1e4 / counts_per_spot
    normalized = counts.multiply(scale[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)

    means = np.asarray(normalized.mean(axis=0)).ravel()
    sq_means = np.asarray(normalized.power(2).mean(axis=0)).ravel()
    variances = np.maximum(sq_means - means**2, 0.0)
    top_idx = np.argsort(variances)[::-1][: min(n_hvg, normalized.shape[1])]
    dense = normalized[:, top_idx].toarray().astype(np.float32, copy=False)
    return dense


def spatial_connectivity_graph(coords: np.ndarray, neighbors: int = 6) -> sparse.csr_matrix:
    nn = NearestNeighbors(n_neighbors=min(neighbors + 1, len(coords)), algorithm="auto")
    nn.fit(coords)
    graph = nn.kneighbors_graph(coords, mode="connectivity")
    graph = graph.maximum(graph.T)
    return graph.tocsr()


def marker_separation_score_topn(x: np.ndarray, labels: np.ndarray, top_n: int) -> float:
    scores: list[float] = []
    for cluster in np.unique(labels):
        in_mask = labels == cluster
        out_mask = ~in_mask
        if in_mask.sum() < 3 or out_mask.sum() < 3:
            continue
        diff = x[in_mask].mean(axis=0) - x[out_mask].mean(axis=0)
        use_n = int(min(int(top_n), diff.shape[0]))
        if use_n < 1:
            continue
        top = np.sort(diff)[-use_n:]
        scores.append(float(np.mean(top)))
    return float(np.median(scores)) if scores else float("nan")


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_id",
        "sample_id",
        "method_id",
        "preprocessing_id",
        "K",
        "seed",
        "top_marker_n",
        "marker_coherence_median",
        "n_spots",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--sample-ids", required=True, help="Comma-separated sample IDs")
    parser.add_argument("--k-grid", default="4,6")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--top-n-grid", default="10,20,50")
    parser.add_argument("--stagate-num-pcs", type=int, default=50)
    parser.add_argument("--stagate-hidden-dim", type=int, default=64)
    parser.add_argument("--stagate-latent-dim", type=int, default=30)
    parser.add_argument("--stagate-max-epochs", type=int, default=120)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--note", default="topn-sensitivity-stagate")
    args = parser.parse_args()

    # Import inside main so the script can be inspected without torch installed.
    from stagate_minimal import STAGATEMinimal, STAGATEMinimalConfig  # type: ignore

    dataset_root = Path(args.dataset_root)
    sample_ids = [s.strip() for s in args.sample_ids.split(",") if s.strip()]
    k_values = [int(x) for x in args.k_grid.split(",") if x.strip()]
    topn_values = sorted({int(x) for x in args.top_n_grid.split(",") if x.strip()})

    rows: list[dict[str, object]] = []
    for sample_id in sample_ids:
        counts, coords = load_flat_sample(dataset_root, sample_id)
        x = normalize_and_hvg(counts, n_hvg=2000)
        graph = spatial_connectivity_graph(coords, neighbors=6)

        config = STAGATEMinimalConfig(
            num_pcs=int(args.stagate_num_pcs),
            hidden_dim=int(args.stagate_hidden_dim),
            latent_dim=int(args.stagate_latent_dim),
            lr=1e-3,
            max_epochs=int(args.stagate_max_epochs),
            weight_decay=0.0,
        )
        model = STAGATEMinimal(config)
        model.fit(x, graph, seed=int(args.seed))

        for k in k_values:
            labels = model.predict_labels(k=int(k), seed=int(args.seed))
            for top_n in topn_values:
                mk = marker_separation_score_topn(x, labels, top_n=int(top_n))
                rows.append(
                    {
                        "dataset_id": args.dataset_id,
                        "sample_id": sample_id,
                        "method_id": "M5_stagate",
                        "preprocessing_id": "log1p_hvg",
                        "K": int(k),
                        "seed": int(args.seed),
                        "top_marker_n": int(top_n),
                        "marker_coherence_median": round(float(mk), 6) if mk == mk else "",
                        "n_spots": int(x.shape[0]),
                        "notes": args.note,
                    }
                )

    write_tsv(Path(args.output_tsv), rows)
    print(f"Wrote {len(rows)} rows to {args.output_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

