#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import NearestNeighbors


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype={"dataset_id": str, "sample_id": str, "method_id": str})


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


@dataclass(frozen=True)
class ImageGrad:
    grad: np.ndarray  # shape (H, W)


def load_detected_tissue_image_grad(extracted_dir: Path, sample_id: str) -> ImageGrad:
    img_gz = extracted_dir / f"{sample_id}_detected_tissue_image.jpg.gz"
    if not img_gz.exists():
        raise FileNotFoundError(img_gz)
    with gzip.open(img_gz, "rb") as handle:
        img = Image.open(io.BytesIO(handle.read())).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).astype(np.float32)

    # Central differences on the interior; pad to keep same shape.
    dx = np.zeros_like(gray)
    dy = np.zeros_like(gray)
    if gray.shape[1] >= 3:
        dx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    if gray.shape[0] >= 3:
        dy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
    grad = np.sqrt(dx**2 + dy**2).astype(np.float32, copy=False)
    return ImageGrad(grad=grad)


def knn_edges(xy: np.ndarray, k: int = 6) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    n = xy.shape[0]
    if n < 3:
        return np.zeros((0, 2), dtype=np.int32)
    nn = NearestNeighbors(n_neighbors=min(n, k + 1), algorithm="auto")
    nn.fit(xy)
    idx = nn.kneighbors(return_distance=False)
    edges: set[tuple[int, int]] = set()
    for i in range(n):
        for j in idx[i, 1:]:
            a, b = (i, int(j))
            if a == b:
                continue
            if a > b:
                a, b = b, a
            edges.add((a, b))
    if not edges:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(sorted(edges), dtype=np.int32)


def sample_grad_at_points(grad: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h, w = grad.shape
    xs = np.rint(xy[:, 0]).astype(int)
    ys = np.rint(xy[:, 1]).astype(int)
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    return grad[ys, xs].astype(np.float32, copy=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Weak external anchor: histology edge alignment for domain boundaries.")
    ap.add_argument(
        "--domain-maps",
        action="append",
        default=[],
        help="TSV containing per-spot domain maps (can be passed multiple times).",
    )
    ap.add_argument(
        "--output-tsv",
        default="results/benchmarks/histology_edge_alignment.tsv",
        help="Output TSV (per method×sample×K).",
    )
    ap.add_argument(
        "--data-root",
        default="data/raw",
        help="Root containing GEO extracted folders.",
    )
    args = ap.parse_args()

    domain_paths = [Path(p) for p in args.domain_maps if str(p).strip()]
    if not domain_paths:
        raise SystemExit("Provide at least one --domain-maps TSV")

    df_all = []
    for p in domain_paths:
        if not p.exists():
            raise FileNotFoundError(p)
        df_all.append(read_tsv(p))
    df = pd.concat(df_all, ignore_index=True)

    required = {"dataset_id", "sample_id", "method_id", "K", "x", "y", "domain_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in domain map table: {', '.join(sorted(missing))}")

    if "seed" not in df.columns:
        df["seed"] = ""

    out_rows: list[dict[str, Any]] = []
    data_root = Path(args.data_root)

    for (dataset_id, sample_id, method_id, k, seed), sub in df.groupby(
        ["dataset_id", "sample_id", "method_id", "K", "seed"], dropna=False
    ):
        dataset_id = str(dataset_id)
        sample_id = str(sample_id)
        method_id = str(method_id)
        k_int = int(k)
        seed_str = "" if pd.isna(seed) else str(seed)

        xy = sub[["x", "y"]].to_numpy(dtype=np.float32, copy=True)
        labels = sub["domain_label"].to_numpy(dtype=np.int32, copy=False)
        if xy.shape[0] < 50:
            continue

        extracted_dir = data_root / dataset_id / "extracted"
        try:
            img_grad = load_detected_tissue_image_grad(extracted_dir, sample_id)
        except Exception as exc:
            out_rows.append(
                {
                    "dataset_id": dataset_id,
                    "sample_id": sample_id,
                    "method_id": method_id,
                    "K": k_int,
                    "seed": seed_str,
                    "n_spots": int(xy.shape[0]),
                    "n_edges": "",
                    "n_boundary_edges": "",
                    "mean_grad_boundary_edges": "",
                    "mean_grad_within_edges": "",
                    "delta_grad_boundary_minus_within": "",
                    "mean_grad_boundary_spots": "",
                    "mean_grad_interior_spots": "",
                    "delta_grad_boundary_minus_interior": "",
                    "status": "missing_image",
                    "notes": str(exc),
                }
            )
            continue

        g_spot = sample_grad_at_points(img_grad.grad, xy)
        edges = knn_edges(xy, k=6)
        if edges.shape[0] == 0:
            continue

        a = edges[:, 0]
        b = edges[:, 1]
        same = labels[a] == labels[b]
        # Edge gradient proxy: mean gradient at the two endpoints.
        g_edge = 0.5 * (g_spot[a] + g_spot[b])

        boundary_edges = ~same
        within_edges = same

        mean_boundary_edges = float(np.mean(g_edge[boundary_edges])) if np.any(boundary_edges) else float("nan")
        mean_within_edges = float(np.mean(g_edge[within_edges])) if np.any(within_edges) else float("nan")
        delta_edges = mean_boundary_edges - mean_within_edges

        boundary_spot = np.zeros(xy.shape[0], dtype=bool)
        if np.any(boundary_edges):
            boundary_spot[a[boundary_edges]] = True
            boundary_spot[b[boundary_edges]] = True
        interior_spot = ~boundary_spot
        mean_boundary_spots = float(np.mean(g_spot[boundary_spot])) if np.any(boundary_spot) else float("nan")
        mean_interior_spots = float(np.mean(g_spot[interior_spot])) if np.any(interior_spot) else float("nan")
        delta_spots = mean_boundary_spots - mean_interior_spots

        out_rows.append(
            {
                "dataset_id": dataset_id,
                "sample_id": sample_id,
                "method_id": method_id,
                "K": k_int,
                "seed": seed_str,
                "n_spots": int(xy.shape[0]),
                "n_edges": int(edges.shape[0]),
                "n_boundary_edges": int(np.sum(boundary_edges)),
                "mean_grad_boundary_edges": round(mean_boundary_edges, 6) if np.isfinite(mean_boundary_edges) else "",
                "mean_grad_within_edges": round(mean_within_edges, 6) if np.isfinite(mean_within_edges) else "",
                "delta_grad_boundary_minus_within": round(delta_edges, 6) if np.isfinite(delta_edges) else "",
                "mean_grad_boundary_spots": round(mean_boundary_spots, 6) if np.isfinite(mean_boundary_spots) else "",
                "mean_grad_interior_spots": round(mean_interior_spots, 6) if np.isfinite(mean_interior_spots) else "",
                "delta_grad_boundary_minus_interior": round(delta_spots, 6) if np.isfinite(delta_spots) else "",
                "status": "success",
                "notes": "",
            }
        )

    out_path = Path(args.output_tsv)
    write_tsv(
        out_path,
        out_rows,
        fieldnames=[
            "dataset_id",
            "sample_id",
            "method_id",
            "K",
            "seed",
            "n_spots",
            "n_edges",
            "n_boundary_edges",
            "mean_grad_boundary_edges",
            "mean_grad_within_edges",
            "delta_grad_boundary_minus_within",
            "mean_grad_boundary_spots",
            "mean_grad_interior_spots",
            "delta_grad_boundary_minus_interior",
            "status",
            "notes",
        ],
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

