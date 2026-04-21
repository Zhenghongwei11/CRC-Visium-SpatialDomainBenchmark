#!/usr/bin/env python3
"""Run a local smoke test for CRC spatial-domain benchmarking."""

from __future__ import annotations

import argparse
import csv
import contextlib
import gzip
import io
import itertools
import math
import pathlib
import random
import time
from datetime import datetime, timezone
from typing import TypedDict

import h5py
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from scipy.io import mmread
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_lines(path: pathlib.Path) -> list[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [line.strip() for line in handle]
    return path.read_text(encoding="utf-8").splitlines()


class SampleEntry(TypedDict):
    sample_id: str
    matrix_path: str
    barcodes_path: str
    coords_path: str


def _discover_mtx_samples(dataset_extract_dir: pathlib.Path) -> list[SampleEntry]:
    entries: list[SampleEntry] = []
    for matrix_path in dataset_extract_dir.rglob("matrix.mtx*"):
        matrix_dir = matrix_path.parent
        if not (matrix_dir / "barcodes.tsv.gz").exists() and not (matrix_dir / "barcodes.tsv").exists():
            continue
        sample_root = matrix_dir.parent
        coords_candidates = list(sample_root.rglob("tissue_positions*.csv*"))
        if not coords_candidates:
            continue
        barcodes_path = resolve_file(matrix_dir, ["barcodes.tsv.gz", "barcodes.tsv"])
        entries.append(
            {
                "sample_id": sample_root.name,
                "matrix_path": str(matrix_path),
                "barcodes_path": str(barcodes_path),
                "coords_path": str(coords_candidates[0]),
            }
        )
    return entries


def _discover_flat_mtx_samples(dataset_extract_dir: pathlib.Path) -> list[SampleEntry]:
    entries: list[SampleEntry] = []
    for matrix_path in dataset_extract_dir.glob("*_matrix.mtx*"):
        matrix_name = matrix_path.name
        prefix = matrix_name.replace("_matrix.mtx.gz", "").replace("_matrix.mtx", "")
        barcodes_candidates = [
            dataset_extract_dir / f"{prefix}_barcodes.tsv.gz",
            dataset_extract_dir / f"{prefix}_barcodes.tsv",
        ]
        coords_candidates = [
            dataset_extract_dir / f"{prefix}_tissue_positions.csv.gz",
            dataset_extract_dir / f"{prefix}_tissue_positions.csv",
            dataset_extract_dir / f"{prefix}_tissue_positions_list.csv.gz",
            dataset_extract_dir / f"{prefix}_tissue_positions_list.csv",
        ]
        barcodes_path = None
        coords_path = None
        for candidate in barcodes_candidates:
            if candidate.exists():
                barcodes_path = candidate
                break
        for candidate in coords_candidates:
            if candidate.exists():
                coords_path = candidate
                break
        if barcodes_path is None or coords_path is None:
            continue
        entries.append(
            {
                "sample_id": prefix,
                "matrix_path": str(matrix_path),
                "barcodes_path": str(barcodes_path),
                "coords_path": str(coords_path),
            }
        )
    return entries


def _discover_h5_samples(dataset_extract_dir: pathlib.Path) -> list[SampleEntry]:
    entries: list[SampleEntry] = []
    for h5_path in dataset_extract_dir.rglob("*_filtered_feature_bc_matrix.h5"):
        sample_id = h5_path.name.replace("_filtered_feature_bc_matrix.h5", "")
        coords_candidates = [
            dataset_extract_dir / f"{sample_id}_tissue_positions_list.csv",
            dataset_extract_dir / f"{sample_id}_tissue_positions_list.csv.gz",
            dataset_extract_dir / f"{sample_id}_tissue_positions.csv",
            dataset_extract_dir / f"{sample_id}_tissue_positions.csv.gz",
        ]
        coords_path = None
        for candidate in coords_candidates:
            if candidate.exists():
                coords_path = candidate
                break
        if coords_path is None:
            continue
        entries.append(
            {
                "sample_id": sample_id,
                "matrix_path": str(h5_path),
                "barcodes_path": "",
                "coords_path": str(coords_path),
            }
        )
    return entries


def find_sample_entries(dataset_extract_dir: pathlib.Path) -> list[SampleEntry]:
    entries = _discover_mtx_samples(dataset_extract_dir)
    entries.extend(_discover_flat_mtx_samples(dataset_extract_dir))
    entries.extend(_discover_h5_samples(dataset_extract_dir))
    dedup: dict[str, SampleEntry] = {}
    for entry in entries:
        dedup[entry["sample_id"]] = entry
    return [dedup[key] for key in sorted(dedup)]


def resolve_file(parent: pathlib.Path, names: list[str]) -> pathlib.Path:
    for name in names:
        candidate = parent / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing one of {names} under {parent}")


def load_sample(sample_entry: SampleEntry) -> tuple[str, sparse.csr_matrix, np.ndarray]:
    matrix_path = pathlib.Path(sample_entry["matrix_path"])
    coords_path = pathlib.Path(sample_entry["coords_path"])

    if matrix_path.suffix == ".h5":
        with h5py.File(matrix_path, "r") as handle:
            group = handle["matrix"]
            data = np.array(group["data"])
            indices = np.array(group["indices"])
            indptr = np.array(group["indptr"])
            shape = tuple(np.array(group["shape"]).tolist())
            matrix = sparse.csc_matrix((data, indices, indptr), shape=shape).tocsr()
            barcodes = [
                value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)
                for value in np.array(group["barcodes"])
            ]
    else:
        barcodes_path = pathlib.Path(sample_entry["barcodes_path"])
        matrix = mmread(matrix_path).tocsr()
        barcodes = _read_lines(barcodes_path)

    if matrix.shape[1] != len(barcodes):
        raise ValueError(
            f"Barcode mismatch for {sample_entry['sample_id']}: matrix spots={matrix.shape[1]}, "
            f"barcodes={len(barcodes)}"
        )

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

    coords = coords[coords["barcode"].isin(barcodes)].copy()
    coords["barcode"] = pd.Categorical(coords["barcode"], categories=barcodes, ordered=True)
    coords = coords.sort_values("barcode")
    if "in_tissue" in coords.columns:
        in_tissue = coords["in_tissue"].to_numpy().astype(int) == 1
    else:
        in_tissue = np.ones(len(coords), dtype=bool)

    # 10x matrix orientation is genes x spots; transpose to spots x genes.
    matrix = matrix.transpose().tocsr()
    matrix = matrix[in_tissue, :]
    coord_array = coords.loc[in_tissue, ["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()

    sample_id = sample_entry["sample_id"]
    return sample_id, matrix, coord_array


def normalize_and_select(matrix: sparse.csr_matrix, max_genes: int) -> np.ndarray:
    counts_per_spot = np.asarray(matrix.sum(axis=1)).ravel()
    counts_per_spot[counts_per_spot == 0] = 1.0
    scale = 1e4 / counts_per_spot
    normalized = matrix.multiply(scale[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)

    means = np.asarray(normalized.mean(axis=0)).ravel()
    sq_means = np.asarray(normalized.power(2).mean(axis=0)).ravel()
    variances = np.maximum(sq_means - means**2, 0.0)
    top_idx = np.argsort(variances)[::-1][:max_genes]
    dense = normalized[:, top_idx].toarray().astype(np.float32)
    return dense


def build_pcs(x: np.ndarray, n_components: int = 20) -> np.ndarray:
    n_components = max(2, min(n_components, x.shape[0] - 1, x.shape[1] - 1))
    model = PCA(n_components=n_components, random_state=0)
    return model.fit_transform(x)


def spatial_coherence(labels: np.ndarray, coords: np.ndarray, neighbors: int = 6) -> float:
    nn = NearestNeighbors(n_neighbors=min(neighbors + 1, len(coords)), algorithm="auto")
    nn.fit(coords)
    idx = nn.kneighbors(return_distance=False)
    neighbor_idx = idx[:, 1:]
    matches = (labels[:, None] == labels[neighbor_idx]).mean()
    return float(matches)


def spatial_connectivity_graph(coords: np.ndarray, neighbors: int = 6) -> sparse.csr_matrix:
    nn = NearestNeighbors(n_neighbors=min(neighbors + 1, len(coords)), algorithm="auto")
    nn.fit(coords)
    graph = nn.kneighbors_graph(coords, mode="connectivity")
    graph = graph.maximum(graph.T)
    return graph.tocsr()


def marker_separation_score(x: np.ndarray, labels: np.ndarray) -> float:
    scores: list[float] = []
    for cluster in np.unique(labels):
        in_mask = labels == cluster
        out_mask = ~in_mask
        if in_mask.sum() < 3 or out_mask.sum() < 3:
            continue
        diff = x[in_mask].mean(axis=0) - x[out_mask].mean(axis=0)
        top = np.sort(diff)[-20:]
        scores.append(float(np.mean(top)))
    return float(np.median(scores)) if scores else float("nan")


def _require_leiden() -> tuple[object, object]:
    try:
        import igraph as ig  # type: ignore
        import leidenalg as la  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency probe
        raise RuntimeError(
            "Leiden baseline requested but dependencies are missing. "
            "Install `python-igraph` and `leidenalg` (see scripts/requirements_smoketest.txt)."
        ) from exc
    return ig, la


def _require_spagcn() -> tuple[object, object, object]:
    try:
        import torch  # type: ignore
        from spagcn_minimal import SpaGCNMinimal, SpaGCNMinimalConfig  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency probe
        raise RuntimeError(
            "SpaGCN baseline requested but dependencies are missing. "
            "Install the torch stack in an isolated environment (see scripts/requirements_spagcn.txt)."
        ) from exc
    return torch, SpaGCNMinimal, SpaGCNMinimalConfig


def _require_stagate() -> tuple[object, object, object]:
    try:
        import torch  # type: ignore
        from stagate_minimal import STAGATEMinimal, STAGATEMinimalConfig  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency probe
        raise RuntimeError(
            "STAGATE baseline requested but dependencies are missing. "
            "Install the torch stack in an isolated environment (see scripts/requirements_spagcn.txt)."
        ) from exc
    return torch, STAGATEMinimal, STAGATEMinimalConfig


def _build_weighted_spatial_igraph(pcs: np.ndarray, spatial_graph: sparse.csr_matrix):
    ig, _ = _require_leiden()
    graph = spatial_graph.tocoo()
    src = graph.row.astype(int)
    dst = graph.col.astype(int)
    keep = src < dst
    src = src[keep]
    dst = dst[keep]
    if len(src) == 0:
        raise ValueError("Spatial graph has no edges")

    x = pcs.astype(np.float64, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = x / norms
    cos = np.sum(x[src] * x[dst], axis=1)
    weights = np.maximum(0.0, (1.0 + cos) / 2.0).astype(float).tolist()

    edges = list(zip(src.tolist(), dst.tolist(), strict=True))
    g = ig.Graph(n=x.shape[0], edges=[(a, b) for a, b in edges], directed=False)
    g.es["weight"] = weights
    return g


def _leiden_membership(g, resolution: float, seed: int) -> np.ndarray:
    _, la = _require_leiden()
    try:
        part = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=float(resolution),
            seed=int(seed),
        )
    except TypeError:  # pragma: no cover - older leidenalg
        part = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=float(resolution),
        )
    return np.asarray(part.membership, dtype=int)


def _select_resolution_for_exact_k(g, target_k: int, seed: int) -> tuple[float, int, bool]:
    candidates = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0]
    evals: list[tuple[float, int]] = []
    for res in candidates:
        labels = _leiden_membership(g, res, seed=seed)
        k_obs = int(len(np.unique(labels)))
        evals.append((res, k_obs))
        if k_obs == target_k:
            return res, k_obs, True

    evals = sorted(evals, key=lambda t: t[0])
    below = [(r, k) for r, k in evals if k < target_k]
    above = [(r, k) for r, k in evals if k > target_k]
    if below and above:
        low = max(below, key=lambda t: t[0])[0]
        high = min(above, key=lambda t: t[0])[0]
        best_res = low
        best_k = dict(evals).get(low, 0)
        best_diff = abs(best_k - target_k)
        for _ in range(18):
            mid = math.sqrt(low * high)
            labels = _leiden_membership(g, mid, seed=seed)
            k_obs = int(len(np.unique(labels)))
            diff = abs(k_obs - target_k)
            if diff < best_diff:
                best_diff = diff
                best_res = mid
                best_k = k_obs
            if k_obs == target_k:
                return mid, k_obs, True
            if k_obs < target_k:
                low = mid
            else:
                high = mid
        # Final attempt: dense scan in the bracket on a log scale.
        scan = np.geomspace(low, high, num=60)
        for res in scan:
            labels = _leiden_membership(g, float(res), seed=seed)
            k_obs = int(len(np.unique(labels)))
            if k_obs == target_k:
                return float(res), k_obs, True
        return best_res, best_k, False

    # Fallback: broad scan on a log scale (robust to non-monotonicity).
    best_res, best_k = min(evals, key=lambda t: (abs(t[1] - target_k), t[0]))
    scan = np.geomspace(0.005, 10.0, num=90)
    for res in scan:
        labels = _leiden_membership(g, float(res), seed=seed)
        k_obs = int(len(np.unique(labels)))
        if k_obs == target_k:
            return float(res), k_obs, True
        diff = abs(k_obs - target_k)
        if diff < abs(best_k - target_k):
            best_res, best_k = float(res), k_obs
    return best_res, best_k, False


def _spagcn_membership(
    x: np.ndarray,
    coords: np.ndarray,
    target_k: int,
    seed: int,
    p_target: float,
    max_epochs: int,
    max_spots: int,
) -> tuple[np.ndarray, str]:
    if x.shape[0] > max_spots:
        raise RuntimeError(
            f"SpaGCN skipped: n_spots={x.shape[0]} exceeds --spagcn-max-spots={max_spots} "
            "(avoid O(n^2) adjacency blow-up)."
        )

    torch_mod, SpaGCNMinimal, SpaGCNMinimalConfig = _require_spagcn()

    coords_f = coords.astype(np.float32, copy=False)
    # Dense pairwise distances (xy-only, no histology) to avoid numba/llvmlite builds on macOS.
    gram = coords_f @ coords_f.T
    sq = np.sum(coords_f**2, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * gram
    dist2[dist2 < 0] = 0
    adj = np.sqrt(dist2).astype(np.float32, copy=False)

    def calc_p(l: float) -> float:
        adj_exp = np.exp(-1.0 * (adj**2) / (2.0 * (l**2)))
        return float(np.mean(np.sum(adj_exp, axis=1)) - 1.0)

    def search_l_silent(target_p: float, start: float = 0.01, end: float = 1000.0, tol: float = 0.01) -> float | None:
        p_low = calc_p(start)
        p_high = calc_p(end)
        if p_low > target_p + tol:
            return None
        if p_high < target_p - tol:
            return None
        if abs(p_low - target_p) <= tol:
            return float(start)
        if abs(p_high - target_p) <= tol:
            return float(end)
        low, high = float(start), float(end)
        for _ in range(60):
            mid = 0.5 * (low + high)
            p_mid = calc_p(mid)
            if abs(p_mid - target_p) <= tol:
                return float(mid)
            if p_mid <= target_p:
                low = mid
            else:
                high = mid
        return float(low)

    l_val = search_l_silent(p_target, start=0.01, end=1000.0, tol=0.01)

    if l_val is None or not np.isfinite(l_val):
        adj_flat = adj.ravel()
        adj_pos = adj_flat[np.isfinite(adj_flat) & (adj_flat > 0)]
        if adj_pos.size == 0:
            raise RuntimeError("SpaGCN failed: adjacency matrix has no positive finite distances")
        l_val = float(np.quantile(adj_pos, 0.05))

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch_mod.manual_seed(int(seed))

    config_obj = SpaGCNMinimalConfig(max_epochs=int(max_epochs))
    clf = SpaGCNMinimal(config=config_obj)
    clf.set_l(float(l_val))
    with contextlib.redirect_stdout(io.StringIO()):
        clf.train(x.astype(np.float32, copy=False), adj, n_clusters=int(target_k))
        labels = clf.predict()
    k_obs = int(len(np.unique(labels)))
    if k_obs != int(target_k):
        raise RuntimeError(f"SpaGCN produced {k_obs} clusters (target K={target_k})")

    note = f"spagcn_p={p_target};l={float(l_val):.4g};max_epochs={int(max_epochs)}"
    return labels, note


def _stagate_membership(
    x: np.ndarray,
    spatial_graph: sparse.csr_matrix,
    target_k: int,
    seed: int,
    num_pcs: int,
    hidden_dim: int,
    latent_dim: int,
    max_epochs: int,
    max_spots: int,
) -> tuple[np.ndarray, str]:
    if x.shape[0] > max_spots:
        raise RuntimeError(
            f"STAGATE skipped: n_spots={x.shape[0]} exceeds --stagate-max-spots={max_spots} "
            "(avoid excessive runtime on large graphs)."
        )

    torch_mod, STAGATEMinimal, STAGATEMinimalConfig = _require_stagate()

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch_mod.manual_seed(int(seed))

    config_obj = STAGATEMinimalConfig(
        num_pcs=int(num_pcs),
        hidden_dim=int(hidden_dim),
        latent_dim=int(latent_dim),
        max_epochs=int(max_epochs),
    )
    clf = STAGATEMinimal(config=config_obj)
    with contextlib.redirect_stdout(io.StringIO()):
        clf.fit(x.astype(np.float32, copy=False), spatial_graph, seed=int(seed))
        labels = clf.predict_labels(k=int(target_k), seed=int(seed))

    k_obs = int(len(np.unique(labels)))
    if k_obs != int(target_k):
        raise RuntimeError(f"STAGATE produced {k_obs} clusters (target K={target_k})")

    note = (
        f"stagate_num_pcs={int(num_pcs)};"
        f"hidden_dim={int(hidden_dim)};"
        f"latent_dim={int(latent_dim)};"
        f"max_epochs={int(max_epochs)}"
    )
    return labels, note


def append_rows(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as existing:
            header = existing.readline().rstrip("\n").split("\t")
        with path.open("r", encoding="utf-8", newline="") as existing:
            reader = csv.DictReader(existing, delimiter="\t")
            seen = {tuple((r.get(k, "") or "") for k in header) for r in reader}
        mode = "a"
    else:
        header = list(rows[0].keys())
        seen = set()
        mode = "w"

    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t", extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        for row in rows:
            out_row = {k: row.get(k, "") for k in header}
            key = tuple((out_row.get(k, "") or "") for k in header)
            if key in seen:
                continue
            seen.add(key)
            writer.writerow(out_row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="GSE285505")
    parser.add_argument("--dataset-root", default="data/raw/GSE285505/extracted")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument(
        "--sample-ids",
        default="",
        help="Optional comma-separated sample_id allowlist (runs only these samples, in the provided order).",
    )
    parser.add_argument("--max-genes", type=int, default=2000)
    parser.add_argument("--k-grid", default="4,6")
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument(
        "--methods",
        default="M0_expr_kmeans,M1_spatial_concat_kmeans",
        help="Comma-separated method IDs to run (baselines only).",
    )
    parser.add_argument("--results-bench", default="results/benchmarks")
    parser.add_argument("--results-fig", default="results/figures")
    parser.add_argument("--hardware-id", default="local-macbook-air-8gb-smoketest")
    parser.add_argument("--note", default="smoke-test")
    parser.add_argument("--spagcn-p", type=float, default=0.50)
    parser.add_argument("--spagcn-max-epochs", type=int, default=200)
    parser.add_argument("--spagcn-max-spots", type=int, default=6500)
    parser.add_argument("--stagate-num-pcs", type=int, default=50)
    parser.add_argument("--stagate-hidden-dim", type=int, default=64)
    parser.add_argument("--stagate-latent-dim", type=int, default=30)
    parser.add_argument("--stagate-max-epochs", type=int, default=200)
    parser.add_argument("--stagate-max-spots", type=int, default=6500)
    args = parser.parse_args()

    dataset_root = pathlib.Path(args.dataset_root)
    bench_dir = pathlib.Path(args.results_bench)
    fig_dir = pathlib.Path(args.results_fig)
    k_grid = [int(x) for x in args.k_grid.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    methods_requested = [m.strip() for m in args.methods.split(",") if m.strip()]

    sample_entries = find_sample_entries(dataset_root)
    allowlist = [s.strip() for s in str(args.sample_ids).split(",") if s.strip()]
    if allowlist:
        entry_by_id = {e["sample_id"]: e for e in sample_entries}
        missing = [sid for sid in allowlist if sid not in entry_by_id]
        if missing:
            raise FileNotFoundError(f"Requested --sample-ids not found under dataset_root: {', '.join(missing)}")
        sample_entries = [entry_by_id[sid] for sid in allowlist]
    else:
        sample_entries = sample_entries[: args.max_samples]
    if not sample_entries:
        raise FileNotFoundError(f"No sample entries found under {dataset_root}")

    process = psutil.Process()
    method_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    spatial_rows: list[dict[str, object]] = []
    marker_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []

    fig1_rows: list[dict[str, object]] = []
    fig2_rows: list[dict[str, object]] = []
    fig3_rows: list[dict[str, object]] = []
    fig4_rows: list[dict[str, object]] = []

    for sample_entry in sample_entries:
        sample_id, matrix, coords = load_sample(sample_entry)
        x = normalize_and_select(matrix, max_genes=args.max_genes)
        pcs = build_pcs(x)
        coords_scaled = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-6)
        spatial_features = np.concatenate([pcs, 0.5 * coords_scaled], axis=1)
        spatial_graph = spatial_connectivity_graph(coords, neighbors=6)

        method_configs: list[tuple[str, str, np.ndarray, list[int]]] = []
        if "M0_expr_kmeans" in methods_requested:
            method_configs.append(("M0_expr_kmeans", "kmeans", pcs, seeds))
        if "M1_spatial_concat_kmeans" in methods_requested:
            method_configs.append(("M1_spatial_concat_kmeans", "kmeans", spatial_features, seeds))
        if "M2_spatial_ward" in methods_requested:
            # Spatially constrained Ward clustering (expression PCs with coordinate connectivity constraint).
            method_configs.append(("M2_spatial_ward", "ward", pcs, [seeds[0]]))
        if "M3_spatial_leiden" in methods_requested:
            # Spatially constrained graph baseline (Leiden on weighted spatial kNN graph).
            method_configs.append(("M3_spatial_leiden", "leiden", pcs, seeds))
        if "M4_spagcn" in methods_requested:
            # SpaGCN (expression + spatial coordinates, no histology; fixed-K via kmeans init).
            method_configs.append(("M4_spagcn", "spagcn", pcs, seeds[:2] if len(seeds) > 1 else seeds))
        if "M5_stagate" in methods_requested:
            # STAGATE (graph-attention autoencoder; portable implementation; fixed-K via k-means).
            method_configs.append(("M5_stagate", "stagate", pcs, seeds[:2] if len(seeds) > 1 else seeds))

        fig1_rows.append(
            {
                "panel_id": "Fig1A",
                "record_type": "sample_overview",
                "dataset_id": args.dataset_id,
                "sample_id": sample_id,
                "role": "smoketest",
                "modality": "Visium",
                "platform": "10x",
                "organism": "Homo sapiens",
                "tissue": "CRC",
                "n_spots": int(matrix.shape[0]),
                "n_genes": int(matrix.shape[1]),
                "notes": f"{args.note} sample",
            }
        )

        method_control_seed_offsets = {
            "M0_expr_kmeans": 0,
            "M1_spatial_concat_kmeans": 100,
            "M2_spatial_ward": 200,
            "M3_spatial_leiden": 300,
            "M4_spagcn": 400,
            "M5_stagate": 500,
        }

        leiden_graph = None
        leiden_resolution_by_k: dict[int, tuple[float, int, bool]] = {}

        for method_id, method_kind, features, method_seeds in method_configs:
            for k in k_grid:
                labels_by_seed: dict[int, np.ndarray] = {}
                run_spatial_scores: list[float] = []
                run_marker_scores: list[float] = []
                run_times: list[float] = []
                run_mem: list[float] = []

                leiden_note = ""
                spagcn_note = ""
                stagate_note = ""
                if method_kind == "leiden":
                    if leiden_graph is None:
                        leiden_graph = _build_weighted_spatial_igraph(pcs, spatial_graph)
                    if k not in leiden_resolution_by_k:
                        res, k_obs, exact = _select_resolution_for_exact_k(leiden_graph, target_k=k, seed=seeds[0])
                        leiden_resolution_by_k[k] = (res, k_obs, exact)
                    res, k_obs, exact = leiden_resolution_by_k[k]
                    leiden_note = f"leiden_res={res:.4g};k_observed={k_obs};exact_k={int(exact)};res_seed={seeds[0]}"
                if method_kind == "spagcn":
                    spagcn_note = (
                        f"spagcn_p={float(args.spagcn_p):.3g};"
                        f"max_epochs={int(args.spagcn_max_epochs)};"
                        f"max_spots={int(args.spagcn_max_spots)}"
                    )
                if method_kind == "stagate":
                    stagate_note = (
                        f"stagate_num_pcs={int(args.stagate_num_pcs)};"
                        f"hidden_dim={int(args.stagate_hidden_dim)};"
                        f"latent_dim={int(args.stagate_latent_dim)};"
                        f"max_epochs={int(args.stagate_max_epochs)};"
                        f"max_spots={int(args.stagate_max_spots)}"
                    )

                for seed in method_seeds:
                    if method_kind in {"spagcn", "stagate"}:
                        print(
                            f"[run] method={method_id} sample={sample_id} K={k} seed={seed} note={args.note}",
                            flush=True,
                        )
                    started = time.perf_counter()
                    started_utc = utc_now()
                    status = "success"
                    error_message = ""
                    labels = None
                    try:
                        if method_kind == "kmeans":
                            model = KMeans(
                                n_clusters=k,
                                random_state=seed,
                                n_init=10,
                                max_iter=300,
                            )
                            labels = model.fit_predict(features)
                        elif method_kind == "ward":
                            model = AgglomerativeClustering(
                                n_clusters=k,
                                linkage="ward",
                                connectivity=spatial_graph,
                            )
                            labels = model.fit_predict(features)
                        elif method_kind == "leiden":
                            if leiden_graph is None:
                                raise RuntimeError("Leiden graph was not initialized")
                            res, _, _ = leiden_resolution_by_k[k]
                            exact = bool(leiden_resolution_by_k[k][2])
                            if (not exact) and str(args.note).startswith("stage3"):
                                raise RuntimeError("Leiden resolution search did not find exact K for this sample")
                            labels = _leiden_membership(leiden_graph, res, seed=seed)
                            if exact and int(len(np.unique(labels))) != int(k):
                                raise RuntimeError(
                                    f"Leiden produced {int(len(np.unique(labels)))} clusters (target K={k})"
                                )
                        else:
                            if method_kind == "spagcn":
                                labels, spagcn_run_note = _spagcn_membership(
                                    x,
                                    coords,
                                    target_k=k,
                                    seed=seed,
                                    p_target=float(args.spagcn_p),
                                    max_epochs=int(args.spagcn_max_epochs),
                                    max_spots=int(args.spagcn_max_spots),
                                )
                                spagcn_note = spagcn_run_note
                            elif method_kind == "stagate":
                                labels, stagate_run_note = _stagate_membership(
                                    x,
                                    spatial_graph,
                                    target_k=k,
                                    seed=seed,
                                    num_pcs=int(args.stagate_num_pcs),
                                    hidden_dim=int(args.stagate_hidden_dim),
                                    latent_dim=int(args.stagate_latent_dim),
                                    max_epochs=int(args.stagate_max_epochs),
                                    max_spots=int(args.stagate_max_spots),
                                )
                                stagate_note = stagate_run_note
                            else:
                                raise ValueError(f"Unknown method_kind={method_kind}")
                    except Exception as exc:  # pragma: no cover - defensive logging
                        status = "failed"
                        error_message = str(exc)
                    elapsed = time.perf_counter() - started
                    finished_utc = utc_now()
                    peak_rss_mb = process.memory_info().rss / (1024 * 1024)

                    runtime_rows.append(
                        {
                            "dataset_id": args.dataset_id,
                            "sample_id": sample_id,
                            "method_id": method_id,
                            "preprocessing_id": "log1p_hvg",
                            "param_set_id": f"K{k}",
                            "K": k,
                            "seed": seed,
                            "status": status,
                            "wall_time_sec": round(elapsed, 6),
                            "cpu_time_sec": round(elapsed, 6),
                            "peak_rss_mb": round(peak_rss_mb, 3),
                            "disk_tmp_bytes": "",
                            "started_utc": started_utc,
                            "finished_utc": finished_utc,
                            "hardware_id": args.hardware_id,
                            "software_versions": "python-smoketest",
                            "notes": (
                                leiden_note
                                if method_kind == "leiden"
                                else (
                                    spagcn_note
                                    if method_kind == "spagcn"
                                    else (stagate_note if method_kind == "stagate" else "")
                                )
                            ),
                        }
                    )

                    if labels is None:
                        failure_rows.append(
                            {
                                "dataset_id": args.dataset_id,
                                "sample_id": sample_id,
                                "method_id": method_id,
                                "preprocessing_id": "log1p_hvg",
                                "param_set_id": f"K{k}",
                                "K": k,
                                "seed": seed,
                                "failure_type": "runtime_error",
                                "error_message": error_message,
                                "log_path": "",
                                "stacktrace_path": "",
                                "wall_time_sec": round(elapsed, 6),
                                "peak_rss_mb": round(peak_rss_mb, 3),
                                "finished_utc": finished_utc,
                            "notes": (
                                leiden_note
                                if method_kind == "leiden"
                                else (
                                    spagcn_note
                                    if method_kind == "spagcn"
                                    else (stagate_note if method_kind == "stagate" else "")
                                )
                            ),
                        }
                    )
                        continue

                    labels_by_seed[seed] = labels
                    sp_score = spatial_coherence(labels, coords)
                    mk_score = marker_separation_score(x, labels)
                    run_spatial_scores.append(sp_score)
                    run_marker_scores.append(mk_score)
                    run_times.append(elapsed)
                    run_mem.append(peak_rss_mb)

                if not labels_by_seed:
                    continue

                ari_values: list[float] = []
                for a, b in itertools.combinations(labels_by_seed, 2):
                    ari = adjusted_rand_score(labels_by_seed[a], labels_by_seed[b])
                    ari_values.append(float(ari))
                    stability_rows.append(
                        {
                            "dataset_id": args.dataset_id,
                            "sample_id": sample_id,
                            "method_id": method_id,
                            "preprocessing_id": "log1p_hvg",
                            "param_set_id": f"K{k}",
                            "K": k,
                            "stability_type": "seed_pair_ari",
                            "replicate_id": f"{a}_vs_{b}",
                            "seed": a,
                            "seed2": b,
                            "metric_name": "ari",
                            "metric_value": round(float(ari), 6),
                            "n_spots": int(matrix.shape[0]),
                            "notes": "",
                        }
                    )

                for seed, labels in labels_by_seed.items():
                    sp_score = spatial_coherence(labels, coords)
                    spatial_rows.append(
                        {
                            "dataset_id": args.dataset_id,
                            "sample_id": sample_id,
                            "method_id": method_id,
                            "preprocessing_id": "log1p_hvg",
                            "param_set_id": f"K{k}",
                            "K": k,
                            "metric_name": "neighbor_agreement",
                            "metric_value": round(sp_score, 6),
                            "bootstrap_ci_lower": "",
                            "bootstrap_ci_upper": "",
                            "n_spots": int(matrix.shape[0]),
                            "notes": "",
                        }
                    )

                reference_seed = method_seeds[0]
                ref_labels = labels_by_seed[reference_seed]
                for cluster in np.unique(ref_labels):
                    in_mask = ref_labels == cluster
                    out_mask = ~in_mask
                    if in_mask.sum() < 3 or out_mask.sum() < 3:
                        continue
                    diff = x[in_mask].mean(axis=0) - x[out_mask].mean(axis=0)
                    marker_rows.append(
                        {
                            "dataset_id": args.dataset_id,
                            "sample_id": sample_id,
                            "method_id": method_id,
                            "preprocessing_id": "log1p_hvg",
                            "param_set_id": f"K{k}",
                            "K": k,
                            "domain_id": int(cluster),
                            "top_marker_n": 20,
                            "marker_reproducibility_within_dataset": round(
                                float(np.mean(np.sort(diff)[-20:])), 6
                            ),
                            "marker_reproducibility_across_datasets": "",
                            "enrichment_coherence_score": round(float(np.max(diff)), 6),
                            "notes": "",
                        }
                    )

                offset = method_control_seed_offsets.get(method_id, 0)
                random_rng = np.random.default_rng(20260211 + k + offset)
                rand_labels = random_rng.integers(0, k, size=ref_labels.shape[0])
                rand_sp = spatial_coherence(rand_labels, coords)
                real_sp = spatial_coherence(ref_labels, coords)
                control_rows.append(
                    {
                        "dataset_id": args.dataset_id,
                        "sample_id": sample_id,
                        "method_id": method_id,
                        "preprocessing_id": "log1p_hvg",
                        "param_set_id": f"K{k}",
                        "K": k,
                        "seed": reference_seed,
                        "control_type": "random_labels",
                        "metric_name": "neighbor_agreement",
                        "metric_value": round(rand_sp, 6),
                        "metric_value_real": round(real_sp, 6),
                        "delta_vs_real": round(rand_sp - real_sp, 6),
                        "notes": "",
                    }
                )

                shuffled_coords = coords.copy()
                random_rng.shuffle(shuffled_coords)
                shuf_sp = spatial_coherence(ref_labels, shuffled_coords)
                control_rows.append(
                    {
                        "dataset_id": args.dataset_id,
                        "sample_id": sample_id,
                        "method_id": method_id,
                        "preprocessing_id": "log1p_hvg",
                        "param_set_id": f"K{k}",
                        "K": k,
                        "seed": reference_seed,
                        "control_type": "shuffled_coordinates",
                        "metric_name": "neighbor_agreement",
                        "metric_value": round(shuf_sp, 6),
                        "metric_value_real": round(real_sp, 6),
                        "delta_vs_real": round(shuf_sp - real_sp, 6),
                        "notes": "",
                    }
                )

                median_ari = float(np.median(ari_values)) if ari_values else float("nan")
                iqr_ari = (
                    float(np.percentile(ari_values, 75) - np.percentile(ari_values, 25))
                    if ari_values
                    else float("nan")
                )
                median_sp = float(np.median(run_spatial_scores)) if run_spatial_scores else float("nan")
                iqr_sp = (
                    float(np.percentile(run_spatial_scores, 75) - np.percentile(run_spatial_scores, 25))
                    if run_spatial_scores
                    else float("nan")
                )
                median_mk = float(np.median(run_marker_scores)) if run_marker_scores else float("nan")
                iqr_mk = (
                    float(np.percentile(run_marker_scores, 75) - np.percentile(run_marker_scores, 25))
                    if run_marker_scores
                    else float("nan")
                )
                failure_rate = 1.0 - (len(labels_by_seed) / len(method_seeds))

                summary_row = {
                    "dataset_id": args.dataset_id,
                    "sample_id": sample_id,
                    "method_id": method_id,
                    "method_family": "baseline",
                    "preprocessing_id": "log1p_hvg",
                    "param_set_id": f"K{k}",
                    "K": k,
                    "seed_count": len(method_seeds),
                    "stability_ari_median": round(median_ari, 6),
                    "stability_ari_iqr": round(iqr_ari, 6),
                    "spatial_coherence_median": round(median_sp, 6),
                    "spatial_coherence_iqr": round(iqr_sp, 6),
                    "marker_coherence_median": round(median_mk, 6),
                    "marker_coherence_iqr": round(iqr_mk, 6),
                    "wall_time_sec_median": round(float(np.median(run_times)), 6),
                    "peak_rss_mb_median": round(float(np.median(run_mem)), 3),
                    "failure_rate": round(failure_rate, 6),
                    "notes": (
                        f"{args.note};{leiden_note}"
                        if leiden_note
                        else (f"{args.note};{spagcn_note}" if spagcn_note else args.note)
                    ),
                }
                method_rows.append(summary_row)
                fig2_rows.append({"panel_id": "Fig2A", **summary_row})
                fig3_rows.append(
                    {
                        "panel_id": "Fig3A",
                        "dataset_id": args.dataset_id,
                        "sample_id": sample_id,
                        "method_id": method_id,
                        "sensitivity_factor": "K",
                        "factor_value": k,
                        "K": k,
                        "control_type": "none",
                        "metric_name": "spatial_coherence_median",
                        "metric_value": round(median_sp, 6),
                        "metric_ci_lower": "",
                        "metric_ci_upper": "",
                        "notes": args.note,
                    }
                )
                fig4_rows.append(
                    {
                        "panel_id": "Fig4A",
                        "dataset_id": args.dataset_id,
                        "sample_id": sample_id,
                        "method_id": method_id,
                        "K": k,
                        "status": "success",
                        "wall_time_sec": round(float(np.median(run_times)), 6),
                        "peak_rss_mb": round(float(np.median(run_mem)), 3),
                        "is_laptop_feasible": int(np.median(run_mem) < 6000),
                        "is_cloud_required": int(np.median(run_mem) >= 6000),
                        "recommendation": "local-first",
                        "notes": args.note,
                    }
                )
                sensitivity_rows.append(
                    {
                        "dataset_id": args.dataset_id,
                        "sample_id": sample_id,
                        "method_id": method_id,
                        "preprocessing_id": "log1p_hvg",
                        "param_set_id": f"K{k}",
                        "sensitivity_factor": "K",
                        "factor_value": k,
                        "K": k,
                        "metric_name": "spatial_coherence_median",
                        "metric_value": round(median_sp, 6),
                        "metric_ci_lower": "",
                        "metric_ci_upper": "",
                        "notes": args.note,
                    }
                )

    if not failure_rows:
        failure_rows.append(
            {
                "dataset_id": args.dataset_id,
                "sample_id": sample_entries[0]["sample_id"],
                "method_id": "none",
                "preprocessing_id": "log1p_hvg",
                "param_set_id": "NA",
                "K": "",
                "seed": "",
                "failure_type": "none",
                "error_message": "",
                "log_path": "",
                "stacktrace_path": "",
                "wall_time_sec": "",
                "peak_rss_mb": "",
                "finished_utc": utc_now(),
                "notes": f"{args.note} completed without runtime failures",
            }
        )

    append_rows(bench_dir / "method_benchmark.tsv", method_rows)
    append_rows(bench_dir / "stability.tsv", stability_rows)
    append_rows(bench_dir / "spatial_coherence.tsv", spatial_rows)
    append_rows(bench_dir / "marker_coherence.tsv", marker_rows)
    append_rows(bench_dir / "runtime_memory.tsv", runtime_rows)
    append_rows(bench_dir / "failure_log.tsv", failure_rows)
    append_rows(bench_dir / "negative_controls.tsv", control_rows)
    append_rows(bench_dir / "sensitivity_summary.tsv", sensitivity_rows)

    append_rows(fig_dir / "fig1_overview.tsv", fig1_rows)
    append_rows(fig_dir / "fig2_benchmark_summary.tsv", fig2_rows)
    append_rows(fig_dir / "fig3_sensitivity.tsv", fig3_rows)
    append_rows(fig_dir / "fig4_compute_and_guidance.tsv", fig4_rows)

    print(
        f"[done] dataset={args.dataset_id} samples={len(sample_entries)} "
        f"method_rows={len(method_rows)} runtime_rows={len(runtime_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
