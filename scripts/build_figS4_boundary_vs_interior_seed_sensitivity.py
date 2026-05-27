#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu


def _read_lines(path: Path) -> list[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [line.strip() for line in handle]
    return path.read_text(encoding="utf-8").splitlines()


def _load_flat_visium_sample(dataset_root: Path, sample_id: str) -> tuple[sparse.csr_matrix, list[str], pd.DataFrame]:
    matrix_path = dataset_root / f"{sample_id}_matrix.mtx.gz"
    barcodes_path = dataset_root / f"{sample_id}_barcodes.tsv.gz"
    features_path = dataset_root / f"{sample_id}_features.tsv.gz"
    coords_path = dataset_root / f"{sample_id}_tissue_positions.csv.gz"
    if not coords_path.exists():
        coords_path = dataset_root / f"{sample_id}_tissue_positions_list.csv.gz"

    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)
    if not barcodes_path.exists():
        raise FileNotFoundError(barcodes_path)
    if not features_path.exists():
        raise FileNotFoundError(features_path)
    if not coords_path.exists():
        raise FileNotFoundError(coords_path)

    counts = mmread(matrix_path).tocsr()  # genes x spots
    barcodes = _read_lines(barcodes_path)
    features = pd.read_csv(features_path, header=None, sep="\t", compression="infer")
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
    if counts.shape[0] != features.shape[0]:
        raise ValueError(f"Feature mismatch: matrix genes={counts.shape[0]} features={features.shape[0]}")

    coords = coords[coords["barcode"].isin(barcodes)].copy()
    coords["barcode"] = pd.Categorical(coords["barcode"], categories=barcodes, ordered=True)
    coords = coords.sort_values("barcode")
    in_tissue = coords["in_tissue"].to_numpy().astype(int) == 1
    if int(in_tissue.sum()) < 50:
        raise ValueError("Too few in-tissue spots")

    counts = counts.transpose().tocsr()  # spots x genes
    counts = counts[in_tissue, :]
    coords = coords.loc[in_tissue].reset_index(drop=True)

    genes = features.iloc[:, 1].astype(str).tolist()
    return counts, genes, coords


def _lognorm_sparse(counts_spots_x_genes: sparse.csr_matrix, gene_idx: list[int], scale: float = 1e4) -> np.ndarray:
    sub = counts_spots_x_genes[:, gene_idx]
    libsize = np.asarray(counts_spots_x_genes.sum(axis=1)).ravel().astype(np.float64)
    libsize[libsize == 0] = 1.0
    dense = sub.toarray().astype(np.float64, copy=False)
    dense = dense / libsize[:, None] * float(scale)
    return np.log1p(dense)


def _bh_fdr(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.empty(n, dtype=np.float64)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        idx = int(order[i])
        rank = i + 1
        val = min(prev, pvals[idx] * n / rank)
        prev = val
        ranked[idx] = val
    return ranked.tolist()


def _bootstrap_median_diff(a: np.ndarray, b: np.ndarray, *, seed: int, n_boot: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    n_a = a.shape[0]
    n_b = b.shape[0]
    diffs = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        aa = rng.choice(a, size=n_a, replace=True)
        bb = rng.choice(b, size=n_b, replace=True)
        diffs[i] = float(np.median(bb) - np.median(aa))
    lo = float(np.quantile(diffs, 0.025))
    hi = float(np.quantile(diffs, 0.975))
    return lo, hi


@dataclass(frozen=True)
class Signature:
    signature_id: str
    label: str
    genes: list[str]


def _boundary_mask(coords_xy: np.ndarray, labels: np.ndarray, *, k: int = 6) -> np.ndarray:
    if coords_xy.ndim != 2 or coords_xy.shape[1] != 2:
        raise ValueError("coords_xy must be n_spots x 2")
    if coords_xy.shape[0] != labels.shape[0]:
        raise ValueError("coords/labels length mismatch")
    if int(k) < 1:
        raise ValueError("k must be >= 1")

    # Query k+1 neighbors to exclude self (distance 0).
    tree = cKDTree(coords_xy.astype(np.float64, copy=False))
    _, nn = tree.query(coords_xy, k=int(k) + 1)
    nn = nn[:, 1:]  # drop self
    neigh_labels = labels[nn]
    boundary = np.any(neigh_labels != labels[:, None], axis=1)
    return boundary.astype(bool)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-tsv",
        default="results/figures/figS3_instability_case_study.tsv",
        help="Case-study TSV (spot-level; includes barcode, seed, domain_label, x, y).",
    )
    parser.add_argument("--dataset-root", required=True, help="Path to data/raw/<GSE>/extracted")
    parser.add_argument("--output-tsv", default="results/figures/figS4_boundary_vs_interior_seed_sensitivity.tsv")
    parser.add_argument("--neighbor-k", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260422)
    parser.add_argument("--note", default="figS4-boundary-vs-interior-seed-sensitivity")
    args = parser.parse_args()

    case_path = Path(args.case_tsv)
    if not case_path.exists():
        raise FileNotFoundError(case_path)

    case = pd.read_csv(case_path, sep="\t")
    case = case[case["notes"].astype(str) == "figS3-bayesspace-instability-case-study"].copy()
    if case.empty:
        raise ValueError("No case-study rows found (notes filter)")

    dataset_id = str(case["dataset_id"].iloc[0])
    sample_id = str(case["sample_id"].iloc[0])
    method_id = str(case["method_id"].iloc[0])
    if method_id != "BayesSpace":
        raise ValueError(f"Expected BayesSpace case study, got method_id={method_id}")

    seeds = sorted({int(s) for s in case["seed"].astype(int).unique().tolist()})
    if len(seeds) < 2:
        raise ValueError("Need at least 2 seeds for sensitivity analysis")
    k_val = int(case["K"].astype(int).iloc[0])

    # Use a consistent coordinate system across seeds (coords are identical per spot).
    ref_seed = int(seeds[0])
    ref = case[case["seed"].astype(int) == ref_seed].copy()
    ref = ref.sort_values("barcode").reset_index(drop=True)
    barcodes = ref["barcode"].astype(str).tolist()
    coords_xy = ref[["x", "y"]].to_numpy().astype(np.float64, copy=False)

    dataset_root = Path(args.dataset_root)
    counts, gene_names, coords = _load_flat_visium_sample(dataset_root, sample_id)
    bc_to_idx = {bc: i for i, bc in enumerate(coords["barcode"].astype(str).tolist())}
    missing = [bc for bc in barcodes if bc not in bc_to_idx]
    if missing:
        raise ValueError(f"Missing {len(missing)} barcodes in extracted matrix/coords (example={missing[0]})")

    spot_idx = np.array([bc_to_idx[bc] for bc in barcodes], dtype=int)

    signatures = [
        Signature("SIG_CAF_FAP", "CAF / FAP-associated (fibroblast activation)", ["FAP", "COL1A1", "DCN", "COL3A1", "LUM"]),
        Signature("SIG_MY_SPP1", "SPP1-myeloid–associated", ["SPP1", "LST1", "TYROBP", "C1QA", "C1QB", "C1QC"]),
        Signature("SIG_TCELL", "T-cell–associated", ["TRAC", "CD3D", "CD3E", "CD247"]),
        Signature(
            "SIG_EXCLUSION_TGFB_CXCL12",
            "Immune-exclusion–adjacent (TGFβ / CXCL12 / myofibroblast barrier; exploratory)",
            ["TGFB1", "CXCL12", "ACTA2", "TAGLN", "FAP", "COL1A1"],
        ),
        Signature(
            "SIG_CYTOTOX",
            "Cytotoxic lymphocyte–associated (exploratory)",
            ["NKG7", "GNLY", "GZMB", "PRF1", "IFNG"],
        ),
    ]

    # Include interpretable gene anchors plus signature genes (deduplicated).
    gene_targets = ["FAP", "SPP1", "EPCAM", "COL1A1", "PTPRC"] + sorted({g for sig in signatures for g in sig.genes})
    gene_to_indices: dict[str, list[int]] = {}
    for i, g in enumerate(gene_names):
        gene_to_indices.setdefault(g, []).append(i)

    def get_gene_lognorm(gene: str) -> np.ndarray | None:
        idxs = gene_to_indices.get(gene)
        if not idxs:
            return None
        mat = _lognorm_sparse(counts, idxs)  # n_spots x n_dups
        if mat.shape[1] == 1:
            return mat[:, 0]
        return mat.sum(axis=1)

    gene_expr: dict[str, np.ndarray] = {}
    for g in gene_targets:
        vec = get_gene_lognorm(g)
        if vec is None:
            continue
        gene_expr[g] = vec.astype(np.float64, copy=False)

    # Precompute per-feature vectors for the barcodes used in the case study.
    features: list[tuple[str, str, str, np.ndarray, str]] = []
    for g in ["FAP", "SPP1", "EPCAM", "COL1A1", "PTPRC"]:
        if g in gene_expr:
            v = gene_expr[g][spot_idx]
            features.append(("gene", g, f"{g} (log-normalized expression)", v, g))

    for sig in signatures:
        used = [g for g in sig.genes if g in gene_expr]
        if not used:
            continue
        mat = np.vstack([gene_expr[g][spot_idx] for g in used]).T
        score = np.mean(mat, axis=1)
        features.append(("signature", sig.signature_id, f"{sig.label} (mean log-normalized expression)", score, ",".join(used)))

    # Immune-exclusion contrast: (CAF + SPP1-myeloid) - T-cell
    sig_map = {s.signature_id: s for s in signatures}
    caf_used = [g for g in sig_map["SIG_CAF_FAP"].genes if g in gene_expr]
    my_used = [g for g in sig_map["SIG_MY_SPP1"].genes if g in gene_expr]
    t_used = [g for g in sig_map["SIG_TCELL"].genes if g in gene_expr]
    if caf_used and my_used and t_used:
        caf = np.mean(np.vstack([gene_expr[g][spot_idx] for g in caf_used]).T, axis=1)
        my = np.mean(np.vstack([gene_expr[g][spot_idx] for g in my_used]).T, axis=1)
        tc = np.mean(np.vstack([gene_expr[g][spot_idx] for g in t_used]).T, axis=1)
        contrast = (caf + my) - tc
        features.append(
            (
                "signature",
                "SIG_EXCLUSION_CONTRAST",
                "CAF+SPP1-myeloid minus T-cell (contrast score)",
                contrast,
                f"CAF:{','.join(caf_used)};MY:{','.join(my_used)};TCELL:{','.join(t_used)}",
            )
        )

    rows: list[dict[str, object]] = []
    for seed in seeds:
        sub = case[case["seed"].astype(int) == int(seed)].copy()
        sub = sub.sort_values("barcode").reset_index(drop=True)
        if sub.shape[0] != len(barcodes):
            raise ValueError(f"Seed {seed}: expected {len(barcodes)} spots, got {sub.shape[0]}")
        if sub["barcode"].astype(str).tolist() != barcodes:
            raise ValueError(f"Seed {seed}: barcode ordering mismatch vs reference seed")

        labels = sub["domain_label"].astype(int).to_numpy()
        boundary = _boundary_mask(coords_xy, labels, k=int(args.neighbor_k))
        interior = ~boundary
        if int(boundary.sum()) < 10 or int(interior.sum()) < 10:
            raise ValueError(f"Seed {seed}: too few boundary/interior spots (boundary={boundary.sum()} interior={interior.sum()})")

        for feature_type, feature_id, feature_label, values, genes_used in features:
            a = values[interior]
            b = values[boundary]
            p = float(mannwhitneyu(b, a, alternative="two-sided").pvalue)
            median_delta = float(np.median(b) - np.median(a))
            ci_lo, ci_hi = _bootstrap_median_diff(a, b, seed=int(args.bootstrap_seed) + int(seed), n_boot=int(args.bootstrap))
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "sample_id": sample_id,
                    "method_id": method_id,
                    "K": int(k_val),
                    "seed": int(seed),
                    "neighbor_k": int(args.neighbor_k),
                    "n_spots_total": int(len(values)),
                    "n_boundary_spots": int(boundary.sum()),
                    "n_interior_spots": int(interior.sum()),
                    "boundary_definition": "boundary = has ≥1 label-discordant neighbor in 6-NN spatial graph",
                    "feature_type": feature_type,
                    "feature_id": feature_id,
                    "feature_label": feature_label,
                    "genes_used": genes_used,
                    "mean_boundary": float(np.mean(b)),
                    "mean_interior": float(np.mean(a)),
                    "mean_delta_boundary_minus_interior": float(np.mean(b) - np.mean(a)),
                    "median_boundary": float(np.median(b)),
                    "median_interior": float(np.median(a)),
                    "median_delta_boundary_minus_interior": median_delta,
                    "median_delta_ci_lower": ci_lo,
                    "median_delta_ci_upper": ci_hi,
                    "pvalue_mannwhitney_two_sided": p,
                    "fdr_bh_within_seed": "",
                    "notes": str(args.note),
                }
            )

    out_df = pd.DataFrame(rows)
    out_df["fdr_bh_within_seed"] = np.nan
    for seed in seeds:
        mask = out_df["seed"].astype(int) == int(seed)
        pvals = [float(x) for x in out_df.loc[mask, "pvalue_mannwhitney_two_sided"].tolist()]
        out_df.loc[mask, "fdr_bh_within_seed"] = _bh_fdr(pvals)

    out_path = Path(args.output_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_df.shape[0]} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
