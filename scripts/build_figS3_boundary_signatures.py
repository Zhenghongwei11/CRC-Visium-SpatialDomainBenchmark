#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
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

    # matrix is genes x spots in 10x output
    counts = mmread(matrix_path).tocsr()
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

    # Align coords to barcodes and filter in-tissue
    coords = coords[coords["barcode"].isin(barcodes)].copy()
    coords["barcode"] = pd.Categorical(coords["barcode"], categories=barcodes, ordered=True)
    coords = coords.sort_values("barcode")
    in_tissue = coords["in_tissue"].to_numpy().astype(int) == 1
    if int(in_tissue.sum()) < 50:
        raise ValueError("Too few in-tissue spots")

    # transpose to spots x genes and filter in-tissue
    counts = counts.transpose().tocsr()
    counts = counts[in_tissue, :]
    coords = coords.loc[in_tissue].reset_index(drop=True)

    # Return spots x genes, in-tissue barcodes, and coords df
    in_tissue_barcodes = coords["barcode"].astype(str).tolist()
    genes = features.iloc[:, 1].astype(str).tolist()
    return counts, genes, coords


def _lognorm_sparse(counts_spots_x_genes: sparse.csr_matrix, gene_idx: list[int], scale: float = 1e4) -> np.ndarray:
    sub = counts_spots_x_genes[:, gene_idx]
    libsize = np.asarray(counts_spots_x_genes.sum(axis=1)).ravel().astype(np.float64)
    libsize[libsize == 0] = 1.0
    # dense output: n_spots x n_genes_selected
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


def _label_align_switch_set(case: pd.DataFrame) -> tuple[int, list[int], dict[str, int], set[str]]:
    """
    Compute unstable spots for a case-study run table.

    Unstable = barcode whose aligned label differs from reference label in ANY non-reference seed.
    Alignment = maximum-overlap label permutation to reference seed.
    """
    seeds = sorted({int(s) for s in case["seed"].astype(int).unique().tolist()})
    if not seeds:
        raise ValueError("No seeds found in case study table")
    ref_seed = int(seeds[0])
    ref = case[case["seed"].astype(int) == ref_seed].copy()
    if ref.empty:
        raise ValueError("Reference seed rows missing")

    ref_labels = dict(zip(ref["barcode"].astype(str), ref["domain_label"].astype(int), strict=True))
    label_set = sorted({int(x) for x in case["domain_label"].astype(int).unique().tolist()})
    K = int(case["K"].astype(int).iloc[0])
    if K and len(label_set) != K:
        label_set = list(range(1, K + 1))

    unstable: set[str] = set()
    for seed in seeds:
        if int(seed) == ref_seed:
            continue
        sub = case[case["seed"].astype(int) == int(seed)].copy()
        # contingency: current_label -> ref_label -> count
        contingency: dict[int, dict[int, int]] = {}
        by_bc: dict[str, int] = {}
        for bc, cur_lab in zip(sub["barcode"].astype(str), sub["domain_label"].astype(int), strict=True):
            if bc not in ref_labels:
                continue
            cur = int(cur_lab)
            by_bc[bc] = cur
            contingency.setdefault(cur, {})
            contingency[cur][ref_labels[bc]] = contingency[cur].get(ref_labels[bc], 0) + 1

        best_mapping: dict[int, int] | None = None
        best_score = -1
        for perm in itertools.permutations(label_set):
            mapping = dict(zip(label_set, perm, strict=True))
            score = 0
            for cur in label_set:
                mapped = mapping[cur]
                score += contingency.get(cur, {}).get(mapped, 0)
            if score > best_score:
                best_score = score
                best_mapping = mapping

        if best_mapping is None:
            continue

        for bc, cur in by_bc.items():
            aligned = best_mapping.get(cur, cur)
            if int(aligned) != int(ref_labels[bc]):
                unstable.add(bc)

    return K, seeds, ref_labels, unstable


@dataclass(frozen=True)
class Signature:
    signature_id: str
    label: str
    genes: list[str]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-tsv",
        default="results/figures/figS3_instability_case_study.tsv",
        help="Case-study TSV (spot-level; includes barcode, seed, domain_label).",
    )
    parser.add_argument("--dataset-root", required=True, help="Path to data/raw/<GSE>/extracted")
    parser.add_argument("--output-tsv", default="results/figures/figS3_boundary_signatures.tsv")
    parser.add_argument("--note", default="figS3-boundary-signatures")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260421)
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

    K, seeds, ref_labels, unstable = _label_align_switch_set(case)
    all_barcodes = set(ref_labels.keys())
    stable = all_barcodes - unstable

    dataset_root = Path(args.dataset_root)
    counts, gene_names, coords = _load_flat_visium_sample(dataset_root, sample_id)
    bc_to_idx = {bc: i for i, bc in enumerate(coords["barcode"].astype(str).tolist())}

    # Ensure we can map all stable/unstable barcodes to the matrix
    missing = [bc for bc in all_barcodes if bc not in bc_to_idx]
    if missing:
        raise ValueError(f"Missing {len(missing)} barcodes in extracted matrix/coords (example={missing[0]})")

    stable_idx = np.array([bc_to_idx[bc] for bc in sorted(stable)], dtype=int)
    unstable_idx = np.array([bc_to_idx[bc] for bc in sorted(unstable)], dtype=int)

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

    gene_targets = ["FAP", "SPP1"] + sorted({g for sig in signatures for g in sig.genes})
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

    # Compute per-gene lognorm vectors
    gene_expr: dict[str, np.ndarray] = {}
    missing_genes: list[str] = []
    for g in gene_targets:
        vec = get_gene_lognorm(g)
        if vec is None:
            missing_genes.append(g)
            continue
        gene_expr[g] = vec.astype(np.float64, copy=False)

    rows: list[dict[str, object]] = []

    def add_feature(feature_type: str, feature_id: str, label: str, values: np.ndarray, genes_used: str) -> None:
        a = values[stable_idx]
        b = values[unstable_idx]
        # Mann–Whitney U (two-sided) as a simple, robust group comparison
        p = float(mannwhitneyu(b, a, alternative="two-sided").pvalue)
        median_delta = float(np.median(b) - np.median(a))
        ci_lo, ci_hi = _bootstrap_median_diff(a, b, seed=int(args.bootstrap_seed), n_boot=int(args.bootstrap))
        mean_stable = float(np.mean(a))
        mean_unstable = float(np.mean(b))
        mean_delta = float(mean_unstable - mean_stable)
        frac_stable = float(np.mean(a > 0))
        frac_unstable = float(np.mean(b > 0))
        frac_delta = float(frac_unstable - frac_stable)
        rows.append(
            {
                "dataset_id": dataset_id,
                "sample_id": sample_id,
                "case_method_id": method_id,
                "case_K": int(K),
                "case_seeds": ",".join(str(s) for s in seeds),
                "reference_seed": int(seeds[0]),
                "n_spots_total": int(len(all_barcodes)),
                "n_stable_spots": int(len(stable)),
                "n_unstable_spots": int(len(unstable)),
                "unstable_definition": "label differs vs reference in any non-reference seed after maximum-overlap alignment",
                "feature_type": feature_type,
                "feature_id": feature_id,
                "feature_label": label,
                "genes_used": genes_used,
                "mean_stable": mean_stable,
                "mean_unstable": mean_unstable,
                "mean_delta_unstable_minus_stable": mean_delta,
                "fraction_nonzero_stable": frac_stable,
                "fraction_nonzero_unstable": frac_unstable,
                "fraction_nonzero_delta_unstable_minus_stable": frac_delta,
                "median_stable": float(np.median(a)),
                "median_unstable": float(np.median(b)),
                "median_delta_unstable_minus_stable": median_delta,
                "median_delta_ci_lower": ci_lo,
                "median_delta_ci_upper": ci_hi,
                "pvalue_mannwhitney_two_sided": p,
                "fdr_bh": "",
                "notes": str(args.note),
            }
        )

    # Gene-level targets (include a few interpretable anchors beyond FAP/SPP1)
    for g in ["FAP", "SPP1", "EPCAM", "COL1A1", "PTPRC"]:
        if g not in gene_expr:
            continue
        add_feature("gene", g, f"{g} (log-normalized expression)", gene_expr[g], g)

    # Signatures: mean of available genes
    for sig in signatures:
        used = [g for g in sig.genes if g in gene_expr]
        if not used:
            continue
        mat = np.vstack([gene_expr[g] for g in used]).T  # n_spots x n_genes
        score = np.mean(mat, axis=1)
        add_feature("signature", sig.signature_id, f"{sig.label} (mean log-normalized expression)", score, ",".join(used))

    # Immune-exclusion contrast: (CAF + SPP1-myeloid) - T-cell
    sig_map = {s.signature_id: s for s in signatures}
    caf_used = [g for g in sig_map["SIG_CAF_FAP"].genes if g in gene_expr]
    my_used = [g for g in sig_map["SIG_MY_SPP1"].genes if g in gene_expr]
    t_used = [g for g in sig_map["SIG_TCELL"].genes if g in gene_expr]
    if caf_used and my_used and t_used:
        caf = np.mean(np.vstack([gene_expr[g] for g in caf_used]).T, axis=1)
        my = np.mean(np.vstack([gene_expr[g] for g in my_used]).T, axis=1)
        tc = np.mean(np.vstack([gene_expr[g] for g in t_used]).T, axis=1)
        contrast = (caf + my) - tc
        add_feature(
            "signature",
            "SIG_EXCLUSION_CONTRAST",
            "CAF+SPP1-myeloid minus T-cell (contrast score)",
            contrast,
            f"CAF:{','.join(caf_used)};MY:{','.join(my_used)};TCELL:{','.join(t_used)}",
        )

    # BH-FDR across reported features
    pvals = [float(r["pvalue_mannwhitney_two_sided"]) for r in rows]
    fdrs = _bh_fdr(pvals)
    for r, f in zip(rows, fdrs, strict=True):
        r["fdr_bh"] = float(f)

    out_path = Path(args.output_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, sep="\t", index=False)

    if missing_genes:
        print(f"[warn] Missing genes (not found in features): {','.join(sorted(set(missing_genes)))}")
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
