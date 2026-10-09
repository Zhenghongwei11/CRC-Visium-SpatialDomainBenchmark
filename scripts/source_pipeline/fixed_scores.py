#!/usr/bin/env python3
"""Compute the four-gene mean from the full original 10x count matrix."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from scientific_functions import SCORE_DEFINITIONS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_counts(root: Path, sample_id: str) -> tuple[sparse.csr_matrix, list[str], list[str], list[Path]]:
    h5 = root / f"{sample_id}_filtered_feature_bc_matrix.h5"
    matrix = root / f"{sample_id}_matrix.mtx.gz"
    barcodes = root / f"{sample_id}_barcodes.tsv.gz"
    features = root / f"{sample_id}_features.tsv.gz"
    if h5.is_file():
        # Read the standard 10x HDF5 layout directly. This keeps the
        # downstream score reconstruction independent of scanpy installation.
        import h5py

        def decode(values: object) -> list[str]:
            return [value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    for value in values]

        with h5py.File(h5, "r") as handle:
            matrix = handle["matrix"]
            shape = tuple(int(value) for value in matrix["shape"][...])
            data = matrix["data"][...]
            indices = matrix["indices"][...]
            indptr = matrix["indptr"][...]
            # 10x stores genes x barcodes as CSC; score code expects cells x genes.
            counts = sparse.csc_matrix((data, indices, indptr), shape=shape).transpose().tocsr()
            names = decode(matrix["barcodes"][...])
            feature_group = matrix["features"]
            gene_values = feature_group["name"][...] if "name" in feature_group else feature_group["id"][...]
            genes = decode(gene_values)
        # Match AnnData.var_names_make_unique for duplicate feature names.
        used = set(genes)
        seen: dict[str, int] = {}
        unique_genes: list[str] = []
        for gene in genes:
            count = seen.get(gene, 0)
            if count == 0:
                unique_genes.append(gene)
            else:
                suffix = count
                while f"{gene}-{suffix}" in used:
                    suffix += 1
                name = f"{gene}-{suffix}"
                unique_genes.append(name)
                used.add(name)
            seen[gene] = count + 1
        return counts, names, unique_genes, [h5]
    if all(path.is_file() for path in (matrix, barcodes, features)):
        counts = mmread(matrix).tocsr().transpose().tocsr()
        with gzip.open(barcodes, "rt", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip()]
        feature_table = pd.read_csv(features, sep="\t", header=None, compression="infer")
        genes = feature_table.iloc[:, 1 if feature_table.shape[1] >= 2 else 0].astype(str).tolist()
        if counts.shape != (len(names), len(genes)):
            raise ValueError(f"{sample_id}: matrix/metadata dimensions disagree")
        # Match AnnData.var_names_make_unique: the first duplicate retains its
        # plain name, and subsequent copies receive a numerical suffix.
        used = set(genes)
        seen: dict[str, int] = {}
        unique_genes: list[str] = []
        for gene in genes:
            count = seen.get(gene, 0)
            if count == 0:
                unique_genes.append(gene)
            else:
                suffix = count
                while f"{gene}-{suffix}" in used:
                    suffix += 1
                name = f"{gene}-{suffix}"
                unique_genes.append(name)
                used.add(name)
            seen[gene] = count + 1
        return counts, names, unique_genes, [matrix, barcodes, features]
    raise FileNotFoundError(f"{sample_id}: original H5 or Matrix Market trio not found under {root}")


def fixed_scores(
    counts: sparse.csr_matrix,
    barcodes: list[str],
    genes: list[str],
    dataset_id: str,
    sample_id: str,
) -> pd.DataFrame:
    """Apply original per-spot 10k normalization, log1p and fixed gene means."""
    if counts.shape != (len(barcodes), len(genes)):
        raise ValueError("counts/barcodes/genes dimensions disagree")
    if len(set(barcodes)) != len(barcodes) or len(set(genes)) != len(genes):
        raise ValueError("duplicate barcode or non-unique feature name")
    if not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError("nonfinite or negative count")
    totals = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    scale = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)
    normalized = counts.astype(float).multiply(scale[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    index = {gene: position for position, gene in enumerate(genes)}
    required = sorted({gene for definition in SCORE_DEFINITIONS.values() for gene in definition["genes"]})
    gene_values = {gene: normalized[:, index[gene]].toarray().ravel() for gene in required if gene in index}
    if not set(SCORE_DEFINITIONS["primary_barrier"]["genes"]).issubset(gene_values):
        missing = sorted(set(SCORE_DEFINITIONS["primary_barrier"]["genes"]) - set(gene_values))
        raise ValueError(f"{sample_id}: primary genes absent: {missing}")

    values = np.mean(np.vstack([gene_values[gene] for gene in SCORE_DEFINITIONS["primary_barrier"]["genes"]]), axis=0).astype(float)
    return pd.DataFrame({"dataset_id": dataset_id, "sample_id": sample_id, "barcode": barcodes,
                         "score_id": "primary_barrier", "score_value": values,
                         "status": "success", "missing_genes": ""})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--expected-package-sha256", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--expected-barcodes", type=Path, required=True,
                        help="Canonical registered in-tissue spots TSV with a barcode column")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.source_package.is_file():
        raise FileNotFoundError(args.source_package)
    actual_sha = sha256_file(args.source_package)
    if actual_sha != args.expected_package_sha256.lower():
        raise ValueError("source package SHA-256 mismatch")
    counts, barcodes, genes, members = read_counts(args.source_root, args.sample_id)
    scores = fixed_scores(counts, barcodes, genes, args.dataset_id, args.sample_id)
    expected_table = pd.read_csv(args.expected_barcodes, sep="\t", dtype={"barcode": str})
    if "barcode" not in expected_table or expected_table["barcode"].isna().any():
        raise ValueError("expected-barcodes file needs nonempty barcode column")
    expected = expected_table["barcode"].astype(str).tolist()
    if len(set(expected)) != len(expected) or not expected:
        raise ValueError("expected-barcodes file has duplicate or zero barcodes")
    if not set(expected).issubset(barcodes):
        raise ValueError(f"{args.sample_id}: expected barcode absent from original expression package")
    scores = scores.set_index("barcode").loc[expected].reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "spot_scores.tsv.gz"
    scores.to_csv(score_path, sep="\t", index=False, compression="gzip")
    manifest = {
        "schema_version": "r4_fixed_scores_v1",
        "dataset_id": args.dataset_id,
        "sample_id": args.sample_id,
        "source_package_sha256": actual_sha,
        "source_files": [{"name": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in members],
        "score_script_sha256": sha256_file(Path(__file__)),
        "score_definition_script_sha256": sha256_file(Path(__file__).with_name("scientific_functions.py")),
        "normalization": "normalize_total(target_sum=1e4);log1p;arithmetic_mean",
        "n_matrix_barcodes": len(barcodes),
        "n_registered_in_tissue_barcodes": len(expected),
        "expected_barcodes_sha256": sha256_file(args.expected_barcodes),
        "n_features": len(genes),
        "score_sha256": sha256_file(score_path),
        "score_size_bytes": score_path.stat().st_size,
    }
    (args.output_dir / "score_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
