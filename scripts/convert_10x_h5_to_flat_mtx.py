#!/usr/bin/env python3
"""Convert 10x Genomics `filtered_feature_bc_matrix.h5` into flat mtx/tsv files.

BayesSpace helper scripts in this repo expect flat files:
  - <sample>_matrix.mtx.gz
  - <sample>_barcodes.tsv.gz
  - <sample>_features.tsv.gz
and an existing tissue-positions file (<sample>_tissue_positions*.csv.gz).

This converter is designed for Visium-style 10x h5 bundles, and is only used
to enable BayesSpace runs on datasets distributed as h5 only.
"""

from __future__ import annotations

import argparse
import gzip
import pathlib

import h5py
import numpy as np
from scipy import sparse
from scipy.io import mmwrite


def _decode_list(arr: np.ndarray) -> list[str]:
    out: list[str] = []
    for v in arr:
        if isinstance(v, (bytes, bytearray)):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return out


def read_10x_h5_matrix(h5_path: pathlib.Path) -> tuple[sparse.csc_matrix, list[str], list[str], list[str]]:
    with h5py.File(h5_path, "r") as handle:
        group = handle["matrix"]
        data = np.array(group["data"])
        indices = np.array(group["indices"])
        indptr = np.array(group["indptr"])
        shape = tuple(np.array(group["shape"]).tolist())
        matrix = sparse.csc_matrix((data, indices, indptr), shape=shape)

        barcodes = _decode_list(np.array(group["barcodes"]))

        feature_group = group.get("features")
        if feature_group is None:
            raise KeyError("Missing /matrix/features in h5")
        feature_ids = _decode_list(np.array(feature_group.get("id"))) if "id" in feature_group else []
        feature_names = _decode_list(np.array(feature_group.get("name"))) if "name" in feature_group else []
        feature_types = _decode_list(np.array(feature_group.get("feature_type"))) if "feature_type" in feature_group else []

        if not feature_names:
            raise KeyError("Missing /matrix/features/name in h5")
        if not feature_ids:
            feature_ids = feature_names
        if not feature_types:
            feature_types = ["Gene Expression"] * len(feature_names)

    return matrix, barcodes, feature_ids, feature_names


def write_gz_lines(path: pathlib.Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def convert_sample(dataset_root: pathlib.Path, sample_id: str) -> None:
    h5_path = dataset_root / f"{sample_id}_filtered_feature_bc_matrix.h5"
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    out_matrix = dataset_root / f"{sample_id}_matrix.mtx.gz"
    out_barcodes = dataset_root / f"{sample_id}_barcodes.tsv.gz"
    out_features = dataset_root / f"{sample_id}_features.tsv.gz"

    if out_matrix.exists() and out_barcodes.exists() and out_features.exists():
        return

    matrix, barcodes, feature_ids, feature_names = read_10x_h5_matrix(h5_path)

    # Write Matrix Market to a temp file first, then gzip to keep output deterministic.
    tmp_mtx = dataset_root / f".{sample_id}_matrix.mtx"
    mmwrite(str(tmp_mtx), matrix, field="integer")
    with tmp_mtx.open("rb") as src, gzip.open(out_matrix, "wb") as dst:
        dst.write(src.read())
    tmp_mtx.unlink(missing_ok=True)

    write_gz_lines(out_barcodes, barcodes)

    # 10x features.tsv typically has 3 columns: id, name, feature_type.
    feature_lines = [f"{fid}\t{fname}\tGene Expression" for fid, fname in zip(feature_ids, feature_names, strict=True)]
    write_gz_lines(out_features, feature_lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, help="Directory containing extracted GEO files")
    ap.add_argument("--sample-id", default="", help="If set, convert only this sample")
    args = ap.parse_args()

    dataset_root = pathlib.Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    if args.sample_id:
        convert_sample(dataset_root, args.sample_id)
        return 0

    h5_paths = sorted(dataset_root.glob("*_filtered_feature_bc_matrix.h5"))
    for h5_path in h5_paths:
        sample_id = h5_path.name.replace("_filtered_feature_bc_matrix.h5", "")
        convert_sample(dataset_root, sample_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

