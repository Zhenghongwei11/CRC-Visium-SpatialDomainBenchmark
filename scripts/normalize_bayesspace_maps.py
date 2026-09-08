#!/usr/bin/env python3
"""Add the standard header to headerless BayesSpace spot-map TSV files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


COLUMNS = [
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


def normalize(path: Path) -> bool:
    first_line = path.open("r", encoding="utf-8").readline().rstrip("\n")
    if first_line.split("\t")[:5] == COLUMNS[:5]:
        return False
    table = pd.read_csv(path, sep="\t", header=None, names=COLUMNS, low_memory=False)
    if table.shape[1] != len(COLUMNS) or table["dataset_id"].isna().any():
        raise ValueError(f"Unexpected BayesSpace map structure: {path}")
    temporary = path.with_name(f".{path.name}.normalized.tmp")
    table.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map-dir",
        default="results/cross_method_boundary/domain_maps",
    )
    args = parser.parse_args()
    paths = sorted(Path(args.map_dir).glob("*_bayes_*.tsv"))
    changed = sum(int(normalize(path)) for path in paths)
    print(f"Normalized {changed} of {len(paths)} BayesSpace map files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
