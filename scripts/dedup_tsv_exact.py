#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def dedup_rows_exact(path_in: Path) -> tuple[list[str], list[list[str]], int]:
    with path_in.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        seen: set[tuple[str, ...]] = set()
        out_rows: list[list[str]] = []
        duplicates = 0
        for row in reader:
            key = tuple(row)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            out_rows.append(row)
    return header, out_rows, duplicates


def write_tsv(path_out: Path, header: list[str], rows: list[list[str]]) -> None:
    path_out.parent.mkdir(parents=True, exist_ok=True)
    with path_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove exact duplicate TSV rows (header preserved).")
    parser.add_argument("input", type=Path, help="Input TSV path.")
    parser.add_argument("--output", type=Path, help="Output TSV path (default: overwrite input).")
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file in-place (ignored if --output is provided).",
    )
    args = parser.parse_args()

    path_in: Path = args.input
    if not path_in.exists():
        raise FileNotFoundError(path_in)

    path_out = args.output if args.output else path_in
    if args.output is None and not args.inplace:
        parser.error("Either pass --output or --inplace.")

    header, rows, duplicates = dedup_rows_exact(path_in)
    write_tsv(path_out, header, rows)
    print(f"[dedup] {path_in} -> {path_out} (removed {duplicates} duplicate rows; kept {len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

