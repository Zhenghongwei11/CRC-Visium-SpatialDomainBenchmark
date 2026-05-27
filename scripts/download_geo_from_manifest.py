#!/usr/bin/env python3
"""Download and unpack GEO RAW tar files declared in a public data manifest."""

from __future__ import annotations

import argparse
import csv
import pathlib
import shutil
import subprocess
import tarfile
import urllib.request


def read_manifest_row(manifest_path: pathlib.Path, dataset_id: str) -> dict[str, str]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("dataset_id") == dataset_id:
                return row
    raise ValueError(f"Dataset {dataset_id!r} not found in {manifest_path}")


def download_file(url: str, destination: pathlib.Path, expected_size: int | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    curl = shutil.which("curl")
    if curl:
        subprocess.run(
            [
                curl,
                "-L",
                "--fail",
                "--silent",
                "--show-error",
                "--retry",
                "8",
                "--retry-delay",
                "5",
                "--retry-all-errors",
                "-C",
                "-",
                "-o",
                str(destination),
                url,
            ],
            check=True,
        )
    else:
        with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    size = destination.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(
            f"Downloaded size mismatch for {destination}: expected {expected_size}, got {size}"
        )


def unpack_tar(tar_path: pathlib.Path, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as archive:
        archive.extractall(output_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="docs/DATA_MANIFEST.tsv")
    parser.add_argument("--dataset-id", default="GSE280318")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-unpack", action="store_true")
    args = parser.parse_args()

    manifest_path = pathlib.Path(args.manifest)
    if not manifest_path.exists() and str(args.manifest) == "docs/DATA_MANIFEST.tsv":
        # Backward-compatible fallback for legacy working trees.
        manifest_path = pathlib.Path("data/manifest.tsv")
    output_root = pathlib.Path(args.output_dir)
    row = read_manifest_row(manifest_path, args.dataset_id)

    raw_url = row["raw_bundle_url"]
    raw_name = row["raw_bundle_filename"]
    expected_size_text = (row.get("raw_bundle_content_length_bytes") or "").strip()
    expected_size = int(expected_size_text) if expected_size_text else None

    dataset_dir = output_root / args.dataset_id
    tar_path = dataset_dir / raw_name
    needs_download = args.force_download or not tar_path.exists()
    if tar_path.exists() and expected_size is not None and tar_path.stat().st_size != expected_size:
        needs_download = True

    if needs_download:
        print(f"[download] {raw_url}")
        download_file(raw_url, tar_path, expected_size)
    else:
        print(f"[skip] file exists: {tar_path}")

    if not args.skip_unpack:
        extract_dir = dataset_dir / "extracted"
        print(f"[unpack] {tar_path} -> {extract_dir}")
        unpack_tar(tar_path, extract_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
