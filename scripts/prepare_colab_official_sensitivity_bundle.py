#!/usr/bin/env python3
"""Prepare the Colab input bundle for official-method sensitivity runs."""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "tmp" / "colab_official_sensitivity"
ALLOWED_SUFFIXES = (
    "_matrix.mtx.gz",
    "_barcodes.tsv.gz",
    "_features.tsv.gz",
    "_tissue_positions.csv.gz",
    "_tissue_positions_list.csv.gz",
    "_tissue_lowres_image.png.gz",
    "_scalefactors.json.gz",
    "_scalefactors_json.json.gz",
)

SAMPLES = [
    {
        "dataset_id": "GSE267401",
        "sample_id": "GSM8265211_CTC21P",
        "source_root": ROOT / "data" / "raw" / "GSE267401" / "extracted",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE267401",
        "sample_id": "GSM8265212_CTC21M",
        "source_root": ROOT / "data" / "raw" / "GSE267401" / "extracted",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE267401",
        "sample_id": "GSM8265213_CTC17P",
        "source_root": ROOT / "data" / "raw" / "GSE267401" / "extracted",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE267401",
        "sample_id": "GSM8265214_CTC17M",
        "source_root": ROOT / "data" / "raw" / "GSE267401" / "extracted",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE285505",
        "sample_id": "GSM8703563_Tumor19",
        "source_root": ROOT / "data" / "raw" / "GSE285505" / "extracted",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE285505",
        "sample_id": "GSM8703564_Tumor20",
        "source_root": ROOT / "data" / "raw" / "GSE285505" / "extracted",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE285505",
        "sample_id": "GSM8703565_Tumor24",
        "source_root": ROOT / "data" / "raw" / "GSE285505" / "extracted",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE285505",
        "sample_id": "GSM8703566_Tumor26",
        "source_root": ROOT / "data" / "raw" / "GSE285505" / "extracted",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE311294",
        "sample_id": "GSM9322957_TR11_206",
        "source_root": ROOT / "data" / "raw" / "GSE311294" / "extracted",
        "rationale": "reviewer_focal_TR11_206",
    },
    {
        "dataset_id": "GSE311294",
        "sample_id": "GSM9322958_TR11_16184",
        "source_root": ROOT / "data" / "raw" / "GSE311294" / "extracted",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE311294",
        "sample_id": "GSM9322959_TR11_18105",
        "source_root": ROOT / "data" / "raw" / "GSE311294" / "extracted",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE311294",
        "sample_id": "GSM9322960_TR11_21723",
        "source_root": ROOT / "data" / "raw" / "GSE311294" / "extracted",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE311294",
        "sample_id": "GSM9322961_TR16_23542",
        "source_root": ROOT / "data" / "raw" / "GSE311294" / "extracted",
        "rationale": "full_cohort_crc_section",
    },
]


def copy_sample_files(stage: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for sample in SAMPLES:
        dataset_id = sample["dataset_id"]
        sample_id = sample["sample_id"]
        source_root = Path(sample["source_root"])
        target_root = stage / "data" / dataset_id / "extracted"
        target_root.mkdir(parents=True, exist_ok=True)
        paths = [
            path
            for path in sorted(source_root.glob(f"{sample_id}_*"))
            if any(path.name.endswith(suffix) for suffix in ALLOWED_SUFFIXES)
        ]
        if not paths:
            raise FileNotFoundError(f"No files found for {sample_id} under {source_root}")
        for path in paths:
            if path.is_file():
                shutil.copy2(path, target_root / path.name)
        rows.append(
            {
                "dataset_id": str(dataset_id),
                "sample_id": str(sample_id),
                "source_root": str(source_root.relative_to(ROOT)),
                "bundle_root": f"data/{dataset_id}/extracted",
                "rationale": str(sample["rationale"]),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(stage / "sample_manifest.tsv", sep="\t", index=False)
    return manifest


def copy_scripts(stage: Path) -> None:
    script_dir = stage / "repo_scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts" / "run_bayesspace_baseline.R", script_dir / "run_bayesspace_baseline.R")


def build_tar(stage: Path, output_tar: Path) -> None:
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    if output_tar.exists():
        output_tar.unlink()
    with tarfile.open(output_tar, "w:gz") as handle:
        for path in sorted(stage.rglob("*")):
            if path == output_tar:
                continue
            if path.is_file():
                handle.add(path, arcname=path.relative_to(stage))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tar-name", default="jbcb_official_sensitivity_input.tar.gz")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    stage = out_dir / "bundle"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    manifest = copy_sample_files(stage)
    copy_scripts(stage)
    output_tar = out_dir / args.tar_name
    build_tar(stage, output_tar)
    size_mb = output_tar.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_tar} ({size_mb:.1f} MB)")
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
