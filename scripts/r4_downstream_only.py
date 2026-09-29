#!/usr/bin/env python3
"""Run the corrected R4 interface readout from fixed scores and saved maps.

This entry point never downloads data, computes PCA, or fits a spatial map.
Input preparation and Drive readback are separate required gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import colab_interface_validation_full as interface
from r4_downstream_input_contract import SectionIdentity, validate_section


ESTIMATORS = ("strict_computational_interface", "anchor_retention")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_tsv(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    frame.to_csv(path, sep="\t", index=False, compression="gzip" if path.suffix == ".gz" else None)
    return {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "rows": len(frame)}


def run_section(
    identity: SectionIdentity,
    section_id: str,
    spots: pd.DataFrame,
    scores: pd.DataFrame,
    maps: pd.DataFrame,
    score_manifest: dict[str, object],
    input_hashes: dict[str, str],
    output_dir: Path,
) -> dict[str, object]:
    report = validate_section(identity, spots, scores, maps)
    if set(scores["score_id"].astype(str)) != set(interface.SCORE_DEFINITIONS):
        raise ValueError("R4 run needs every locked primary and secondary score/status")
    if score_manifest.get("dataset_id") != identity.dataset_id or score_manifest.get("sample_id") != identity.sample_id:
        raise ValueError("score manifest identity mismatch")
    if not score_manifest.get("source_package_sha256"):
        raise ValueError("score manifest lacks verified source package SHA-256")
    if not section_id.strip():
        raise ValueError("section_id is empty")
    if output_dir.exists():
        raise FileExistsError(f"versioned output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    interface.DATASET_ID = identity.dataset_id
    barcodes = spots["barcode"].astype(str).to_numpy()
    score_values: dict[str, np.ndarray | None] = {}
    score_status: dict[str, dict[str, str]] = {}
    for score_id, group in scores.groupby("score_id", sort=True):
        aligned = group.assign(barcode=group["barcode"].astype(str)).set_index("barcode").loc[barcodes]
        status = str(aligned["status"].iloc[0])
        score_values[str(score_id)] = aligned["score_value"].to_numpy(dtype=float) if status == "success" else None
        score_status[str(score_id)] = {
            "status": status,
            "missing_genes": str(aligned["missing_genes"].iloc[0]) if status != "success" else "",
        }
    coords = spots[["x_fullres", "y_fullres"]].to_numpy(dtype=float)
    array_coords = spots[["array_row", "array_col"]].to_numpy(dtype=float)
    diameter = float(spots["spot_diameter_fullres"].iloc[0])
    sample = {
        "sample_id": identity.sample_id,
        "patient_id": identity.patient_id,
        "section_id": section_id,
        "barcodes": barcodes,
        "coords": coords,
        "array_coords": array_coords,
        "raw_labels": spots["raw_label"].astype(str).to_numpy(),
        "coarse_labels": spots["coarse_label"].astype(str).to_numpy(),
        "microns_per_pixel": interface.NOMINAL_SPOT_DIAMETER_UM / diameter,
        "spot_diameter_fullres": diameter,
        "score_values": score_values,
        "score_status": score_status,
    }
    successful_maps = maps[maps["status"].astype(str).eq("success")].copy()
    failed_maps = maps[~maps["status"].astype(str).eq("success")].copy()
    log_rows: list[dict[str, object]] = []
    effect_parts: list[pd.DataFrame] = []
    mapping_parts: list[pd.DataFrame] = []
    membership_parts: list[pd.DataFrame] = []
    anchor_parts: list[pd.DataFrame] = []
    for score_id in interface.SCORE_DEFINITIONS:
        for estimator_id in ESTIMATORS:
            effects, mapping, membership, anchors = interface.compute_effects(
                {identity.sample_id: sample}, successful_maps, log_rows, score_id, estimator_id
            )
            if not effects.empty:
                effect_parts.append(effects)
            # Domain choices and near/far memberships depend on labels, maps,
            # and estimator, not on the expression score. Retain one exact
            # copy per estimator instead of six identical spot-level copies.
            if score_id == "primary_barrier" and not mapping.empty:
                mapping_parts.append(mapping)
            if score_id == "primary_barrier" and not membership.empty:
                membership.insert(0, "score_id", score_id)
                membership_parts.append(membership)
            if not anchors.empty:
                anchors.insert(1, "estimator_id", estimator_id)
                anchor_parts.append(anchors)
    effects = pd.concat(effect_parts, ignore_index=True) if effect_parts else pd.DataFrame()
    mapping = pd.concat(mapping_parts, ignore_index=True).drop_duplicates() if mapping_parts else pd.DataFrame()
    membership = pd.concat(membership_parts, ignore_index=True) if membership_parts else pd.DataFrame()
    anchors = pd.concat(anchor_parts, ignore_index=True) if anchor_parts else pd.DataFrame()
    patient = interface.patient_summary(effects)
    summary = interface.analysis_summary(effects)
    trace = spots.copy()
    trace["spatial_block"] = interface.block_ids(array_coords)
    artifacts = []
    for name, frame in (
        ("spot_inputs.tsv.gz", trace),
        ("spot_scores.tsv.gz", scores),
        ("input_failed_maps.tsv.gz", failed_maps),
        ("section_effects.tsv", effects),
        ("domain_mapping.tsv.gz", mapping),
        ("spot_membership.tsv.gz", membership),
        ("anchor_effects.tsv", anchors),
        ("patient_effects.tsv", patient),
        ("analysis_summary.tsv", summary),
        ("validation_log.tsv", pd.DataFrame(log_rows)),
    ):
        artifacts.append(_write_tsv(frame, output_dir / name))
    manifest = {
        "schema_version": "r4_downstream_result_v1",
        "run_type": "post_outcome_corrected_downstream_only",
        "identity": report,
        "section_id": section_id,
        "score_source_package_sha256": score_manifest["source_package_sha256"],
        "score_manifest": score_manifest,
        "input_sha256": input_hashes,
        "code_sha256": {
            "runner": sha256_file(Path(__file__)),
            "input_contract": sha256_file(Path(__file__).with_name("r4_downstream_input_contract.py")),
            "effect_implementation": sha256_file(Path(interface.__file__)),
        },
        "locked_settings": {
            "spot_diameter_um": interface.NOMINAL_SPOT_DIAMETER_UM,
            "bootstrap_replicates": interface.BOOTSTRAP_REPLICATES,
            "min_group_spots": interface.MIN_GROUP_SPOTS,
            "min_spatial_blocks": interface.MIN_BLOCKS,
            "K": interface.K_VALUES,
            "estimators": ESTIMATORS,
            "scores": list(interface.SCORE_DEFINITIONS),
        },
        "artifacts": artifacts,
    }
    (output_dir / "result_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--section-id", required=True)
    parser.add_argument("--spots", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--score-manifest", required=True, type=Path)
    parser.add_argument("--maps", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    score_manifest = json.loads(args.score_manifest.read_text())
    input_hashes = {name: sha256_file(path) for name, path in (
        ("spots", args.spots), ("scores", args.scores),
        ("score_manifest", args.score_manifest), ("maps", args.maps),
    )}
    if input_hashes["scores"] != score_manifest.get("score_sha256"):
        raise ValueError("score table SHA-256 differs from its source manifest")
    run_section(
        SectionIdentity(args.dataset_id, args.sample_id, args.patient_id),
        args.section_id,
        pd.read_csv(args.spots, sep="\t", low_memory=False, keep_default_na=False),
        pd.read_csv(args.scores, sep="\t", low_memory=False),
        pd.read_csv(args.maps, sep="\t", low_memory=False),
        score_manifest,
        input_hashes,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
