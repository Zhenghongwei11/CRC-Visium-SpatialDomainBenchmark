#!/usr/bin/env python3
"""Independently reconcile application exports with the historical audit.

This script does not call the application adapter. It checks all retained
effect values, added non-evaluable rows, patient counts, event denominators,
and the exact continuous decomposition.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEY = ["sample_id", "estimator_id", "method_id", "K", "partition_id"]
NUMERIC = ["primary_scale_mad", "morphology_anchor_delta", "morphology_anchor_low", "morphology_anchor_high",
           "computational_delta", "computational_low", "computational_high", "change_vs_reference", "change_low", "change_high",
           "anchor_change", "anchor_change_low", "anchor_change_high", "n_anchor_near", "n_anchor_far",
           "n_computational_near", "n_computational_far", "n_blocks", "n_change_blocks",
           "change_draws_attempted", "change_draws_accepted", "change_draws_used",
           "anchor_change_draws_attempted", "anchor_change_draws_accepted", "anchor_change_draws_used"]
CATEGORY = ["patient_id", "section_id", "is_reference", "status", "status_reason", "anchor_conclusion", "computational_conclusion"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(output: Path, historical: Path) -> dict:
    manifest = json.loads((output / "run_manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        path = output / artifact["name"]
        check(path.stat().st_size == artifact["size_bytes"] and sha256(path) == artifact["sha256"], f"Export changed: {path.name}")
        check(len(read(path)) == artifact["rows"], f"Export row count changed: {path.name}")
    effects = read(output / "real_cohort_effects.tsv")
    anchors = read(output / "anchor_sections.tsv")
    patients = read(output / "patient_summary.tsv")
    check(not effects.duplicated(["resource", *KEY]).any(), "Duplicate application effect key")
    check(not anchors.duplicated(["resource", "sample_id"]).any(), "Duplicate anchor section")
    check(not patients.duplicated(["resource", "patient_id", "estimator_id"]).any(), "Duplicate patient summary")
    check(effects.loc[~effects.historical_row_present, "status"].eq("not_evaluable").all(), "Added grid row promoted to eligible")
    check(effects.label_provenance_class.eq("morphology_based_blinding_unknown").all(), "Unknown blinding incorrectly promoted")
    check(effects.interval_interpretation.eq("descriptive_historical_bootstrap_transportability_unestablished").all(), "Uncalibrated interval promoted")
    check(effects.domain_matching_uses_tissue_labels.all(), "Domain-matching label dependence omitted")
    numeric_checks = category_checks = 0
    resources = []
    for resource, config in manifest["config"]["resources"].items():
        prefix = config["historical_audit_prefix"]
        old_path = historical / f"{prefix}_all_section_effects.tsv"
        old = read(old_path)
        old = old.loc[old.score_id.eq("primary_barrier")].set_index(KEY).sort_index()
        new = effects.loc[effects.resource.eq(resource) & effects.historical_row_present].set_index(KEY).sort_index()
        check(old.index.equals(new.index), f"Historical keys changed: {resource}")
        for column in NUMERIC:
            left = pd.to_numeric(old[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(new[column], errors="coerce").to_numpy(float)
            check(np.isclose(left, right, atol=1e-10, rtol=1e-10, equal_nan=True).all(), f"Historical numeric changed: {resource}/{column}")
            numeric_checks += len(left)
        for column in CATEGORY:
            check(old[column].fillna("").astype(str).eq(new[column].fillna("").astype(str)).all(), f"Historical category changed: {resource}/{column}")
            category_checks += len(old)
        primary_audit = json.loads((historical / f"{prefix}_primary_claim_audit.json").read_text())
        counts = []
        for estimator, previous in primary_audit["estimators"].items():
            group = effects.loc[effects.resource.eq(resource) & effects.estimator_id.eq(estimator)]
            alternatives = group.loc[~group.is_reference]
            checks = {
                "all_rows": int(group.historical_row_present.sum()),
                "successful_rows": int(group.status.eq("success").sum()),
                "successful_nonreference_rows": int(alternatives.status.eq("success").sum()),
                "map_to_anchor_loss_nonreference_rows": int(alternatives.map_to_anchor_support_loss.sum()),
                "additional_loss_rows": int(alternatives.additional_support_loss.sum()),
                "material_paired_additional_loss_rows": int(alternatives.flagged_paired_additional_loss.sum()),
            }
            for label, value in checks.items():
                check(value == previous[label], f"Historical count changed: {resource}/{estimator}/{label}")
            ok = group.status.eq("success")
            refs = group.loc[group.is_reference].set_index(["sample_id", "method_id", "K"])
            reference_delta = group.set_index(["sample_id", "method_id", "K"]).index.map(refs.computational_delta)
            check(np.isclose((group.computational_delta - group.morphology_anchor_delta)[ok], group.anchor_change[ok], atol=1e-10).all(), "Map-anchor identity failed")
            check(np.isclose((group.computational_delta.to_numpy() - np.asarray(reference_delta, float))[ok], group.change_vs_reference[ok], atol=1e-10).all(), "Paired change identity failed")
            # Every additional-loss row must have an eligible, same-direction reference.
            event = group.additional_support_loss
            check((group.additional_loss_eligible[event] & group.reference_reproduces_anchor[event]).all(), "Additional loss has wrong denominator")
            check((~group.is_reference[event]).all(), "Reference counted as an alternative event")
            p = patients.loc[patients.resource.eq(resource) & patients.estimator_id.eq(estimator)]
            check(len(p) == config["n_patients"], "Repeated sections inflate patient n")
            check(int(p.n_attempted_sections.sum()) == config["n_sections"], "Sections missing from patient summary")
            expected_patients = set(group.loc[group.flagged_paired_additional_loss, "patient_id"])
            check(set(p.loc[p.any_flagged_paired_additional_loss, "patient_id"]) == expected_patients, "Patient event aggregation mismatch")
            counts.append({"estimator": estimator, **checks, "declared_rows": len(group), "additional_loss_eligible_rows": int(group.additional_loss_eligible.sum()),
                           "independent_patients": len(p), "flagged_additional_loss_patients": sorted(expected_patients)})
        a = anchors.loc[anchors.resource.eq(resource)]
        check(a.sample_id.nunique() == config["n_sections"] and a.patient_id.nunique() == config["n_patients"], "Anchor inventory mismatch")
        marker = read(output / "marker_sensitivity.tsv")
        marker = marker.loc[marker.resource.eq(resource)]
        check(len(marker) == config["n_sections"] and marker.sensitivity_status.eq("not_evaluable").all(), "Missing marker inputs represented as a result")
        resources.append({"resource": resource, "historical_effects_sha256": sha256(old_path),
                          "historical_rows": len(old), "added_non_evaluable_rows": int((effects.resource.eq(resource) & ~effects.historical_row_present).sum()),
                          "anchor_eligible_sections": int(a.anchor_status.eq("evaluable").sum()), "estimators": counts})
    return {"schema_version": "tissue_anchor_application_audit_v1", "status": "pass",
            "audited_utc": datetime.now(timezone.utc).isoformat(), "run_manifest_sha256": sha256(output / "run_manifest.json"),
            "numeric_fields_checked": numeric_checks, "category_fields_checked": category_checks,
            "resources": resources,
            "intentional_changes": [
                "All attempted sections retained: Valdeolivas two sections without pure tumor/stroma anchors add 144 non-evaluable estimator-map rows",
                "GSE declared grid includes seven absent saved maps, adding 14 non-evaluable estimator-map rows",
                "Label eligibility, anchor support eligibility, own-map eligibility and historical reference-comparison eligibility separated",
                "Additional-loss denominator restricted to anchor/reference-supported eligible alternatives",
                "Continuous effects standardized by section MAD; intervals retain descriptive interpretation",
                "Available method-resolution settings summarized within sections before equal-section patient summaries",
                "Score and estimator display labels may be clarified; historical numeric results and classifications retained",
            ]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/interface_validation/reframed/real_application_v1")
    parser.add_argument("--historical", type=Path, default=ROOT / "results/interface_validation/r4_undirected_20260926_audit_v1")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report_path = args.report or args.output / "independent_audit.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report = audit(args.output, args.historical)
    report["audit_script_sha256"] = sha256(Path(__file__))
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
