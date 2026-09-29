#!/usr/bin/env python3
"""Decompose saved tissue-anchored contrasts and summarize within patients.

No data retrieval, map fitting, or new bootstrap run occurs here. Continuous
contrasts and membership diagnostics are reconstructed from registered spots,
fixed scores and maps. Historical intervals retain their descriptive status.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import colab_interface_validation_full as interface
from audit_r4_independent_geometry import independent_anchor, independent_neighbors
from r4_downstream_only import sha256_file
from r4_downstream_input_contract import SectionIdentity, validate_section


ROOT = Path(__file__).resolve().parents[1]
SUPPORT = {"near_enrichment", "near_depletion"}
MAP_KEY = ["estimator_id", "method_id", "K", "partition_id"]
EFFECT_KEY = ["resource", "sample_id", *MAP_KEY]
ESTIMATOR_LABELS = {
    "strict_computational_interface": "Computational adjacency and stromal retention",
    "anchor_retention": "Stromal retention within fixed morphology bands",
}


def read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t", low_memory=False, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def median(values: object) -> float:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    return float(np.median(array[np.isfinite(array)])) if np.isfinite(array).any() else np.nan


def numeric(value: object) -> float:
    return float(pd.to_numeric(value, errors="coerce"))


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else np.nan


def contrast(values: np.ndarray, near: np.ndarray, far: np.ndarray) -> float:
    if not near.any() or not far.any():
        return np.nan
    return float(np.median(values[near]) - np.median(values[far]))


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = int((left | right).sum())
    return ratio(int((left & right).sum()), union)


def numeric_equal(left: float, right: float, label: str) -> None:
    if not np.isclose(left, right, atol=1e-10, rtol=1e-10, equal_nan=True):
        raise ValueError(f"Numeric replay mismatch for {label}: {left} vs {right}")


def load_contract(config_path: Path, root: Path) -> tuple[dict, dict]:
    config = json.loads(config_path.read_text())
    protocol = root / config["protocol"]
    # Read and bind the scientific contract, including its original freeze.
    if not protocol.read_text().strip() or sha256_file(protocol) != config["protocol_sha256"]:
        raise ValueError("Protocol fingerprint mismatch")
    freeze = json.loads((root / config["simulation_freeze"]).read_text())
    for relative, expected in freeze["file_sha256"].items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")
    sim_path = root / config["simulation_manifest"]
    if sha256_file(sim_path) != freeze["manifest_sha256"]:
        raise ValueError("Frozen simulation manifest changed")
    sim = json.loads(sim_path.read_text())
    if config["score_genes"] != interface.PRIMARY_GENES:
        raise ValueError("Primary score differs from frozen implementation")
    for field, observed in (("bootstrap_replicates", interface.BOOTSTRAP_REPLICATES),
                            ("min_group_spots", interface.MIN_GROUP_SPOTS),
                            ("min_blocks", interface.MIN_BLOCKS)):
        if sim["estimation"][field] != observed:
            raise ValueError(f"Frozen estimator setting mismatch: {field}")
    if config["legacy_support_margin_mad"] != 0.5 or config["legacy_paired_change_margin_mad"] != 0.25:
        raise ValueError("Historical comparison thresholds changed")
    if config["K"] != interface.K_VALUES or set(config["estimators"]) != set(ESTIMATOR_LABELS):
        raise ValueError("Declared map or estimator grid changed")
    return config, sim


def memberships(spots: pd.DataFrame, maps: pd.DataFrame, config: dict) -> tuple[dict, dict]:
    coords = spots[["array_row", "array_col"]].to_numpy(float)
    neighbors = independent_neighbors(coords)
    near, far, _ = independent_anchor(spots, neighbors)
    coarse = spots.coarse_label.astype(str).to_numpy()
    xy = spots[["x_fullres", "y_fullres"]].to_numpy(float)
    diameter = float(spots.spot_diameter_fullres.iloc[0])
    distance = (cKDTree(xy[coarse == "tumor"]).query(xy)[0] * 55 / diameter
                if (coarse == "tumor").any() else np.full(len(spots), np.nan))
    anchor = {"near": near, "far": far, "distance_um": distance,
              "xy_um": xy * 55 / diameter, "blocks": spots.spatial_block.to_numpy(str),
              "stroma_mask": coarse == "stroma"}
    records = {}
    order = spots.barcode.astype(str).tolist()
    for (method, k, partition), group in maps[maps.status.eq("success")].groupby(
        ["method_id", "K", "replicate_id"], sort=True
    ):
        labels = group.set_index("barcode").loc[order, "domain_label"].to_numpy(int)
        selected, stats = interface.domain_selection(labels, coarse)
        for estimator in config["estimators"]:
            record = {"selection_status": selected["status"], "selection_reason": selected["reason"]}
            if selected["status"] == "success":
                n, f = interface.computational_groups(
                    near, far, labels, selected["tumor_domain"], selected["stroma_domain"], neighbors, estimator
                )
                record.update(near=n, far=f, stroma=labels == selected["stroma_domain"],
                              tumor=labels == selected["tumor_domain"], selected=selected)
                tied_tumor = sum(row["tumor_fraction"] == selected["tumor_fraction"] for row in stats)
                tied_stroma = sum(row["domain_label"] != selected["tumor_domain"]
                                 and row["stroma_n"] > 0 and row["stroma_fraction"] == selected["stroma_fraction"]
                                 for row in stats)
                record.update(n_tied_tumor_domains=tied_tumor, n_tied_stroma_domains=tied_stroma)
            records[(estimator, method, int(k), str(partition))] = record
    return anchor, records


def geometry_diagnostics(record: dict, reference: dict, anchor: dict) -> dict:
    if "near" not in record:
        return {"geometry_status": "domain_mapping_unavailable"}
    n, f = record["near"], record["far"]
    result = {
        "geometry_status": "available", "n_selected_near": int(n.sum()), "n_selected_far": int(f.sum()),
        "near_retention": ratio(int(n.sum()), int(anchor["near"].sum())),
        "far_retention": ratio(int(f.sum()), int(anchor["far"].sum())),
        "n_near_blocks": len(set(anchor["blocks"][n])), "n_far_blocks": len(set(anchor["blocks"][f])),
        "n_selected_stroma_domain": int(record["stroma"].sum()),
        "selected_tumor_domain": record["selected"]["tumor_domain"],
        "selected_stroma_domain": record["selected"]["stroma_domain"],
        "selected_tumor_purity": record["selected"]["tumor_fraction"],
        "selected_stroma_purity": record["selected"]["stroma_fraction"],
        "n_tied_tumor_domains": record["n_tied_tumor_domains"],
        "n_tied_stroma_domains": record["n_tied_stroma_domains"],
        "domain_tie_policy": "Descending morphology purity, then ascending domain label",
    }
    if "near" not in reference:
        result["membership_change_scope"] = "reference_geometry_unavailable"
        return result
    changed_stroma = record["stroma"] ^ reference["stroma"]
    changed_near = n ^ reference["near"]
    changed_far = f ^ reference["far"]
    overlap = int((record["stroma"] & reference["stroma"]).sum())
    if not changed_stroma.any() and not changed_near.any() and not changed_far.any() and np.array_equal(record["tumor"], reference["tumor"]):
        scope = "identical_selected_compartments_and_bands"
    elif not changed_stroma.any() and not changed_far.any() and changed_near.any():
        scope = "adjacency_only_near_selection_change"
    elif overlap == 0:
        scope = "disjoint_selected_stromal_domains"
    elif changed_stroma.any():
        scope = "stromal_domain_change_extent_reported"
    else:
        scope = "selected_tumor_domain_change_without_band_change"
    centroid_shift = np.linalg.norm(
        anchor["xy_um"][record["stroma"]].mean(axis=0)
        - anchor["xy_um"][reference["stroma"]].mean(axis=0)
    )
    changed_distance = anchor["distance_um"][changed_stroma]
    result.update({
        "membership_change_scope": scope,
        "stroma_domain_jaccard_vs_reference": jaccard(record["stroma"], reference["stroma"]),
        "tumor_domain_jaccard_vs_reference": jaccard(record["tumor"], reference["tumor"]),
        "near_jaccard_vs_reference": jaccard(n, reference["near"]),
        "far_jaccard_vs_reference": jaccard(f, reference["far"]),
        "n_changed_near": int(changed_near.sum()), "n_changed_far": int(changed_far.sum()),
        "n_changed_stroma_domain": int(changed_stroma.sum()),
        "n_changed_stroma_outside_anchor_bands": int((changed_stroma & ~(anchor["near"] | anchor["far"])).sum()),
        "changed_stroma_near_fraction": ratio(int((changed_stroma & anchor["near"]).sum()), int(changed_stroma.sum())),
        "changed_stroma_median_distance_um": median(changed_distance),
        "stroma_domain_centroid_shift_um": float(centroid_shift),
    })
    return result


def decompose(frame: pd.DataFrame, support_margin: float = 0.5, change_margin: float = 0.25) -> pd.DataFrame:
    rows = frame.copy()
    for column in ("computational_delta", "computational_low", "computational_high", "primary_scale_mad",
                   "morphology_anchor_delta", "morphology_anchor_low", "morphology_anchor_high",
                   "change_vs_reference", "change_low", "change_high", "anchor_change", "anchor_change_low", "anchor_change_high"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    key = ["resource", "sample_id", "estimator_id", "method_id", "K"]
    if rows.duplicated(EFFECT_KEY).any():
        raise ValueError("Duplicate section-map key")
    numeric = ["computational_delta", "computational_low", "computational_high", "anchor_change",
               "anchor_change_low", "anchor_change_high", "primary_scale_mad"]
    refs = rows.loc[rows.is_reference, key + ["status", "computational_conclusion", "partition_id", *numeric]]
    if refs.duplicated(key).any() or len(refs) != len(rows[key].drop_duplicates()):
        raise ValueError("Every method-resolution group must contain exactly one declared reference")
    refs = refs.rename(columns={column: f"reference_{column}" for column in refs if column not in key})
    rows = rows.merge(refs, on=key, validate="many_to_one")
    ok = rows.status.eq("success")
    anchor_ok = rows.anchor_status.eq("evaluable")
    rows["anchor_supported"] = anchor_ok & rows.anchor_conclusion.isin(SUPPORT)
    rows["map_comparison_eligible"] = ok
    rows["reference_mismatch_eligible"] = rows.reference_status.eq("success") & anchor_ok
    rows["reference_mismatch"] = rows.reference_anchor_change
    rows["reference_mismatch_low"] = rows.reference_anchor_change_low
    rows["reference_mismatch_high"] = rows.reference_anchor_change_high
    rows["reference_reproduces_anchor"] = (rows.reference_mismatch_eligible & rows.anchor_supported
                                            & rows.reference_computational_conclusion.eq(rows.anchor_conclusion))
    rows["reference_support_loss"] = (rows.reference_mismatch_eligible & rows.anchor_supported
                                       & rows.reference_computational_conclusion.eq("no_supported_difference"))
    rows["map_to_anchor_support_loss"] = ok & rows.anchor_supported & rows.computational_conclusion.eq("no_supported_difference")
    rows["additional_loss_eligible"] = ~rows.is_reference & ok & rows.reference_reproduces_anchor
    rows["additional_support_loss"] = rows.additional_loss_eligible & rows.computational_conclusion.eq("no_supported_difference")
    rows["paired_change_eligible"] = (~rows.is_reference & ok
                                     & np.isfinite(rows.change_vs_reference) & np.isfinite(rows.change_low)
                                     & np.isfinite(rows.change_high))
    rows["paired_interval_excludes_zero"] = rows.paired_change_eligible & ((rows.change_low > 0) | (rows.change_high < 0))
    rows["legacy_paired_magnitude_flag"] = rows.paired_change_eligible & rows.change_vs_reference.abs().ge(change_margin * rows.primary_scale_mad)
    rows["paired_change_flag"] = rows.paired_interval_excludes_zero & rows.legacy_paired_magnitude_flag
    rows["flagged_paired_additional_loss"] = rows.additional_support_loss & rows.paired_change_flag
    rows["supported_reversal_vs_anchor"] = (ok & rows.anchor_supported & rows.computational_conclusion.isin(SUPPORT)
                                            & ~rows.computational_conclusion.eq(rows.anchor_conclusion))
    rows["supported_reversal_vs_reference"] = (~rows.is_reference & ok & rows.reference_status.eq("success")
                                              & rows.reference_computational_conclusion.isin(SUPPORT)
                                              & rows.computational_conclusion.isin(SUPPORT)
                                              & ~rows.computational_conclusion.eq(rows.reference_computational_conclusion))
    rows["opposite_point_sign_vs_reference"] = rows.paired_change_eligible & (rows.computational_delta * rows.reference_computational_delta < 0)
    rows["legacy_support_margin_mad"] = support_margin
    rows["legacy_paired_change_margin_mad"] = change_margin
    for source, target in (("morphology_anchor_delta", "anchor_standardized"), ("morphology_anchor_low", "anchor_standardized_low"),
                           ("morphology_anchor_high", "anchor_standardized_high"), ("computational_delta", "map_standardized"),
                           ("computational_low", "map_standardized_low"), ("computational_high", "map_standardized_high"),
                           ("reference_mismatch", "reference_mismatch_standardized"),
                           ("reference_mismatch_low", "reference_mismatch_standardized_low"),
                           ("reference_mismatch_high", "reference_mismatch_standardized_high"),
                           ("change_vs_reference", "paired_change_standardized"), ("change_low", "paired_change_standardized_low"),
                           ("change_high", "paired_change_standardized_high"), ("anchor_change", "map_discrepancy_standardized")):
        rows[target] = rows[source] / rows.primary_scale_mad.where(rows.primary_scale_mad > 0)
    rows["map_anchor_relation"] = "not_evaluable"
    rows.loc[ok & ~rows.anchor_supported, "map_anchor_relation"] = "anchor_not_supported"
    rows.loc[ok & rows.reference_reproduces_anchor & rows.computational_conclusion.eq(rows.anchor_conclusion), "map_anchor_relation"] = "supported_same_direction"
    rows.loc[ok & rows.anchor_supported & rows.computational_conclusion.eq(rows.anchor_conclusion), "map_anchor_relation"] = "supported_same_direction"
    rows.loc[rows.map_to_anchor_support_loss, "map_anchor_relation"] = "support_rule_not_met"
    rows.loc[rows.supported_reversal_vs_anchor, "map_anchor_relation"] = "supported_opposite_direction"
    rows["attenuated_point_toward_zero"] = ok & (rows.computational_delta * rows.morphology_anchor_delta > 0) & (rows.computational_delta.abs() < rows.morphology_anchor_delta.abs())
    return rows


def verify_section(source: Path, config: dict) -> tuple[dict, list[dict]]:
    manifest = json.loads((source / "result_manifest.json").read_text())
    checked = []
    for artifact in manifest["artifacts"]:
        path = source / artifact["name"]
        if path.stat().st_size != artifact["size_bytes"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Historical artifact changed: {path}")
        checked.append({"path": str(path), "sha256": artifact["sha256"], "size_bytes": artifact["size_bytes"]})
    maps = source / "input_maps.tsv.gz"
    if sha256_file(maps) != manifest["input_sha256"]["maps"]:
        raise ValueError(f"Saved map fingerprint mismatch: {source.name}")
    if manifest["score_manifest"]["normalization"] != config["normalization"]:
        raise ValueError("Score normalization mismatch")
    if manifest["code_sha256"]["effect_implementation"] != sha256_file(Path(interface.__file__)):
        raise ValueError("Historical estimator differs from frozen estimator")
    checked.extend({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                   for path in (maps, source / "result_manifest.json"))
    return manifest, checked


def adapt_section(resource: str, source: Path, config: dict) -> tuple[pd.DataFrame, dict, list[dict]]:
    resource_config = config["resources"][resource]
    manifest, checks = verify_section(source, config)
    identity = manifest["identity"]
    spots = read_table(source / "spot_inputs.tsv.gz")
    scores = read_table(source / "spot_scores.tsv.gz")
    maps = read_table(source / "input_maps.tsv.gz")
    validate_section(SectionIdentity(identity["dataset_id"], identity["sample_id"], identity["patient_id"]), spots, scores, maps)
    historical = read_table(source / "section_effects.tsv")
    if not historical.empty:
        historical = historical.loc[historical.score_id.eq(config["score_id"])].copy()
        if historical.duplicated(MAP_KEY).any():
            raise ValueError("Duplicate historical primary effect")
    anchor_table = read_table(source / "anchor_effects.tsv")
    anchor_table = anchor_table.loc[anchor_table.score_id.eq(config["score_id"])].drop(columns="estimator_id").drop_duplicates()
    if len(anchor_table) != 1:
        raise ValueError(f"Expected one unique section anchor: {source.name}")
    saved_anchor = anchor_table.iloc[0].to_dict()
    anchor, map_records = memberships(spots, maps, config)
    primary_scores = scores.loc[scores.score_id.eq(config["score_id"])].set_index("barcode").loc[spots.barcode.astype(str)]
    if not primary_scores.status.eq("success").all():
        raise ValueError("Retained primary score is not evaluable")
    values = primary_scores.score_value.to_numpy(float)
    computed_anchor = contrast(values, anchor["near"], anchor["far"])
    scale = interface.score_scale(values, anchor)
    if "anchor_delta" in saved_anchor:
        anchor_blocks = min(len(set(anchor["blocks"][anchor["near"]])), len(set(anchor["blocks"][anchor["far"]])))
        archived_point = computed_anchor if anchor_blocks >= interface.MIN_BLOCKS else np.nan
        numeric_equal(archived_point, numeric(saved_anchor["anchor_delta"]), f"{source.name}/anchor")
    anchor_ok = saved_anchor["status"] == "evaluable"
    provenance = {
        "resource": resource, "dataset_id": identity["dataset_id"], "sample_id": identity["sample_id"],
        "patient_id": identity["patient_id"], "section_id": manifest["section_id"],
        "score_id": config["score_id"], "score_label": config["score_label"],
        "label_provenance_class": resource_config["label_provenance_class"],
        "label_source": resource_config["label_source"], "expression_blinding": resource_config["expression_blinding"],
        "anchor_selection_uses_score": resource_config["anchor_selection_uses_score"],
        "domain_matching_uses_tissue_labels": resource_config["domain_matching_uses_tissue_labels"],
        "map_learning_uses_expression": resource_config["map_learning_uses_expression"],
        "map_score_feature_independence": "not_established",
        "domain_mapping_rule": "One tumor and one distinct stromal domain maximizing morphology purity; ties by domain label",
        "annotation_registration_status": "barcode_and_coordinate_registered",
        "interval_interpretation": "descriptive_historical_bootstrap_transportability_unestablished",
        "source_result_manifest_sha256": sha256_file(source / "result_manifest.json"),
        "source_expression_package_sha256": manifest["score_source_package_sha256"],
        "source_maps_sha256": manifest["input_sha256"]["maps"],
        "source_effects_sha256": sha256_file(source / "section_effects.tsv"),
        "protocol_sha256": config["protocol_sha256"],
    }
    section = {**provenance, "anchor_status": saved_anchor["status"], "anchor_reason": saved_anchor["reason"],
               "n_registered_spots": len(spots), "n_anchor_near": int(anchor["near"].sum()),
               "n_anchor_far": int(anchor["far"].sum()), "primary_scale_mad": scale,
               "morphology_anchor_delta": computed_anchor if anchor_ok else np.nan,
               "morphology_anchor_low": numeric(saved_anchor.get("anchor_low", "")) if anchor_ok else np.nan,
               "morphology_anchor_high": numeric(saved_anchor.get("anchor_high", "")) if anchor_ok else np.nan,
               "anchor_conclusion": saved_anchor.get("anchor_conclusion", "uncertain") if anchor_ok else "uncertain"}
    rows = []
    historical_lookup = {tuple(row[field] for field in MAP_KEY): row for row in historical.to_dict("records")}
    for estimator, method, k in itertools.product(config["estimators"], resource_config["methods"], config["K"]):
        partitions = config["ward_partitions"] if method == "M2_spatial_ward" else config["stochastic_partitions"]
        reference_id = "neighbors_6" if method == "M2_spatial_ward" else "seed_11"
        reference = map_records.get((estimator, method, k, reference_id), {})
        for partition in partitions:
            key = (estimator, method, k, partition)
            record = map_records.get(key, {})
            old = historical_lookup.get(key)
            row = {**section, "estimator_id": estimator, "estimator_label": ESTIMATOR_LABELS[estimator],
                   "method_id": method, "method_label": interface.METHOD_LABELS[method], "K": k,
                   "partition_id": partition, "is_reference": partition == reference_id,
                   "historical_row_present": old is not None, "saved_map_present": bool(record),
                   "status": "not_evaluable", "status_reason": "saved_partition_unavailable",
                   "computational_conclusion": "uncertain", "computational_delta": np.nan,
                   "computational_low": np.nan, "computational_high": np.nan,
                   "anchor_change": np.nan, "anchor_change_low": np.nan, "anchor_change_high": np.nan,
                   "change_vs_reference": np.nan, "change_low": np.nan, "change_high": np.nan}
            if old is not None:
                row.update(old)
                row.update(provenance)
                row["anchor_status"] = saved_anchor["status"]
            elif not anchor_ok:
                row["status_reason"] = saved_anchor["reason"]
            elif record:
                raise ValueError(f"Eligible anchor and saved map lack historical effect: {source.name}/{key}")
            row["domain_mapping_status"] = record.get("selection_status", "unavailable")
            row["domain_mapping_reason"] = record.get("selection_reason", "saved_partition_unavailable")
            geometry = geometry_diagnostics(record, reference, anchor)
            row.update(geometry)
            own_ok = (anchor_ok and record.get("selection_status") == "success"
                      and geometry.get("n_selected_near", 0) >= interface.MIN_GROUP_SPOTS
                      and geometry.get("n_selected_far", 0) >= interface.MIN_GROUP_SPOTS
                      and min(geometry.get("n_near_blocks", 0), geometry.get("n_far_blocks", 0)) >= interface.MIN_BLOCKS)
            row["map_self_eligible"] = own_ok
            if "near" in record and anchor_ok:
                d = contrast(values, record["near"], record["far"])
                if old is not None:
                    archived_point = d if min(geometry["n_near_blocks"], geometry["n_far_blocks"]) >= interface.MIN_BLOCKS else np.nan
                    numeric_equal(archived_point, numeric(old["computational_delta"]), f"{source.name}/{key}")
                if row["status"] == "success":
                    numeric_equal(d - computed_anchor, float(row["anchor_change"]), f"{source.name}/{key}/map-anchor")
                row["replayed_map_delta"] = d
            else:
                row["replayed_map_delta"] = np.nan
            rows.append(row)
    if len(historical_lookup) != sum(row["historical_row_present"] for row in rows):
        raise ValueError("Historical rows outside declared grid")
    section["anchor_supported"] = anchor_ok and section["anchor_conclusion"] in SUPPORT
    section["anchor_standardized"] = ratio(section["morphology_anchor_delta"], scale)
    return pd.DataFrame(rows), section, checks


def summarize_methods(effects: pd.DataFrame) -> pd.DataFrame:
    output = []
    group_key = ["resource", "patient_id", "sample_id", "estimator_id", "method_id", "K"]
    for key, group in effects.groupby(group_key, sort=True):
        successful = group.loc[group.map_comparison_eligible]
        alternatives = successful.loc[~successful.is_reference]
        reference = group.loc[group.is_reference].iloc[0]
        row = dict(zip(group_key, key, strict=True))
        row.update(n_declared_maps=len(group), n_historical_maps=int(group.historical_row_present.sum()),
                   n_eligible_maps=len(successful), n_eligible_alternatives=len(alternatives),
                   reference_eligible=bool(reference.map_comparison_eligible),
                   anchor_supported=bool(reference.anchor_supported),
                   reference_support_loss=bool(reference.reference_support_loss),
                   reference_mismatch_standardized=reference.reference_mismatch_standardized if reference.map_comparison_eligible else np.nan,
                   reference_map_standardized=reference.map_standardized if reference.map_comparison_eligible else np.nan)
        for field in ("paired_change_standardized", "map_standardized", "map_discrepancy_standardized", "near_retention", "far_retention"):
            row[f"alternative_median_{field}"] = median(alternatives[field])
        for field in ("additional_loss_eligible", "additional_support_loss", "paired_change_eligible", "paired_change_flag",
                      "flagged_paired_additional_loss", "supported_reversal_vs_anchor", "supported_reversal_vs_reference",
                      "map_to_anchor_support_loss"):
            row[f"n_{field}"] = int(alternatives[field].sum())
        output.append(row)
    return pd.DataFrame(output)


SUMMARY_VALUES = ("reference_mismatch_standardized", "reference_map_standardized", "alternative_median_paired_change_standardized",
                  "alternative_median_map_standardized", "alternative_median_map_discrepancy_standardized")
EVENTS = ("additional_support_loss", "paired_change_flag", "flagged_paired_additional_loss", "supported_reversal_vs_anchor",
          "supported_reversal_vs_reference", "map_to_anchor_support_loss")


def nested_summaries(effects: pd.DataFrame, anchors: pd.DataFrame) -> dict[str, pd.DataFrame]:
    methods = summarize_methods(effects)
    section_rows = []
    for key, group in methods.groupby(["resource", "patient_id", "sample_id", "estimator_id"], sort=True):
        row = dict(zip(["resource", "patient_id", "sample_id", "estimator_id"], key, strict=True))
        row.update(n_declared_method_K=len(group), n_eligible_reference_method_K=int(group.reference_eligible.sum()),
                   n_eligible_alternative_method_K=int(group.n_eligible_alternatives.gt(0).sum()),
                   n_eligible_maps=int(group.n_eligible_maps.sum()), n_eligible_alternatives=int(group.n_eligible_alternatives.sum()),
                   any_reference_support_loss=bool(group.reference_support_loss.any()),
                   additional_loss_comparisons=int(group.n_additional_loss_eligible.sum()),
                   paired_change_comparisons=int(group.n_paired_change_eligible.sum()))
        for field in SUMMARY_VALUES:
            row[field] = median(group[field])
        for field in EVENTS:
            row[f"any_{field}"] = bool(group[f"n_{field}"].gt(0).any())
        section_rows.append(row)
    sections = pd.DataFrame(section_rows).merge(anchors[["resource", "patient_id", "sample_id", "anchor_status", "anchor_reason",
                                                        "anchor_supported", "anchor_conclusion", "anchor_standardized"]],
                                               on=["resource", "patient_id", "sample_id"], validate="many_to_one")
    patients = []
    for key, group in sections.groupby(["resource", "patient_id", "estimator_id"], sort=True):
        row = dict(zip(["resource", "patient_id", "estimator_id"], key, strict=True))
        eligible_anchor = group.loc[group.anchor_status.eq("evaluable")]
        row.update(n_attempted_sections=len(group), n_anchor_eligible_sections=len(eligible_anchor),
                   n_anchor_supported_sections=int(group.anchor_supported.sum()),
                   n_map_eligible_sections=int(group.n_eligible_maps.gt(0).sum()),
                   n_reference_eligible_sections=int(group.n_eligible_reference_method_K.gt(0).sum()),
                   n_additional_loss_eligible_sections=int(group.additional_loss_comparisons.gt(0).sum()),
                   n_paired_change_eligible_sections=int(group.paired_change_comparisons.gt(0).sum()),
                   n_eligible_maps=int(group.n_eligible_maps.sum()),
                   n_eligible_alternatives=int(group.n_eligible_alternatives.sum()),
                   independent_unit="patient", aggregation="equal-section median after equal-method-K median; available settings",
                   uncertainty="section range; no patient inferential interval")
        for field in (*SUMMARY_VALUES, "anchor_standardized"):
            values = pd.to_numeric(group[field], errors="coerce").dropna()
            row[field] = median(values)
            row[f"{field}_section_min"] = float(values.min()) if len(values) else np.nan
            row[f"{field}_section_max"] = float(values.max()) if len(values) else np.nan
        for field in ("reference_support_loss", *EVENTS):
            row[f"any_{field}"] = bool(group[f"any_{field}"].any())
        row["anchor_supported_directions"] = ";".join(sorted(set(group.loc[group.anchor_supported, "anchor_conclusion"])))
        row["anchor_supported_direction_conflict"] = len(set(group.loc[group.anchor_supported, "anchor_conclusion"])) > 1
        patients.append(row)
    patient_methods = methods.groupby(["resource", "patient_id", "estimator_id", "method_id", "K"], as_index=False).agg(
        n_sections=("sample_id", "size"), n_reference_eligible_sections=("reference_eligible", "sum"),
        reference_mismatch_standardized=("reference_mismatch_standardized", "median"),
        alternative_median_paired_change_standardized=("alternative_median_paired_change_standardized", "median"),
        n_additional_support_loss=("n_additional_support_loss", "sum"),
        n_flagged_paired_additional_loss=("n_flagged_paired_additional_loss", "sum"))
    return {"section_method_summary": methods, "section_summary": sections,
            "patient_summary": pd.DataFrame(patients), "patient_method_summary": patient_methods}


def cohort_summary(effects: pd.DataFrame, anchors: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (resource, estimator), group in effects.groupby(["resource", "estimator_id"], sort=True):
        a = anchors.loc[anchors.resource.eq(resource)]
        p = patients.loc[patients.resource.eq(resource) & patients.estimator_id.eq(estimator)]
        refs = group.loc[group.is_reference]
        alts = group.loc[~group.is_reference]
        row = {"resource": resource, "estimator_id": estimator,
               "n_attempted_sections": len(a), "n_label_eligible_sections": int((~a.anchor_reason.eq("missing_pure_tumor_or_stroma_label")).sum()),
               "n_anchor_eligible_sections": int(a.anchor_status.eq("evaluable").sum()),
               "n_anchor_supported_sections": int(a.anchor_supported.sum()),
               "n_attempted_patients": len(p), "n_anchor_eligible_patients": int(p.n_anchor_eligible_sections.gt(0).sum()),
               "n_anchor_supported_patients": int(p.n_anchor_supported_sections.gt(0).sum()),
               "n_map_eligible_patients": int(p.n_map_eligible_sections.gt(0).sum()),
               "n_declared_rows": len(group), "n_historical_rows": int(group.historical_row_present.sum()),
               "n_eligible_rows": int(group.map_comparison_eligible.sum()), "n_eligible_alternative_rows": int(alts.map_comparison_eligible.sum()),
               "n_reference_mismatch_eligible_rows": int(refs.reference_mismatch_eligible.sum()),
               "n_anchor_supported_reference_rows": int((refs.map_comparison_eligible & refs.anchor_supported).sum()),
               "n_reference_support_loss_rows": int(refs.reference_support_loss.sum()),
               "n_additional_loss_eligible_rows": int(alts.additional_loss_eligible.sum()),
               "n_additional_loss_eligible_patients": int(p.n_additional_loss_eligible_sections.gt(0).sum()),
               "n_paired_change_eligible_patients": int(p.n_paired_change_eligible_sections.gt(0).sum())}
        for field in EVENTS:
            row[f"n_{field}_alternative_rows"] = int(alts[field].sum())
            row[f"n_patients_with_{field}"] = int(p[f"any_{field}"].sum())
        row["n_patients_with_reference_support_loss"] = int(p.any_reference_support_loss.sum())
        row["interpretation"] = "Descriptive available-resource application; map counts are repeated observations"
        rows.append(row)
    return pd.DataFrame(rows)


def marker_gate(config: dict, root: Path, anchors: pd.DataFrame) -> pd.DataFrame:
    gate = read_table(root / config["marker_input_gate"])
    if gate.resource.duplicated().any() or set(gate.resource) != set(config["resources"]):
        raise ValueError("Marker input gate does not cover declared resources")
    if not gate.contains_single_gene_values.eq("no").all() or not gate.can_calculate_TGFB1_CXCL12_now.eq("no").all():
        raise ValueError("Expression input gate changed; implement same-mask sensitivity before export")
    output = anchors[["resource", "patient_id", "sample_id"]].merge(gate, on="resource", validate="many_to_one")
    output["sensitivity_status"] = "not_evaluable"
    output["sensitivity_reason"] = "Compatible normalized single-gene values unavailable in retained replay inputs"
    output["interpretation"] = "ACTA2/TAGLN component dependence unresolved; no inference of absent muscle confounding"
    output["protocol_sha256"] = config["protocol_sha256"]
    output["input_gate_sha256"] = sha256_file(root / config["marker_input_gate"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / "docs/interface_validation/real_resource_application_v1.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config, _ = load_contract(args.config, args.project_root)
    output = args.output or args.project_root / config["output_root"]
    if output.exists():
        raise FileExistsError(f"Use a new versioned output directory: {output}")
    output.mkdir(parents=True)
    all_effects, all_anchors, checks = [], [], []
    for resource, settings in config["resources"].items():
        source_root = args.project_root / settings["source_root"]
        sources = sorted(path for path in source_root.iterdir() if (path / "result_manifest.json").is_file())
        if len(sources) != settings["n_sections"]:
            raise ValueError(f"Section inventory mismatch: {resource}")
        for source in sources:
            frame, anchor, section_checks = adapt_section(resource, source, config)
            all_effects.append(frame)
            all_anchors.append(anchor)
            checks.extend(section_checks)
            print(json.dumps({"resource": resource, "sample_id": source.name, "rows": len(frame), "anchor_status": anchor["anchor_status"]}), flush=True)
    effects = decompose(pd.concat(all_effects, ignore_index=True), config["legacy_support_margin_mad"], config["legacy_paired_change_margin_mad"])
    anchors = pd.DataFrame(all_anchors)
    for resource, settings in config["resources"].items():
        if anchors.loc[anchors.resource.eq(resource), "patient_id"].nunique() != settings["n_patients"]:
            raise ValueError(f"Patient inventory mismatch: {resource}")
    tables = {"real_cohort_effects": effects, "anchor_sections": anchors, **nested_summaries(effects, anchors)}
    tables["cohort_summary"] = cohort_summary(effects, anchors, tables["patient_summary"])
    tables["marker_sensitivity"] = marker_gate(config, args.project_root, anchors)
    tables["non_evaluable_settings"] = effects.loc[~effects.map_comparison_eligible].copy()
    tables["input_checksums"] = pd.DataFrame(checks)
    artifacts = []
    for name, table in tables.items():
        path = output / f"{name}.tsv"
        table.to_csv(path, sep="\t", index=False)
        artifacts.append({"name": path.name, "rows": len(table), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {"schema_version": "tissue_anchor_real_application_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "analysis_role": config["analysis_role"], "config": config,
                "config_sha256": sha256_file(args.config), "script_sha256": sha256_file(Path(__file__)),
                "dependencies_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in
                    ("audit_r4_independent_geometry.py", "r4_downstream_input_contract.py", "r4_downstream_only.py", "colab_interface_validation_full.py")},
                "input_checks": len(checks), "historical_intervals_reused": True, "new_bootstrap_computation": False,
                "software": {"numpy": np.__version__, "pandas": pd.__version__}, "artifacts": artifacts}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "cohort_summary": tables["cohort_summary"].to_dict("records")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
