#!/usr/bin/env python3
"""Build molecular tumor-stroma proxy interface sensitivity tables for JBCB R3.

The analysis is derived from existing partition maps and Visium expression
matrices. It does not rerun spatial-domain models.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from build_r3_cross_method_evidence import (
    DEFAULT_MAP_DIR,
    METHOD_LABELS,
    ROOT,
    _block_ids,
    _boundary_mask,
    _neighbor_indices,
    _read_map,
    _reference_id,
    _sample_data,
)


FEATURE_ORDER = [
    "COL1A1",
    "FAP",
    "EPCAM",
    "SPP1",
    "PTPRC",
    "SIG_CAF_FAP",
    "SIG_MY_SPP1",
    "SIG_TCELL",
    "SIG_EXCLUSION_TGFB_CXCL12",
    "SIG_CYTOTOX",
    "SIG_EXCLUSION_CONTRAST",
]

FULL_COHORT_MAPS = sorted(DEFAULT_MAP_DIR.glob("*.tsv"))
OFFICIAL_TARGETED_MAPS = [
    ROOT
    / "results"
    / "colab_official_full13"
    / "domain_maps_all"
    / "official_spagcn_stagate_bayesspace_maps.tsv"
]


def _stable_seed(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _bootstrap_median_delta(
    group_a: np.ndarray,
    group_b: np.ndarray,
    *,
    key: str,
    replicates: int,
) -> tuple[float, float]:
    if group_a.size == 0 or group_b.size == 0:
        return float("nan"), float("nan")
    if replicates <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(_stable_seed(key))
    estimates = np.empty(replicates, dtype=float)
    batch_size = 100
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        size = stop - start
        aa = rng.choice(group_a, size=(size, group_a.size), replace=True)
        bb = rng.choice(group_b, size=(size, group_b.size), replace=True)
        estimates[start:stop] = np.median(bb, axis=1) - np.median(aa, axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _block_median_delta(
    values: np.ndarray,
    group_a: np.ndarray,
    group_b: np.ndarray,
    blocks: np.ndarray,
    *,
    min_per_group: int,
) -> tuple[int, float, float, float]:
    deltas: list[float] = []
    for block in sorted(set(blocks.tolist())):
        in_block = blocks == block
        a = in_block & group_a
        b = in_block & group_b
        if int(a.sum()) < min_per_group or int(b.sum()) < min_per_group:
            continue
        deltas.append(float(np.median(values[b]) - np.median(values[a])))
    if not deltas:
        return 0, float("nan"), float("nan"), float("nan")
    arr = np.asarray(deltas, dtype=float)
    return (
        int(arr.size),
        float(np.median(arr)),
        float(np.quantile(arr, 0.025)),
        float(np.quantile(arr, 0.975)),
    )


def _any_neighbor(mask: np.ndarray, neighbor_indices: np.ndarray) -> np.ndarray:
    return np.any(mask[neighbor_indices], axis=1)


def _select_proxy_domains(
    labels: np.ndarray,
    epcam: np.ndarray,
    col1a1: np.ndarray,
    fap: np.ndarray | None,
    *,
    min_domain_spots: int,
) -> dict[str, object]:
    domain_rows: list[dict[str, object]] = []
    stroma_score_values = col1a1 if fap is None else (col1a1 + fap) / 2.0
    for label in sorted(np.unique(labels).tolist()):
        in_domain = labels == int(label)
        if int(in_domain.sum()) < min_domain_spots:
            continue
        domain_rows.append(
            {
                "label": int(label),
                "n_spots": int(in_domain.sum()),
                "epcam_median": float(np.median(epcam[in_domain])),
                "col1a1_median": float(np.median(col1a1[in_domain])),
                "fap_median": float(np.median(fap[in_domain])) if fap is not None else float("nan"),
                "stroma_proxy_median": float(np.median(stroma_score_values[in_domain])),
            }
        )
    if len(domain_rows) < 2:
        return {"status": "not_evaluable", "reason": "fewer_than_two_domains_after_size_filter"}

    domain_table = pd.DataFrame(domain_rows)
    tumor = domain_table.sort_values(["epcam_median", "n_spots"], ascending=[False, False]).iloc[0]
    stroma_candidates = domain_table[domain_table["label"].ne(int(tumor["label"]))].copy()
    if stroma_candidates.empty:
        return {"status": "not_evaluable", "reason": "no_non_tumor_candidate_domain"}
    stroma = stroma_candidates.sort_values(
        ["stroma_proxy_median", "n_spots"], ascending=[False, False]
    ).iloc[0]

    if float(tumor["epcam_median"]) <= float(stroma["epcam_median"]):
        return {"status": "not_evaluable", "reason": "selected_epcam_domain_not_higher_than_stroma_domain"}
    if float(stroma["stroma_proxy_median"]) <= float(tumor["stroma_proxy_median"]):
        return {"status": "not_evaluable", "reason": "selected_stroma_domain_not_higher_than_epcam_domain"}

    return {
        "status": "success",
        "reason": "",
        "tumor_label": int(tumor["label"]),
        "stroma_label": int(stroma["label"]),
        "tumor_domain_n": int(tumor["n_spots"]),
        "stroma_domain_n": int(stroma["n_spots"]),
        "tumor_epcam_median": float(tumor["epcam_median"]),
        "stroma_epcam_median": float(stroma["epcam_median"]),
        "tumor_stroma_proxy_median": float(tumor["stroma_proxy_median"]),
        "stroma_proxy_median": float(stroma["stroma_proxy_median"]),
        "domain_selection_note": "EPCAM-high domain paired with highest COL1A1/FAP proxy domain among remaining domains.",
    }


def _method_label(method_id: str) -> str:
    if method_id in METHOD_LABELS:
        return METHOD_LABELS[method_id]
    return method_id.replace("_", " ")


def _prepare_maps(group: pd.DataFrame, expected_barcodes: list[str]) -> dict[str, pd.DataFrame]:
    expected_set = set(expected_barcodes)
    maps: dict[str, pd.DataFrame] = {}
    for replicate_id in sorted(group["replicate_id"].astype(str).unique().tolist()):
        replicate = group[group["replicate_id"].astype(str).eq(replicate_id)].copy()
        replicate["barcode"] = replicate["barcode"].astype(str)
        if set(replicate["barcode"]) != expected_set or replicate["barcode"].duplicated().any():
            continue
        maps[replicate_id] = replicate.set_index("barcode").loc[expected_barcodes].reset_index()
    return maps


def _process_table(
    table: pd.DataFrame,
    *,
    analysis_scope: str,
    cache: dict[tuple[str, str], object],
    min_domain_spots: int,
    min_group_spots: int,
    bootstrap: int,
    compute_pvalue: bool,
    rows: list[dict[str, object]],
) -> None:
    for (dataset_id, sample_id, method_id, k_value), group in table.groupby(
        ["dataset_id", "sample_id", "method_id", "K"], sort=True
    ):
        dataset_id = str(dataset_id)
        sample_id = str(sample_id)
        method_id = str(method_id)
        k_value = int(k_value)
        sample = _sample_data(dataset_id, sample_id, cache)
        maps = _prepare_maps(group, sample.barcodes)
        reference_id = _reference_id(method_id)
        if reference_id not in maps or len(maps) < 2:
            continue

        feature_values = {
            feature_id: values
            for feature_id, (_feature_type, _genes_used, values) in sample.features.items()
        }
        if "EPCAM" not in feature_values or "COL1A1" not in feature_values:
            continue
        epcam = feature_values["EPCAM"]
        col1a1 = feature_values["COL1A1"]
        fap = feature_values.get("FAP")
        coords = maps[reference_id][["x", "y"]].to_numpy(dtype=float)
        neighbor_indices = _neighbor_indices(coords, neighbors=6)
        blocks = _block_ids(coords)

        for partition_id, partition in sorted(maps.items()):
            labels = partition["domain_label"].to_numpy(dtype=int)
            selected = _select_proxy_domains(
                labels,
                epcam,
                col1a1,
                fap,
                min_domain_spots=min_domain_spots,
            )
            base = {
                "analysis_scope": analysis_scope,
                "dataset_id": dataset_id,
                "sample_id": sample_id,
                "method_id": method_id,
                "method_label": _method_label(method_id),
                "K": k_value,
                "partition_id": partition_id,
                "is_reference": partition_id == reference_id,
                "status": selected["status"],
                "status_reason": selected["reason"],
            }
            if selected["status"] != "success":
                rows.append(base)
                continue

            tumor_mask = labels == int(selected["tumor_label"])
            stroma_mask = labels == int(selected["stroma_label"])
            boundary = _boundary_mask(labels, neighbor_indices)
            tumor_interface = tumor_mask & _any_neighbor(stroma_mask, neighbor_indices)
            stroma_interface = stroma_mask & _any_neighbor(tumor_mask, neighbor_indices)
            tumor_interior = tumor_mask & ~boundary
            stroma_interior = stroma_mask & ~boundary
            if int(tumor_interface.sum()) < min_group_spots or int(stroma_interface.sum()) < min_group_spots:
                row = dict(base)
                row.update(selected)
                row["status"] = "not_evaluable"
                row["status_reason"] = "selected_domains_not_adjacent_with_enough_interface_spots"
                row["n_tumor_interface"] = int(tumor_interface.sum())
                row["n_stroma_interface"] = int(stroma_interface.sum())
                rows.append(row)
                continue
            if int(stroma_interior.sum()) < min_group_spots:
                row = dict(base)
                row.update(selected)
                row["status"] = "not_evaluable"
                row["status_reason"] = "too_few_stroma_interior_spots"
                row["n_tumor_interface"] = int(tumor_interface.sum())
                row["n_stroma_interface"] = int(stroma_interface.sum())
                row["n_stroma_interior"] = int(stroma_interior.sum())
                rows.append(row)
                continue

            for feature_id in FEATURE_ORDER:
                if feature_id not in sample.features:
                    continue
                feature_type, genes_used, values = sample.features[feature_id]
                stroma_interface_values = values[stroma_interface]
                stroma_interior_values = values[stroma_interior]
                tumor_interface_values = values[tumor_interface]
                ci_lo, ci_hi = _bootstrap_median_delta(
                    stroma_interior_values,
                    stroma_interface_values,
                    key=f"{analysis_scope}|{dataset_id}|{sample_id}|{method_id}|{k_value}|{partition_id}|{feature_id}|stroma",
                    replicates=bootstrap,
                )
                n_blocks, block_delta, block_lo, block_hi = _block_median_delta(
                    values,
                    stroma_interior,
                    stroma_interface,
                    blocks,
                    min_per_group=min_group_spots,
                )
                pvalue = float("nan")
                if compute_pvalue:
                    pvalue = float(
                        mannwhitneyu(
                            stroma_interface_values,
                            stroma_interior_values,
                            alternative="two-sided",
                        ).pvalue
                    )
                row = dict(base)
                row.update(selected)
                row.update(
                    {
                        "contrast_id": "stroma_interface_minus_stroma_interior",
                        "contrast_label": "COL1A1/FAP-high side adjacent to EPCAM-high domain minus its clear interior",
                        "feature_type": feature_type,
                        "feature_id": feature_id,
                        "genes_used": genes_used,
                        "n_tumor_interface": int(tumor_interface.sum()),
                        "n_stroma_interface": int(stroma_interface.sum()),
                        "n_stroma_interior": int(stroma_interior.sum()),
                        "n_tumor_interior": int(tumor_interior.sum()),
                        "median_stroma_interface": float(np.median(stroma_interface_values)),
                        "median_stroma_interior": float(np.median(stroma_interior_values)),
                        "median_tumor_interface": float(np.median(tumor_interface_values)),
                        "median_delta": float(np.median(stroma_interface_values) - np.median(stroma_interior_values)),
                        "bootstrap_ci_lower": ci_lo,
                        "bootstrap_ci_upper": ci_hi,
                        "bootstrap_replicates": bootstrap,
                        "pvalue_mannwhitney_spot_level": pvalue,
                        "n_mixed_blocks": n_blocks,
                        "block_median_delta": block_delta,
                        "block_interval_lower": block_lo,
                        "block_interval_upper": block_hi,
                    }
                )
                rows.append(row)


def _add_reference_comparisons(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "feature_id" not in table.columns:
        return table
    out = table.copy()
    success = out[out["status"].eq("success") & out["feature_id"].notna()].copy()
    key_cols = ["analysis_scope", "dataset_id", "sample_id", "method_id", "K", "contrast_id", "feature_id"]
    refs = success[success["is_reference"].eq(True)][
        key_cols
        + [
            "partition_id",
            "median_delta",
            "bootstrap_ci_lower",
            "bootstrap_ci_upper",
            "block_median_delta",
        ]
    ].rename(
        columns={
            "partition_id": "reference_id",
            "median_delta": "reference_delta",
            "bootstrap_ci_lower": "reference_ci_lower",
            "bootstrap_ci_upper": "reference_ci_upper",
            "block_median_delta": "reference_block_delta",
        }
    )
    out = out.merge(refs, on=key_cols, how="left")
    out["opposite_sign_vs_reference"] = (
        out["status"].eq("success")
        & out["reference_delta"].notna()
        & out["median_delta"].notna()
        & ((np.sign(out["reference_delta"].astype(float)) * np.sign(out["median_delta"].astype(float))) < 0)
    )
    out["reference_interval_excludes_zero"] = (
        out["reference_ci_lower"].notna()
        & out["reference_ci_upper"].notna()
        & ((out["reference_ci_lower"].astype(float) > 0) | (out["reference_ci_upper"].astype(float) < 0))
    )
    out["partition_interval_excludes_zero"] = (
        out["bootstrap_ci_lower"].notna()
        & out["bootstrap_ci_upper"].notna()
        & ((out["bootstrap_ci_lower"].astype(float) > 0) | (out["bootstrap_ci_upper"].astype(float) < 0))
    )
    out["interval_supported_opposite_sign_vs_reference"] = (
        out["opposite_sign_vs_reference"]
        & out["reference_interval_excludes_zero"]
        & out["partition_interval_excludes_zero"]
    )
    ref_abs = out["reference_delta"].abs()
    out["attenuated_below_25pct_reference"] = (
        out["status"].eq("success")
        & out["reference_delta"].notna()
        & ref_abs.gt(0.05)
        & out["median_delta"].abs().le(0.25 * ref_abs)
    )
    out["effect_relation_to_reference"] = "not_evaluable"
    out.loc[out["status"].eq("success") & out["is_reference"].eq(True), "effect_relation_to_reference"] = "reference"
    out.loc[
        out["status"].eq("success") & out["opposite_sign_vs_reference"],
        "effect_relation_to_reference",
    ] = "opposite_sign_point_estimate"
    out.loc[
        out["status"].eq("success") & out["interval_supported_opposite_sign_vs_reference"],
        "effect_relation_to_reference",
    ] = "opposite_sign_interval_supported"
    out.loc[
        out["status"].eq("success")
        & ~out["opposite_sign_vs_reference"]
        & out["attenuated_below_25pct_reference"]
        & ~out["is_reference"],
        "effect_relation_to_reference",
    ] = "attenuated_below_25pct_reference"
    out.loc[
        out["status"].eq("success")
        & ~out["is_reference"]
        & out["effect_relation_to_reference"].eq("not_evaluable"),
        "effect_relation_to_reference",
    ] = "same_direction_or_zero"
    return out


def build_table(
    out_path: Path,
    *,
    bootstrap: int = 1000,
    include_official: bool = False,
    map_files: list[Path] | None = None,
    analysis_scope: str = "custom_input",
    compute_pvalue: bool = False,
) -> Path:
    rows: list[dict[str, object]] = []
    cache: dict[tuple[str, str], object] = {}
    if map_files is None:
        map_files = FULL_COHORT_MAPS
        analysis_scope = "full_cohort_matched_input"
    for path in map_files:
        if path.exists():
            _process_table(
                _read_map(path),
                analysis_scope=analysis_scope,
                cache=cache,
                min_domain_spots=30,
                min_group_spots=5,
                bootstrap=bootstrap,
                compute_pvalue=compute_pvalue,
                rows=rows,
            )
    if include_official:
        for path in OFFICIAL_TARGETED_MAPS:
            if path.exists():
                _process_table(
                    _read_map(path),
                    analysis_scope="targeted_official_implementation",
                    cache=cache,
                    min_domain_spots=30,
                    min_group_spots=5,
                    bootstrap=bootstrap,
                    compute_pvalue=compute_pvalue,
                    rows=rows,
                )

    table = _add_reference_comparisons(pd.DataFrame(rows))
    sort_cols = [
        "analysis_scope",
        "method_id",
        "dataset_id",
        "sample_id",
        "K",
        "contrast_id",
        "feature_id",
        "partition_id",
    ]
    existing_sort = [col for col in sort_cols if col in table.columns]
    table = table.sort_values(existing_sort, na_position="last")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, sep="\t", index=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(ROOT / "results" / "r3_cross_method" / "tumor_stroma_proxy_interface.tsv"),
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--map-file",
        action="append",
        default=None,
        help="Explicit partition-map TSV to process. May be supplied more than once.",
    )
    parser.add_argument("--analysis-scope", default="custom_input")
    parser.add_argument(
        "--compute-pvalue",
        action="store_true",
        help="Compute spot-level Mann-Whitney p-values. Disabled by default to avoid pseudo-replication-heavy slow tests.",
    )
    parser.add_argument(
        "--include-official",
        action="store_true",
        help="Also append targeted official-implementation partitions when available.",
    )
    args = parser.parse_args()
    path = build_table(
        Path(args.out),
        bootstrap=int(args.bootstrap),
        include_official=bool(args.include_official),
        map_files=[Path(p) for p in args.map_file] if args.map_file else None,
        analysis_scope=str(args.analysis_scope),
        compute_pvalue=bool(args.compute_pvalue),
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
