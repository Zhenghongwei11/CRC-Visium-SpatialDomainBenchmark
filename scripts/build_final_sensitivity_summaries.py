#!/usr/bin/env python3
"""Build final-release S14/S15 sensitivity summaries from released TSV inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / "results" / "cross_method_boundary"
OFFICIAL = ROOT / "results" / "official_sensitivity"


def build_targeted_official_sensitivity_table() -> None:
    stability = pd.read_csv(OFFICIAL / "evidence_all" / "stability_summary.tsv", sep="\t")
    switching = pd.read_csv(OFFICIAL / "evidence_all" / "switching_boundary_relationship.tsv", sep="\t")
    downstream = pd.read_csv(OFFICIAL / "evidence_all" / "downstream_robustness_summary.tsv", sep="\t")
    comparison = pd.read_csv(OFFICIAL / "comparison_all" / "downstream_sign_comparison.tsv", sep="\t")

    keys = ["dataset_id", "sample_id", "method_id", "K"]
    switching_summary = (
        switching.assign(rr_above_one=switching["risk_ratio"].gt(1))
        .groupby(keys, as_index=False)
        .agg(
            pairwise_median_switching_fraction=("switching_fraction", "median"),
            pairwise_median_fraction_switching_in_boundary=("fraction_switching_in_union_boundary", "median"),
            pairwise_median_risk_ratio=("risk_ratio", "median"),
            pairwise_fraction_rr_above_one=("rr_above_one", "mean"),
        )
    )
    downstream_summary = (
        downstream.groupby(keys, as_index=False)
        .agg(
            n_feature_units=("feature_id", "size"),
            fraction_with_direction_reversal=("any_direction_reversal_vs_reference", "mean"),
            fraction_with_sign_state_change=("any_sign_state_change_vs_reference", "mean"),
            fraction_with_block_direction_reversal=("any_block_direction_reversal_vs_reference", "mean"),
            median_effect_range=("delta_range", "median"),
            fraction_all_partitions_same_direction=("all_partitions_same_direction", "mean"),
        )
    )
    comparison_summary = (
        comparison.groupby(["dataset_id", "sample_id", "official_method_id", "K"], as_index=False)
        .agg(
            current_method_id=("current_method_id", "first"),
            n_comparison_feature_units=("feature_id", "size"),
            fraction_reference_delta_sign_agrees=("reference_delta_sign_agrees", "mean"),
            official_fraction_direction_reversal=("official_any_direction_reversal_vs_reference", "mean"),
            current_fraction_direction_reversal=("current_any_direction_reversal_vs_reference", "mean"),
            official_fraction_sign_state_change=("official_any_sign_state_change_vs_reference", "mean"),
            current_fraction_sign_state_change=("current_any_sign_state_change_vs_reference", "mean"),
        )
    )

    table = (
        stability.merge(switching_summary, on=keys, how="left")
        .merge(downstream_summary, on=keys, how="left")
        .merge(
            comparison_summary,
            left_on=["dataset_id", "sample_id", "method_id", "K"],
            right_on=["dataset_id", "sample_id", "official_method_id", "K"],
            how="left",
        )
        .drop(columns=["official_method_id"])
    )
    method_scope = {
        "Official_BayesSpace_nrep1000": "Bioconductor BayesSpace; nrep=1000; fixed K=4/K=6; seeds 11, 23, 37.",
        "Official_STAGATE_pyG": "STAGATE-linked PyG implementation; KNN spatial graph; 1000 epochs; KMeans fixed-K clustering of learned embedding.",
        "Official_SpaGCN_v1_2_7": "Official SpaGCN 1.2.7 SpaGCN class; histology graph; fixed-K k-means initialization.",
    }
    table.insert(
        table.columns.get_loc("method_label") + 1,
        "targeted_run_scope",
        table["method_id"].map(method_scope),
    )
    table["representative_slice_set"] = "TR11_206, CTC21P, TR11_18105"
    table["execution_environment"] = "Colab T4 GPU where needed; no heavy graph-neural-network or high-nrep BayesSpace training was run locally."
    table["interpretive_scope"] = "Targeted sensitivity check only; not a replacement for the full 13-section matched-input analysis."

    ordered = [
        "dataset_id",
        "sample_id",
        "K",
        "method_id",
        "method_label",
        "targeted_run_scope",
        "n_alternatives",
        "median_ari",
        "min_ari",
        "max_ari",
        "n_below_0_60",
        "n_union_switching",
        "union_switching_fraction",
        "pairwise_median_switching_fraction",
        "pairwise_median_fraction_switching_in_boundary",
        "pairwise_median_risk_ratio",
        "pairwise_fraction_rr_above_one",
        "n_feature_units",
        "fraction_with_direction_reversal",
        "fraction_with_sign_state_change",
        "fraction_with_block_direction_reversal",
        "median_effect_range",
        "fraction_all_partitions_same_direction",
        "current_method_id",
        "n_comparison_feature_units",
        "fraction_reference_delta_sign_agrees",
        "official_fraction_direction_reversal",
        "current_fraction_direction_reversal",
        "official_fraction_sign_state_change",
        "current_fraction_sign_state_change",
        "representative_slice_set",
        "execution_environment",
        "interpretive_scope",
    ]
    table = table[ordered].sort_values(["method_id", "dataset_id", "sample_id", "K"])
    table.to_csv(CROSS / "targeted_official_sensitivity.tsv", sep="\t", index=False)


def build_stability_stratified_downstream_table() -> None:
    s9 = pd.read_csv(CROSS / "downstream_opposite_sign_evidence.tsv", sep="\t")
    s4 = pd.read_csv(CROSS / "stability_summary.tsv", sep="\t")
    keys = ["dataset_id", "sample_id", "method_id", "K"]
    df = s9.merge(s4[keys + ["median_ari"]], on=keys, how="left")
    df["stability_band"] = pd.cut(
        df["median_ari"],
        bins=[-np.inf, 0.60, 0.80, np.inf],
        labels=["ARI_lt_0_60", "ARI_0_60_to_0_80", "ARI_gt_0_80"],
        right=False,
    )
    df["stability_band_note"] = df["stability_band"].map(
        {
            "ARI_lt_0_60": "Below the author-defined descriptive stability screen.",
            "ARI_0_60_to_0_80": "Above the screen but not a high-stability setting.",
            "ARI_gt_0_80": "High-stability setting in this descriptive stratification.",
        }
    ).astype(str)
    group_cols = ["method_id", "method_label", "stability_band", "stability_band_note"]
    by_method = (
        df.groupby(group_cols, observed=True)
        .agg(
            n_feature_settings=("feature_id", "size"),
            n_distinct_section_k_settings=("sample_id", lambda x: int(df.loc[x.index, keys].drop_duplicates().shape[0])),
            median_ari=("median_ari", "median"),
            n_opposite_sign_point_estimates=("any_opposite_sign_point_estimate", "sum"),
            fraction_opposite_sign_point_estimates=("any_opposite_sign_point_estimate", "mean"),
            n_interval_supported_opposite_sign=("any_interval_supported_opposite_sign_estimate", "sum"),
            fraction_interval_supported_opposite_sign=("any_interval_supported_opposite_sign_estimate", "mean"),
        )
        .reset_index()
    )
    combined = (
        df.groupby(["stability_band", "stability_band_note"], observed=True)
        .agg(
            n_feature_settings=("feature_id", "size"),
            n_distinct_section_k_settings=("sample_id", lambda x: int(df.loc[x.index, keys].drop_duplicates().shape[0])),
            median_ari=("median_ari", "median"),
            n_opposite_sign_point_estimates=("any_opposite_sign_point_estimate", "sum"),
            fraction_opposite_sign_point_estimates=("any_opposite_sign_point_estimate", "mean"),
            n_interval_supported_opposite_sign=("any_interval_supported_opposite_sign_estimate", "sum"),
            fraction_interval_supported_opposite_sign=("any_interval_supported_opposite_sign_estimate", "mean"),
        )
        .reset_index()
    )
    combined.insert(0, "method_id", "All_methods_combined")
    combined.insert(1, "method_label", "All methods combined")
    table = pd.concat([combined, by_method], ignore_index=True)
    table["interpretation_note"] = (
        "Post-review descriptive stratification; ARI bands do not define validity thresholds and spot-bootstrap intervals do not resolve spatial dependence."
    )
    table.to_csv(CROSS / "stability_stratified_downstream_sensitivity.tsv", sep="\t", index=False)


def main() -> None:
    build_targeted_official_sensitivity_table()
    build_stability_stratified_downstream_table()


if __name__ == "__main__":
    main()
