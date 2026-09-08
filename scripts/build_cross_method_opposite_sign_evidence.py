#!/usr/bin/env python3
"""Stratify downstream opposite-sign estimates by spot-bootstrap interval support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "cross_method_boundary"
UNIT_KEYS = [
    "dataset_id",
    "sample_id",
    "method_id",
    "method_label",
    "K",
    "feature_type",
    "feature_id",
]


def interval_excludes_zero_in_effect_direction(row: pd.Series) -> bool:
    effect = float(row["median_delta_boundary_minus_interior"])
    lower = float(row["bootstrap_ci_lower"])
    upper = float(row["bootstrap_ci_upper"])
    return bool((effect > 0 and lower > 0) or (effect < 0 and upper < 0))


def build_unit_table() -> pd.DataFrame:
    robustness = pd.read_csv(DATA / "downstream_robustness_summary.tsv", sep="\t")
    contrasts = pd.read_csv(DATA / "downstream_contrasts.tsv", sep="\t")
    contrast_groups = {
        key: group.copy()
        for key, group in contrasts.groupby(UNIT_KEYS, sort=False, dropna=False)
    }

    rows: list[dict[str, object]] = []
    for _, unit in robustness.iterrows():
        key = tuple(unit[column] for column in UNIT_KEYS)
        group = contrast_groups[key]
        reference = group.loc[group["is_reference"].eq(True)]
        if len(reference) != 1:
            raise AssertionError(f"Expected one reference partition for {key}; found {len(reference)}")
        reference = reference.iloc[0]
        reference_effect = float(reference["median_delta_boundary_minus_interior"])
        alternatives = group.loc[group["is_reference"].eq(False)].copy()
        alternative_effects = alternatives["median_delta_boundary_minus_interior"]
        reversing = alternatives.loc[
            (reference_effect != 0)
            & (alternative_effects != 0)
            & (np.sign(alternative_effects) == -np.sign(reference_effect))
        ].copy()

        reference_supported = interval_excludes_zero_in_effect_direction(reference)
        supported_ids = [
            str(row["partition_id"])
            for _, row in reversing.iterrows()
            if interval_excludes_zero_in_effect_direction(row)
        ]
        reversing_ids = reversing["partition_id"].astype(str).tolist()
        alternative_interval_includes_zero = [
            not interval_excludes_zero_in_effect_direction(row)
            for _, row in reversing.iterrows()
        ]
        any_point_opposite = bool(reversing_ids)
        any_interval_supported = bool(reference_supported and supported_ids)
        both_sides_include_zero = bool(
            any_point_opposite
            and not reference_supported
            and any(alternative_interval_includes_zero)
        )

        if not any_point_opposite:
            evidence_class = "no opposite-sign point estimate"
        elif any_interval_supported:
            evidence_class = "reference and alternative intervals exclude zero"
        elif not reference_supported and any(alternative_interval_includes_zero):
            evidence_class = "reference and alternative intervals include zero"
        elif not reference_supported:
            evidence_class = "reference interval includes zero"
        else:
            evidence_class = "alternative interval includes zero"

        rows.append(
            {
                **{column: unit[column] for column in UNIT_KEYS},
                "reference_id": reference["partition_id"],
                "reference_delta": reference_effect,
                "reference_ci_lower": reference["bootstrap_ci_lower"],
                "reference_ci_upper": reference["bootstrap_ci_upper"],
                "reference_interval_excludes_zero": reference_supported,
                "n_alternatives": int(len(alternatives)),
                "n_opposite_sign_alternatives": int(len(reversing)),
                "opposite_sign_alternative_ids": ";".join(reversing_ids),
                "n_interval_supported_opposite_sign_alternatives": int(len(supported_ids))
                if reference_supported
                else 0,
                "interval_supported_opposite_sign_alternative_ids": ";".join(supported_ids)
                if reference_supported
                else "",
                "any_opposite_sign_point_estimate": any_point_opposite,
                "any_interval_supported_opposite_sign_estimate": any_interval_supported,
                "reference_interval_includes_zero_among_point_changes": bool(
                    any_point_opposite and not reference_supported
                ),
                "reference_and_alternative_intervals_include_zero": both_sides_include_zero,
                "evidence_class": evidence_class,
                "interpretation_note": (
                    "Post-review descriptive stratification based on spot-bootstrap intervals; "
                    "it is not an independent-sample or spatial-dependence-resolved test."
                ),
            }
        )

    result = pd.DataFrame(rows)
    if len(result) != 1947:
        raise AssertionError(f"Expected 1,947 feature-settings; found {len(result)}")
    checks = {
        "point-estimate opposite signs": int(result["any_opposite_sign_point_estimate"].sum()),
        "interval-supported opposite signs": int(
            result["any_interval_supported_opposite_sign_estimate"].sum()
        ),
        "reference interval includes zero": int(
            result["reference_interval_includes_zero_among_point_changes"].sum()
        ),
        "both intervals include zero": int(
            result["reference_and_alternative_intervals_include_zero"].sum()
        ),
    }
    expected = {
        "point-estimate opposite signs": 176,
        "interval-supported opposite signs": 38,
        "reference interval includes zero": 106,
        "both intervals include zero": 77,
    }
    if checks != expected:
        raise AssertionError(f"Unexpected downstream evidence counts: {checks}")
    return result


def build_method_summary(unit_table: pd.DataFrame) -> pd.DataFrame:
    effect_summary = pd.read_csv(DATA / "downstream_method_summary.tsv", sep="\t").set_index(
        "method_id"
    )
    rows: list[dict[str, object]] = []
    for (method_id, method_label), group in unit_table.groupby(
        ["method_id", "method_label"], sort=False
    ):
        point = group["any_opposite_sign_point_estimate"]
        supported = group["any_interval_supported_opposite_sign_estimate"]
        effect = effect_summary.loc[method_id]
        rows.append(
            {
                "method_id": method_id,
                "method_label": method_label,
                "n_feature_settings": int(len(group)),
                "n_with_opposite_sign_point_estimate": int(point.sum()),
                "fraction_with_opposite_sign_point_estimate": float(point.mean()),
                "n_with_interval_supported_opposite_sign_estimate": int(supported.sum()),
                "fraction_with_interval_supported_opposite_sign_estimate": float(supported.mean()),
                "fraction_interval_supported_among_point_changes": float(
                    supported.sum() / point.sum()
                )
                if point.sum()
                else 0.0,
                "n_point_changes_with_reference_interval_including_zero": int(
                    group["reference_interval_includes_zero_among_point_changes"].sum()
                ),
                "n_point_changes_with_reference_and_alternative_intervals_including_zero": int(
                    group["reference_and_alternative_intervals_include_zero"].sum()
                ),
                "fraction_with_sign_state_change": effect["fraction_with_sign_state_change"],
                "fraction_with_block_direction_change": effect[
                    "fraction_with_block_direction_reversal"
                ],
                "median_effect_range": effect["median_effect_range"],
                "fraction_all_partitions_same_direction": effect[
                    "fraction_all_partitions_same_direction"
                ],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    unit_table = build_unit_table()
    method_summary = build_method_summary(unit_table)
    unit_table.to_csv(DATA / "downstream_opposite_sign_evidence.tsv", sep="\t", index=False)
    method_summary.to_csv(
        DATA / "downstream_opposite_sign_method_summary.tsv", sep="\t", index=False
    )
    print(
        "Built downstream opposite-sign evidence: "
        f"{len(unit_table)} units, "
        f"{int(unit_table['any_opposite_sign_point_estimate'].sum())} point changes, "
        f"{int(unit_table['any_interval_supported_opposite_sign_estimate'].sum())} interval-supported"
    )


if __name__ == "__main__":
    main()
