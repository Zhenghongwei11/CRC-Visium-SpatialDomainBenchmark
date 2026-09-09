#!/usr/bin/env python3
"""Compare targeted official Colab sensitivity summaries with current R3 evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL = ROOT / "results" / "colab_official_sensitivity" / "evidence"
DEFAULT_CURRENT = ROOT / "results" / "r3_cross_method"
DEFAULT_OUT = ROOT / "results" / "colab_official_sensitivity" / "comparison"

METHOD_MAP = {
    "Official_SpaGCN_v1_2_7": "M4_spagcn",
    "Official_STAGATE_pyG": "M5_stagate",
    "Official_BayesSpace_nrep1000": "BayesSpace",
}


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", low_memory=False)


def compare_stability(official_dir: Path, current_dir: Path, out_dir: Path) -> None:
    official = read_tsv(official_dir / "stability_summary.tsv")
    current = read_tsv(current_dir / "stability_summary.tsv")
    if official.empty or current.empty:
        return
    official = official[official["method_id"].isin(METHOD_MAP)].copy()
    official["current_method_id"] = official["method_id"].map(METHOD_MAP)
    current = current.rename(
        columns={
            "method_id": "current_method_id",
            "median_ari": "current_median_ari",
            "min_ari": "current_min_ari",
            "max_ari": "current_max_ari",
            "union_switching_fraction": "current_union_switching_fraction",
        }
    )
    official = official.rename(
        columns={
            "method_id": "official_method_id",
            "median_ari": "official_median_ari",
            "min_ari": "official_min_ari",
            "max_ari": "official_max_ari",
            "union_switching_fraction": "official_union_switching_fraction",
        }
    )
    cols = [
        "dataset_id",
        "sample_id",
        "current_method_id",
        "K",
        "current_median_ari",
        "current_min_ari",
        "current_max_ari",
        "current_union_switching_fraction",
    ]
    merged = official.merge(current[cols], on=["dataset_id", "sample_id", "current_method_id", "K"], how="left")
    merged["delta_median_ari_official_minus_current"] = merged["official_median_ari"] - merged["current_median_ari"]
    merged["delta_union_switching_fraction_official_minus_current"] = (
        merged["official_union_switching_fraction"] - merged["current_union_switching_fraction"]
    )
    merged.sort_values(["official_method_id", "dataset_id", "sample_id", "K"]).to_csv(
        out_dir / "stability_comparison.tsv", sep="\t", index=False
    )


def compare_switching(official_dir: Path, current_dir: Path, out_dir: Path) -> None:
    official = read_tsv(official_dir / "switching_boundary_relationship.tsv")
    current = read_tsv(current_dir / "switching_boundary_relationship.tsv")
    if official.empty or current.empty:
        return
    group_cols = ["dataset_id", "sample_id", "method_id", "K"]
    metrics = {
        "switching_fraction": "median",
        "fraction_switching_in_union_boundary": "median",
        "risk_ratio": "median",
        "jaccard_switching_union_boundary": "median",
    }
    off = official[official["method_id"].isin(METHOD_MAP)].groupby(group_cols, as_index=False).agg(metrics)
    cur = current.groupby(group_cols, as_index=False).agg(metrics)
    off["current_method_id"] = off["method_id"].map(METHOD_MAP)
    off = off.rename(columns={c: f"official_{c}" for c in metrics})
    off = off.rename(columns={"method_id": "official_method_id"})
    cur = cur.rename(columns={c: f"current_{c}" for c in metrics})
    cur = cur.rename(columns={"method_id": "current_method_id"})
    merged = off.merge(cur, on=["dataset_id", "sample_id", "current_method_id", "K"], how="left")
    for c in metrics:
        merged[f"delta_{c}_official_minus_current"] = merged[f"official_{c}"] - merged[f"current_{c}"]
    merged.sort_values(["official_method_id", "dataset_id", "sample_id", "K"]).to_csv(
        out_dir / "switching_comparison.tsv", sep="\t", index=False
    )


def compare_downstream(official_dir: Path, current_dir: Path, out_dir: Path) -> None:
    official = read_tsv(official_dir / "downstream_robustness_summary.tsv")
    current = read_tsv(current_dir / "downstream_robustness_summary.tsv")
    if official.empty or current.empty:
        return
    keys = ["dataset_id", "sample_id", "method_id", "K", "feature_type", "feature_id"]
    off = official[official["method_id"].isin(METHOD_MAP)].copy()
    off["current_method_id"] = off["method_id"].map(METHOD_MAP)
    off = off.rename(
        columns={
            "method_id": "official_method_id",
            "reference_delta": "official_reference_delta",
            "delta_range": "official_delta_range",
            "any_direction_reversal_vs_reference": "official_any_direction_reversal_vs_reference",
            "any_sign_state_change_vs_reference": "official_any_sign_state_change_vs_reference",
        }
    )
    cur = current.rename(
        columns={
            "method_id": "current_method_id",
            "reference_delta": "current_reference_delta",
            "delta_range": "current_delta_range",
            "any_direction_reversal_vs_reference": "current_any_direction_reversal_vs_reference",
            "any_sign_state_change_vs_reference": "current_any_sign_state_change_vs_reference",
        }
    )
    keep = [
        "dataset_id",
        "sample_id",
        "current_method_id",
        "K",
        "feature_type",
        "feature_id",
        "current_reference_delta",
        "current_delta_range",
        "current_any_direction_reversal_vs_reference",
        "current_any_sign_state_change_vs_reference",
    ]
    merged = off.merge(cur[keep], on=["dataset_id", "sample_id", "current_method_id", "K", "feature_type", "feature_id"], how="left")
    merged["reference_delta_sign_agrees"] = (
        merged["official_reference_delta"].astype(float).gt(0) == merged["current_reference_delta"].astype(float).gt(0)
    )
    merged.sort_values(["official_method_id", "dataset_id", "sample_id", "K", "feature_id"]).to_csv(
        out_dir / "downstream_sign_comparison.tsv", sep="\t", index=False
    )
    summary = (
        merged.groupby(["official_method_id", "current_method_id"], as_index=False)
        .agg(
            n_feature_units=("feature_id", "size"),
            fraction_reference_delta_sign_agrees=("reference_delta_sign_agrees", "mean"),
            official_fraction_direction_reversal=("official_any_direction_reversal_vs_reference", "mean"),
            current_fraction_direction_reversal=("current_any_direction_reversal_vs_reference", "mean"),
            official_fraction_sign_state_change=("official_any_sign_state_change_vs_reference", "mean"),
            current_fraction_sign_state_change=("current_any_sign_state_change_vs_reference", "mean"),
        )
    )
    summary.to_csv(out_dir / "downstream_sign_summary.tsv", sep="\t", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", default=str(DEFAULT_OFFICIAL))
    parser.add_argument("--current-dir", default=str(DEFAULT_CURRENT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    official_dir = Path(args.official_dir)
    current_dir = Path(args.current_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_stability(official_dir, current_dir, out_dir)
    compare_switching(official_dir, current_dir, out_dir)
    compare_downstream(official_dir, current_dir, out_dir)
    print(f"Wrote comparison tables under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
