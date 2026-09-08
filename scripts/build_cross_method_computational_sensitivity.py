#!/usr/bin/env python3
"""Consolidate method-relevant computational sensitivity evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from build_cross_method_boundary_evidence import METHOD_LABELS, _align_labels, _read_map


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "cross_method_boundary"


def _append(
    rows: list[dict[str, object]],
    *,
    dataset_id: str,
    sample_id: str,
    method_id: str,
    k_value: int,
    factor: str,
    reference: str,
    alternative: str,
    metric: str,
    value: float,
    scope: str,
    source: str,
    status: str = "success",
) -> None:
    rows.append(
        {
            "dataset_id": dataset_id,
            "sample_id": sample_id,
            "method_id": method_id,
            "method_label": METHOD_LABELS.get(method_id, method_id),
            "K": int(k_value),
            "sensitivity_factor": factor,
            "reference_setting": reference,
            "alternative_setting": alternative,
            "metric_name": metric,
            "metric_value": value,
            "status": status,
            "scope": scope,
            "source_file": source,
        }
    )


def _main_partition_sensitivity(results_dir: Path, rows: list[dict[str, object]]) -> None:
    path = results_dir / "stability_pairwise.tsv"
    table = pd.read_csv(path, sep="\t")
    for row in table.itertuples(index=False):
        factor = "graph_neighbors" if row.stability_type == "graph_specification_ari" else "random_seed"
        _append(
            rows,
            dataset_id=str(row.dataset_id),
            sample_id=str(row.sample_id),
            method_id=str(row.method_id),
            k_value=int(row.K),
            factor=factor,
            reference=str(row.reference_id),
            alternative=str(row.alternative_id),
            metric="ARI_to_reference_partition",
            value=float(row.ari),
            scope="all 13 CRC sections",
            source=path.name,
        )


def _k_sensitivity(results_dir: Path, rows: list[dict[str, object]]) -> None:
    path = results_dir / "global_characterization.tsv"
    table = pd.read_csv(path, sep="\t")
    metrics = ["spatial_coherence_median", "marker_coherence_median", "stability_ari_median"]
    for keys, group in table.groupby(["dataset_id", "sample_id", "method_id"], sort=True):
        dataset_id, sample_id, method_id = map(str, keys)
        by_k = group.set_index("K")
        if 4 not in by_k.index or 6 not in by_k.index:
            continue
        for metric in metrics:
            reference = by_k.at[4, metric]
            alternative = by_k.at[6, metric]
            if not np.isfinite(reference) or not np.isfinite(alternative):
                continue
            _append(
                rows,
                dataset_id=dataset_id,
                sample_id=sample_id,
                method_id=method_id,
                k_value=6,
                factor="domain_resolution_K",
                reference="K4",
                alternative="K6",
                metric=f"delta_{metric}_K6_minus_K4",
                value=float(alternative - reference),
                scope="all available CRC sections",
                source=path.name,
            )


def _leiden_graph_sensitivity(results_dir: Path, rows: list[dict[str, object]]) -> None:
    paths = sorted((results_dir / "sensitivity").glob("*_leiden_graph.tsv"))
    for path in paths:
        raw = pd.read_csv(path, sep="\t", low_memory=False)
        for keys, failed in raw[~raw["status"].astype(str).eq("success")].groupby(
            ["dataset_id", "sample_id", "method_id", "K", "replicate_id"], sort=True
        ):
            dataset_id, sample_id, method_id, k_value, replicate_id = keys
            _append(
                rows,
                dataset_id=str(dataset_id),
                sample_id=str(sample_id),
                method_id=str(method_id),
                k_value=int(k_value),
                factor="graph_neighbors",
                reference="neighbors_6",
                alternative=str(replicate_id),
                metric="ARI_to_reference_partition",
                value=float("nan"),
                scope="all 13 CRC sections",
                source=path.name,
                status="not_evaluable_nonexact_K",
            )
        table = _read_map(path)
        for keys, group in table.groupby(["dataset_id", "sample_id", "method_id", "K"], sort=True):
            dataset_id, sample_id, method_id, k_value = keys
            maps: dict[str, pd.DataFrame] = {}
            for replicate_id, replicate in group.groupby("replicate_id"):
                replicate = replicate.copy()
                replicate["barcode"] = replicate["barcode"].astype(str)
                maps[str(replicate_id)] = replicate.set_index("barcode")
            if "neighbors_6" not in maps:
                continue
            reference = maps["neighbors_6"]
            for alternative_id in ("neighbors_4", "neighbors_8"):
                if alternative_id not in maps:
                    continue
                common = reference.index.intersection(maps[alternative_id].index)
                ref_labels = reference.loc[common, "domain_label"].to_numpy(dtype=int)
                alt_labels = maps[alternative_id].loc[common, "domain_label"].to_numpy(dtype=int)
                _align_labels(ref_labels, alt_labels)
                ari = float(adjusted_rand_score(ref_labels, alt_labels))
                _append(
                    rows,
                    dataset_id=str(dataset_id),
                    sample_id=str(sample_id),
                    method_id=str(method_id),
                    k_value=int(k_value),
                    factor="graph_neighbors",
                    reference="neighbors_6",
                    alternative=alternative_id,
                    metric="ARI_to_reference_partition",
                    value=ari,
                    scope="all 13 CRC sections",
                    source=path.name,
                )


def _bayesspace_depth(rows: list[dict[str, object]]) -> None:
    sensitivity_path = ROOT / "results" / "benchmarks" / "bayesspace_nrep1000_sensitivity.tsv"
    main_path = ROOT / "results" / "benchmarks" / "method_benchmark_locked.tsv"
    sensitivity = pd.read_csv(sensitivity_path, sep="\t")
    main = pd.read_csv(main_path, sep="\t")
    main = main[main["method_id"].eq("BayesSpace")]
    metrics = [
        "spatial_coherence_median",
        "marker_coherence_median",
        "stability_ari_median",
        "wall_time_sec_median",
    ]
    merged = sensitivity.merge(
        main[["dataset_id", "sample_id", "method_id", "K"] + metrics],
        on=["dataset_id", "sample_id", "method_id", "K"],
        how="left",
        suffixes=("_nrep1000", "_nrep100"),
    )
    for row in merged.itertuples(index=False):
        for metric in metrics:
            main_value = float(getattr(row, f"{metric}_nrep100"))
            sensitivity_value = float(getattr(row, f"{metric}_nrep1000"))
            _append(
                rows,
                dataset_id=str(row.dataset_id),
                sample_id=str(row.sample_id),
                method_id="BayesSpace",
                k_value=int(row.K),
                factor="MCMC_iterations",
                reference="nrep_100",
                alternative="nrep_1000",
                metric=f"delta_{metric}_nrep1000_minus_nrep100",
                value=sensitivity_value - main_value,
                scope="two preselected GSE311294 sections",
                source=sensitivity_path.name,
            )


def _deep_hyperparameters(rows: list[dict[str, object]]) -> None:
    path = ROOT / "results" / "benchmarks" / "baseline_hyperparam_sensitivity.tsv"
    table = pd.read_csv(path, sep="\t")
    metrics = [
        "spatial_coherence_median",
        "marker_coherence_median",
        "stability_ari_median",
        "wall_time_sec_median",
    ]
    for row in table.itertuples(index=False):
        method_id = str(row.method_id)
        if method_id == "M4_spagcn":
            alternative = f"p={row.spagcn_p}"
            factor = "spatial_weight_p"
        else:
            alternative = f"latent={int(row.stagate_latent_dim)};epochs={int(row.stagate_max_epochs)}"
            factor = "latent_dimension_and_epochs"
        for metric in metrics:
            value = getattr(row, metric)
            if not np.isfinite(value):
                continue
            _append(
                rows,
                dataset_id=str(row.dataset_id),
                sample_id=str(row.sample_id),
                method_id=method_id,
                k_value=int(row.K),
                factor=factor,
                reference="predefined sensitivity grid",
                alternative=alternative,
                metric=metric,
                value=float(value),
                scope="two representative CRC sections",
                source=path.name,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    rows: list[dict[str, object]] = []
    _main_partition_sensitivity(results_dir, rows)
    _k_sensitivity(results_dir, rows)
    _leiden_graph_sensitivity(results_dir, rows)
    _bayesspace_depth(rows)
    _deep_hyperparameters(rows)
    output = pd.DataFrame(rows).sort_values(
        ["method_id", "sensitivity_factor", "dataset_id", "sample_id", "K", "metric_name", "alternative_setting"]
    )
    output.to_csv(results_dir / "computational_sensitivity.tsv", sep="\t", index=False)

    ari = output[output["metric_name"].eq("ARI_to_reference_partition")].copy()
    summary = (
        ari.groupby(["method_id", "method_label", "sensitivity_factor"], as_index=False)
        .agg(
            n_comparisons=("metric_value", "size"),
            median_ari=("metric_value", "median"),
            q1_ari=("metric_value", lambda values: values.quantile(0.25)),
            q3_ari=("metric_value", lambda values: values.quantile(0.75)),
            fraction_below_0_60=("metric_value", lambda values: float(np.mean(np.asarray(values) < 0.60))),
        )
    )
    summary.to_csv(results_dir / "computational_sensitivity_summary.tsv", sep="\t", index=False)
    print(f"Wrote {len(output)} sensitivity rows under {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
