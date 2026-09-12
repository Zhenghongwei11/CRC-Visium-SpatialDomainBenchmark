#!/usr/bin/env python3
"""Build the JBCB R3 cross-method boundary-reliability evidence tables."""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from build_histology_edge_alignment import load_detected_tissue_image_grad, sample_grad_at_points
from build_jbcb_robustness_tables import _feature_vectors
from build_figS3_boundary_signatures import _load_flat_visium_sample


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP_DIR = ROOT / "results" / "r3_cross_method" / "domain_maps"
DEFAULT_OUT_DIR = ROOT / "results" / "r3_cross_method"

METHOD_ORDER = [
    "BayesSpace",
    "M5_stagate",
    "M4_spagcn",
    "M3_spatial_leiden",
    "M2_spatial_ward",
    "M0_expr_kmeans",
    "M1_spatial_concat_kmeans",
]
METHOD_LABELS = {
    "BayesSpace": "BayesSpace",
    "M5_stagate": "STAGATE-style",
    "M4_spagcn": "SpaGCN-style",
    "M3_spatial_leiden": "spatial Leiden",
    "M2_spatial_ward": "spatial Ward",
    "M0_expr_kmeans": "expression-only k-means",
    "M1_spatial_concat_kmeans": "coordinate-augmented k-means",
}
DATASET_ROOTS = {
    "GSE267401": ROOT / "data" / "raw" / "GSE267401" / "extracted",
    "GSE311294": ROOT / "data" / "raw" / "GSE311294" / "extracted",
    "GSE285505": ROOT / "data" / "raw" / "GSE285505" / "extracted",
}


@dataclass(frozen=True)
class SampleData:
    barcodes: list[str]
    coords: np.ndarray
    features: dict[str, tuple[str, str, np.ndarray]]


def _bh_fdr(values: list[float]) -> list[float]:
    if not values:
        return []
    p = np.asarray(values, dtype=float)
    valid = np.isfinite(p)
    out = np.full(p.shape, np.nan, dtype=float)
    if not np.any(valid):
        return out.tolist()
    idx = np.flatnonzero(valid)
    order = idx[np.argsort(p[idx])]
    adjusted = np.empty(order.size, dtype=float)
    previous = 1.0
    for reverse_rank in range(order.size - 1, -1, -1):
        original_index = int(order[reverse_rank])
        rank = reverse_rank + 1
        value = min(previous, p[original_index] * order.size / rank)
        adjusted[reverse_rank] = value
        previous = value
    for position, original_index in enumerate(order):
        out[int(original_index)] = adjusted[position]
    return out.tolist()


def _read_map(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", low_memory=False)
    required = {"dataset_id", "sample_id", "method_id", "K", "barcode", "x", "y", "domain_label"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    table["dataset_id"] = table["dataset_id"].replace({"GSE280318": "GSE285505"})
    table = table[table["dataset_id"].isin(DATASET_ROOTS)].copy()
    table["K"] = table["K"].astype(int)
    table["domain_label"] = table["domain_label"].astype(int)
    if "replicate_type" not in table.columns:
        table["replicate_type"] = "seed"
    if "replicate_id" not in table.columns:
        if "seed" not in table.columns:
            raise ValueError(f"{path} has neither replicate_id nor seed")
        table["replicate_id"] = table["seed"].map(lambda value: f"seed_{int(value)}")
    if "config_id" not in table.columns:
        table["config_id"] = table.apply(
            lambda row: f"K{int(row['K'])}_{row['replicate_id']}", axis=1
        )
    if "status" not in table.columns:
        table["status"] = "success"
    return table[table["status"].astype(str).eq("success")].copy()


def _reference_id(method_id: str) -> str:
    return "neighbors_6" if method_id == "M2_spatial_ward" else "seed_11"


def _align_labels(reference: np.ndarray, alternative: np.ndarray) -> np.ndarray:
    reference_values = np.unique(reference)
    alternative_values = np.unique(alternative)
    contingency = np.zeros((alternative_values.size, reference_values.size), dtype=np.int64)
    for i, alt_label in enumerate(alternative_values):
        mask = alternative == alt_label
        for j, ref_label in enumerate(reference_values):
            contingency[i, j] = int(np.sum(mask & (reference == ref_label)))
    rows, cols = linear_sum_assignment(-contingency)
    mapping = {int(alternative_values[i]): int(reference_values[j]) for i, j in zip(rows, cols, strict=True)}
    return np.asarray([mapping.get(int(value), int(value)) for value in alternative], dtype=int)


def _neighbor_indices(coords: np.ndarray, neighbors: int = 6) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=min(coords.shape[0], neighbors + 1))
    model.fit(coords)
    return model.kneighbors(return_distance=False)[:, 1:]


def _boundary_mask(labels: np.ndarray, neighbor_indices: np.ndarray) -> np.ndarray:
    return np.any(labels[:, None] != labels[neighbor_indices], axis=1)


def _odds_ratio_interval(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    cells = np.asarray([a, b, c, d], dtype=float)
    if np.any(cells == 0):
        cells += 0.5
    aa, bb, cc, dd = cells.tolist()
    odds_ratio = (aa * dd) / (bb * cc)
    standard_error = math.sqrt(1.0 / aa + 1.0 / bb + 1.0 / cc + 1.0 / dd)
    lower = math.exp(math.log(odds_ratio) - 1.96 * standard_error)
    upper = math.exp(math.log(odds_ratio) + 1.96 * standard_error)
    return float(odds_ratio), float(lower), float(upper)


def _median_nearest_distance(coords: np.ndarray, targets: np.ndarray, query_mask: np.ndarray) -> float:
    if not np.any(targets) or not np.any(query_mask):
        return float("nan")
    model = NearestNeighbors(n_neighbors=1)
    model.fit(coords[targets])
    distances = model.kneighbors(coords[query_mask], return_distance=True)[0][:, 0]
    return float(np.median(distances))


def _dataset_image_root(dataset_id: str) -> Path:
    return DATASET_ROOTS[dataset_id]


def _sample_data(dataset_id: str, sample_id: str, cache: dict[tuple[str, str], SampleData]) -> SampleData:
    key = (dataset_id, sample_id)
    if key in cache:
        return cache[key]
    counts, genes, coords_table = _load_flat_visium_sample(_dataset_image_root(dataset_id), sample_id)
    vectors = _feature_vectors(counts, genes)
    features = {
        vector.feature_id: (vector.feature_type, vector.genes_used, vector.values.astype(float, copy=False))
        for vector in vectors
    }
    value = SampleData(
        barcodes=coords_table["barcode"].astype(str).tolist(),
        coords=coords_table[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float),
        features=features,
    )
    cache[key] = value
    return value


def _block_ids(coords: np.ndarray, grid: int = 6) -> np.ndarray:
    x_edges = np.linspace(float(coords[:, 0].min()), float(coords[:, 0].max()), grid + 1)
    y_edges = np.linspace(float(coords[:, 1].min()), float(coords[:, 1].max()), grid + 1)
    x_bins = np.clip(np.digitize(coords[:, 0], x_edges[1:-1], right=False), 0, grid - 1)
    y_bins = np.clip(np.digitize(coords[:, 1], y_edges[1:-1], right=False), 0, grid - 1)
    return np.asarray([f"x{int(x)}_y{int(y)}" for x, y in zip(x_bins, y_bins, strict=True)], dtype=object)


def _block_contrast(
    values: np.ndarray,
    boundary: np.ndarray,
    blocks: np.ndarray,
    min_per_group: int = 5,
) -> tuple[int, float, float, float]:
    contrasts: list[float] = []
    for block in np.unique(blocks):
        in_block = blocks == block
        boundary_group = in_block & boundary
        interior_group = in_block & ~boundary
        if int(boundary_group.sum()) < min_per_group or int(interior_group.sum()) < min_per_group:
            continue
        contrasts.append(float(np.median(values[boundary_group]) - np.median(values[interior_group])))
    if not contrasts:
        return 0, float("nan"), float("nan"), float("nan")
    array = np.asarray(contrasts, dtype=float)
    return (
        int(array.size),
        float(np.median(array)),
        float(np.quantile(array, 0.025)),
        float(np.quantile(array, 0.975)),
    )


def _bootstrap_median_delta(
    boundary_values: np.ndarray,
    interior_values: np.ndarray,
    key: str,
    replicates: int = 1000,
) -> tuple[float, float]:
    """Return a deterministic percentile interval for the median difference."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    batch_size = 100
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        size = stop - start
        boundary_sample = rng.choice(
            boundary_values, size=(size, boundary_values.size), replace=True
        )
        interior_sample = rng.choice(
            interior_values, size=(size, interior_values.size), replace=True
        )
        estimates[start:stop] = np.median(boundary_sample, axis=1) - np.median(
            interior_sample, axis=1
        )
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _global_characterization(out_dir: Path) -> None:
    locked_path = ROOT / "results" / "benchmarks" / "method_benchmark_locked.tsv"
    locked = pd.read_csv(locked_path, sep="\t")
    locked["source_origin"] = str(locked_path.relative_to(ROOT))
    r3_parts: list[pd.DataFrame] = []
    r3_paths = sorted((out_dir / "runs").glob("*/*/bench/method_benchmark.tsv"))
    r3_paths += sorted((out_dir / "runs" / "bayesspace").glob("*.tsv"))
    for path in r3_paths:
        part = pd.read_csv(path, sep="\t")
        if not {"dataset_id", "sample_id", "method_id", "K"}.issubset(part.columns):
            continue
        part["source_origin"] = str(path.relative_to(ROOT))
        r3_parts.append(part)
    if r3_parts:
        r3 = pd.concat(r3_parts, ignore_index=True, sort=False)
        keys = ["dataset_id", "sample_id", "method_id", "K"]
        r3["dataset_id"] = r3["dataset_id"].replace({"GSE280318": "GSE285505"})
        locked["dataset_id"] = locked["dataset_id"].replace({"GSE280318": "GSE285505"})
        r3_keys = set(map(tuple, r3[keys].itertuples(index=False, name=None)))
        keep_locked = ~locked[keys].apply(tuple, axis=1).isin(r3_keys)
        source = pd.concat([locked[keep_locked], r3], ignore_index=True, sort=False)
    else:
        source = locked
    source["dataset_id"] = source["dataset_id"].replace({"GSE280318": "GSE285505"})
    source = source[source["dataset_id"].isin(DATASET_ROOTS)].copy()
    source = source[source["method_id"].isin(METHOD_ORDER)].copy()
    units = source[["dataset_id", "sample_id", "K"]].drop_duplicates()
    expected = units.merge(pd.DataFrame({"method_id": METHOD_ORDER}), how="cross")
    source = expected.merge(
        source,
        on=["dataset_id", "sample_id", "method_id", "K"],
        how="left",
        validate="one_to_one",
    )
    source["method_label"] = source["method_id"].map(METHOD_LABELS)
    source["status"] = np.where(source["seed_count"].notna(), "success", "failed")
    failure_path = ROOT / "results" / "benchmarks" / "failure_log.tsv"
    failure_reason: dict[tuple[str, str, str, int], str] = {}
    failed_replicates: dict[tuple[str, str, str, int], str] = {}
    if failure_path.exists():
        failures = pd.read_csv(failure_path, sep="\t")
        failures["dataset_id"] = failures["dataset_id"].replace({"GSE280318": "GSE285505"})
        failures = failures[
            failures["dataset_id"].isin(DATASET_ROOTS)
            & failures["method_id"].isin(METHOD_ORDER)
        ].copy()
        for key, group in failures.groupby(["dataset_id", "sample_id", "method_id", "K"]):
            messages = sorted(set(group["error_message"].dropna().astype(str)))
            normalized_key = (str(key[0]), str(key[1]), str(key[2]), int(key[3]))
            failure_reason[normalized_key] = "; ".join(messages)
            if "seed" in group.columns:
                seeds = sorted({int(value) for value in group["seed"].dropna().tolist()})
                failed_replicates[normalized_key] = ",".join(f"seed_{seed}" for seed in seeds)
    source["failure_reason"] = source.apply(
        lambda row: failure_reason.get(
            (str(row["dataset_id"]), str(row["sample_id"]), str(row["method_id"]), int(row["K"])),
            "",
        ),
        axis=1,
    )
    columns = [
        "dataset_id",
        "sample_id",
        "method_id",
        "method_label",
        "K",
        "seed_count",
        "spatial_coherence_median",
        "spatial_coherence_iqr",
        "marker_coherence_median",
        "marker_coherence_iqr",
        "stability_ari_median",
        "stability_ari_iqr",
        "failure_rate",
        "status",
        "failure_reason",
        "source_origin",
    ]
    source[columns].sort_values(["method_id", "dataset_id", "sample_id", "K"]).to_csv(
        out_dir / "global_characterization.tsv", sep="\t", index=False
    )
    summary = (
        source.groupby(["method_id", "method_label"], as_index=False)
        .agg(
            n_section_k_expected=("sample_id", "size"),
            n_section_k_available=("status", lambda values: int(np.sum(np.asarray(values) == "success"))),
            spatial_coherence_median=("spatial_coherence_median", "median"),
            spatial_coherence_q1=("spatial_coherence_median", lambda values: values.quantile(0.25)),
            spatial_coherence_q3=("spatial_coherence_median", lambda values: values.quantile(0.75)),
            marker_coherence_median=("marker_coherence_median", "median"),
            marker_coherence_q1=("marker_coherence_median", lambda values: values.quantile(0.25)),
            marker_coherence_q3=("marker_coherence_median", lambda values: values.quantile(0.75)),
        )
    )
    summary.to_csv(out_dir / "global_characterization_summary.tsv", sep="\t", index=False)


def _process_map_file(
    path: Path,
    cache: dict[tuple[str, str], SampleData],
    coverage_rows: list[dict[str, object]],
    stability_rows: list[dict[str, object]],
    switching_rows: list[dict[str, object]],
    relation_rows: list[dict[str, object]],
    histology_rows: list[dict[str, object]],
    downstream_rows: list[dict[str, object]],
) -> None:
    table = _read_map(path)
    for (dataset_id, sample_id, method_id, k_value), group in table.groupby(
        ["dataset_id", "sample_id", "method_id", "K"], sort=True
    ):
        dataset_id = str(dataset_id)
        sample_id = str(sample_id)
        method_id = str(method_id)
        k_value = int(k_value)
        replicate_ids = sorted(group["replicate_id"].astype(str).unique().tolist())
        reference_id = _reference_id(method_id)
        sample = _sample_data(dataset_id, sample_id, cache)
        expected_barcodes = sample.barcodes
        expected_set = set(expected_barcodes)
        maps: dict[str, pd.DataFrame] = {}
        invalid_replicates: list[str] = []
        for replicate_id in replicate_ids:
            replicate = group[group["replicate_id"].astype(str) == replicate_id].copy()
            replicate["barcode"] = replicate["barcode"].astype(str)
            if set(replicate["barcode"]) != expected_set or replicate["barcode"].duplicated().any():
                invalid_replicates.append(replicate_id)
                continue
            maps[replicate_id] = replicate.set_index("barcode").loc[expected_barcodes].reset_index()

        coverage_rows.append(
            {
                "dataset_id": dataset_id,
                "sample_id": sample_id,
                "method_id": method_id,
                "method_label": METHOD_LABELS.get(method_id, method_id),
                "K": k_value,
                "expected_replicates": 3,
                "successful_replicates": len(maps),
                "replicate_ids": ",".join(sorted(maps)),
                "invalid_replicates": ",".join(invalid_replicates),
                "observed_domain_counts": ";".join(
                    f"{replicate_id}:{int(replicate['domain_label'].nunique())}"
                    for replicate_id, replicate in sorted(maps.items())
                ),
                "n_spots": len(expected_barcodes),
                "reference_id": reference_id,
                "status": "complete" if len(maps) == 3 and reference_id in maps else "incomplete",
                "source_file": path.name,
                "failure_reason": "",
            }
        )
        if reference_id not in maps or len(maps) < 2:
            continue

        reference = maps[reference_id]
        reference_labels = reference["domain_label"].to_numpy(dtype=int)
        coords = reference[["x", "y"]].to_numpy(dtype=float)
        neighbor_indices = _neighbor_indices(coords, neighbors=6)
        reference_boundary = _boundary_mask(reference_labels, neighbor_indices)
        blocks = _block_ids(coords)

        try:
            image_grad = load_detected_tissue_image_grad(_dataset_image_root(dataset_id), sample_id)
            gradient_values = sample_grad_at_points(image_grad.grad, coords)
            gradient_status = "success"
        except Exception as exc:
            gradient_values = np.full(coords.shape[0], np.nan, dtype=float)
            gradient_status = f"missing_image:{exc}"

        partition_ids = sorted(maps)
        for partition_id in partition_ids:
            partition = maps[partition_id]
            labels = partition["domain_label"].to_numpy(dtype=int)
            boundary = _boundary_mask(labels, neighbor_indices)
            for feature_id, (feature_type, genes_used, values_by_sample) in sample.features.items():
                value_by_barcode = dict(zip(sample.barcodes, values_by_sample.tolist(), strict=True))
                values = np.asarray([value_by_barcode[barcode] for barcode in expected_barcodes], dtype=float)
                boundary_values = values[boundary]
                interior_values = values[~boundary]
                if boundary_values.size == 0 or interior_values.size == 0:
                    continue
                pvalue = float(mannwhitneyu(boundary_values, interior_values, alternative="two-sided").pvalue)
                bootstrap_lower, bootstrap_upper = _bootstrap_median_delta(
                    boundary_values,
                    interior_values,
                    key=(
                        f"{dataset_id}|{sample_id}|{method_id}|{k_value}|"
                        f"{partition_id}|{feature_id}"
                    ),
                )
                n_blocks, block_median, block_lower, block_upper = _block_contrast(values, boundary, blocks)
                downstream_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "sample_id": sample_id,
                        "method_id": method_id,
                        "method_label": METHOD_LABELS.get(method_id, method_id),
                        "K": k_value,
                        "partition_id": partition_id,
                        "is_reference": partition_id == reference_id,
                        "feature_type": feature_type,
                        "feature_id": feature_id,
                        "genes_used": genes_used,
                        "n_boundary": int(boundary.sum()),
                        "n_interior": int((~boundary).sum()),
                        "median_boundary": float(np.median(boundary_values)),
                        "median_interior": float(np.median(interior_values)),
                        "median_delta_boundary_minus_interior": float(
                            np.median(boundary_values) - np.median(interior_values)
                        ),
                        "bootstrap_ci_lower": bootstrap_lower,
                        "bootstrap_ci_upper": bootstrap_upper,
                        "bootstrap_replicates": 1000,
                        "pvalue_mannwhitney_spot_level": pvalue,
                        "n_mixed_blocks": n_blocks,
                        "block_median_delta": block_median,
                        "block_interval_lower": block_lower,
                        "block_interval_upper": block_upper,
                    }
                )

        for alternative_id in partition_ids:
            if alternative_id == reference_id:
                continue
            alternative = maps[alternative_id]
            alternative_labels_raw = alternative["domain_label"].to_numpy(dtype=int)
            alternative_labels = _align_labels(reference_labels, alternative_labels_raw)
            switching = alternative_labels != reference_labels
            alternative_boundary = _boundary_mask(alternative_labels, neighbor_indices)
            union_boundary = reference_boundary | alternative_boundary
            ari = float(adjusted_rand_score(reference_labels, alternative_labels_raw))
            stability_rows.append(
                {
                    "dataset_id": dataset_id,
                    "sample_id": sample_id,
                    "method_id": method_id,
                    "method_label": METHOD_LABELS.get(method_id, method_id),
                    "K": k_value,
                    "reference_id": reference_id,
                    "alternative_id": alternative_id,
                    "stability_type": "graph_specification_ari" if method_id == "M2_spatial_ward" else "seed_pair_ari",
                    "ari": ari,
                    "below_0_60": ari < 0.60,
                    "n_spots": int(switching.size),
                    "n_switching": int(switching.sum()),
                    "switching_fraction": float(switching.mean()),
                }
            )

            a = int(np.sum(switching & union_boundary))
            b = int(np.sum(switching & ~union_boundary))
            c = int(np.sum(~switching & union_boundary))
            d = int(np.sum(~switching & ~union_boundary))
            risk_switching = a / (a + b) if (a + b) else float("nan")
            risk_stable = c / (c + d) if (c + d) else float("nan")
            risk_ratio = risk_switching / risk_stable if risk_stable > 0 else float("nan")
            odds_ratio, odds_lower, odds_upper = _odds_ratio_interval(a, b, c, d)
            intersection = int(np.sum(switching & union_boundary))
            union = int(np.sum(switching | union_boundary))
            relation_rows.append(
                {
                    "dataset_id": dataset_id,
                    "sample_id": sample_id,
                    "method_id": method_id,
                    "method_label": METHOD_LABELS.get(method_id, method_id),
                    "K": k_value,
                    "reference_id": reference_id,
                    "alternative_id": alternative_id,
                    "n_spots": int(switching.size),
                    "n_switching": int(switching.sum()),
                    "switching_fraction": float(switching.mean()),
                    "n_reference_boundary": int(reference_boundary.sum()),
                    "n_alternative_boundary": int(alternative_boundary.sum()),
                    "n_union_boundary": int(union_boundary.sum()),
                    "n_switching_in_union_boundary": a,
                    "fraction_switching_in_union_boundary": risk_switching,
                    "fraction_stable_in_union_boundary": risk_stable,
                    "risk_ratio": risk_ratio,
                    "odds_ratio": odds_ratio,
                    "odds_ratio_ci_lower": odds_lower,
                    "odds_ratio_ci_upper": odds_upper,
                    "jaccard_switching_union_boundary": intersection / union if union else float("nan"),
                    "median_distance_switching_to_reference_boundary": _median_nearest_distance(
                        coords, reference_boundary, switching
                    ),
                    "median_distance_stable_to_reference_boundary": _median_nearest_distance(
                        coords, reference_boundary, ~switching
                    ),
                }
            )

            if gradient_status == "success" and np.any(switching) and np.any(~switching):
                pvalue = float(
                    mannwhitneyu(gradient_values[switching], gradient_values[~switching], alternative="two-sided").pvalue
                )
                histology_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "sample_id": sample_id,
                        "method_id": method_id,
                        "method_label": METHOD_LABELS.get(method_id, method_id),
                        "K": k_value,
                        "reference_id": reference_id,
                        "alternative_id": alternative_id,
                        "n_switching": int(switching.sum()),
                        "n_stable": int((~switching).sum()),
                        "median_gradient_switching": float(np.median(gradient_values[switching])),
                        "median_gradient_stable": float(np.median(gradient_values[~switching])),
                        "median_delta_switching_minus_stable": float(
                            np.median(gradient_values[switching]) - np.median(gradient_values[~switching])
                        ),
                        "pvalue_mannwhitney_spot_level": pvalue,
                        "status": gradient_status,
                    }
                )
            else:
                histology_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "sample_id": sample_id,
                        "method_id": method_id,
                        "method_label": METHOD_LABELS.get(method_id, method_id),
                        "K": k_value,
                        "reference_id": reference_id,
                        "alternative_id": alternative_id,
                        "n_switching": int(switching.sum()),
                        "n_stable": int((~switching).sum()),
                        "status": gradient_status if gradient_status != "success" else "not_evaluable_no_switching",
                    }
                )

            for index in np.flatnonzero(switching):
                switching_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "sample_id": sample_id,
                        "method_id": method_id,
                        "K": k_value,
                        "reference_id": reference_id,
                        "alternative_id": alternative_id,
                        "barcode": expected_barcodes[int(index)],
                        "x": float(coords[int(index), 0]),
                        "y": float(coords[int(index), 1]),
                        "reference_label": int(reference_labels[int(index)]),
                        "alternative_label_aligned": int(alternative_labels[int(index)]),
                        "reference_boundary": bool(reference_boundary[int(index)]),
                        "alternative_boundary": bool(alternative_boundary[int(index)]),
                        "union_boundary": bool(union_boundary[int(index)]),
                        "histology_gradient": float(gradient_values[int(index)])
                        if np.isfinite(gradient_values[int(index)])
                        else float("nan"),
                    }
                )


def _write_summaries(
    out_dir: Path,
    coverage_rows: list[dict[str, object]],
    stability_rows: list[dict[str, object]],
    switching_rows: list[dict[str, object]],
    relation_rows: list[dict[str, object]],
    histology_rows: list[dict[str, object]],
    downstream_rows: list[dict[str, object]],
) -> None:
    coverage = pd.DataFrame(coverage_rows)
    global_table = pd.read_csv(out_dir / "global_characterization.tsv", sep="\t")
    expected_units = global_table[["dataset_id", "sample_id", "K"]].drop_duplicates()
    expected_rows: list[dict[str, object]] = []
    failure_reason: dict[tuple[str, str, str, int], str] = {}
    failed_replicates: dict[tuple[str, str, str, int], str] = {}
    for failure_path in sorted((out_dir / "runs").glob("**/failure_log.tsv")):
        failures = pd.read_csv(failure_path, sep="\t")
        if not {"dataset_id", "sample_id", "method_id", "K", "failure_type"}.issubset(failures.columns):
            continue
        failures["dataset_id"] = failures["dataset_id"].replace({"GSE280318": "GSE285505"})
        failures = failures[
            failures["method_id"].isin(METHOD_ORDER)
            & ~failures["failure_type"].astype(str).eq("none")
        ]
        for key, group in failures.groupby(["dataset_id", "sample_id", "method_id", "K"]):
            messages = sorted(set(group["error_message"].dropna().astype(str)))
            normalized_key = (str(key[0]), str(key[1]), str(key[2]), int(key[3]))
            failure_reason[normalized_key] = "; ".join(messages)
            if "seed" in group.columns:
                seeds = sorted({int(value) for value in group["seed"].dropna().tolist()})
                failed_replicates[normalized_key] = ",".join(f"seed_{seed}" for seed in seeds)
    present_keys = {
        (str(row.dataset_id), str(row.sample_id), str(row.method_id), int(row.K))
        for row in coverage.itertuples(index=False)
    }
    for unit in expected_units.itertuples(index=False):
        for method_id in METHOD_ORDER:
            key = (str(unit.dataset_id), str(unit.sample_id), method_id, int(unit.K))
            if key in present_keys:
                continue
            expected_rows.append(
                {
                    "dataset_id": str(unit.dataset_id),
                    "sample_id": str(unit.sample_id),
                    "method_id": method_id,
                    "method_label": METHOD_LABELS[method_id],
                    "K": int(unit.K),
                    "expected_replicates": 3,
                    "successful_replicates": 0,
                    "replicate_ids": "",
                    "invalid_replicates": "",
                    "observed_domain_counts": "",
                    "n_spots": "",
                    "reference_id": _reference_id(method_id),
                    "status": "failed" if key in failure_reason else "missing",
                    "source_file": "",
                    "failure_reason": failure_reason.get(key, ""),
                    "failed_replicates": failed_replicates.get(key, ""),
                }
            )
    if expected_rows:
        coverage = pd.concat([coverage, pd.DataFrame(expected_rows)], ignore_index=True)
    coverage["failure_reason"] = coverage.apply(
        lambda row: failure_reason.get(
            (str(row["dataset_id"]), str(row["sample_id"]), str(row["method_id"]), int(row["K"])),
            str(row.get("failure_reason", "")) if pd.notna(row.get("failure_reason", "")) else "",
        ),
        axis=1,
    )
    coverage["failed_replicates"] = coverage.apply(
        lambda row: failed_replicates.get(
            (str(row["dataset_id"]), str(row["sample_id"]), str(row["method_id"]), int(row["K"])),
            str(row.get("failed_replicates", "")) if pd.notna(row.get("failed_replicates", "")) else "",
        ),
        axis=1,
    )
    coverage = coverage.sort_values(["method_id", "dataset_id", "sample_id", "K"])
    coverage.to_csv(out_dir / "coverage_manifest.tsv", sep="\t", index=False)

    stability = pd.DataFrame(stability_rows).sort_values(
        ["method_id", "dataset_id", "sample_id", "K", "alternative_id"]
    )
    stability.to_csv(out_dir / "stability_pairwise.tsv", sep="\t", index=False)
    stability_summary = (
        stability.groupby(["dataset_id", "sample_id", "method_id", "method_label", "K"], as_index=False)
        .agg(
            n_alternatives=("alternative_id", "size"),
            median_ari=("ari", "median"),
            min_ari=("ari", "min"),
            max_ari=("ari", "max"),
            n_below_0_60=("below_0_60", "sum"),
        )
    )
    switching_table = pd.DataFrame(switching_rows)
    union_counts = (
        switching_table.groupby(["dataset_id", "sample_id", "method_id", "K"])["barcode"]
        .nunique()
        .rename("n_union_switching")
        .reset_index()
    )
    unit_spots = stability[["dataset_id", "sample_id", "method_id", "K", "n_spots"]].drop_duplicates()
    union_counts = union_counts.merge(unit_spots, on=["dataset_id", "sample_id", "method_id", "K"], how="left")
    union_counts["union_switching_fraction"] = union_counts["n_union_switching"] / union_counts["n_spots"]
    stability_summary = stability_summary.merge(
        union_counts.drop(columns=["n_spots"]),
        on=["dataset_id", "sample_id", "method_id", "K"],
        how="left",
    )
    stability_summary.to_csv(out_dir / "stability_summary.tsv", sep="\t", index=False)

    switching_table.to_csv(
        out_dir / "switching_spots.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    relation = pd.DataFrame(relation_rows).sort_values(
        ["method_id", "dataset_id", "sample_id", "K", "alternative_id"]
    )
    relation.to_csv(out_dir / "switching_boundary_relationship.tsv", sep="\t", index=False)
    relation_summary = (
        relation.groupby(["method_id", "method_label"], as_index=False)
        .agg(
            n_comparisons=("alternative_id", "size"),
            median_switching_fraction=("switching_fraction", "median"),
            median_fraction_switching_in_boundary=("fraction_switching_in_union_boundary", "median"),
            median_fraction_stable_in_boundary=("fraction_stable_in_union_boundary", "median"),
            median_risk_ratio=("risk_ratio", "median"),
            median_odds_ratio=("odds_ratio", "median"),
            median_jaccard=("jaccard_switching_union_boundary", "median"),
            fraction_rr_above_one=("risk_ratio", lambda values: float(np.mean(np.asarray(values) > 1))),
        )
    )
    relation_summary.to_csv(out_dir / "switching_boundary_summary.tsv", sep="\t", index=False)

    histology = pd.DataFrame(histology_rows)
    histology["fdr_bh"] = _bh_fdr(histology.get("pvalue_mannwhitney_spot_level", pd.Series(dtype=float)).tolist())
    histology.sort_values(["method_id", "dataset_id", "sample_id", "K", "alternative_id"]).to_csv(
        out_dir / "histology_gradient_localization.tsv", sep="\t", index=False
    )
    histology_success = histology[histology["status"].eq("success")].copy()
    histology_summary = (
        histology_success.groupby(["method_id", "method_label"], as_index=False)
        .agg(
            n_comparisons=("alternative_id", "size"),
            median_gradient_delta=("median_delta_switching_minus_stable", "median"),
            fraction_positive_gradient_delta=(
                "median_delta_switching_minus_stable",
                lambda values: float(np.mean(np.asarray(values) > 0)),
            ),
            fraction_fdr_below_0_05=("fdr_bh", lambda values: float(np.mean(np.asarray(values) < 0.05))),
        )
    )
    histology_summary.to_csv(out_dir / "histology_gradient_summary.tsv", sep="\t", index=False)

    downstream = pd.DataFrame(downstream_rows)
    downstream["fdr_bh_within_partition"] = downstream.groupby(
        ["dataset_id", "sample_id", "method_id", "K", "partition_id"]
    )["pvalue_mannwhitney_spot_level"].transform(lambda values: _bh_fdr(values.tolist()))
    downstream.sort_values(
        ["method_id", "dataset_id", "sample_id", "K", "feature_id", "partition_id"]
    ).to_csv(out_dir / "downstream_contrasts.tsv", sep="\t", index=False)

    summary_rows: list[dict[str, object]] = []
    for keys, group in downstream.groupby(
        ["dataset_id", "sample_id", "method_id", "method_label", "K", "feature_type", "feature_id", "genes_used"],
        sort=True,
    ):
        dataset_id, sample_id, method_id, method_label, k_value, feature_type, feature_id, genes_used = keys
        reference_id = _reference_id(str(method_id))
        reference = group[group["partition_id"].eq(reference_id)]
        if reference.empty:
            continue
        reference_delta = float(reference["median_delta_boundary_minus_interior"].iloc[0])
        alternatives = group[~group["partition_id"].eq(reference_id)].copy()
        alternative_values = alternatives["median_delta_boundary_minus_interior"].to_numpy(dtype=float)
        sign_reference = int(np.sign(reference_delta))
        alternative_signs = np.sign(alternative_values).astype(int)
        sign_state_changes = (
            int(np.sum(alternative_signs != sign_reference)) if alternative_values.size else 0
        )
        direction_reversals = (
            int(np.sum((sign_reference * alternative_signs) < 0)) if alternative_values.size else 0
        )
        zero_involved_changes = sign_state_changes - direction_reversals
        reference_block = reference["block_median_delta"].iloc[0]
        evaluable_block_alternatives = alternatives["block_median_delta"].dropna().to_numpy(dtype=float)
        block_direction_reversals = 0
        if pd.notna(reference_block) and evaluable_block_alternatives.size:
            block_direction_reversals = int(
                np.sum((np.sign(float(reference_block)) * np.sign(evaluable_block_alternatives)) < 0)
            )
        summary_rows.append(
            {
                "dataset_id": dataset_id,
                "sample_id": sample_id,
                "method_id": method_id,
                "method_label": method_label,
                "K": int(k_value),
                "feature_type": feature_type,
                "feature_id": feature_id,
                "genes_used": genes_used,
                "reference_id": reference_id,
                "reference_delta": reference_delta,
                "n_alternatives": int(alternative_values.size),
                "minimum_delta": float(group["median_delta_boundary_minus_interior"].min()),
                "maximum_delta": float(group["median_delta_boundary_minus_interior"].max()),
                "delta_range": float(
                    group["median_delta_boundary_minus_interior"].max()
                    - group["median_delta_boundary_minus_interior"].min()
                ),
                "n_sign_state_changes_vs_reference": sign_state_changes,
                "any_sign_state_change_vs_reference": sign_state_changes > 0,
                "n_direction_reversals_vs_reference": direction_reversals,
                "any_direction_reversal_vs_reference": direction_reversals > 0,
                "n_zero_involved_changes_vs_reference": zero_involved_changes,
                "all_partitions_same_direction": int(
                    np.unique(np.sign(group["median_delta_boundary_minus_interior"].to_numpy(dtype=float))).size == 1
                ),
                "reference_block_delta": float(reference_block) if pd.notna(reference_block) else float("nan"),
                "median_block_delta": float(group["block_median_delta"].median()),
                "n_partitions_with_evaluable_blocks": int(group["block_median_delta"].notna().sum()),
                "n_block_direction_reversals_vs_reference": block_direction_reversals,
                "any_block_direction_reversal_vs_reference": block_direction_reversals > 0,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "downstream_robustness_summary.tsv", sep="\t", index=False)
    method_summary = (
        summary.groupby(["method_id", "method_label"], as_index=False)
        .agg(
            n_section_k_features=("feature_id", "size"),
            fraction_with_direction_reversal=("any_direction_reversal_vs_reference", "mean"),
            fraction_with_sign_state_change=("any_sign_state_change_vs_reference", "mean"),
            fraction_with_block_direction_reversal=(
                "any_block_direction_reversal_vs_reference",
                "mean",
            ),
            median_effect_range=("delta_range", "median"),
            fraction_all_partitions_same_direction=("all_partitions_same_direction", "mean"),
        )
    )
    method_summary.to_csv(out_dir / "downstream_method_summary.tsv", sep="\t", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", default=str(DEFAULT_MAP_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    map_dir = Path(args.map_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(map_dir.glob("*.tsv")) + sorted(map_dir.glob("*.tsv.gz"))
    if not paths:
        raise FileNotFoundError(f"No map TSV files found under {map_dir}")

    _global_characterization(out_dir)
    cache: dict[tuple[str, str], SampleData] = {}
    coverage_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    switching_rows: list[dict[str, object]] = []
    relation_rows: list[dict[str, object]] = []
    histology_rows: list[dict[str, object]] = []
    downstream_rows: list[dict[str, object]] = []

    for path in paths:
        print(f"[evidence] {path.name}", flush=True)
        _process_map_file(
            path,
            cache,
            coverage_rows,
            stability_rows,
            switching_rows,
            relation_rows,
            histology_rows,
            downstream_rows,
        )

    _write_summaries(
        out_dir,
        coverage_rows,
        stability_rows,
        switching_rows,
        relation_rows,
        histology_rows,
        downstream_rows,
    )
    print(f"Wrote R3 evidence tables under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
