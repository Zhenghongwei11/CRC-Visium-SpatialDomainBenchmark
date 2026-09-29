#!/usr/bin/env python3
"""Independently check R4 v4 geometry using KD-tree and sparse graph APIs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/interface_validation/r4_undirected_20260926_audit_v1/INDEPENDENT_GEOMETRY_AUDIT_20260926.json"
SOURCES = {
    "valdeolivas": ROOT / "tmp/r4_valdeolivas_undirected_20260926_v1",
    "gse": ROOT / "tmp/r4_gse_undirected_20260926_v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def independent_neighbors(coords: np.ndarray, count: int = 6) -> np.ndarray:
    tree = cKDTree(coords)
    k = min(count + 1, len(coords))
    distances, _ = tree.query(coords, k=k)
    if k == 1:
        return np.empty((len(coords), 0), dtype=int)
    radii = np.nextafter(np.asarray(distances)[:, -1], np.inf)
    candidates = tree.query_ball_point(coords, radii)
    result = np.empty((len(coords), k - 1), dtype=int)
    for row, possible in enumerate(candidates):
        ranked = sorted(
            (int(index) for index in possible if index != row),
            key=lambda index: (float(np.sum((coords[index] - coords[row]) ** 2)), index),
        )
        if len(ranked) < k - 1:
            raise ValueError("KD-tree radius omitted a nearest neighbor")
        result[row] = ranked[: k - 1]
    return result


def independent_anchor(spots: pd.DataFrame, neighbors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coarse = spots.coarse_label.astype(str).to_numpy()
    tumor = coarse == "tumor"
    stroma = coarse == "stroma"
    n = len(spots)
    if not tumor.any() or not stroma.any():
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.full(n, -1, dtype=int)
    xy = spots[["x_fullres", "y_fullres"]].to_numpy(float)
    distance = cKDTree(xy[tumor]).query(xy)[0] * 55.0 / float(spots.spot_diameter_fullres.iloc[0])
    near = stroma & (distance <= 100.0)
    stroma_rows = np.repeat(np.arange(n), neighbors.shape[1])
    stroma_cols = neighbors.ravel()
    keep = stroma[stroma_rows] & stroma[stroma_cols]
    graph = coo_matrix(
        (np.ones(int(keep.sum()), dtype=np.uint8), (stroma_rows[keep], stroma_cols[keep])),
        shape=(n, n),
    ).tocsr()
    _, components = connected_components(graph, directed=False)
    components[~stroma] = -1
    far = stroma & (distance >= 200.0) & (distance <= 400.0) & np.isin(components, np.unique(components[near]))
    return near, far, components


def compare_section(cohort: str, source: Path) -> dict[str, object]:
    spots = pd.read_csv(source / "spot_inputs.tsv.gz", sep="\t", low_memory=False, keep_default_na=False)
    try:
        membership = pd.read_csv(source / "spot_membership.tsv.gz", sep="\t", low_memory=False)
    except pd.errors.EmptyDataError:
        membership = pd.DataFrame()
    try:
        effects = pd.read_csv(source / "section_effects.tsv", sep="\t", low_memory=False)
    except pd.errors.EmptyDataError:
        effects = pd.DataFrame()
    coords = spots[["array_row", "array_col"]].to_numpy(float)
    neighbors = independent_neighbors(coords)
    near, far, components = independent_anchor(spots, neighbors)
    result: dict[str, object] = {
        "cohort": cohort,
        "sample_id": source.name,
        "n_spots": len(spots),
        "n_partitions": 0,
        "membership_rows": len(membership),
        "input_spots_sha256": sha256(source / "spot_inputs.tsv.gz"),
        "membership_sha256": sha256(source / "spot_membership.tsv.gz"),
        "effect_sha256": sha256(source / "section_effects.tsv"),
        "mismatches": {},
    }
    if membership.empty:
        if not effects.empty:
            if not effects.status.eq("not_evaluable").all():
                raise ValueError(f"{source.name}: successful effects without membership")
            result["mismatches"] = {
                "n_anchor_near": int((effects.n_anchor_near != int(near.sum())).sum()),
                "n_anchor_far": int((effects.n_anchor_far != int(far.sum())).sum()),
            }
        return result
    ordered_barcodes = spots.barcode.astype(str).tolist()
    index = {barcode: row for row, barcode in enumerate(ordered_barcodes)}
    base = membership[["barcode", "morphology_near", "morphology_far", "stromal_component"]].drop_duplicates("barcode")
    if len(base) != len(spots):
        raise ValueError(f"{source.name}: incomplete membership barcode coverage")
    base = base.set_index("barcode").loc[ordered_barcodes]
    checks = {
        "morphology_near": int(np.count_nonzero(base.morphology_near.to_numpy(bool) != near)),
        "morphology_far": int(np.count_nonzero(base.morphology_far.to_numpy(bool) != far)),
    }
    # Component integer IDs may differ; compare the minimum input row in each component.
    observed_components = base.stromal_component.to_numpy(int)
    def canonical(ids: np.ndarray) -> np.ndarray:
        minima = {int(label): int(np.flatnonzero(ids == label).min()) for label in np.unique(ids) if label >= 0}
        return np.array([minima.get(int(label), -1) for label in ids], dtype=int)
    checks["stromal_component"] = int(np.count_nonzero(canonical(observed_components) != canonical(components)))
    block = spots.spatial_block.astype(str).to_numpy()
    grouped = membership.groupby(["estimator_id", "method_id", "K", "partition_id"], sort=True)
    for key, frame in grouped:
        if len(frame) != len(spots) or frame.barcode.duplicated().any():
            raise ValueError(f"{source.name}/{key}: incomplete partition membership")
        aligned = frame.assign(_row=frame.barcode.map(index)).sort_values("_row")
        if aligned._row.isna().any():
            raise ValueError(f"{source.name}/{key}: unknown barcode")
        labels = aligned.domain_label.to_numpy(int)
        stroma_domain = int(aligned.selected_stroma_domain.iloc[0])
        tumor_domain = int(aligned.selected_tumor_domain.iloc[0])
        computational_far = far & (labels == stroma_domain)
        computational_near = near & (labels == stroma_domain)
        if key[0] == "strict_computational_interface":
            computational_near &= (labels[neighbors] == tumor_domain).any(axis=1)
        boundary = (labels[:, None] != labels[neighbors]).any(axis=1)
        for field, expected in (("computational_near", computational_near),
                                ("computational_far", computational_far),
                                ("computational_boundary", boundary)):
            checks[field] = checks.get(field, 0) + int(np.count_nonzero(aligned[field].to_numpy(bool) != expected))
        if not effects.empty:
            effect = effects[
                effects.score_id.eq("primary_barrier") & effects.estimator_id.eq(key[0])
                & effects.method_id.eq(key[1]) & effects.K.eq(key[2]) & effects.partition_id.eq(key[3])
            ]
            if len(effect) != 1:
                raise ValueError(f"{source.name}/{key}: missing primary effect")
            effect = effect.iloc[0]
            for field, value in (("n_anchor_near", near.sum()), ("n_anchor_far", far.sum()),
                                 ("n_computational_near", computational_near.sum()),
                                 ("n_computational_far", computational_far.sum())):
                checks[field] = checks.get(field, 0) + int(int(effect[field]) != int(value))
            nblocks = min(len(set(block[computational_near])), len(set(block[computational_far])))
            checks["n_blocks"] = checks.get("n_blocks", 0) + int(int(effect.n_blocks) != nblocks)
    result["n_partitions"] = len(grouped)
    result["mismatches"] = checks
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    results = []
    for cohort, root in SOURCES.items():
        for section in sorted(root.iterdir()):
            if section.is_dir() and (section / "result_manifest.json").exists():
                row = compare_section(cohort, section)
                results.append(row)
                print(json.dumps({"cohort": cohort, "sample_id": section.name, "mismatches": row["mismatches"]}), flush=True)
    total = {field: sum(int(row["mismatches"].get(field, 0)) for row in results)
             for field in {key for row in results for key in row["mismatches"]}}
    report = {"schema_version": "r4_independent_geometry_audit_v1", "sections": results,
              "total_mismatches": total, "status": "pass" if all(value == 0 for value in total.values()) else "mismatch",
              "scope": "all 22 sections; independent KD-tree neighbors and sparse undirected components; primary effect support counts; intervals not independently recomputed"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "sections": len(results), "total_mismatches": total}, sort_keys=True))


if __name__ == "__main__":
    main()
