#!/usr/bin/env python3
"""Paired new map fits with readout features allowed or withheld."""
from __future__ import annotations

import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings

import h5py
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import scipy
from scipy.io import mmread
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
import sklearn
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

GENES = ("TGFB1", "CXCL12", "ACTA2", "TAGLN")
ARMS = ("allowed", "withheld")
METHODS = ("M0_expr_kmeans", "M1_spatial_concat_kmeans", "M2_spatial_ward", "M3_spatial_leiden")
MASTER = "jbcb-readout-withholding-20261007-v1"
REPS = 1000
METRICS = ["selected_delta_score", "abs_deviation_score", "abs_deviation_mad", "near_retention",
           "far_retention", "selected_stroma_fraction", "count_only_excess_mad", "block_matched_excess_mad"]


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identity(path):
    return dict(name=Path(path).name, bytes=Path(path).stat().st_size, sha256=sha(path))


def read(path):
    return pd.read_csv(path, sep="\t", float_precision="round_trip", low_memory=False)


def unique_names(symbols):
    reserved = set(symbols); seen = {}; result = []
    for symbol in symbols:
        n = seen.get(symbol, 0)
        if n == 0:
            result.append(symbol)
        else:
            while symbol + "-" + str(n) in reserved:
                n += 1
            name = symbol + "-" + str(n); reserved.add(name); result.append(name)
        seen[symbol] = n + 1
    return result


def load_counts(directory, section):
    sid = section["sample_id"]; members = []
    for expected in section["source_files"]:
        p = directory / expected["name"]
        record = identity(p)
        if record["bytes"] != expected["size_bytes"] or record["sha256"] != expected["sha256"]:
            raise ValueError("Original source identity differs: " + str(p))
        members.append(record)
    h5 = directory / (sid + "_filtered_feature_bc_matrix.h5")
    if h5.exists():
        with h5py.File(h5) as f:
            m = f["matrix"]
            counts = csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]),
                                shape=tuple(m["shape"][:])).T.tocsr().astype(np.float64)
            barcodes = [x.decode() for x in m["barcodes"][:]]
            symbols = [x.decode() for x in m["features/name"][:]]
            gene_ids = [x.decode() for x in m["features/id"][:]]
    else:
        with gzip.open(directory / (sid + "_matrix.mtx.gz"), "rb") as f:
            counts = mmread(f).T.tocsr().astype(np.float64)
        with gzip.open(directory / (sid + "_barcodes.tsv.gz"), "rt") as f:
            barcodes = [line.strip() for line in f if line.strip()]
        features = pd.read_csv(directory / (sid + "_features.tsv.gz"), sep="\t", header=None)
        gene_ids = features.iloc[:, 0].astype(str).tolist()
        symbols = features.iloc[:, 1].astype(str).tolist()
    if counts.shape != (section["matrix_barcode_count"], section["feature_count"]):
        raise ValueError("Source matrix dimensions differ")
    if len(set(barcodes)) != len(barcodes) or counts.nnz == 0 or not np.isfinite(counts.data).all():
        raise ValueError("Invalid source count matrix/barcodes")
    return counts, barcodes, symbols, gene_ids, members


def geometry(spots, values):
    lattice = spots[["array_row", "array_col"]].to_numpy(float)
    coords = spots[["x_fullres", "y_fullres"]].to_numpy(float)
    stroma = spots.coarse_label.eq("stroma").to_numpy(); tumor = spots.coarse_label.eq("tumor").to_numpy()
    nn = np.empty((len(spots), 6), int); qmin = np.full(len(spots), np.nan)
    for start in range(0, len(spots), 128):
        d = cdist(lattice[start:start + 128], lattice, "sqeuclidean")
        d[np.arange(len(d)), np.arange(start, start + len(d))] = np.inf
        nn[start:start + len(d)] = np.argsort(d, axis=1, kind="stable")[:, :6]
        if tumor.any():
            difference = lattice[start:start + 128, None, :] - lattice[tumor][None, :, :]
            qmin[start:start + len(d)] = (3*difference[:, :, 0]**2 + difference[:, :, 1]**2).min(axis=1)
    rows = np.repeat(np.arange(len(spots)), 6); cols = nn.ravel()
    keep = stroma[rows] & stroma[cols]
    graph = csr_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(len(spots), len(spots)))
    _, comp = connected_components(graph, directed=False)
    near = stroma & (qmin <= 4)
    far = stroma & (qmin >= 16) & (qmin <= 64) & np.isin(comp, np.unique(comp[near]))
    axes = [np.searchsorted(np.linspace(lattice[:, i].min(), lattice[:, i].max(), 7)[1:-1],
                            lattice[:, i], side="right") for i in range(2)]
    blocks = axes[0]*6 + axes[1]
    mad = float(np.median(abs(values[stroma] - np.median(values[stroma])))) if stroma.any() else np.nan
    return coords, stroma, tumor, near, far, blocks, mad


def eligible(near, far, blocks):
    return bool(min(near.sum(), far.sum()) >= 20 and
                min(np.unique(blocks[near]).size, np.unique(blocks[far]).size) >= 3)


def delta(values, near, far):
    return float(np.median(values[near])-np.median(values[far])) if near.any() and far.any() else np.nan


def coordinate_graph(coords, n):
    fit = NearestNeighbors(n_neighbors=n+1).fit(coords)
    graph = fit.kneighbors_graph(coords, mode="connectivity")
    return graph.maximum(graph.T)


def exact_leiden(graph, target, seed):
    def fit(resolution):
        p = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition,
                                     weights="weight", resolution_parameter=float(resolution), seed=int(seed))
        return np.asarray(p.membership, int)
    evaluations = []
    for resolution in [.01, .02, .05, .10, .20, .50, 1, 2, 5, 10]:
        labels = fit(resolution); observed = np.unique(labels).size
        evaluations.append((resolution, observed))
        if observed == target:
            return labels+1, resolution, True
    below = [(r, k) for r, k in evaluations if k < target]
    above = [(r, k) for r, k in evaluations if k > target]
    if below and above:
        low = max(below)[0]; high = min(above)[0]; best = (low, dict(evaluations)[low])
        for _ in range(18):
            middle = math.sqrt(low*high); labels = fit(middle); observed = np.unique(labels).size
            if abs(observed-target) < abs(best[1]-target):
                best = (middle, observed)
            if observed == target:
                return labels+1, middle, True
            if observed < target:
                low = middle
            else:
                high = middle
        return fit(best[0])+1, best[0], False
    best = min(evaluations, key=lambda x: (abs(x[1]-target), x[0]))[0]
    return fit(best)+1, best, False


def control(values, blocks, near, far, sn, sf, kind, context):
    seed = int.from_bytes(hashlib.sha256(context.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed); medians = []
    for full, selected in [(near, sn), (far, sf)]:
        if kind == "count_only":
            pools = [(np.flatnonzero(full), int(selected.sum()))]
        else:
            pools = [(np.flatnonzero(full & (blocks == b)), int((selected & (blocks == b)).sum()))
                     for b in np.unique(blocks[selected])]
        pieces = [np.broadcast_to(pool, (REPS, n)) if len(pool) == n else
                  np.stack([rng.choice(pool, n, replace=False) for _ in range(REPS)])
                  for pool, n in pools]
        medians.append(np.median(values[np.concatenate(pieces, axis=1)], axis=1))
    return medians[0]-medians[1]


def section_job(section, args):
    started = time.monotonic(); workspace = Path(args["workspace"])
    sid = section["sample_id"]; resource = section["resource"]
    directory = Path(args["count_root"]) / resource / sid / "members"
    destination = Path(args["output"]) / "sections" / resource / sid
    checkpoint=destination/'manifest.json'
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        if saved.get('source_script_sha256')!=sha(__file__):
            raise ValueError('Withholding checkpoint belongs to different code')
        for record in saved['files']:
            if sha(destination/record['name'])!=record['sha256']:raise ValueError('Changed withholding output')
        for name in ['spot_inputs','spot_scores']:
            if sha(workspace/section['source_root']/saved[name]['name'])!=saved[name]['sha256']:raise ValueError('Changed registered input')
        for record in section['source_files']:
            if sha(directory/record['name'])!=record['sha256']:raise ValueError('Changed original count input')
        return saved
    destination.mkdir(parents=True, exist_ok=True)
    counts, barcodes, symbols, gene_ids, member_ids = load_counts(directory, section)
    spots_path = workspace / section["source_root"] / "spot_inputs.tsv.gz"
    scores_path = workspace / section["source_root"] / "spot_scores.tsv.gz"
    spots = read(spots_path); scores = read(scores_path)
    if spots.barcode.duplicated().any() or set(spots.barcode) != set(barcodes):
        raise ValueError("Registered barcode universe differs: " + sid)
    order = pd.Index(barcodes).get_indexer(spots.barcode)
    counts = counts[order].tocsr(); names = unique_names(symbols)
    gene_indices = [names.index(gene) for gene in GENES]
    original_total = np.asarray(counts.sum(axis=1)).ravel()
    original_readout = np.log1p(counts[:, gene_indices].toarray()/original_total[:, None]*10000).mean(axis=1)
    primary = scores[scores.score_id.eq("primary_barrier") & scores.status.eq("success")].set_index("barcode")
    if primary.index.duplicated().any() or set(primary.index) != set(spots.barcode):
        raise ValueError("Frozen readout barcode mismatch")
    values = primary.loc[spots.barcode, "score_value"].to_numpy(float)
    if not np.allclose(original_readout, values, rtol=2e-6, atol=2e-7):
        raise ValueError("Raw-count recomposition differs from frozen measurement")
    excluded = np.isin(symbols, GENES)
    fit_total = np.asarray(counts[:, ~excluded].sum(axis=1)).ravel()
    if (fit_total <= 0).any():
        raise ValueError("Non-readout normalization total is zero")
    x = counts.multiply((10000/fit_total)[:, None]).tocsr(); x.data = np.log1p(x.data)
    means = np.asarray(x.mean(axis=0)).ravel()
    variances = np.maximum(np.asarray(x.power(2).mean(axis=0)).ravel()-means**2, 0)
    selected = {}
    for arm in ARMS:
        candidates = np.arange(len(names)) if arm == "allowed" else np.flatnonzero(~excluded)
        selected[arm] = candidates[np.argsort(variances[candidates])[::-1][:2000]]
    universe = pd.DataFrame(dict(feature_index=np.arange(len(names)), feature_id=gene_ids,
                                 gene_symbol=symbols, feature_name=names, fitting_denominator_member=~excluded,
                                 normalized_variance=variances))
    for arm in ARMS:
        ranks = np.full(len(names), np.nan); ranks[selected[arm]] = np.arange(1, len(selected[arm])+1)
        universe[arm + "_PCA_rank"] = ranks
    universe.to_csv(destination / "feature_universe.tsv.gz", sep="\t", index=False)
    coords, stroma, tumor, near, far, blocks, mad = geometry(spots, values)
    anchor = delta(values, near, far); full_ok = eligible(near, far, blocks)
    np.savez_compressed(destination / "registered_measurement.npz", barcodes=spots.barcode.to_numpy(str),
                        values=values, near=near, far=far, blocks=blocks, stroma=stroma, tumor=tumor,
                        coords=coords, fit_normalization_total=fit_total, original_total=original_total)
    fixed_graphs = {n: coordinate_graph(coords, n) for n in [4, 6, 8]}
    scaled = (coords-coords.mean(axis=0))/(coords.std(axis=0)+1e-6)
    rows = []; labels_saved = {}; masks_saved = {}; draws_saved = {}
    for arm in ARMS:
        dense = x[:, selected[arm]].toarray().astype(np.float32)
        pcs = PCA(n_components=min(20, len(spots)-1, dense.shape[1]-1), random_state=0).fit_transform(dense).astype(np.float32)
        np.save(destination / (arm + "_pca.npy"), pcs)
        augmented = np.concatenate([pcs, .5*scaled], axis=1)
        edges = fixed_graphs[6].tocoo(); edge_mask = edges.row < edges.col
        source = edges.row[edge_mask]; target = edges.col[edge_mask]
        normalized = pcs.astype(float); norm = np.linalg.norm(normalized, axis=1, keepdims=True)
        normalized /= np.where(norm == 0, 1, norm)
        weights = np.maximum(0, (1+np.sum(normalized[source]*normalized[target], axis=1))/2)
        lg = ig.Graph(n=len(spots), edges=list(zip(source.tolist(), target.tolist())), directed=False)
        lg.es["weight"] = weights.tolist()
        for method in METHODS:
            for k in [4, 6]:
                for replicate in ([4, 6, 8] if method == "M2_spatial_ward" else [11, 23, 37]):
                    partition = ("neighbors_" if method == "M2_spatial_ward" else "seed_") + str(replicate)
                    uid = "|".join(map(str, [arm, method, k, partition])); resolution = np.nan
                    rec = dict(resource=resource, patient_id=section["patient_id"], sample_id=sid, arm=arm,
                               method_id=method, K=k, partition_id=partition, fit_status="success", fit_reason="",
                               status="not_evaluable", reason="", n_registered=len(spots),
                               n_readout_PCA_features=int(excluded[selected[arm]].sum()),
                               n_anchor_near=int(near.sum()), n_anchor_far=int(far.sum()), anchor_delta_score=anchor,
                               stroma_mad=mad, n_selected_near=0, n_selected_far=0,
                               n_near_blocks=0, n_far_blocks=0, tumor_domain=np.nan, stroma_domain=np.nan,
                               same_PCA_features=bool(np.array_equal(selected["allowed"], selected["withheld"])))
                    for column in METRICS:
                        rec[column] = np.nan
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            if method in METHODS[:2]:
                                features = pcs if method == METHODS[0] else augmented
                                labels = KMeans(n_clusters=k, n_init=50, random_state=replicate).fit_predict(features)+1
                                exact = True
                            elif method == METHODS[2]:
                                labels = AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=fixed_graphs[replicate]).fit_predict(pcs)+1
                                exact = True
                            else:
                                labels, resolution, exact = exact_leiden(lg, k, replicate)
                            rec["fit_warnings"] = " | ".join(dict.fromkeys(str(w.message) for w in caught))
                        rec["fit_observed_K"] = int(np.unique(labels).size); rec["leiden_resolution"] = resolution
                        labels_saved[uid] = labels.astype(np.int16)
                        if not exact:
                            rec.update(fit_status="not_evaluable", fit_reason="no_exact_K_resolution", reason="fit_unavailable")
                        else:
                            fractions = [(int(label), float(tumor[labels == label].mean()),
                                          float(stroma[labels == label].mean()), int(stroma[labels == label].sum()))
                                         for label in np.unique(labels)]
                            if not tumor.any() or not stroma.any():
                                rec["reason"] = "annotation_class_absent"
                            else:
                                td = min(fractions, key=lambda f: (-f[1], f[0]))[0]
                                candidates = [f for f in fractions if f[0] != td and f[3] > 0]
                                if not candidates:
                                    rec["reason"] = "no_distinct_stromal_domain"
                                else:
                                    sd = min(candidates, key=lambda f: (-f[2], f[0]))
                                    sn = near & (labels == sd[0]); sf = far & (labels == sd[0])
                                    masks_saved[uid + "|near"] = sn; masks_saved[uid + "|far"] = sf
                                    masks_saved[uid + "|stroma"] = stroma & (labels == sd[0])
                                    point = delta(values, sn, sf); departure = point-anchor
                                    rec.update(tumor_domain=td, stroma_domain=sd[0], selected_stroma_fraction=sd[2],
                                               n_selected_near=int(sn.sum()), n_selected_far=int(sf.sum()),
                                               n_near_blocks=np.unique(blocks[sn]).size, n_far_blocks=np.unique(blocks[sf]).size,
                                               near_retention=sn.sum()/near.sum() if near.any() else np.nan,
                                               far_retention=sf.sum()/far.sum() if far.any() else np.nan,
                                               selected_delta_score=point, abs_deviation_score=abs(departure),
                                               abs_deviation_mad=abs(departure)/mad if mad > 0 else np.nan)
                                    if not full_ok:
                                        rec["reason"] = "full_band_support_below_threshold"
                                    elif not eligible(sn, sf, blocks):
                                        rec["reason"] = "selected_band_support_below_threshold"
                                    elif not mad > 0:
                                        rec["reason"] = "zero_readout_MAD"
                                    else:
                                        rec["status"] = "success"
                                        for kind in ["count_only", "block_matched"]:
                                            context = "|".join(map(str, [MASTER, resource, sid, method, k, partition, kind]))
                                            draws = control(values, blocks, near, far, sn, sf, kind, context)
                                            draws_saved[uid + "|" + kind] = draws
                                            center = float(np.median(abs(draws-anchor)))
                                            rec[kind + "_control_median_abs_departure_score"] = center
                                            rec[kind + "_excess_mad"] = (abs(departure)-center)/mad
                                            rec[kind + "_rng_key"] = context
                                            rec[kind + "_draws"] = len(draws)
                    except Exception as exc:
                        rec.update(fit_status="failed", fit_reason=str(exc), reason="execution_failed",
                                   exception_trace=traceback.format_exc())
                    rows.append(rec)
        print(json.dumps(dict(sample_id=sid, arm=arm, completed_attempts=sum(r["arm"] == arm for r in rows))), flush=True)
    table = pd.DataFrame(rows); table.to_csv(destination / "map_results.tsv", sep="\t", index=False)
    np.savez_compressed(destination / "fitted_labels.npz", **labels_saved)
    np.savez_compressed(destination / "selected_memberships.npz", **masks_saved)
    np.savez_compressed(destination / "control_draws.npz", **draws_saved)
    manifest = dict(resource=resource, sample_id=sid, patient_id=section["patient_id"],
                    source_script_sha256=sha(__file__),
                    original_members=member_ids, spot_inputs=identity(spots_path), spot_scores=identity(scores_path),
                    raw_readout_max_abs_error=float(np.max(abs(original_readout-values))),
                    n_registered=len(spots), n_source_features=len(names), n_withheld_source_features=int(excluded.sum()),
                    attempts=len(table), fit_success=int(table.fit_status.eq("success").sum()),
                    region_success=int(table.status.eq("success").sum()), seconds=time.monotonic()-started,
                    files=[identity(p) for p in sorted(destination.iterdir()) if p.is_file()])
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def summarize(output):
    table = pd.concat([read(p) for p in sorted((output / "sections").glob("*/*/map_results.tsv"))], ignore_index=True)
    if len(table) != 1056 or table.duplicated(["resource", "sample_id", "arm", "method_id", "K", "partition_id"]).any():
        raise ValueError("Incomplete attempt grid")
    table.to_csv(output / "map_results.tsv", sep="\t", index=False)
    paired = []; allowed = table[table.arm.eq("allowed")]; withheld = table[table.arm.eq("withheld")]
    keys = ["resource", "patient_id", "sample_id", "method_id", "K", "partition_id"]
    merged = allowed.merge(withheld, on=keys, suffixes=("_allowed", "_withheld"), validate="one_to_one")
    for section, group in merged.groupby(["resource", "sample_id"]):
        directory = output / "sections" / section[0] / section[1]
        with np.load(directory / "fitted_labels.npz") as labels, np.load(directory / "selected_memberships.npz") as masks:
            for r in group.to_dict("records"):
                rec = {k: r[k] for k in keys}
                rec.update(allowed_fit_status=r["fit_status_allowed"], withheld_fit_status=r["fit_status_withheld"],
                           allowed_status=r["status_allowed"], withheld_status=r["status_withheld"],
                           common_evaluable=r["status_allowed"] == r["status_withheld"] == "success",
                           same_PCA_features=r["same_PCA_features_allowed"], n_readout_PCA_features=r["n_readout_PCA_features_allowed"],
                           whole_map_ARI=np.nan, stromal_domain_Jaccard=np.nan, band_selection_Jaccard=np.nan)
                a = "|".join(map(str, ["allowed", r["method_id"], r["K"], r["partition_id"]]))
                b = a.replace("allowed|", "withheld|", 1)
                if r["fit_status_allowed"] == r["fit_status_withheld"] == "success":
                    rec["whole_map_ARI"] = adjusted_rand_score(labels[a], labels[b])
                    if a + "|stroma" in masks and b + "|stroma" in masks:
                        for name, suffix in [("stromal_domain_Jaccard", "stroma"), ("band_selection_Jaccard", None)]:
                            left = masks[a + "|" + suffix] if suffix else masks[a + "|near"] | masks[a + "|far"]
                            right = masks[b + "|" + suffix] if suffix else masks[b + "|near"] | masks[b + "|far"]
                            union = (left | right).sum()
                            rec[name] = (left & right).sum()/union if union else np.nan
                for metric in METRICS:
                    rec[metric + "_allowed"] = r[metric + "_allowed"]
                    rec[metric + "_withheld"] = r[metric + "_withheld"]
                    rec[metric + "_change"] = r[metric + "_withheld"]-r[metric + "_allowed"]
                paired.append(rec)
    pairs = pd.DataFrame(paired); pairs.to_csv(output / "paired_map_results.tsv", sep="\t", index=False)
    common_keys = set(map(tuple, pairs.loc[pairs.common_evaluable, keys].to_numpy()))
    table["common_evaluable"] = [tuple(row) in common_keys for row in table[keys].to_numpy()]
    table.to_csv(output / "map_results.tsv", sep="\t", index=False)
    patient_rows = []; section_rows = []; resource_rows = []
    for frame in ["arm_specific", "common_configurations"]:
        valid = table[table.status.eq("success") & (table.common_evaluable if frame == "common_configurations" else True)]
        cells = valid.groupby(keys[:-1]+["arm"])[METRICS].median().reset_index()
        sec = cells.groupby(["resource", "patient_id", "sample_id", "arm"])[METRICS].median().reset_index()
        base = table.groupby(["resource", "patient_id", "sample_id", "arm"]).size().rename("n_declared_maps").reset_index()
        counts = valid.groupby(["resource", "patient_id", "sample_id", "arm"]).size().rename("n_evaluable_maps").reset_index()
        sec = base.merge(counts, how="left").merge(sec, how="left"); sec.n_evaluable_maps = sec.n_evaluable_maps.fillna(0).astype(int)
        sec["frame"] = frame; section_rows.append(sec)
        for (resource, patient, arm), group in sec.groupby(["resource", "patient_id", "arm"]):
            rec = dict(resource=resource, patient_id=patient, arm=arm, frame=frame,
                       n_attempted_sections=len(group), n_evaluable_sections=int(group.n_evaluable_maps.gt(0).sum()),
                       n_declared_maps=int(group.n_declared_maps.sum()), n_evaluable_maps=int(group.n_evaluable_maps.sum()))
            rec.update(group[METRICS].median().to_dict()); patient_rows.append(rec)
    patients = pd.DataFrame(patient_rows)
    for (frame, resource, arm), group in patients.groupby(["frame", "resource", "arm"]):
        rec = dict(frame=frame, resource=resource, arm=arm, n_attempted_patients=len(group),
                   n_evaluable_patients=int(group.n_evaluable_maps.gt(0).sum()), n_evaluable_maps=int(group.n_evaluable_maps.sum()),
                   n_evaluable_sections=int(group.n_evaluable_sections.sum()))
        rec.update(group[METRICS].median().to_dict()); resource_rows.append(rec)
    patients.to_csv(output / "patient_summary.tsv", sep="\t", index=False)
    pd.concat(section_rows).to_csv(output / "section_summary.tsv", sep="\t", index=False)
    pd.DataFrame(resource_rows).to_csv(output / "resource_summary.tsv", sep="\t", index=False)
    table.groupby(["resource", "arm", "method_id", "fit_status", "status", "reason"], dropna=False).size().rename("n_maps").reset_index().to_csv(output / "availability.tsv", sep="\t", index=False)
    return dict(attempts=len(table), paired_configurations=len(pairs), common_evaluable=int(pairs.common_evaluable.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["workspace", "count-root", "source-manifest", "plan", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=2); args = p.parse_args()
    source = json.loads(args.source_manifest.read_text()); sections = source["sections"]
    if len(sections) != 22 or source["schema_version"] != "source_counts_manifest_v1":
        raise ValueError("Unexpected source manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    context = {k: str(v.resolve()) for k, v in vars(args).items() if isinstance(v, Path)}
    start = dict(created_utc=datetime.now(timezone.utc).isoformat(), analysis_class="post_outcome_paired_feature_withholding",
                 plan=identity(args.plan), source_manifest=identity(args.source_manifest), source_script=identity(__file__),
                 genes=list(GENES), master_seed=MASTER, workers=args.workers,
                 software=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                               sklearn=sklearn.__version__, h5py=h5py.__version__, igraph=ig.__version__,
                               leidenalg=getattr(leidenalg, "__version__", "unknown"), platform=platform.platform()))
    (args.output / "started_manifest.json").write_text(json.dumps(start, indent=2)+"\n")
    manifests = []
    with threadpool_limits(limits=1), ProcessPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(section_job, s, context) for s in sections]):
            m = future.result(); manifests.append(m)
            print(json.dumps(dict(sample_id=m["sample_id"], completed_sections=len(manifests), seconds=m["seconds"])), flush=True)
    summary = summarize(args.output)
    finish = {**start, **summary, "sections": manifests,
              "files": [dict(relative_path=str(f.relative_to(args.output)), **identity(f))
                        for f in sorted(args.output.rglob("*")) if f.is_file()]}
    (args.output / "run_manifest.json").write_text(json.dumps(finish, indent=2)+"\n")
    print(json.dumps(dict(complete=True, **summary)), flush=True)


if __name__ == "__main__":
    main()
