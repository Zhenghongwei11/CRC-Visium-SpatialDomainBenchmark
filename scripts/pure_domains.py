#!/usr/bin/env python3
"""Construct connected high-purity regions using finite clustering settings."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "2")

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parent/'analysis_pipeline'))
from run_jbcb_readout_withholding import load_counts, unique_names

ROOT = Path(__file__).resolve().parents[1]
OFFSETS = ((0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1))
FOUR = ("TGFB1", "CXCL12", "ACTA2", "TAGLN")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def graph_and_geometry(spots):
    lattice = spots[["array_row", "array_col"]].to_numpy(int)
    if len(set(map(tuple, lattice))) != len(lattice):
        raise ValueError("Repeated registered lattice locations")
    lookup = {tuple(x): i for i, x in enumerate(lattice)}
    edges = []
    for i, (r, c) in enumerate(lattice):
        for dr, dc in OFFSETS:
            j = lookup.get((r + dr, c + dc))
            if j is not None and i < j:
                edges.append((i, j))
    edges = np.asarray(edges, int).reshape(-1, 2)
    row = np.r_[edges[:, 0], edges[:, 1]]
    col = np.r_[edges[:, 1], edges[:, 0]]
    graph = csr_matrix((np.ones(len(row)), (row, col)), shape=(len(spots), len(spots)))
    stroma = spots.coarse_label.eq("stroma").to_numpy()
    tumor = spots.coarse_label.eq("tumor").to_numpy()
    near, far = np.zeros(len(spots), bool), np.zeros(len(spots), bool)
    if tumor.any() and stroma.any():
        xy = np.c_[lattice[:, 1] / 2, np.sqrt(3) * lattice[:, 0] / 2]
        _, nearest = cKDTree(xy[tumor]).query(xy[stroma])
        delta = lattice[stroma] - lattice[tumor][nearest]
        q = np.full(len(spots), np.inf)
        q[stroma] = delta[:, 1] ** 2 + 3 * delta[:, 0] ** 2
        _, comp = connected_components(graph[stroma][:, stroma], directed=False)
        fullcomp = np.full(len(spots), -1)
        fullcomp[stroma] = comp
        near = stroma & (q <= 4)
        far = stroma & (q >= 16) & (q <= 64) & np.isin(fullcomp, np.unique(fullcomp[near]))
    axes = [np.searchsorted(np.linspace(lattice[:, i].min(), lattice[:, i].max(), 7)[1:-1],
                            lattice[:, i], side="right") for i in (0, 1)]
    blocks = axes[0] * 6 + axes[1]
    return graph, edges, stroma, near, far, blocks


def eligible(near, far, blocks):
    return bool(min(near.sum(), far.sum()) >= 20 and
                min(len(np.unique(blocks[near])), len(np.unique(blocks[far]))) >= 3)


def split_connected(labels, graph):
    result = np.full(len(labels), -1, int)
    nextid = 0
    for label in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == label)
        count, components = connected_components(graph[indices][:, indices], directed=False)
        result[indices] = components + nextid
        nextid += count
    assert (result >= 0).all()
    return result


def fit_ward(pcs, graph, requested_k):
    ncomp, components = connected_components(graph, directed=False)
    sizes = np.bincount(components)
    allocation = np.maximum(1, np.floor(requested_k * sizes / len(pcs)).astype(int))
    allocation = np.minimum(allocation, sizes)
    target = max(requested_k, ncomp)
    while allocation.sum() < target:
        candidates = np.flatnonzero(allocation < sizes)
        if not len(candidates):
            break
        i = candidates[np.argmax(requested_k * sizes[candidates] / len(pcs) - allocation[candidates])]
        allocation[i] += 1
    labels = np.full(len(pcs), -1, int)
    offset = 0
    for comp, k in enumerate(allocation):
        indices = np.flatnonzero(components == comp)
        if k == 1:
            local = np.zeros(len(indices), int)
        elif k == len(indices):
            local = np.arange(len(indices))
        else:
            local = AgglomerativeClustering(n_clusters=int(k), linkage="ward",
                        connectivity=graph[indices][:, indices]).fit_predict(pcs[indices])
        labels[indices] = local + offset
        offset += int(k)
    return labels


def merge_regions(initial, edges, pcs, stroma, quantile):
    regions = {}
    for label in np.unique(initial):
        members = np.flatnonzero(initial == label)
        regions[int(label)] = dict(members=members, total=len(members), ns=int(stroma[members].sum()),
                                   sum=pcs[members].sum(axis=0), lowest=int(label), neighbors=set())
    adjacency = set()
    for i, j in edges:
        a, b = sorted((int(initial[i]), int(initial[j])))
        if a != b:
            adjacency.add((a, b))
    distances = []
    for a, b in sorted(adjacency):
        regions[a]["neighbors"].add(b)
        regions[b]["neighbors"].add(a)
        distances.append(float(np.linalg.norm(regions[a]["sum"] / regions[a]["total"] -
                                              regions[b]["sum"] / regions[b]["total"])))
    threshold = float(np.quantile(distances, quantile)) if distances else 0.0
    queue = []

    def propose(a, b):
        a, b = sorted((a, b))
        x, y = regions[a], regions[b]
        if (x["ns"] + y["ns"]) / (x["total"] + y["total"]) < .90:
            return
        distance = float(np.linalg.norm(x["sum"] / x["total"] - y["sum"] / y["total"]))
        if distance <= threshold:
            cost = distance * distance * x["total"] * y["total"] / (x["total"] + y["total"])
            heapq.heappush(queue, (cost, a, b))

    for a, b in sorted(adjacency):
        propose(a, b)
    nextid = max(regions) + 1
    merges = 0
    while queue:
        _, a, b = heapq.heappop(queue)
        if a not in regions or b not in regions or b not in regions[a]["neighbors"]:
            continue
        x, y = regions.pop(a), regions.pop(b)
        neighbors = (x["neighbors"] | y["neighbors"]) - {a, b}
        z = dict(members=np.r_[x["members"], y["members"]], total=x["total"] + y["total"],
                 ns=x["ns"] + y["ns"], sum=x["sum"] + y["sum"],
                 lowest=min(x["lowest"], y["lowest"]), neighbors=set(neighbors))
        regions[nextid] = z
        for neighbor in sorted(neighbors):
            regions[neighbor]["neighbors"].difference_update({a, b})
            regions[neighbor]["neighbors"].add(nextid)
            propose(neighbor, nextid)
        merges += 1
        nextid += 1
    return regions, threshold, merges


def summarize_region(regions, near, far, blocks, n):
    pure = [r for r in regions.values() if r["ns"] / r["total"] >= .90]
    pure.sort(key=lambda r: (-r["ns"], -r["total"], r["lowest"]))
    selected = np.zeros(n, bool)
    if pure:
        selected[pure[0]["members"]] = True
    any_eligible = False
    for r in pure:
        mask = np.zeros(n, bool)
        mask[r["members"]] = True
        any_eligible |= eligible(mask & near, mask & far, blocks)
    sn, sf = selected & near, selected & far
    return selected, dict(n_final_regions=len(regions), n_pure_regions=len(pure),
        selected_region_size=int(selected.sum()), selected_stroma_count=pure[0]["ns"] if pure else 0,
        whole_domain_stroma_fraction=pure[0]["ns"] / pure[0]["total"] if pure else np.nan,
        n_selected_near=int(sn.sum()), n_selected_far=int(sf.sum()),
        n_selected_near_blocks=len(np.unique(blocks[sn])), n_selected_far_blocks=len(np.unique(blocks[sf])),
        primary_evaluable=eligible(sn, sf, blocks), any_pure_region_evaluable=any_eligible)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "config/regional_profiles.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pure_domains")
    parser.add_argument("--sample", action="append")
    parser.add_argument("--counts", type=Path, default=ROOT / "original_inputs/counts")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    lock = json.loads((args.protocol.parent / "PROTOCOL_LOCK_v1.json").read_text())
    assert sha(args.protocol) == lock["sha256"]
    manifest_path = ROOT / protocol["source_manifest"]
    sections = json.loads(manifest_path.read_text())["sections"]
    if args.sample:
        sections = [s for s in sections if s["sample_id"] in args.sample]
    args.output.mkdir(parents=True, exist_ok=True)
    rows, provenance = [], []
    start = time.time()
    with threadpool_limits(limits=2):
        for section in sections:
            sid, resource = section["sample_id"], section["resource"]
            out = args.output / "sections" / resource / sid
            done = out / "DONE.json"
            if done.exists():
                checkpoint = json.loads(done.read_text())
                if checkpoint["protocol_sha256"] != sha(args.protocol) or checkpoint["script_sha256"] != sha(__file__):
                    raise ValueError("Checkpoint belongs to different code or protocol")
                for name, digest in checkpoint["files"].items():
                    assert sha(out / name) == digest
                rows.extend(pd.read_csv(out / "coverage.tsv", sep="\t").to_dict("records"))
                provenance.append(checkpoint)
                print("REUSED", resource, sid, flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            counts, barcodes, symbols, _, members = load_counts(args.counts.resolve() / resource / sid / "members", section)
            spot_path = ROOT / section["source_root"] / "spot_inputs.tsv.gz"
            spots = pd.read_csv(spot_path, sep="\t")
            order = pd.Index(barcodes).get_indexer(spots.barcode)
            assert len(order) == len(barcodes) and min(order) >= 0 and len(set(order)) == len(order)
            counts = counts[order].tocsr()
            graph, edges, stroma, near, far, blocks = graph_and_geometry(spots)
            saved_geo = ROOT / "data/reference_inputs/geometry" / resource / sid / "geometry_arrays.npz"
            with np.load(saved_geo) as saved:
                assert np.array_equal(near, saved["nominal_honeycomb_exact|baseline|near"])
                assert np.array_equal(far, saved["nominal_honeycomb_exact|baseline|far"])
            excluded = np.isin(symbols, FOUR)
            denom = np.asarray(counts[:, ~excluded].sum(axis=1)).ravel()
            assert (denom > 0).all()
            x = counts.multiply((1e4 / denom)[:, None]).tocsr()
            x.data = np.log1p(x.data)
            means = np.asarray(x.mean(axis=0)).ravel()
            variance = np.maximum(np.asarray(x.power(2).mean(axis=0)).ravel() - means ** 2, 0)
            candidates = np.flatnonzero(~excluded)
            indices = candidates[np.argsort(-variance[candidates], kind="stable")[:2000]]
            dense = x[:, indices].toarray().astype(np.float32)
            pcs = PCA(n_components=min(20, len(spots)-1, len(indices)-1), svd_solver="randomized", random_state=11).fit_transform(dense).astype(float)
            pd.DataFrame({"feature_index": indices, "gene_symbol": np.asarray(symbols)[indices], "variance": variance[indices]}).to_csv(out / "fit_features.tsv", sep="\t", index=False)
            del dense, x
            names = unique_names(symbols)
            values = np.log1p(counts[:, [names.index(g) for g in FOUR]].toarray() /
                             np.asarray(counts.sum(axis=1)).ravel()[:, None] * 1e4)
            masks, localrows = {}, []
            for method in protocol["domain_pilot"]["methods"]:
                for k in protocol["domain_pilot"]["K"]:
                    labels = KMeans(n_clusters=k, n_init=10, random_state=11).fit_predict(pcs) if method == "expression_kmeans" else fit_ward(pcs, graph, k)
                    initial = split_connected(labels, graph)
                    for quantile in (.25, .50):
                        regions, threshold, merges = merge_regions(initial, edges, pcs, stroma, quantile)
                        selected, summary = summarize_region(regions, near, far, blocks, len(spots))
                        key = f"{method}|K{k}|q{int(quantile*100)}"
                        masks[key] = selected
                        summary.update(resource=resource, patient_id=section["patient_id"], sample_id=sid,
                            method=method, requested_K=k, actual_fitted_K=len(np.unique(labels)),
                            merge_quantile=quantile, merge_threshold=threshold, n_merges=merges,
                            n_initial_connected_regions=len(np.unique(initial)),
                            anatomical_evaluable=eligible(near, far, blocks),
                            n_full_near=int(near.sum()), n_full_far=int(far.sum()), mask_key=key)
                        localrows.append(summary)
            pd.DataFrame(localrows).to_csv(out / "coverage.tsv", sep="\t", index=False)
            np.savez_compressed(out / "selected_regions.npz", **masks)
            np.savez_compressed(out / "measurement.npz", values=values, near=near, far=far, blocks=blocks,
                                barcodes=spots.barcode.to_numpy(str))
            checkpoint = dict(resource=resource, sample_id=sid, protocol_sha256=sha(args.protocol),
                script_sha256=sha(__file__), spot_input_sha256=sha(spot_path), counts=members,
                files={name: sha(out / name) for name in ("coverage.tsv", "selected_regions.npz", "measurement.npz", "fit_features.tsv")})
            done.write_text(json.dumps(checkpoint, indent=2) + "\n")
            rows.extend(localrows)
            provenance.append(checkpoint)
            print("DONE", resource, sid, "eligible", sum(r["primary_evaluable"] for r in localrows), "of", len(localrows), flush=True)
    coverage = pd.DataFrame(rows)
    coverage.to_csv(args.output / "coverage.tsv", sep="\t", index=False)
    summaries = []
    for keys, group in coverage.groupby(["resource", "method", "requested_K", "merge_quantile"]):
        good = group[group.primary_evaluable.astype(bool)]
        anygood = group[group.any_pure_region_evaluable.astype(bool)]
        summaries.append(dict(zip(["resource", "method", "K", "merge_quantile"], keys),
             attempted_sections=len(group), attempted_patients=group.patient_id.nunique(),
             eligible_sections=len(good), eligible_patients=good.patient_id.nunique(),
             any_region_eligible_sections=len(anygood), any_region_eligible_patients=anygood.patient_id.nunique()))
    pd.DataFrame(summaries).to_csv(args.output / "resource_coverage.tsv", sep="\t", index=False)
    coverage_lock = dict(locked_utc=datetime.now(timezone.utc).isoformat(), coverage_sha256=sha(args.output / "coverage.tsv"),
                         note="Coverage fixed before calculating new selected-region expression contrasts.")
    (args.output / "COVERAGE_LOCK.json").write_text(json.dumps(coverage_lock, indent=2) + "\n")
    # Only after the coverage output is fixed, describe contrasts without using
    # them to select the construction or its retained regions.
    contrasts = []
    for section in sections:
        out = args.output / "sections" / section["resource"] / section["sample_id"]
        local = coverage[coverage.sample_id.eq(section["sample_id"])]
        with np.load(out / "measurement.npz") as measure, np.load(out / "selected_regions.npz") as masks:
            near, far, v = measure["near"], measure["far"], measure["values"]
            readouts = {g: v[:, i] for i, g in enumerate(FOUR)}
            readouts.update(four_gene_mean=v.mean(axis=1), signaling_pair=v[:, :2].mean(axis=1), contractile_pair=v[:, 2:].mean(axis=1))
            for _, row in local.iterrows():
                if not row.primary_evaluable:
                    continue
                mask = masks[row.mask_key]
                for name, value in readouts.items():
                    full = float(np.median(value[near]) - np.median(value[far]))
                    selected = float(np.median(value[near & mask]) - np.median(value[far & mask]))
                    contrasts.append(dict(resource=row.resource, patient_id=row.patient_id, sample_id=row.sample_id,
                        method=row.method, requested_K=row.requested_K, merge_quantile=row.merge_quantile,
                        readout=name, full_contrast=full, selected_contrast=selected, selected_minus_full=selected-full))
    pd.DataFrame(contrasts).to_csv(args.output / "contrasts.tsv", sep="\t", index=False)
    final = dict(status="completed_finite_domain_pilot_not_cell_identity_validation", n_sections=len(sections),
        elapsed_seconds=time.time()-start, protocol_sha256=sha(args.protocol), script_sha256=sha(__file__),
        source_manifest_sha256=sha(manifest_path), python=platform.python_version(), sections=provenance,
        outputs={p.name:sha(p) for p in args.output.iterdir() if p.is_file() and p.name != "RUN_MANIFEST.json"})
    (args.output / "RUN_MANIFEST.json").write_text(json.dumps(final, indent=2) + "\n")
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
