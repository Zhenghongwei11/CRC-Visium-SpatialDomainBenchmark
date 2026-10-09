#!/usr/bin/env python3
"""Independently rebuild all new fitting inputs, maps, controls and summaries."""
import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import time
import warnings

import h5py
import igraph
import leidenalg
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

GENES = {"TGFB1", "CXCL12", "ACTA2", "TAGLN"}
METRICS = ["selected_delta_score", "abs_deviation_score", "abs_deviation_mad", "near_retention",
           "far_retention", "selected_stroma_fraction", "count_only_excess_mad", "block_matched_excess_mad"]


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def close(a, b, label, tolerance=1e-10):
    if not np.allclose(a, b, rtol=0, atol=tolerance, equal_nan=True):
        raise AssertionError(label)


def table(path):
    return pd.read_csv(path, sep="\t", float_precision="round_trip", low_memory=False)


def controls(values, blocks, near, far, sn, sf, blockwise, key):
    generator = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
    answers = []
    for whole, selected in [(near, sn), (far, sf)]:
        selections = []
        cells = np.unique(blocks[selected]) if blockwise else [None]
        for cell in cells:
            pool = np.where(whole if cell is None else whole & (blocks == cell))[0]
            n = int(selected.sum()) if cell is None else int((selected & (blocks == cell)).sum())
            sample = np.tile(pool, (1000, 1)) if len(pool) == n else np.array([
                generator.choice(pool, size=n, replace=False) for _ in range(1000)])
            selections.append(sample)
        selected_indices = np.hstack(selections)
        answers.append(np.median(values[selected_indices], axis=1))
    return answers[0]-answers[1]


def verify_section(section, options):
    sid = section["sample_id"]; resource = section["resource"]
    directory = Path(options["output"]) / "sections" / resource / sid
    manifest = directory / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    record = json.loads(manifest.read_text())
    for file in record["files"]:
        p = directory / file["name"]
        assert p.stat().st_size == file["bytes"] and digest(p) == file["sha256"], file["name"]
    raw = Path(options["count_root"]) / resource / sid / "members"
    for file in section["source_files"]:
        p = raw / file["name"]
        assert digest(p) == file["sha256"] and p.stat().st_size == file["size_bytes"]
    h5 = raw / (sid + "_filtered_feature_bc_matrix.h5")
    if h5.exists():
        with h5py.File(h5) as f:
            node = f["matrix"]
            matrix = csc_matrix((node["data"][:], node["indices"][:], node["indptr"][:]),
                                shape=tuple(node["shape"][:])).transpose().tocsr().astype(float)
            barcodes = [x.decode() for x in node["barcodes"][:]]
            symbols = [x.decode() for x in node["features/name"][:]]
    else:
        with gzip.open(raw / (sid + "_matrix.mtx.gz"), "rb") as f:
            matrix = mmread(f).transpose().tocsr().astype(float)
        with gzip.open(raw / (sid + "_barcodes.tsv.gz"), "rt") as f:
            barcodes = [line.strip() for line in f if line.strip()]
        symbols = pd.read_csv(raw / (sid + "_features.tsv.gz"), header=None, sep="\t").iloc[:, 1].astype(str).tolist()
    root = Path(options["workspace"]) / section["source_root"]
    spots = table(root / "spot_inputs.tsv.gz"); scores = table(root / "spot_scores.tsv.gz")
    assert spots.barcode.is_unique and set(spots.barcode) == set(barcodes)
    barcode_lookup = {b: i for i, b in enumerate(barcodes)}
    matrix = matrix[[barcode_lookup[b] for b in spots.barcode]].tocsr()
    frozen = scores[scores.score_id.eq("primary_barrier")].set_index("barcode").loc[spots.barcode, "score_value"].to_numpy()
    raw_total = np.asarray(matrix.sum(axis=1)).flatten()
    gene_index = [symbols.index(g) for g in ["TGFB1", "CXCL12", "ACTA2", "TAGLN"]]
    recomposed = np.log1p(matrix[:, gene_index].toarray()*10000/raw_total[:, None]).mean(axis=1)
    assert np.allclose(recomposed, frozen, rtol=2e-6, atol=2e-7)
    held = np.array([name in GENES for name in symbols])
    normalization = raw_total - np.asarray(matrix[:, held].sum(axis=1)).flatten()
    normalized = matrix.multiply((10000/normalization)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    mean = np.asarray(normalized.sum(axis=0)).flatten()/matrix.shape[0]
    var = np.maximum(np.asarray(normalized.multiply(normalized).sum(axis=0)).flatten()/matrix.shape[0]-mean**2, 0)
    features = table(directory / "feature_universe.tsv.gz")
    assert features.gene_symbol.tolist() == symbols
    assert np.array_equal(features.fitting_denominator_member.to_numpy(), ~held)
    close(features.normalized_variance, var, "full feature variance", 1e-10)
    pcs = {}; selected = {}
    for arm in ["allowed", "withheld"]:
        candidates = np.arange(len(symbols)) if arm == "allowed" else np.where(~held)[0]
        selected[arm] = candidates[np.argsort(var[candidates])[::-1][:2000]]
        reported = features[features[arm + "_PCA_rank"].notna()].sort_values(arm + "_PCA_rank").feature_index.to_numpy(int)
        assert np.array_equal(reported, selected[arm]), "actual PCA gene selection differs"
        if arm == "withheld":
            assert not held[reported].any(), "Readout gene entered withheld PCA"
        fitted = PCA(n_components=min(20, matrix.shape[0]-1, len(reported)-1), random_state=0).fit_transform(
            normalized[:, reported].toarray().astype(np.float32)).astype(np.float32)
        saved = np.load(directory / (arm + "_pca.npy"))
        close(fitted, saved, "PCA scores", 2e-5); pcs[arm] = saved
    lattice = spots[["array_row", "array_col"]].to_numpy(); n = len(spots)
    tree = cKDTree(lattice); distances, _ = tree.query(lattice, k=7)
    edges = []
    for i in range(n):
        candidates = np.asarray(tree.query_ball_point(lattice[i], np.nextafter(distances[i, -1], np.inf)))
        candidates = candidates[candidates != i]
        sqdist = ((lattice[candidates]-lattice[i])**2).sum(axis=1)
        targets = candidates[np.lexsort((candidates, sqdist))[:6]]
        assert len(targets) == 6
        edges.extend((i, int(j)) for j in targets)
    stroma = spots.coarse_label.eq("stroma").to_numpy(); tumor = spots.coarse_label.eq("tumor").to_numpy()
    src, dst = np.array(edges).T; keep = stroma[src] & stroma[dst]
    graph = csr_matrix((np.ones(keep.sum()), (src[keep], dst[keep])), shape=(n, n))
    _, component = connected_components(graph, directed=False)
    q = np.full(n, np.nan)
    if tumor.any():
        canonical = lattice*np.array([np.sqrt(3), 1])
        # Use integer q directly at the nearest point identified in canonical space.
        _, neighbor = cKDTree(canonical[tumor]).query(canonical)
        difference = lattice-lattice[tumor][neighbor]
        q = 3*difference[:, 0]**2+difference[:, 1]**2
    near = stroma & (q <= 4)
    far = stroma & (q >= 16) & (q <= 64) & np.isin(component, component[near])
    bins = []
    for axis in range(2):
        boundaries = np.linspace(lattice[:, axis].min(), lattice[:, axis].max(), 7)[1:-1]
        bins.append(np.sum(lattice[:, axis, None] >= boundaries, axis=1))
    blocks = 6*bins[0]+bins[1]
    measurement = np.load(directory / "registered_measurement.npz")
    for field, expected in [("values", frozen), ("near", near), ("far", far), ("blocks", blocks),
                            ("stroma", stroma), ("tumor", tumor), ("fit_normalization_total", normalization),
                            ("original_total", raw_total)]:
        close(measurement[field], expected, field)
    anchor = np.median(frozen[near])-np.median(frozen[far]) if near.any() and far.any() else np.nan
    mad = np.median(abs(frozen[stroma]-np.median(frozen[stroma]))) if stroma.any() else np.nan
    old = table(Path(options["reference_anchors"]))
    old = old[old.resource.eq(resource) & old.sample_id.eq(sid)].iloc[0]
    close([near.sum(), far.sum(), anchor, mad],
          [old.n_anchor_near, old.n_anchor_far, old.morphology_anchor_delta, old.primary_scale_mad], "published anchor")
    coords = spots[["x_fullres", "y_fullres"]].to_numpy(float)
    graphs = {}
    for neighbors in [4, 6, 8]:
        knn = NearestNeighbors(n_neighbors=neighbors+1).fit(coords).kneighbors_graph(coords, mode="connectivity")
        graphs[neighbors] = knn.maximum(knn.T)
    labels = np.load(directory / "fitted_labels.npz"); masks = np.load(directory / "selected_memberships.npz")
    draws = np.load(directory / "control_draws.npz"); rows = table(directory / "map_results.tsv")
    assert len(rows) == 48 and not rows.fit_status.eq("failed").any(), "An execution exception remains"
    count_draws = 0; fits_verified = 0
    for row in rows.to_dict("records"):
        arm, method, k, partition = row["arm"], row["method_id"], row["K"], row["partition_id"]
        uid = "|".join(map(str, [arm, method, k, partition])); replicate = int(partition.split("_")[-1])
        if uid not in labels:
            raise AssertionError("Fit attempt has no saved labels")
        fit_input = pcs[arm]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "M0_expr_kmeans":
                expected_labels = KMeans(n_clusters=k, n_init=50, random_state=replicate).fit_predict(fit_input)+1
            elif method == "M1_spatial_concat_kmeans":
                augmented = np.column_stack([fit_input, .5*(coords-coords.mean(0))/(coords.std(0)+1e-6)])
                expected_labels = KMeans(n_clusters=k, n_init=50, random_state=replicate).fit_predict(augmented)+1
            elif method == "M2_spatial_ward":
                expected_labels = AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=graphs[replicate]).fit_predict(fit_input)+1
            else:
                g = graphs[6].tocoo(); use = g.row < g.col; a, b = g.row[use], g.col[use]
                values = fit_input.astype(float); lengths = np.linalg.norm(values, axis=1)
                values /= np.where(lengths == 0, 1, lengths)[:, None]
                weights = np.maximum(0, (1+(values[a]*values[b]).sum(1))/2)
                lg = igraph.Graph(n=n, edges=list(zip(a.tolist(), b.tolist())), directed=False)
                lg.es["weight"] = weights.tolist()
                expected_labels = np.array(leidenalg.find_partition(lg, leidenalg.RBConfigurationVertexPartition,
                    weights="weight", resolution_parameter=row["leiden_resolution"], seed=replicate).membership)+1
        assert np.array_equal(expected_labels, labels[uid]), "Full map replay differs: " + uid
        fits_verified += 1
        close(row["n_readout_PCA_features"], held[selected[arm]].sum(), "readout feature incidence")
        if row["fit_status"] != "success":
            assert np.unique(labels[uid]).size != k and row["fit_reason"] == "no_exact_K_resolution"
            continue
        assert np.unique(labels[uid]).size == k
        if not stroma.any() or not tumor.any():
            assert row["reason"] == "annotation_class_absent"; continue
        labs = labels[uid]
        fractions = {int(l): (tumor[labs == l].mean(), stroma[labs == l].mean(), stroma[labs == l].sum())
                     for l in np.unique(labs)}
        td = sorted(fractions, key=lambda l: (-fractions[l][0], l))[0]
        candidates = [l for l in fractions if l != td and fractions[l][2] > 0]
        if not candidates:
            assert row["reason"] == "no_distinct_stromal_domain"; continue
        sd = sorted(candidates, key=lambda l: (-fractions[l][1], l))[0]
        sn = near & (labs == sd); sf = far & (labs == sd)
        close([row["tumor_domain"], row["stroma_domain"], row["selected_stroma_fraction"]], [td, sd, fractions[sd][1]], "domain mapping")
        for suffix, group in [("near", sn), ("far", sf), ("stroma", stroma & (labs == sd))]:
            assert np.array_equal(masks[uid + "|" + suffix], group), "Selected mask differs"
        point = np.median(frozen[sn])-np.median(frozen[sf]) if sn.any() and sf.any() else np.nan
        good = lambda a, b: min(a.sum(), b.sum()) >= 20 and min(np.unique(blocks[a]).size, np.unique(blocks[b]).size) >= 3
        should_be_evaluable = good(near, far) and good(sn, sf) and mad > 0
        assert (row["status"] == "success") == should_be_evaluable
        close([row["selected_delta_score"], row["abs_deviation_score"], row["abs_deviation_mad"],
               row["near_retention"], row["far_retention"], row["n_selected_near"], row["n_selected_far"],
               row["n_near_blocks"], row["n_far_blocks"]],
              [point, abs(point-anchor), abs(point-anchor)/mad if mad > 0 else np.nan,
               sn.sum()/near.sum() if near.any() else np.nan, sf.sum()/far.sum() if far.any() else np.nan,
               sn.sum(), sf.sum(), np.unique(blocks[sn]).size, np.unique(blocks[sf]).size], "Regional values")
        if should_be_evaluable:
            for kind in ["count_only", "block_matched"]:
                key = "|".join(map(str, ["jbcb-readout-withholding-20261007-v1", resource, sid, method, k, partition, kind]))
                assert row[kind + "_rng_key"] == key and row[kind + "_draws"] == 1000
                expected_draws = controls(frozen, blocks, near, far, sn, sf, kind == "block_matched", key)
                close(draws[uid + "|" + kind], expected_draws, "Every matched draw", 0)
                center = np.median(abs(expected_draws-anchor))
                close([row[kind + "_control_median_abs_departure_score"], row[kind + "_excess_mad"]],
                      [center, (abs(point-anchor)-center)/mad], "Control summary")
                count_draws += 1000
    return dict(resource=resource, sample_id=sid, fits_verified=fits_verified,
                control_draws_verified=count_draws, source_features_verified=len(symbols),
                readout_gene_features_withheld=int(held.sum()), status="pass")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["workspace", "count-root", "source-manifest", "plan", "output", "reference-anchors"]:
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=1); args = p.parse_args()
    sections = json.loads(args.source_manifest.read_text())["sections"]
    options = {k: str(v.resolve()) for k, v in vars(args).items() if isinstance(v, Path)}
    checked = []
    with threadpool_limits(limits=1), ProcessPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(verify_section, s, options) for s in sections]):
            result = future.result(); checked.append(result)
            print(json.dumps(dict(sample_id=result["sample_id"], independently_verified_sections=len(checked))), flush=True)
    if not (args.output / "run_manifest.json").exists():
        raise FileNotFoundError("Complete the withholding analysis before checking it")
    run = json.loads((args.output / "run_manifest.json").read_text())
    assert run["plan"]["sha256"] == digest(args.plan) and run["source_manifest"]["sha256"] == digest(args.source_manifest)
    for record in run["files"]:
        path = args.output / record["relative_path"]
        assert digest(path) == record["sha256"] and path.stat().st_size == record["bytes"]
    maps = table(args.output / "map_results.tsv"); pairs = table(args.output / "paired_map_results.tsv")
    keys = ["resource", "patient_id", "sample_id", "method_id", "K", "partition_id"]
    assert len(maps) == 1056 and len(pairs) == 528
    pair_keys = set()
    for s in sections:
        sid, resource = s["sample_id"], s["resource"]
        directory = args.output / "sections" / resource / sid
        with np.load(directory / "fitted_labels.npz") as labs, np.load(directory / "selected_memberships.npz") as masks:
            for row in pairs[pairs.resource.eq(resource) & pairs.sample_id.eq(sid)].to_dict("records"):
                a = "|".join(map(str, ["allowed", row["method_id"], row["K"], row["partition_id"]])); b = a.replace("allowed|", "withheld|", 1)
                common = row["allowed_status"] == row["withheld_status"] == "success"
                assert bool(row["common_evaluable"]) == common
                if common:
                    pair_keys.add(tuple(row[k] for k in keys))
                if row["allowed_fit_status"] == row["withheld_fit_status"] == "success":
                    close(row["whole_map_ARI"], adjusted_rand_score(labs[a], labs[b]), "Map overlap")
                    for field, suffix in [("stromal_domain_Jaccard", "stroma"), ("band_selection_Jaccard", None)]:
                        if a + "|stroma" in masks and b + "|stroma" in masks:
                            left = masks[a + "|"+suffix] if suffix else masks[a + "|near"] | masks[a + "|far"]
                            right = masks[b + "|"+suffix] if suffix else masks[b + "|near"] | masks[b + "|far"]
                            union = (left | right).sum(); expected = (left & right).sum()/union if union else np.nan
                            close(row[field], expected, field)
    patients = table(args.output / "patient_summary.tsv"); resources = table(args.output / "resource_summary.tsv")
    all_patients = {(s["resource"], s["patient_id"]) for s in sections}
    for frame in ["arm_specific", "common_configurations"]:
        reconstructed = {}
        for resource, patient in sorted(all_patients):
            for arm in ["allowed", "withheld"]:
                g = maps[maps.resource.eq(resource) & maps.patient_id.eq(patient) & maps.arm.eq(arm) & maps.status.eq("success")]
                if frame == "common_configurations":
                    g = g[np.asarray([tuple(row) in pair_keys for row in g[keys].to_numpy()], dtype=bool)]
                section_values = []
                for _, sg in g.groupby("sample_id"):
                    settings = [cg[METRICS].median().to_numpy() for _, cg in sg.groupby(["method_id", "K"])]
                    section_values.append(np.median(settings, axis=0))
                value = np.median(section_values, axis=0) if section_values else np.full(len(METRICS), np.nan)
                published = patients[patients.frame.eq(frame) & patients.resource.eq(resource) & patients.patient_id.eq(patient) & patients.arm.eq(arm)].iloc[0]
                close(published[METRICS].to_numpy(float), value, "Patient hierarchy")
                assert published.n_evaluable_maps == len(g) and published.n_evaluable_sections == g.sample_id.nunique()
                reconstructed[(resource, patient, arm)] = value
        for resource in ["Valdeolivas", "GSE294385"]:
            for arm in ["allowed", "withheld"]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    value = np.nanmedian([v for (r, _, a), v in reconstructed.items() if r == resource and a == arm], axis=0)
                published = resources[resources.frame.eq(frame) & resources.resource.eq(resource) & resources.arm.eq(arm)].iloc[0]
                close(published[METRICS].to_numpy(float), value, "Resource hierarchy")
    report = dict(status="pass", section_count=len(checked), fits_replayed=sum(r["fits_verified"] for r in checked),
                  control_draws_replayed=sum(r["control_draws_verified"] for r in checked),
                  original_source_features_checked=sum(r["source_features_verified"] for r in checked),
                  paired_configurations=len(pairs), common_evaluable_configurations=len(pair_keys), sections=checked,
                  independent_source_sha256=digest(__file__), plan_sha256=digest(args.plan),
                  verification_scope="Every source section, fitted PCA input/map, selected mask and matched-control draw; paired overlap and patient/resource reporting. No producer functions imported.")
    (args.output / "independent_verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sections"}), flush=True)


if __name__ == "__main__":
    main()
