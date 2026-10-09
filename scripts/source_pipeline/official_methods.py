"""Fit SpaGCN and STAGATE with the study's expression and spatial parameters."""

from __future__ import annotations

import gzip, json, sys, time, traceback

from pathlib import Path

import numpy as np

import pandas as pd

SEEDS = [11, 23, 37]

K_VALUES = [4, 6]

METHOD_SPAGCN = "Official_SpaGCN_v1_2_7"

METHOD_STAGATE = "Official_STAGATE_pyG"

METHOD_BAYES = "Official_BayesSpace_nrep1000"

STAGATE_COMMIT = "ae1158ca8cf1eb6bb8ee198298552d44c9ac21db"

SPAGCN_HVG_N = 2000

SPAGCN_TRAIN_EPOCHS = 80

def log_event(rows: list[dict[str, object]], **kwargs: object) -> None:
    base = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    base.update(kwargs)
    rows.append(base)
    print("[log]", json.dumps(base, sort_keys=True), flush=True)

def read_scalefactors(root: Path, sample_id: str) -> dict[str, float]:
    for suffix in ["_scalefactors_json.json.gz", "_scalefactors.json.gz"]:
        path = root / f"{sample_id}{suffix}"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
    return {}

def lowres_scale(root: Path, sample_id: str) -> float:
    scale = read_scalefactors(root, sample_id).get("tissue_lowres_scalef", 1.0)
    try:
        return float(scale)
    except Exception:
        return 1.0

def read_lines(path: Path) -> list[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [line.rstrip("\n") for line in handle]
    return path.read_text(encoding="utf-8").splitlines()

def read_visium_flat(root: Path, sample_id: str):
    import anndata as ad
    import scanpy as sc
    from scipy import sparse
    from scipy.io import mmread

    matrix_path = root / f"{sample_id}_matrix.mtx.gz"
    barcodes_path = root / f"{sample_id}_barcodes.tsv.gz"
    features_path = root / f"{sample_id}_features.tsv.gz"
    coords_path = root / f"{sample_id}_tissue_positions.csv.gz"
    if not coords_path.exists():
        coords_path = root / f"{sample_id}_tissue_positions_list.csv.gz"
    h5_path = root / f"{sample_id}_filtered_feature_bc_matrix.h5"
    if matrix_path.exists():
        counts = mmread(matrix_path).tocsr().transpose().tocsr()
        barcodes = read_lines(barcodes_path)
        features = pd.read_csv(features_path, sep="\t", header=None, compression="infer")
        genes = features.iloc[:, 1].astype(str).tolist()
        obs_names = [str(value) for value in barcodes]
    elif h5_path.exists():
        h5 = sc.read_10x_h5(str(h5_path))
        h5.var_names_make_unique()
        counts = sparse.csr_matrix(h5.X)
        obs_names = h5.obs_names.astype(str).tolist()
        barcodes = obs_names
        genes = h5.var_names.astype(str).tolist()
    else:
        raise FileNotFoundError(f"Missing flat matrix or 10x H5 for {sample_id} under {root}")
    coords = pd.read_csv(coords_path, header=None, compression="infer")
    if str(coords.iloc[0, 0]) == "barcode":
        coords = pd.read_csv(coords_path, compression="infer")
    else:
        coords.columns = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ]
    coords = coords[coords["barcode"].isin(barcodes)].copy()
    coords["barcode"] = pd.Categorical(coords["barcode"], categories=barcodes, ordered=True)
    coords = coords.sort_values("barcode").reset_index(drop=True)
    mask = coords["in_tissue"].astype(int).to_numpy() == 1
    coords = coords.loc[mask].reset_index(drop=True)
    counts = counts[mask, :]
    obs = pd.DataFrame(index=pd.Index([str(value) for value in coords["barcode"].tolist()], dtype="object"))
    var = pd.DataFrame(index=pd.Index([str(value) for value in genes], dtype="object"))
    adata = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)
    adata.var_names_make_unique()
    adata.obs["array_row"] = coords["array_row"].astype(float).to_numpy()
    adata.obs["array_col"] = coords["array_col"].astype(float).to_numpy()
    adata.obs["x_array"] = adata.obs["array_row"]
    adata.obs["y_array"] = adata.obs["array_col"]
    adata.obs["x_pixel"] = coords["pxl_row_in_fullres"].astype(float).to_numpy()
    adata.obs["y_pixel"] = coords["pxl_col_in_fullres"].astype(float).to_numpy()
    adata.obsm["spatial"] = coords[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy(dtype=float)
    return adata, coords

def read_image(root: Path, sample_id: str):
    import cv2

    for suffix in ["_tissue_lowres_image.png.gz", "_tissue_hires_image.png.gz", "_detected_tissue_image.jpg.gz"]:
        path = root / f"{sample_id}{suffix}"
        if path.exists():
            with gzip.open(path, "rb") as handle:
                data = np.frombuffer(handle.read(), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is not None:
                if suffix == "_tissue_lowres_image.png.gz":
                    return image, lowres_scale(root, sample_id), "lowres_histology"
                return image, 1.0, "fullres_or_detected_histology"
    return None, 1.0, "no_histology"

def append_map(rows: list[pd.DataFrame], dataset_id: str, sample_id: str, method_id: str, k: int, seed: int, coords: pd.DataFrame, labels, note: str) -> None:
    labels = np.asarray(labels)
    if not np.issubdtype(labels.dtype, np.integer):
        labels = pd.Categorical(labels).codes
    labels = labels.astype(int)
    labels = labels + (0 if labels.size and labels.min() >= 1 else 1)
    rows.append(
        pd.DataFrame(
            {
                "dataset_id": dataset_id,
                "sample_id": sample_id,
                "method_id": method_id,
                "K": int(k),
                "seed": int(seed),
                "replicate_type": "seed",
                "replicate_id": f"seed_{int(seed)}",
                "config_id": f"K{int(k)}_seed_{int(seed)}",
                "barcode": coords["barcode"].astype(str).tolist(),
                "x": coords["pxl_col_in_fullres"].astype(float).tolist(),
                "y": coords["pxl_row_in_fullres"].astype(float).tolist(),
                "domain_label": labels.astype(int).tolist(),
                "status": "success",
                "notes": note,
            }
        )
    )

def run_spagcn(dataset_id: str, sample_id: str, root: Path, map_rows: list[pd.DataFrame], log_rows: list[dict[str, object]]) -> None:
    import random
    import scanpy as sc
    import SpaGCN as spg
    import torch
    from scipy import sparse

    np.float = float  # SpaGCN 1.2.x compatibility with current numpy.
    np.int = int
    if not hasattr(sparse.csr_matrix, "A"):
        sparse.csr_matrix.A = property(lambda self: self.toarray())
    if not hasattr(sparse.csc_matrix, "A"):
        sparse.csc_matrix.A = property(lambda self: self.toarray())
    base, coords = read_visium_flat(root, sample_id)
    image, image_scale, image_note = read_image(root, sample_id)
    histology = image is not None
    x_pixel = [int(round(float(value) * image_scale)) for value in base.obs["x_pixel"].tolist()]
    y_pixel = [int(round(float(value) * image_scale)) for value in base.obs["y_pixel"].tolist()]
    x_array = base.obs["x_array"].astype(float).tolist()
    y_array = base.obs["y_array"].astype(float).tolist()
    adj = spg.calculate_adj_matrix(
        x=x_pixel,
        y=y_pixel,
        x_pixel=x_pixel,
        y_pixel=y_pixel,
        image=image,
        beta=49,
        alpha=1,
        histology=histology,
    )
    l_value = spg.search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    try:
        valid_l = l_value is not None and bool(np.isfinite(float(l_value)))
    except (TypeError, ValueError):
        valid_l = False
    if not valid_l:
        if not histology:
            raise RuntimeError(f"SpaGCN search_l returned an invalid value for {sample_id} without histology")
        log_event(
            log_rows,
            method=METHOD_SPAGCN,
            dataset_id=dataset_id,
            sample_id=sample_id,
            status="histology_fallback",
            note="official_spagcn_spatial_only_after_invalid_histology_l",
        )
        adj = spg.calculate_adj_matrix(
            x=x_pixel,
            y=y_pixel,
            x_pixel=x_pixel,
            y_pixel=y_pixel,
            image=None,
            beta=49,
            alpha=1,
            histology=False,
        )
        l_value = spg.search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)
        try:
            valid_l = l_value is not None and bool(np.isfinite(float(l_value)))
        except (TypeError, ValueError):
            valid_l = False
        image_note = "spatial_only_histology_invalid"
    if not valid_l:
        raise RuntimeError(f"SpaGCN search_l returned an invalid value for {sample_id}")
    for k in K_VALUES:
        for seed in SEEDS:
            try:
                adata = base.copy()
                adata.var_names_make_unique()
                spg.prefilter_genes(adata, min_cells=3)
                spg.prefilter_specialgenes(adata)
                adata.X = adata.X.astype(np.float32)
                sc.pp.normalize_per_cell(adata)
                sc.pp.log1p(adata)
                n_top = min(SPAGCN_HVG_N, max(100, adata.n_vars - 1))
                sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="cell_ranger")
                if "highly_variable" in adata.var:
                    adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()
                    adata.X = adata.X.astype(np.float32)
                random.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                clf = spg.SpaGCN()
                clf.set_l(l_value)
                clf.train(
                    adata,
                    adj,
                    init_spa=True,
                    init="kmeans",
                    n_clusters=k,
                    tol=5e-3,
                    lr=0.05,
                    max_epochs=SPAGCN_TRAIN_EPOCHS,
                )
                labels, _prob = clf.predict()
                if len(set(labels)) != int(k):
                    log_event(log_rows, method=METHOD_SPAGCN, dataset_id=dataset_id, sample_id=sample_id, K=k, seed=seed, status="wrong_k", observed_k=len(set(labels)))
                    continue
                append_map(map_rows, dataset_id, sample_id, METHOD_SPAGCN, k, seed, coords, labels, f"official_spagcn_1.2.7_kmeans_init_{image_note}" if histology else "official_spagcn_1.2.7_kmeans_init_xy")
                log_event(log_rows, method=METHOD_SPAGCN, dataset_id=dataset_id, sample_id=sample_id, K=k, seed=seed, status="success")
            except Exception as exc:
                log_event(log_rows, method=METHOD_SPAGCN, dataset_id=dataset_id, sample_id=sample_id, K=k, seed=seed, status="failed", error=str(exc), traceback=traceback.format_exc()[-2000:])

def run_stagate(dataset_id: str, sample_id: str, root: Path, map_rows: list[pd.DataFrame], log_rows: list[dict[str, object]]) -> None:
    import scanpy as sc
    from sklearn.cluster import KMeans
    STAGATE = import_stagate_module()

    base, coords = read_visium_flat(root, sample_id)
    for seed in SEEDS:
        try:
            adata = base.copy()
            sc.pp.filter_genes(adata, min_cells=3)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            n_top = min(3000, max(500, adata.n_vars - 1))
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="cell_ranger")
            STAGATE.Cal_Spatial_Net(adata, k_cutoff=6, model="KNN", verbose=False)
            adata = STAGATE.train_STAGATE(adata, random_seed=seed, n_epochs=1000, verbose=False)
            embedding = np.asarray(adata.obsm["STAGATE"], dtype=float)
            for k in K_VALUES:
                labels = KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(embedding)
                append_map(map_rows, dataset_id, sample_id, METHOD_STAGATE, k, seed, coords, labels, "official_stagate_pyg_embedding_kmeans_fixedK")
                log_event(log_rows, method=METHOD_STAGATE, dataset_id=dataset_id, sample_id=sample_id, K=k, seed=seed, status="success")
        except Exception as exc:
            for k in K_VALUES:
                log_event(log_rows, method=METHOD_STAGATE, dataset_id=dataset_id, sample_id=sample_id, K=k, seed=seed, status="failed", error=str(exc), traceback=traceback.format_exc()[-2000:])

def import_stagate_module():
    import importlib
    module = importlib.import_module("STAGATE_pyG")
    return module

