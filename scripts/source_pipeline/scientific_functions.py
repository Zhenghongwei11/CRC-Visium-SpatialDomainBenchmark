"""Register expression and pathology annotations, then fit spatial domain maps."""


from __future__ import annotations


import gzip, hashlib, json, math, time


from pathlib import Path


import numpy as np


import pandas as pd


DATASET_ID = ""


SAMPLE_META = {}


BAYESPACE_NREP = 1000


NOMINAL_SPOT_DIAMETER_UM = 55.0


SEEDS = [11, 23, 37]


K_VALUES = [4, 6]


PRIMARY_GENES = ["TGFB1", "CXCL12", "ACTA2", "TAGLN"]


SCORE_DEFINITIONS = {"primary_barrier": {"label": "Exploratory four-gene mean", "genes": PRIMARY_GENES, "role": "primary"}}


BOOTSTRAP_REPLICATES = 1000


MIN_GROUP_SPOTS = 20


MIN_BLOCKS = 3


ESTIMATOR_LABELS = {
    "strict_computational_interface": "strict computational interface",
    "anchor_retention": "morphology-anchor retention",
}


METHOD_LABELS = {
    "M0_expr_kmeans": "expression-only k-means",
    "M1_spatial_concat_kmeans": "coordinate-augmented k-means",
    "M2_spatial_ward": "spatial Ward",
    "M3_spatial_leiden": "spatial Leiden",
    "M4_spagcn_official": "official SpaGCN v1.2.7",
    "M5_stagate_official": "official STAGATE_pyG",
    "M6_bayesspace_official": f"official BayesSpace (nrep={BAYESPACE_NREP})",
}


BASE_METHOD_IDS = (
    "M0_expr_kmeans",
    "M1_spatial_concat_kmeans",
    "M2_spatial_ward",
    "M3_spatial_leiden",
    "M4_spagcn_official",
    "M5_stagate_official",
)


def read_positions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, compression="infer")
    if str(frame.iloc[0, 0]).lower() == "barcode":
        frame = pd.read_csv(path, compression="infer")
    else:
        frame.columns = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ]
    frame["barcode"] = frame["barcode"].astype(str)
    return frame


def read_scalefactors(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def read_annotation(path: Path) -> pd.DataFrame:
    ann = pd.read_csv(path, compression="infer")
    ann.columns = [str(column).strip() for column in ann.columns]
    barcode_col = next(column for column in ann.columns if column.lower() == "barcode")
    label_cols = [column for column in ann.columns if column != barcode_col]
    if len(label_cols) != 1:
        raise RuntimeError(f"Expected one pathology label column in {path}, found {label_cols}")
    ann = ann.rename(columns={barcode_col: "barcode", label_cols[0]: "raw_label"})
    ann["barcode"] = ann["barcode"].astype(str)
    ann["raw_label"] = ann["raw_label"].fillna("").astype(str).str.strip()
    return ann[["barcode", "raw_label"]]


def map_label(raw_label: str) -> str:
    normalized = " ".join(str(raw_label).replace("\u00a0", " ").lower().split())
    if not normalized:
        return "unresolved"
    if "exclude" in normalized:
        return "exclude"
    if "tumor stroma" in normalized and "tumor cells" not in normalized:
        return "stroma"
    if "tumor cells" in normalized and "tumor stroma" not in normalized:
        return "tumor"
    has_tumor = "tumor" in normalized
    has_stroma = "stroma" in normalized or "fibroblastic" in normalized or "desmoplastic" in normalized
    if has_tumor and has_stroma:
        return "mixed"
    if has_tumor:
        return "tumor"
    if has_stroma:
        if " or " in normalized or ("muscularis" in normalized and "stroma" in normalized):
            return "unresolved"
        return "stroma"
    if "non neo epithelium" in normalized or "non-neoplastic" in normalized or "non neoplastic" in normalized:
        return "normal epithelium"
    if any(token in normalized for token in ("epithelium", "submucosa", "lamina propria", "glandular", "muscularis", "connective tissue")):
        return "other"
    return "unresolved"


def load_expression(root: Path, sample_id: str, annotation_path: Path) -> dict[str, object]:
    import anndata as ad
    import scanpy as sc
    from scipy import sparse
    from scipy.io import mmread
    from sklearn.decomposition import PCA

    h5 = root / f"{sample_id}_filtered_feature_bc_matrix.h5"
    matrix_path = root / f"{sample_id}_matrix.mtx.gz"
    barcodes_path = root / f"{sample_id}_barcodes.tsv.gz"
    features_path = root / f"{sample_id}_features.tsv.gz"
    positions_path = root / f"{sample_id}_tissue_positions_list.csv.gz"
    if not positions_path.exists():
        positions_path = root / f"{sample_id}_tissue_positions.csv.gz"
    scale_path = root / f"{sample_id}_scalefactors_json.json.gz"
    if not scale_path.exists():
        scale_path = root / f"{sample_id}_scalefactors.json.gz"
    if h5.exists():
        adata = sc.read_10x_h5(str(h5), gex_only=True)
    elif matrix_path.exists() and barcodes_path.exists() and features_path.exists():
        counts = mmread(matrix_path).tocsr().transpose().tocsr()
        with gzip.open(barcodes_path, "rt", encoding="utf-8") as handle:
            barcodes = [line.strip() for line in handle if line.strip()]
        features = pd.read_csv(features_path, sep="\t", header=None, compression="infer")
        if counts.shape[0] != len(barcodes) or counts.shape[1] != len(features):
            raise RuntimeError(
                f"Matrix Market dimensions do not match metadata for {sample_id}: "
                f"counts={counts.shape}, barcodes={len(barcodes)}, features={len(features)}"
            )
        adata = ad.AnnData(X=counts)
        adata.obs_names = pd.Index(barcodes, dtype="string")
        gene_names = features.iloc[:, 1 if features.shape[1] >= 2 else 0].astype(str).tolist()
        adata.var_names = pd.Index(gene_names, dtype="string")
        if features.shape[1] >= 1:
            adata.var["gene_ids"] = features.iloc[:, 0].astype(str).to_numpy()
        if features.shape[1] >= 3:
            adata.var["feature_types"] = features.iloc[:, 2].astype(str).to_numpy()
    else:
        raise FileNotFoundError(f"Missing 10x H5 or Matrix Market trio for {sample_id} under {root}")
    adata.var_names_make_unique()
    matrix_barcodes = pd.Index(adata.obs_names.astype(str), dtype="string")
    positions = read_positions(positions_path)
    if positions["barcode"].duplicated().any():
        raise RuntimeError(f"Duplicate tissue-position barcodes for {sample_id}")
    position_barcodes = pd.Index(positions["barcode"].astype(str), dtype="string")
    missing_position_barcodes = sorted(set(matrix_barcodes) - set(position_barcodes))
    if missing_position_barcodes:
        raise RuntimeError(
            f"Tissue-position file is missing {len(missing_position_barcodes)} matrix barcodes for {sample_id}"
        )
    positions = positions[positions["barcode"].isin(adata.obs_names)].copy()
    positions["barcode"] = pd.Categorical(positions["barcode"], categories=adata.obs_names, ordered=True)
    positions = positions.sort_values("barcode").reset_index(drop=True)
    in_tissue = positions["in_tissue"].astype(int).to_numpy() == 1
    positions = positions.loc[in_tissue].reset_index(drop=True)
    adata = adata[positions["barcode"].astype(str).tolist()].copy()
    if adata.n_obs != len(positions):
        raise RuntimeError(f"Expression/position mismatch for {sample_id}")
    annotation = read_annotation(annotation_path)
    if annotation["barcode"].duplicated().any():
        raise RuntimeError(f"Duplicate pathology annotation barcodes for {sample_id}")
    annotation_barcodes = pd.Index(annotation["barcode"].astype(str), dtype="string")
    annotation_outside_matrix = sorted(set(annotation_barcodes) - set(matrix_barcodes))
    if annotation_outside_matrix:
        raise RuntimeError(
            f"Pathology annotation contains {len(annotation_outside_matrix)} barcodes absent from the matrix for {sample_id}"
        )
    in_tissue_barcodes = pd.Index(adata.obs_names.astype(str), dtype="string")
    annotation_in_tissue = annotation_barcodes.intersection(in_tissue_barcodes)
    registration = {
        "matrix_barcode_count": int(len(matrix_barcodes)),
        "position_barcode_count": int(len(position_barcodes)),
        "annotation_row_count": int(len(annotation)),
        "annotation_matrix_intersection": int(len(annotation_barcodes.intersection(matrix_barcodes))),
        "annotation_in_tissue_count": int(len(annotation_in_tissue)),
        "annotation_outside_matrix_count": int(len(annotation_outside_matrix)),
        "matrix_in_tissue_count": int(len(in_tissue_barcodes)),
        "in_tissue_unlabelled_count": int(len(set(in_tissue_barcodes) - set(annotation_barcodes))),
    }
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    gene_values: dict[str, np.ndarray] = {}
    for gene in sorted({gene for definition in SCORE_DEFINITIONS.values() for gene in definition["genes"]}):
        if gene not in adata.var_names:
            continue
        values = adata[:, gene].X
        gene_values[gene] = values.toarray().ravel() if sparse.issparse(values) else np.asarray(values).ravel()
    if not set(PRIMARY_GENES).issubset(gene_values):
        missing = sorted(set(PRIMARY_GENES) - set(gene_values))
        raise RuntimeError(f"{sample_id} missing primary genes after normalization: {missing}")
    primary_score = np.mean(np.vstack([gene_values[gene] for gene in PRIMARY_GENES]), axis=0).astype(float)
    secondary_scores: dict[str, np.ndarray | None] = {}
    score_status: dict[str, dict[str, object]] = {"primary_barrier": {"status": "success", "missing_genes": ""}}
    for score_id, definition in SCORE_DEFINITIONS.items():
        if score_id == "primary_barrier":
            continue
        genes = [str(gene) for gene in definition["genes"]]
        if score_id == "immune_exclusion":
            component_ids = ["CAF_FAP", "SPP1_myeloid", "T_cell"]
            component_genes = [str(gene) for component_id in component_ids for gene in SCORE_DEFINITIONS[component_id]["genes"]]
            missing = sorted(set(component_genes) - set(gene_values))
            if missing:
                secondary_scores[score_id] = None
                score_status[score_id] = {"status": "not_evaluable", "missing_genes": ";".join(missing)}
            else:
                caf = np.mean(np.vstack([gene_values[gene] for gene in SCORE_DEFINITIONS["CAF_FAP"]["genes"]]), axis=0)
                myeloid = np.mean(np.vstack([gene_values[gene] for gene in SCORE_DEFINITIONS["SPP1_myeloid"]["genes"]]), axis=0)
                t_cell = np.mean(np.vstack([gene_values[gene] for gene in SCORE_DEFINITIONS["T_cell"]["genes"]]), axis=0)
                secondary_scores[score_id] = (caf + myeloid) - t_cell
                score_status[score_id] = {"status": "success", "missing_genes": ""}
            continue
        missing = sorted(set(genes) - set(gene_values))
        if missing:
            secondary_scores[score_id] = None
            score_status[score_id] = {"status": "not_evaluable", "missing_genes": ";".join(missing)}
        else:
            secondary_scores[score_id] = np.mean(np.vstack([gene_values[gene] for gene in genes]), axis=0).astype(float)
            score_status[score_id] = {"status": "success", "missing_genes": ""}
    x = adata.X
    means = np.asarray(x.mean(axis=0)).ravel() if sparse.issparse(x) else np.asarray(x).mean(axis=0)
    if sparse.issparse(x):
        sq_means = np.asarray(x.power(2).mean(axis=0)).ravel()
    else:
        sq_means = np.asarray(x, dtype=float).mean(axis=0) ** 2
    variances = np.maximum(sq_means - means**2, 0.0)
    n_features = min(2000, max(2, x.shape[1] - 1))
    top_idx = np.argsort(variances)[::-1][:n_features]
    dense = x[:, top_idx].toarray().astype(np.float32) if sparse.issparse(x) else np.asarray(x[:, top_idx], dtype=np.float32)
    n_components = max(2, min(20, dense.shape[0] - 1, dense.shape[1] - 1))
    pcs = PCA(n_components=n_components, random_state=0).fit_transform(dense).astype(np.float32)
    coords = positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)
    ann = annotation.set_index("barcode")
    raw_labels = np.asarray([ann["raw_label"].get(barcode, "") for barcode in adata.obs_names], dtype=object)
    coarse_labels = np.asarray([map_label(value) for value in raw_labels], dtype=object)
    scale = read_scalefactors(scale_path)
    spot_diameter = float(scale.get("spot_diameter_fullres", float("nan")))
    if not np.isfinite(spot_diameter) or spot_diameter <= 0:
        raise RuntimeError(f"{sample_id} has no finite spot_diameter_fullres scalefactor")
    return {
        "sample_id": sample_id,
        "patient_id": SAMPLE_META[sample_id]["patient_id"],
        "section_id": SAMPLE_META[sample_id]["section_id"],
        "barcodes": np.asarray(adata.obs_names.astype(str)),
        "coords": coords,
        "array_coords": positions[["array_row", "array_col"]].to_numpy(dtype=float),
        "primary_score": primary_score,
        "score_values": {"primary_barrier": primary_score, **secondary_scores},
        "score_status": score_status,
        "pcs": pcs,
        "raw_labels": raw_labels,
        "coarse_labels": coarse_labels,
        "microns_per_pixel": NOMINAL_SPOT_DIAMETER_UM / spot_diameter,
        "spot_diameter_fullres": spot_diameter,
        "registration": registration,
    }


def make_map(sample: dict[str, object], labels: np.ndarray, method_id: str, k_value: int, replicate_id: str, replicate_type: str, notes: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": DATASET_ID,
            "sample_id": sample["sample_id"],
            "method_id": method_id,
            "method_label": METHOD_LABELS[method_id],
            "K": int(k_value),
            "replicate_type": replicate_type,
            "replicate_id": replicate_id,
            "config_id": f"K{int(k_value)}_{replicate_id}",
            "barcode": sample["barcodes"],
            "x": sample["coords"][:, 0],
            "y": sample["coords"][:, 1],
            "domain_label": np.asarray(labels, dtype=int) + 1,
            "status": "success",
            "notes": notes,
        }
    )


def weighted_leiden_graph(pcs: np.ndarray, coords: np.ndarray, neighbors: int = 6):
    import igraph as ig
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(len(coords), neighbors + 1))
    nn.fit(coords)
    graph = nn.kneighbors_graph(coords, mode="connectivity")
    graph = graph.maximum(graph.T).tocoo()
    source = graph.row.astype(int)
    target = graph.col.astype(int)
    keep = source < target
    source = source[keep]
    target = target[keep]
    normalized = pcs.astype(float)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized /= norms
    weights = np.maximum(0.0, (1.0 + np.sum(normalized[source] * normalized[target], axis=1)) / 2.0)
    result = ig.Graph(n=len(coords), edges=list(zip(source.tolist(), target.tolist(), strict=True)), directed=False)
    result.es["weight"] = weights.tolist()
    return result


def leiden_membership(graph, resolution: float, seed: int) -> np.ndarray:
    import leidenalg

    try:
        partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=float(resolution), seed=int(seed))
    except TypeError:
        partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=float(resolution))
    return np.asarray(partition.membership, dtype=int)


def exact_leiden(graph, target_k: int, seed: int) -> tuple[np.ndarray, float, bool]:
    candidates = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0]
    evaluations: list[tuple[float, int]] = []
    for resolution in candidates:
        labels = leiden_membership(graph, resolution, seed)
        observed = int(np.unique(labels).size)
        evaluations.append((resolution, observed))
        if observed == target_k:
            return labels, resolution, True
    below = [(r, k) for r, k in evaluations if k < target_k]
    above = [(r, k) for r, k in evaluations if k > target_k]
    if below and above:
        low = max(below, key=lambda item: item[0])[0]
        high = min(above, key=lambda item: item[0])[0]
        best = (low, dict(evaluations)[low])
        for _ in range(18):
            middle = math.sqrt(low * high)
            labels = leiden_membership(graph, middle, seed)
            observed = int(np.unique(labels).size)
            if abs(observed - target_k) < abs(best[1] - target_k):
                best = (middle, observed)
            if observed == target_k:
                return labels, middle, True
            if observed < target_k:
                low = middle
            else:
                high = middle
        return leiden_membership(graph, best[0], seed), float(best[0]), False
    best_resolution, best_k = min(evaluations, key=lambda item: (abs(item[1] - target_k), item[0]))
    return leiden_membership(graph, best_resolution, seed), float(best_resolution), False


