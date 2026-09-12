#!/usr/bin/env python3
"""Run official SpaGCN/STAGATE/BayesSpace sensitivity on Colab."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


WORK = Path("/content/jbcb_official_sensitivity")
SEEDS = [11, 23, 37]
K_VALUES = [4, 6]
METHOD_SPAGCN = "Official_SpaGCN_v1_2_7"
METHOD_STAGATE = "Official_STAGATE_pyG"
METHOD_BAYES = "Official_BayesSpace_nrep1000"
SPAGCN_HVG_N = 2000
SPAGCN_TRAIN_EPOCHS = 80
ALLOWED_SUFFIXES = (
    "_matrix.mtx.gz",
    "_barcodes.tsv.gz",
    "_features.tsv.gz",
    "_tissue_positions.csv.gz",
    "_tissue_positions_list.csv.gz",
    "_tissue_lowres_image.png.gz",
    "_tissue_hires_image.png.gz",
    "_detected_tissue_image.jpg.gz",
    "_scalefactors.json.gz",
    "_scalefactors_json.json.gz",
    "_filtered_feature_bc_matrix.h5",
)
REMOTE_SERIES = {
    "GSE267401": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE267nnn/GSE267401/suppl/GSE267401_RAW.tar",
    "GSE311294": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE311nnn/GSE311294/suppl/GSE311294_RAW.tar",
    "GSE285505": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE285nnn/GSE285505/suppl/GSE285505_RAW.tar",
}
REMOTE_SAMPLES = [
    {
        "dataset_id": "GSE267401",
        "sample_id": "GSM8265211_CTC21P",
        "series_id": "GSE267401",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE267401",
        "sample_id": "GSM8265212_CTC21M",
        "series_id": "GSE267401",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE267401",
        "sample_id": "GSM8265213_CTC17P",
        "series_id": "GSE267401",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE267401",
        "sample_id": "GSM8265214_CTC17M",
        "series_id": "GSE267401",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE285505",
        "sample_id": "GSM8703563_Tumor19",
        "series_id": "GSE285505",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE285505",
        "sample_id": "GSM8703564_Tumor20",
        "series_id": "GSE285505",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE285505",
        "sample_id": "GSM8703565_Tumor24",
        "series_id": "GSE285505",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE285505",
        "sample_id": "GSM8703566_Tumor26",
        "series_id": "GSE285505",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE311294",
        "sample_id": "GSM9322957_TR11_206",
        "series_id": "GSE311294",
        "rationale": "reviewer_focal_TR11_206",
    },
    {
        "dataset_id": "GSE311294",
        "sample_id": "GSM9322958_TR11_16184",
        "series_id": "GSE311294",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE311294",
        "sample_id": "GSM9322959_TR11_18105",
        "series_id": "GSE311294",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE311294",
        "sample_id": "GSM9322960_TR11_21723",
        "series_id": "GSE311294",
        "rationale": "full_cohort_crc_section",
    },
    {
        "dataset_id": "GSE311294",
        "sample_id": "GSM9322961_TR16_23542",
        "series_id": "GSE311294",
        "rationale": "full_cohort_crc_section",
    },
]


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    print("[cmd]", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, timeout=timeout)


def log_event(rows: list[dict[str, object]], **kwargs: object) -> None:
    base = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    base.update(kwargs)
    rows.append(base)
    print("[log]", json.dumps(base, sort_keys=True), flush=True)


def install_python_stack(log_rows: list[dict[str, object]], *, include_pyg: bool = True) -> None:
    packages = [
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas>=2.2,<2.3",
        "anndata>=0.10,<0.12",
        "scanpy>=1.10,<1.11",
        "opencv-python-headless",
        "Pillow",
        "python-igraph",
        "louvain",
        "leidenalg",
        "SpaGCN==1.2.7",
    ]
    if include_pyg:
        packages.append("torch-geometric")
    run([sys.executable, "-m", "pip", "install", "-q", *packages], timeout=3600)
    if include_pyg:
        try:
            import torch

            torch_version = str(torch.__version__).split("+")[0]
            cuda_version = torch.version.cuda
            cuda_tag = "cpu" if cuda_version is None else "cu" + str(cuda_version).replace(".", "")
            wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"
            run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "torch-scatter",
                    "torch-sparse",
                    "-f",
                    wheel_url,
                ],
                timeout=1800,
            )
            log_event(log_rows, step="install_pyg_optional_deps", status="success", wheel_url=wheel_url)
        except Exception as exc:
            log_event(log_rows, step="install_pyg_optional_deps", status="failed", error=str(exc))
    log_event(log_rows, step="install_python_stack", status="success", include_pyg=include_pyg)


def install_h5_reader_stack(log_rows: list[dict[str, object]]) -> None:
    packages = [
        "numpy",
        "scipy",
        "pandas>=2.2,<2.3",
        "anndata>=0.10,<0.12",
        "scanpy>=1.10,<1.11",
    ]
    run([sys.executable, "-m", "pip", "install", "-q", *packages], timeout=1800)
    log_event(log_rows, step="install_h5_reader_stack", status="success")


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


def install_stagate(log_rows: list[dict[str, object]]) -> Path:
    repo = WORK / "external" / "STAGATE_pyG"
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/QIFEIDKN/STAGATE_pyG.git", str(repo)], timeout=600)
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)], timeout=900)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    log_event(log_rows, step="install_stagate_pyg", status="success", repo=str(repo))
    return repo


def import_stagate_module():
    repo = WORK / "external" / "STAGATE_pyG"
    if repo.exists() and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import STAGATE_pyG as STAGATE

        return STAGATE
    except ModuleNotFoundError:
        import STAGATE

        return STAGATE


def install_r_stack(log_rows: list[dict[str, object]]) -> bool:
    code = (
        'options(repos=c(CRAN="https://cloud.r-project.org")); '
        'if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager"); '
        'BiocManager::install(c("BayesSpace","SingleCellExperiment","mclust","hdf5r"), '
        'ask=FALSE, update=FALSE)'
    )
    try:
        run(["Rscript", "-e", code], timeout=7200)
        log_event(log_rows, step="install_r_bayesspace", status="success")
        return True
    except Exception as exc:
        log_event(log_rows, step="install_r_bayesspace", status="failed", error=str(exc))
        return False


def ensure_bayesspace_runner(extracted: Path) -> None:
    script_dir = extracted / "repo_scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    target = script_dir / "run_bayesspace_baseline.R"
    if target.exists():
        return
    target.write_text(
        r'''#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(Matrix)
  library(SingleCellExperiment)
  library(BayesSpace)
  library(mclust)
})
parse_args <- function(input) {
  result <- list(dataset_id="", dataset_root="", sample_id="", k_grid="4,6", seeds="11,23,37",
                 nrep="1000", gamma="3", output_domain_map_tsv="", output_tsv="", note="official_bayesspace")
  i <- 1L
  while (i <= length(input)) {
    key <- input[[i]]
    if (startsWith(key, "--") && i < length(input)) {
      result[[gsub("-", "_", gsub("^--", "", key))]] <- input[[i + 1L]]
      i <- i + 2L
    } else {
      i <- i + 1L
    }
  }
  result
}
args <- parse_args(commandArgs(trailingOnly=TRUE))
k_values <- as.integer(strsplit(args$k_grid, ",")[[1]])
seed_values <- as.integer(strsplit(args$seeds, ",")[[1]])
nrep <- as.integer(args$nrep)
dataset_root <- args$dataset_root
prefix <- args$sample_id
matrix_file <- file.path(dataset_root, paste0(prefix, "_matrix.mtx.gz"))
barcodes_file <- file.path(dataset_root, paste0(prefix, "_barcodes.tsv.gz"))
features_file <- file.path(dataset_root, paste0(prefix, "_features.tsv.gz"))
coords_file <- file.path(dataset_root, paste0(prefix, "_tissue_positions.csv.gz"))
if (!file.exists(coords_file)) coords_file <- file.path(dataset_root, paste0(prefix, "_tissue_positions_list.csv.gz"))
counts <- readMM(gzfile(matrix_file))
barcodes <- read.delim(gzfile(barcodes_file), header=FALSE, stringsAsFactors=FALSE)
features <- read.delim(gzfile(features_file), header=FALSE, stringsAsFactors=FALSE)
rownames(counts) <- make.unique(features[[2]])
colnames(counts) <- barcodes[[1]]
coords <- read.csv(coords_file, stringsAsFactors=FALSE)
if (!("barcode" %in% colnames(coords))) {
  colnames(coords) <- c("barcode","in_tissue","array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres")
}
coords <- coords[coords$barcode %in% colnames(counts), , drop=FALSE]
coords <- coords[match(colnames(counts), coords$barcode), , drop=FALSE]
in_tissue <- coords$in_tissue == 1
counts <- counts[, in_tissue, drop=FALSE]
coords <- coords[in_tissue, , drop=FALSE]
sce <- SingleCellExperiment(assays=list(counts=counts))
colData(sce)$array_row <- coords$array_row
colData(sce)$array_col <- coords$array_col
colData(sce)$row <- coords$array_row
colData(sce)$col <- coords$array_col
colData(sce)$imagerow <- coords$pxl_row_in_fullres
colData(sce)$imagecol <- coords$pxl_col_in_fullres
set.seed(seed_values[[1]])
sce_pre <- spatialPreprocess(sce, platform="Visium", n.HVGs=min(2000, nrow(sce)), n.PCs=min(15, ncol(sce)-1L), log.normalize=TRUE)
bench_rows <- list()
for (q in k_values) {
  labels_by_seed <- list()
  times <- c()
  for (seed in seed_values) {
    cluster_args <- list(sce=sce_pre, q=q, platform="Visium", d=min(15, ncol(reducedDim(sce_pre, "PCA"))),
                         init.method="mclust", model="t", gamma=as.numeric(args$gamma),
                         nrep=nrep, save.chain=FALSE)
    if ("burn.in" %in% names(formals(BayesSpace::spatialCluster))) cluster_args[["burn.in"]] <- min(100L, nrep - 1L)
    if ("verbose" %in% names(formals(BayesSpace::spatialCluster))) cluster_args$verbose <- FALSE
    set.seed(seed)
    start <- proc.time()[["elapsed"]]
    sce_q <- do.call(BayesSpace::spatialCluster, cluster_args)
    times <- c(times, proc.time()[["elapsed"]] - start)
    labels <- as.integer(colData(sce_q)$spatial.cluster)
    labels_by_seed[[as.character(seed)]] <- labels
    map_out <- data.frame(dataset_id=args$dataset_id, sample_id=prefix, method_id="BayesSpace", K=q, seed=seed,
                          barcode=colnames(sce_q), x=as.numeric(colData(sce_q)$imagecol),
                          y=as.numeric(colData(sce_q)$imagerow), domain_label=labels,
                          notes=args$note, stringsAsFactors=FALSE)
    append_mode <- file.exists(args$output_domain_map_tsv) && file.info(args$output_domain_map_tsv)$size > 0
    write.table(map_out, file=args$output_domain_map_tsv, sep="\t", quote=FALSE, row.names=FALSE,
                col.names=!append_mode, append=append_mode)
  }
  ari <- c()
  keys <- names(labels_by_seed)
  for (i in seq_len(length(keys)-1L)) for (j in seq((i+1L), length(keys))) {
    ari <- c(ari, mclust::adjustedRandIndex(labels_by_seed[[keys[[i]]]], labels_by_seed[[keys[[j]]]]))
  }
  bench_rows[[length(bench_rows)+1L]] <- data.frame(dataset_id=args$dataset_id, sample_id=prefix, method_id="BayesSpace",
    method_family="bayesian_spatial", preprocessing_id="bayesspace_default", param_set_id=paste0("K", q), K=q,
    seed_count=length(seed_values), stability_ari_median=median(ari), stability_ari_iqr=IQR(ari),
    spatial_coherence_median=NA, spatial_coherence_iqr=NA, marker_coherence_median=NA, marker_coherence_iqr=NA,
    wall_time_sec_median=median(times), peak_rss_mb_median=NA, failure_rate=0, notes=args$note)
}
write.table(do.call(rbind, bench_rows), file=args$output_tsv, sep="\t", quote=FALSE, row.names=FALSE)
''',
        encoding="utf-8",
    )


def ensure_flat_matrix_for_bayesspace(root: Path, sample_id: str, log_rows: list[dict[str, object]]) -> None:
    matrix_file = root / f"{sample_id}_matrix.mtx.gz"
    barcodes_file = root / f"{sample_id}_barcodes.tsv.gz"
    features_file = root / f"{sample_id}_features.tsv.gz"
    if matrix_file.exists() and barcodes_file.exists() and features_file.exists():
        return
    h5_file = root / f"{sample_id}_filtered_feature_bc_matrix.h5"
    if not h5_file.exists():
        raise FileNotFoundError(f"Missing flat 10x matrix files and H5 fallback for {sample_id}")
    import scipy.io
    from scipy import sparse
    import scanpy as sc

    adata = sc.read_10x_h5(str(h5_file), gex_only=True)
    counts = sparse.coo_matrix(adata.X.T)
    plain_matrix = matrix_file.with_suffix("")
    scipy.io.mmwrite(str(plain_matrix), counts)
    with plain_matrix.open("rb") as src, gzip.open(matrix_file, "wb") as dst:
        shutil.copyfileobj(src, dst)
    plain_matrix.unlink()
    with gzip.open(barcodes_file, "wt", encoding="utf-8") as handle:
        for barcode in adata.obs_names.astype(str):
            handle.write(f"{barcode}\n")
    gene_ids = adata.var.get("gene_ids", adata.var_names).astype(str)
    feature_types = adata.var.get("feature_types", pd.Series(["Gene Expression"] * adata.n_vars, index=adata.var_names)).astype(str)
    with gzip.open(features_file, "wt", encoding="utf-8") as handle:
        for gene_id, gene_name, feature_type in zip(gene_ids, adata.var_names.astype(str), feature_types):
            handle.write(f"{gene_id}\t{gene_name}\t{feature_type}\n")
    log_event(log_rows, step="materialize_h5_for_bayesspace", dataset_root=str(root), sample_id=sample_id, status="success")


def build_input_from_geo(extracted: Path, log_rows: list[dict[str, object]]) -> pd.DataFrame:
    raw_dir = WORK / "geo_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(REMOTE_SAMPLES)
    for series_id, group in manifest.groupby("series_id"):
        url = REMOTE_SERIES[str(series_id)]
        tar_path = raw_dir / f"{series_id}_RAW.tar"
        if not tar_path.exists():
            run(["wget", "-q", "-O", str(tar_path), url], timeout=7200)
        log_event(log_rows, step="download_geo_tar", series_id=str(series_id), status="success", path=str(tar_path))
        with tarfile.open(tar_path, "r") as handle:
            members = handle.getmembers()
            wanted: list[tarfile.TarInfo] = []
            wanted_names: set[str] = set()
            for sample_id in group["sample_id"].astype(str).tolist():
                target_root = extracted / "data" / str(series_id) / "extracted"
                target_root.mkdir(parents=True, exist_ok=True)
                for member in members:
                    name = Path(member.name).name
                    if name.startswith(f"{sample_id}_") and any(name.endswith(suffix) for suffix in ALLOWED_SUFFIXES):
                        member.name = name
                        wanted.append(member)
                        wanted_names.add(name)
            handle.extractall(extracted / "data" / str(series_id) / "extracted", members=wanted)
            log_event(log_rows, step="extract_geo_members", series_id=str(series_id), status="success", n_files=len(wanted_names))
    manifest["bundle_root"] = manifest["series_id"].map(lambda value: f"data/{value}/extracted")
    manifest["source"] = "geo_ncbi_supplementary_raw_tar"
    manifest.to_csv(extracted / "sample_manifest.tsv", sep="\t", index=False)
    ensure_bayesspace_runner(extracted)
    return manifest


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


def run_bayesspace(manifest: pd.DataFrame, extracted: Path, map_path: Path, log_rows: list[dict[str, object]], nrep: int) -> None:
    r_script = extracted / "repo_scripts" / "run_bayesspace_baseline.R"
    tmp_map = map_path.with_name("bayesspace_raw_domain_maps.tsv")
    tmp_bench = map_path.with_name("bayesspace_nrep1000_benchmark.tsv")
    if tmp_map.exists():
        tmp_map.unlink()
    frames: list[pd.DataFrame] = []
    for row in manifest.itertuples(index=False):
        dataset_id = str(row.dataset_id)
        sample_id = str(row.sample_id)
        root = extracted / str(row.bundle_root)
        ensure_flat_matrix_for_bayesspace(root, sample_id, log_rows)
        try:
            bench = tmp_bench.with_name(f"bayesspace_{dataset_id}_{sample_id}_benchmark.tsv")
            run(
                [
                    "Rscript",
                    str(r_script),
                    "--dataset-id",
                    dataset_id,
                    "--dataset-root",
                    str(root),
                    "--sample-id",
                    sample_id,
                    "--k-grid",
                    ",".join(map(str, K_VALUES)),
                    "--seeds",
                    ",".join(map(str, SEEDS)),
                    "--nrep",
                    str(int(nrep)),
                    "--output-domain-map-tsv",
                    str(tmp_map),
                    "--output-tsv",
                    str(bench),
                    "--note",
                    f"official_bayesspace_nrep{int(nrep)}",
                ],
                timeout=14400,
            )
            if bench.exists():
                b = pd.read_csv(bench, sep="\t")
                b["method_id"] = METHOD_BAYES
                frames.append(b)
            log_event(log_rows, method=METHOD_BAYES, dataset_id=dataset_id, sample_id=sample_id, status="success")
        except Exception as exc:
            log_event(log_rows, method=METHOD_BAYES, dataset_id=dataset_id, sample_id=sample_id, status="failed", error=str(exc), traceback=traceback.format_exc()[-2000:])
    if tmp_map.exists():
        maps = pd.read_csv(tmp_map, sep="\t")
        maps["method_id"] = METHOD_BAYES
        maps["notes"] = f"official_bayesspace_nrep{int(nrep)}"
        if map_path.exists():
            old = pd.read_csv(map_path, sep="\t")
            maps = pd.concat([old, maps], ignore_index=True, sort=False)
        maps.to_csv(map_path, sep="\t", index=False)
    if frames:
        pd.concat(frames, ignore_index=True, sort=False).to_csv(tmp_bench, sep="\t", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tar", default="jbcb_official_sensitivity_input.tar.gz")
    parser.add_argument("--output-dir", default="colab_official_output")
    parser.add_argument("--bayesspace-nrep", type=int, default=1000)
    parser.add_argument("--run-bayesspace", action="store_true")
    parser.add_argument("--run-mode", choices=["none", "spagcn", "stagate", "all"], default="spagcn")
    parser.add_argument("--reuse-extracted", action="store_true")
    parser.add_argument("--skip-python-install", action="store_true")
    parser.add_argument("--sample-filter", default="")
    args, _unknown = parser.parse_known_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = WORK / "input" / "bundle"
    if extracted.exists() and not args.reuse_extracted:
        shutil.rmtree(extracted)
    extracted.mkdir(parents=True, exist_ok=True)
    log_rows: list[dict[str, object]] = []
    if args.reuse_extracted and (extracted / "sample_manifest.tsv").exists():
        manifest = pd.read_csv(extracted / "sample_manifest.tsv", sep="\t")
        log_event(log_rows, step="reuse_extracted_input", status="success", n_samples=int(manifest.shape[0]))
    elif Path(args.input_tar).exists():
        with tarfile.open(args.input_tar, "r:gz") as handle:
            handle.extractall(extracted)
        manifest = pd.read_csv(extracted / "sample_manifest.tsv", sep="\t")
    else:
        manifest = build_input_from_geo(extracted, log_rows=log_rows)

    map_rows: list[pd.DataFrame] = []
    if not (extracted / "sample_manifest.tsv").exists():
        manifest.to_csv(extracted / "sample_manifest.tsv", sep="\t", index=False)
    if args.sample_filter.strip():
        selected = {value.strip() for value in args.sample_filter.split(",") if value.strip()}
        manifest = manifest[manifest["sample_id"].astype(str).isin(selected)].copy()
        log_event(log_rows, step="sample_filter", status="success", n_samples=int(manifest.shape[0]), sample_filter=",".join(sorted(selected)))

    try:
        if not args.skip_python_install:
            if args.run_mode == "none":
                install_h5_reader_stack(log_rows)
            else:
                install_python_stack(log_rows, include_pyg=args.run_mode in {"stagate", "all"})
        else:
            log_event(log_rows, step="install_python_stack", status="skipped_by_wrapper")
        if args.run_mode in {"stagate", "all"}:
            install_stagate(log_rows)
    except Exception as exc:
        log_event(log_rows, step="install_python_or_stagate", status="failed", error=str(exc), traceback=traceback.format_exc()[-2000:])

    for row in manifest.itertuples(index=False):
        dataset_id = str(row.dataset_id)
        sample_id = str(row.sample_id)
        root = extracted / str(row.bundle_root)
        if args.run_mode in {"spagcn", "all"}:
            try:
                run_spagcn(dataset_id, sample_id, root, map_rows, log_rows)
            except Exception as exc:
                log_event(log_rows, method=METHOD_SPAGCN, dataset_id=dataset_id, sample_id=sample_id, status="sample_failed", error=str(exc), traceback=traceback.format_exc()[-2000:])
        if args.run_mode in {"stagate", "all"}:
            try:
                run_stagate(dataset_id, sample_id, root, map_rows, log_rows)
            except Exception as exc:
                log_event(log_rows, method=METHOD_STAGATE, dataset_id=dataset_id, sample_id=sample_id, status="sample_failed", error=str(exc), traceback=traceback.format_exc()[-2000:])

    map_path = output_dir / "official_domain_maps.tsv"
    if map_rows:
        pd.concat(map_rows, ignore_index=True, sort=False).to_csv(map_path, sep="\t", index=False)
    if args.run_bayesspace:
        install_r_stack(log_rows)
        run_bayesspace(manifest, extracted, map_path, log_rows, int(args.bayesspace_nrep))
    else:
        log_event(log_rows, step="bayesspace", status="skipped_default_use_run_bayesspace_for_r_task")
    pd.DataFrame(log_rows).to_csv(output_dir / "run_log.tsv", sep="\t", index=False)

    archive = output_dir.with_suffix(".tar.gz")
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(output_dir.rglob("*")):
            handle.add(path, arcname=path.relative_to(output_dir))
    print(f"Wrote {archive}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
