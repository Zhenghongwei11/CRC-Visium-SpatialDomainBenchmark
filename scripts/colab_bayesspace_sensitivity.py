#!/usr/bin/env python3
"""Run targeted BayesSpace nrep=1000 sensitivity on Colab."""

from __future__ import annotations

import os
import subprocess
import tarfile
import time
from pathlib import Path

import pandas as pd


WORK = Path("/content/jbcb_bayesspace_sensitivity")
OUT = Path("colab_bayesspace_output")
NREP = 1000
K_VALUES = "4,6"
SEEDS = "11,23,37"

REMOTE_SERIES = {
    "GSE267401": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE267nnn/GSE267401/suppl/GSE267401_RAW.tar",
    "GSE311294": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE311nnn/GSE311294/suppl/GSE311294_RAW.tar",
}
SAMPLES = [
    ("GSE311294", "GSM9322957_TR11_206", "GSE311294"),
    ("GSE267401", "GSM8265211_CTC21P", "GSE267401"),
    ("GSE311294", "GSM9322959_TR11_18105", "GSE311294"),
]
ALLOWED_SUFFIXES = (
    "_matrix.mtx.gz",
    "_barcodes.tsv.gz",
    "_features.tsv.gz",
    "_tissue_positions.csv.gz",
    "_tissue_positions_list.csv.gz",
)


def run(cmd: list[str], timeout: int | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("[cmd]", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, timeout=timeout)


def log(rows: list[dict[str, object]], **kwargs: object) -> None:
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    row.update(kwargs)
    rows.append(row)
    print("[log]", row, flush=True)


def prepare_data(rows: list[dict[str, object]]) -> pd.DataFrame:
    data_root = WORK / "data"
    raw_root = WORK / "raw"
    data_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(
        [
            {"dataset_id": dataset_id, "sample_id": sample_id, "series_id": series_id}
            for dataset_id, sample_id, series_id in SAMPLES
        ]
    )
    for series_id, group in manifest.groupby("series_id"):
        tar_path = raw_root / f"{series_id}_RAW.tar"
        if not tar_path.exists():
            run(["wget", "-q", "-O", str(tar_path), REMOTE_SERIES[str(series_id)]], timeout=7200)
        target = data_root / str(series_id) / "extracted"
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r") as handle:
            members = handle.getmembers()
            wanted = []
            for sample_id in group["sample_id"].astype(str):
                for member in members:
                    name = Path(member.name).name
                    if name.startswith(f"{sample_id}_") and any(name.endswith(suffix) for suffix in ALLOWED_SUFFIXES):
                        member.name = name
                        wanted.append(member)
            handle.extractall(target, members=wanted)
        log(rows, step="extract_geo_members", series_id=str(series_id), n_files=len(wanted), status="success")
    manifest["dataset_root"] = manifest["series_id"].map(lambda value: str(data_root / str(value) / "extracted"))
    manifest.to_csv(WORK / "sample_manifest.tsv", sep="\t", index=False)
    return manifest


def write_r_script(path: Path) -> None:
    path.write_text(
        r'''
suppressPackageStartupMessages({
  library(Matrix)
  library(SingleCellExperiment)
  library(BayesSpace)
  library(mclust)
})
args <- commandArgs(trailingOnly=TRUE)
dataset_id <- args[[1]]
dataset_root <- args[[2]]
sample_id <- args[[3]]
out_map <- args[[4]]
out_bench <- args[[5]]
nrep <- as.integer(args[[6]])
k_values <- as.integer(strsplit(args[[7]], ",")[[1]])
seed_values <- as.integer(strsplit(args[[8]], ",")[[1]])

matrix_file <- file.path(dataset_root, paste0(sample_id, "_matrix.mtx.gz"))
barcodes_file <- file.path(dataset_root, paste0(sample_id, "_barcodes.tsv.gz"))
features_file <- file.path(dataset_root, paste0(sample_id, "_features.tsv.gz"))
coords_file <- file.path(dataset_root, paste0(sample_id, "_tissue_positions.csv.gz"))
if (!file.exists(coords_file)) coords_file <- file.path(dataset_root, paste0(sample_id, "_tissue_positions_list.csv.gz"))

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
                         init.method="mclust", model="t", gamma=3, nrep=nrep, save.chain=FALSE)
    if ("burn.in" %in% names(formals(BayesSpace::spatialCluster))) cluster_args[["burn.in"]] <- min(100L, nrep - 1L)
    if ("verbose" %in% names(formals(BayesSpace::spatialCluster))) cluster_args$verbose <- FALSE
    set.seed(seed)
    start <- proc.time()[["elapsed"]]
    sce_q <- do.call(BayesSpace::spatialCluster, cluster_args)
    times <- c(times, proc.time()[["elapsed"]] - start)
    labels <- as.integer(colData(sce_q)$spatial.cluster)
    labels_by_seed[[as.character(seed)]] <- labels
    map_out <- data.frame(dataset_id=dataset_id, sample_id=sample_id, method_id="Official_BayesSpace_nrep1000",
                          K=q, seed=seed, replicate_type="seed", replicate_id=paste0("seed_", seed),
                          config_id=paste0("K", q, "_seed_", seed), barcode=colnames(sce_q),
                          x=as.numeric(colData(sce_q)$imagecol), y=as.numeric(colData(sce_q)$imagerow),
                          domain_label=labels, status="success", notes=paste0("official_bayesspace_nrep", nrep),
                          stringsAsFactors=FALSE)
    append_mode <- file.exists(out_map) && file.info(out_map)$size > 0
    write.table(map_out, file=out_map, sep="\t", quote=FALSE, row.names=FALSE,
                col.names=!append_mode, append=append_mode)
  }
  ari <- c()
  keys <- names(labels_by_seed)
  for (i in seq_len(length(keys)-1L)) {
    for (j in seq((i+1L), length(keys))) {
      ari <- c(ari, mclust::adjustedRandIndex(labels_by_seed[[keys[[i]]]], labels_by_seed[[keys[[j]]]]))
    }
  }
  bench_rows[[length(bench_rows)+1L]] <- data.frame(dataset_id=dataset_id, sample_id=sample_id,
    method_id="Official_BayesSpace_nrep1000", K=q, seed_count=length(seed_values),
    stability_ari_median=median(ari), stability_ari_iqr=IQR(ari),
    wall_time_sec_median=median(times), failure_rate=0,
    notes=paste0("official_bayesspace_nrep", nrep), stringsAsFactors=FALSE)
}
append_mode <- file.exists(out_bench) && file.info(out_bench)$size > 0
write.table(do.call(rbind, bench_rows), file=out_bench, sep="\t", quote=FALSE, row.names=FALSE,
            col.names=!append_mode, append=append_mode)
''',
        encoding="utf-8",
    )


def install_bayesspace(rows: list[dict[str, object]]) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    micromamba = WORK / "bin" / "micromamba"
    env = WORK / "mamba_env"
    if not micromamba.exists():
        run(["bash", "-lc", f"curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C {WORK} bin/micromamba"], timeout=600)
    if not (env / "bin" / "Rscript").exists():
        run(
            [
                str(micromamba),
                "create",
                "-y",
                "-p",
                str(env),
                "-c",
                "conda-forge",
                "-c",
                "bioconda",
                "r-base",
                "bioconductor-bayesspace",
                "bioconductor-singlecellexperiment",
                "r-mclust",
                "r-matrix",
            ],
            timeout=7200,
        )
    log(rows, step="install_bayesspace_env", status="success", env=str(env))
    return env / "bin" / "Rscript"


def main() -> int:
    rows: list[dict[str, object]] = []
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    manifest = prepare_data(rows)
    rscript = install_bayesspace(rows)
    r_file = WORK / "run_bayesspace_targeted.R"
    write_r_script(r_file)
    map_path = OUT / "official_bayesspace_domain_maps.tsv"
    bench_path = OUT / "official_bayesspace_benchmark.tsv"
    for path in [map_path, bench_path]:
        if path.exists():
            path.unlink()
    for row in manifest.itertuples(index=False):
        try:
            run(
                [
                    str(rscript),
                    str(r_file),
                    str(row.dataset_id),
                    str(row.dataset_root),
                    str(row.sample_id),
                    str(map_path),
                    str(bench_path),
                    str(NREP),
                    K_VALUES,
                    SEEDS,
                ],
                timeout=14400,
            )
            log(rows, method="Official_BayesSpace_nrep1000", dataset_id=str(row.dataset_id), sample_id=str(row.sample_id), status="success")
        except Exception as exc:
            log(rows, method="Official_BayesSpace_nrep1000", dataset_id=str(row.dataset_id), sample_id=str(row.sample_id), status="failed", error=str(exc))
    pd.DataFrame(rows).to_csv(OUT / "run_log.tsv", sep="\t", index=False)
    archive = Path("colab_bayesspace_output.tar.gz")
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(OUT.rglob("*")):
            handle.add(path, arcname=path.relative_to(OUT))
    print(f"Wrote {archive}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
