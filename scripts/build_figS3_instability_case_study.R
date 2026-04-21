#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Matrix)
  library(SingleCellExperiment)
  library(BayesSpace)
  library(mclust)
})

parse_args <- function(input) {
  result <- list(
    dataset_id = "GSE311294",
    dataset_root = "data/raw/GSE311294/extracted",
    sample_id = "GSM9322957_TR11_206",
    k = "4",
    seeds = "11,23,67,101",
    reference_method_tsv = "results/benchmarks/method_benchmark_locked.tsv",
    output_tsv = "results/figures/figS3_instability_case_study.tsv",
    note = "figS3-bayesspace-instability-case-study"
  )
  i <- 1L
  while (i <= length(input)) {
    key <- input[[i]]
    if (startsWith(key, "--") && i < length(input)) {
      value <- input[[i + 1L]]
      name <- gsub("^--", "", key)
      name <- gsub("-", "_", name)
      result[[name]] <- value
      i <- i + 2L
    } else {
      i <- i + 1L
    }
  }
  result
}

select_marker <- function(gene_names, candidates) {
  for (g in candidates) {
    if (g %in% gene_names) {
      return(g)
    }
  }
  return("")
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
dataset_root <- args$dataset_root
dataset_id <- args$dataset_id
sample_id <- args$sample_id
q <- as.integer(args$k)

seed_values <- as.integer(strsplit(args$seeds, ",")[[1]])
seed_values <- unique(seed_values[!is.na(seed_values)])
if (length(seed_values) < 2) {
  stop("Need at least two seeds for instability case study")
}

ref_ari_median <- NA_real_
ref_ari_iqr <- NA_real_
ref_seed_list <- ""
ref_seed_count <- NA_integer_
ref_runtime_median <- NA_real_
if (!is.null(args$reference_method_tsv) && nzchar(args$reference_method_tsv) && file.exists(args$reference_method_tsv)) {
  ref <- read.delim(args$reference_method_tsv, stringsAsFactors = FALSE)
  if (nrow(ref) > 0) {
    ref_row <- ref[
      ref$dataset_id == dataset_id &
        ref$sample_id == sample_id &
        ref$method_id == "BayesSpace" &
        as.integer(ref$K) == q,
      , drop = FALSE
    ]
    if (nrow(ref_row) >= 1) {
      ref_ari_median <- suppressWarnings(as.numeric(ref_row$stability_ari_median[[1]]))
      ref_ari_iqr <- suppressWarnings(as.numeric(ref_row$stability_ari_iqr[[1]]))
      ref_seed_count <- suppressWarnings(as.integer(ref_row$seed_count[[1]]))
      ref_runtime_median <- suppressWarnings(as.numeric(ref_row$wall_time_sec_median[[1]]))
    }
  }
}

matrix_file <- file.path(dataset_root, paste0(sample_id, "_matrix.mtx.gz"))
barcodes_file <- file.path(dataset_root, paste0(sample_id, "_barcodes.tsv.gz"))
features_file <- file.path(dataset_root, paste0(sample_id, "_features.tsv.gz"))
coords_file <- file.path(dataset_root, paste0(sample_id, "_tissue_positions_list.csv.gz"))
if (!file.exists(coords_file)) {
  coords_file <- file.path(dataset_root, paste0(sample_id, "_tissue_positions.csv.gz"))
}

if (!file.exists(matrix_file) || !file.exists(barcodes_file) || !file.exists(features_file) || !file.exists(coords_file)) {
  stop("Missing flat-matrix Visium files for requested sample_id under dataset_root")
}

counts <- readMM(gzfile(matrix_file))
barcodes <- read.delim(gzfile(barcodes_file), header = FALSE, stringsAsFactors = FALSE)
features <- read.delim(gzfile(features_file), header = FALSE, stringsAsFactors = FALSE)
coords_raw <- read.csv(gzfile(coords_file), stringsAsFactors = FALSE, header = FALSE)

if (is.character(coords_raw[1, 1]) && coords_raw[1, 1] == "barcode") {
  coords <- read.csv(gzfile(coords_file), stringsAsFactors = FALSE)
} else {
  coords <- coords_raw
  colnames(coords) <- c("barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres")
}

rownames(counts) <- make.unique(features[[2]])
colnames(counts) <- barcodes[[1]]

coords <- coords[coords$barcode %in% colnames(counts), , drop = FALSE]
coords <- coords[match(colnames(counts), coords$barcode), , drop = FALSE]
in_tissue <- coords$in_tissue == 1
if (sum(in_tissue) < 50) {
  stop("Too few in-tissue spots for BayesSpace case study")
}

counts <- counts[, in_tissue, drop = FALSE]
coords <- coords[in_tissue, , drop = FALSE]

sce <- SingleCellExperiment(assays = list(counts = counts))
colData(sce)$array_row <- coords$array_row
colData(sce)$array_col <- coords$array_col
colData(sce)$row <- coords$array_row
colData(sce)$col <- coords$array_col
colData(sce)$imagerow <- coords$pxl_row_in_fullres
colData(sce)$imagecol <- coords$pxl_col_in_fullres

set.seed(seed_values[[1]])
sce_pre <- spatialPreprocess(
  sce,
  platform = "Visium",
  n.HVGs = min(2000, nrow(sce)),
  n.PCs = min(15, ncol(sce) - 1L),
  log.normalize = TRUE
)

gene_names <- rownames(sce_pre)
epi_marker <- select_marker(gene_names, c("EPCAM", "KRT19", "KRT8", "KRT18"))
str_marker <- select_marker(gene_names, c("COL1A1", "DCN", "COL3A1", "COL1A2"))
imm_marker <- select_marker(gene_names, c("PTPRC", "CD3D", "MS4A1", "LYZ"))

if (epi_marker == "" || str_marker == "" || imm_marker == "") {
  missing <- c()
  if (epi_marker == "") missing <- c(missing, "epithelial")
  if (str_marker == "") missing <- c(missing, "stromal")
  if (imm_marker == "") missing <- c(missing, "immune")
  stop(sprintf("Missing marker category genes in this sample: %s", paste(missing, collapse = ", ")))
}

log_expr <- assay(sce_pre, "logcounts")
if (is.null(rownames(log_expr))) {
  rownames(log_expr) <- rownames(sce_pre)
}
if (is.null(colnames(log_expr))) {
  colnames(log_expr) <- colnames(sce_pre)
}
epi_expr <- as.numeric(log_expr[epi_marker, ])
str_expr <- as.numeric(log_expr[str_marker, ])
imm_expr <- as.numeric(log_expr[imm_marker, ])

cluster_args_base <- list(
  sce = sce_pre,
  q = q,
  platform = "Visium",
  d = min(15, {
    dim_names <- reducedDimNames(sce_pre)
    if ("PCA" %in% dim_names) {
      ncol(reducedDim(sce_pre, "PCA"))
    } else if (length(dim_names) > 0) {
      ncol(reducedDim(sce_pre, dim_names[[1]]))
    } else {
      15L
    }
  }),
  init.method = "mclust",
  model = "t",
  gamma = 3,
  nrep = 200,
  save.chain = FALSE
)

if ("burn.in" %in% names(formals(BayesSpace::spatialCluster))) {
  cluster_args_base[["burn.in"]] <- min(100L, as.integer(cluster_args_base$nrep) - 1L)
}
if ("verbose" %in% names(formals(BayesSpace::spatialCluster))) {
  cluster_args_base$verbose <- FALSE
}

rows <- list()
labels_by_seed <- list()
run_times <- c()

for (seed in seed_values) {
  cat(sprintf("[figS3] Running BayesSpace seed=%d (q=%d)\n", seed, q))
  run_start <- proc.time()[["elapsed"]]
  set.seed(seed)
  sce_q <- tryCatch(
    do.call(BayesSpace::spatialCluster, cluster_args_base),
    error = function(e) {
      stop(sprintf("BayesSpace spatialCluster failed for seed=%d: %s", seed, as.character(e)))
    }
  )
  run_elapsed <- proc.time()[["elapsed"]] - run_start
  run_times <- c(run_times, run_elapsed)

  labels <- as.integer(colData(sce_q)$spatial.cluster)
  labels_by_seed[[as.character(seed)]] <- labels

  rows[[length(rows) + 1L]] <- data.frame(
    dataset_id = dataset_id,
    sample_id = sample_id,
    method_id = "BayesSpace",
    K = q,
    seed = seed,
    barcode = colnames(sce_q),
    x = as.numeric(colData(sce_q)$imagecol),
    y = as.numeric(colData(sce_q)$imagerow),
    domain_label = labels,
    marker_epithelial_name = epi_marker,
    marker_stromal_name = str_marker,
    marker_immune_name = imm_marker,
    expr_epithelial = epi_expr,
    expr_stromal = str_expr,
    expr_immune = imm_expr,
    runtime_sec = run_elapsed,
    notes = args$note,
    stringsAsFactors = FALSE
  )
}

ari_values <- c()
seed_keys <- names(labels_by_seed)
for (i in seq_len(length(seed_keys) - 1L)) {
  for (j in seq((i + 1L), length(seed_keys))) {
    ari_values <- c(
      ari_values,
      mclust::adjustedRandIndex(labels_by_seed[[seed_keys[[i]]]], labels_by_seed[[seed_keys[[j]]]])
    )
  }
}

case_ari_median <- median(ari_values)
case_ari_iqr <- IQR(ari_values)
case_runtime_median <- median(run_times)

ari_median <- if (!is.na(ref_ari_median)) ref_ari_median else case_ari_median
ari_iqr <- if (!is.na(ref_ari_iqr)) ref_ari_iqr else case_ari_iqr
runtime_median <- if (!is.na(ref_runtime_median)) ref_runtime_median else case_runtime_median

out <- do.call(rbind, rows)
out$stability_ari_median <- ari_median
out$stability_ari_iqr <- ari_iqr
out$runtime_median_sec <- runtime_median
out$seed_list <- args$seeds
out$case_ari_median <- case_ari_median
out$case_ari_iqr <- case_ari_iqr
out$case_runtime_median_sec <- case_runtime_median
out$reference_seed_count <- ref_seed_count
out$reference_method_tsv <- args$reference_method_tsv

out_dir <- dirname(args$output_tsv)
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
}

write.table(out, file = args$output_tsv, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("Wrote %d case-study rows to %s\n", nrow(out), args$output_tsv))
