#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Matrix)
  library(SingleCellExperiment)
  library(BayesSpace)
  library(mclust)
})

parse_args <- function(input) {
  result <- list(
    dataset_id = "",
    dataset_root = "",
    sample_ids = "",
    k_grid = "4,6",
    seed = "11",
    top_n_grid = "10,20,50",
    output_tsv = "results/benchmarks/marker_coherence_topn_sensitivity.tsv",
    note = "topn-sensitivity-bayesspace"
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

args <- parse_args(commandArgs(trailingOnly = TRUE))
dataset_id <- args$dataset_id
dataset_root <- args$dataset_root

if (is.null(dataset_id) || !nzchar(dataset_id)) stop("Missing --dataset-id")
if (is.null(dataset_root) || !nzchar(dataset_root)) stop("Missing --dataset-root")

sample_ids <- strsplit(args$sample_ids, ",")[[1]]
sample_ids <- unique(sample_ids[nzchar(sample_ids)])
if (length(sample_ids) == 0) stop("Missing --sample-ids")

k_values <- as.integer(strsplit(args$k_grid, ",")[[1]])
k_values <- k_values[!is.na(k_values) & k_values >= 2]
if (length(k_values) == 0) stop("No valid K values")

seed <- suppressWarnings(as.integer(args$seed))
if (is.na(seed)) stop("Invalid --seed")

topn_values <- suppressWarnings(as.integer(strsplit(args$top_n_grid, ",")[[1]]))
topn_values <- topn_values[!is.na(topn_values) & topn_values >= 1]
topn_values <- unique(topn_values)
if (length(topn_values) == 0) stop("No valid top_n values")

read_lines <- function(path) {
  if (grepl("\\.gz$", path)) {
    return(readLines(gzfile(path), warn = FALSE))
  }
  readLines(path, warn = FALSE)
}

load_flat_sample <- function(root, sample_id) {
  matrix_file <- file.path(root, paste0(sample_id, "_matrix.mtx.gz"))
  barcodes_file <- file.path(root, paste0(sample_id, "_barcodes.tsv.gz"))
  features_file <- file.path(root, paste0(sample_id, "_features.tsv.gz"))
  coords_file <- file.path(root, paste0(sample_id, "_tissue_positions.csv.gz"))
  if (!file.exists(coords_file)) {
    coords_file <- file.path(root, paste0(sample_id, "_tissue_positions_list.csv.gz"))
  }
  if (!file.exists(matrix_file)) stop(sprintf("Missing matrix file: %s", matrix_file))
  if (!file.exists(barcodes_file)) stop(sprintf("Missing barcodes file: %s", barcodes_file))
  if (!file.exists(features_file)) stop(sprintf("Missing features file: %s", features_file))
  if (!file.exists(coords_file)) stop(sprintf("Missing coords file: %s", coords_file))

  counts <- readMM(gzfile(matrix_file))
  barcodes <- read.delim(gzfile(barcodes_file), header = FALSE, stringsAsFactors = FALSE)
  features <- read.delim(gzfile(features_file), header = FALSE, stringsAsFactors = FALSE)
  coords <- read.csv(gzfile(coords_file), stringsAsFactors = FALSE)
  if (!("barcode" %in% colnames(coords))) {
    colnames(coords) <- c(
      "barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"
    )
  }

  rownames(counts) <- make.unique(features[[2]])
  colnames(counts) <- barcodes[[1]]

  coords <- coords[coords$barcode %in% colnames(counts), , drop = FALSE]
  coords <- coords[match(colnames(counts), coords$barcode), , drop = FALSE]
  in_tissue <- coords$in_tissue == 1
  if (sum(in_tissue) < 50) stop("Too few in-tissue spots")

  counts <- counts[, in_tissue, drop = FALSE]
  coords <- coords[in_tissue, , drop = FALSE]

  sce <- SingleCellExperiment(assays = list(counts = counts))
  colData(sce)$array_row <- coords$array_row
  colData(sce)$array_col <- coords$array_col
  colData(sce)$row <- coords$array_row
  colData(sce)$col <- coords$array_col
  colData(sce)$imagerow <- coords$pxl_row_in_fullres
  colData(sce)$imagecol <- coords$pxl_col_in_fullres
  sce
}

marker_coherence_topn <- function(sce_obj, labels, top_n) {
  expr <- assay(sce_obj, "logcounts")
  label_values <- sort(unique(labels))
  cluster_scores <- c()
  for (cluster in label_values) {
    in_mask <- labels == cluster
    out_mask <- !in_mask
    if (sum(in_mask) < 3 || sum(out_mask) < 3) next
    in_mean <- Matrix::rowMeans(expr[, in_mask, drop = FALSE])
    out_mean <- Matrix::rowMeans(expr[, out_mask, drop = FALSE])
    diff <- in_mean - out_mean
    use_n <- min(as.integer(top_n), length(diff))
    if (use_n < 1) next
    top_vals <- sort(diff, decreasing = TRUE)[seq_len(use_n)]
    cluster_scores <- c(cluster_scores, mean(top_vals))
  }
  if (length(cluster_scores) == 0) return(NA_real_)
  median(cluster_scores)
}

rows <- list()
for (sample_id in sample_ids) {
  sce <- load_flat_sample(dataset_root, sample_id)
  set.seed(seed)
  sce_pre <- spatialPreprocess(
    sce,
    platform = "Visium",
    n.HVGs = min(2000, nrow(sce)),
    n.PCs = min(15, ncol(sce) - 1L),
    log.normalize = TRUE
  )

  for (q in k_values) {
    cluster_args <- list(
      sce = sce_pre,
      q = q,
      platform = "Visium",
      d = min(15, ncol(reducedDim(sce_pre, "PCA"))),
      init.method = "mclust",
      model = "t",
      gamma = 3,
      nrep = 200,
      save.chain = FALSE
    )
    if ("burn.in" %in% names(formals(BayesSpace::spatialCluster))) {
      cluster_args[["burn.in"]] <- min(100L, as.integer(cluster_args$nrep) - 1L)
    }
    if ("verbose" %in% names(formals(BayesSpace::spatialCluster))) {
      cluster_args$verbose <- FALSE
    }

    set.seed(seed)
    sce_q <- do.call(BayesSpace::spatialCluster, cluster_args)
    labels <- as.integer(colData(sce_q)$spatial.cluster)

    for (top_n in topn_values) {
      mk <- marker_coherence_topn(sce_q, labels, top_n)
      rows[[length(rows) + 1L]] <- data.frame(
        dataset_id = dataset_id,
        sample_id = sample_id,
        method_id = "BayesSpace",
        preprocessing_id = "bayesspace_default",
        K = q,
        seed = seed,
        top_marker_n = top_n,
        marker_coherence_median = as.numeric(mk),
        n_spots = ncol(assay(sce_q, "logcounts")),
        notes = args$note,
        stringsAsFactors = FALSE
      )
    }
  }
}

output <- do.call(rbind, rows)
dir.create(dirname(args$output_tsv), recursive = TRUE, showWarnings = FALSE)
write.table(
  output,
  file = args$output_tsv,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
cat(sprintf("Wrote %d rows to %s\n", nrow(output), args$output_tsv))

