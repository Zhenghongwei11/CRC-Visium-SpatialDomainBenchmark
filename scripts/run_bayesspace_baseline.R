#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Matrix)
  library(SingleCellExperiment)
  library(BayesSpace)
  library(mclust)
  library(hdf5r)
})

parse_args <- function(input) {
  result <- list(
    dataset_id = "GSE311294",
    dataset_root = "data/raw/GSE311294/extracted",
    sample_id = "",
    k_grid = "4",
    seed = "11",
    seeds = "",
    nrep = "100",
    gamma = "3",
    output_domain_map_tsv = "",
    domain_map_seed = "",
    output_tsv = "results/benchmarks/bayesspace_stage3.tsv",
    note = "stage3-bayesspace"
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
dataset_root <- args$dataset_root
dataset_id <- args$dataset_id
k_values <- as.integer(strsplit(args$k_grid, ",")[[1]])
nrep <- suppressWarnings(as.integer(args$nrep))
if (is.na(nrep) || nrep < 100) {
  nrep <- 100L
}
gamma_val <- suppressWarnings(as.numeric(args$gamma))
if (!is.finite(gamma_val) || gamma_val <= 0) {
  gamma_val <- 3
}

if (!is.null(args$seeds) && nzchar(args$seeds)) {
  seed_values <- as.integer(strsplit(args$seeds, ",")[[1]])
} else {
  seed_values <- as.integer(args$seed)
}
seed_values <- unique(seed_values[!is.na(seed_values)])
if (length(seed_values) == 0) {
  stop("No valid seed value(s) provided")
}

neighbor_agreement <- function(sce_obj, labels) {
  neighbor_str <- as.character(colData(sce_obj)$spot.neighbors)
  all_matches <- c()
  for (i in seq_along(neighbor_str)) {
    entry <- neighbor_str[[i]]
    if (is.na(entry) || !nzchar(entry)) {
      next
    }
    idx <- suppressWarnings(as.integer(strsplit(entry, ",")[[1]]))
    idx <- idx[!is.na(idx)]
    idx <- idx[idx >= 1 & idx <= length(labels)]
    if (length(idx) == 0) {
      next
    }
    all_matches <- c(all_matches, labels[[i]] == labels[idx])
  }
  if (length(all_matches) == 0) {
    return(NA_real_)
  }
  mean(all_matches)
}

marker_separation_score <- function(sce_obj, labels) {
  expr <- assay(sce_obj, "logcounts")
  label_values <- sort(unique(labels))
  cluster_scores <- c()
  for (cluster in label_values) {
    in_mask <- labels == cluster
    out_mask <- !in_mask
    if (sum(in_mask) < 3 || sum(out_mask) < 3) {
      next
    }
    in_mean <- Matrix::rowMeans(expr[, in_mask, drop = FALSE])
    out_mean <- Matrix::rowMeans(expr[, out_mask, drop = FALSE])
    diff <- in_mean - out_mean
    top_n <- min(20L, length(diff))
    if (top_n < 1) {
      next
    }
    top_vals <- sort(diff, decreasing = TRUE)[seq_len(top_n)]
    cluster_scores <- c(cluster_scores, mean(top_vals))
  }
  if (length(cluster_scores) == 0) {
    return(NA_real_)
  }
  median(cluster_scores)
}

read_10x_h5_counts <- function(h5_path) {
  f <- hdf5r::H5File$new(h5_path, mode = "r")
  on.exit({
    try(f$close_all(), silent = TRUE)
  })

  group <- f[["matrix"]]
  data <- group[["data"]][]
  indices <- group[["indices"]][]
  indptr <- group[["indptr"]][]
  shape <- group[["shape"]][]
  barcodes <- group[["barcodes"]][]

  feat <- group[["features"]]
  feat_name <- feat[["name"]][]
  feat_id <- NULL
  if ("id" %in% names(feat)) {
    feat_id <- feat[["id"]][]
  }

  barcodes <- as.character(barcodes)
  feat_name <- as.character(feat_name)
  if (!is.null(feat_id)) feat_id <- as.character(feat_id)

  m <- new(
    "dgCMatrix",
    Dim = as.integer(shape),
    p = as.integer(indptr),
    i = as.integer(indices),
    x = as.numeric(data)
  )
  rownames(m) <- make.unique(feat_name)
  colnames(m) <- barcodes
  m
}

matrix_files <- sort(list.files(dataset_root, pattern = "_matrix\\.mtx\\.gz$", full.names = TRUE))
h5_files <- sort(list.files(dataset_root, pattern = "_filtered_feature_bc_matrix\\.h5$", full.names = TRUE))
if (length(matrix_files) == 0 && length(h5_files) == 0) {
  stop(sprintf("No Visium matrix files found under %s (expected *_matrix.mtx.gz or *_filtered_feature_bc_matrix.h5)", dataset_root))
}

selected_prefix <- args$sample_id
if (!is.null(selected_prefix) && nzchar(selected_prefix)) {
  selected_path <- file.path(dataset_root, paste0(selected_prefix, "_matrix.mtx.gz"))
  selected_h5 <- file.path(dataset_root, paste0(selected_prefix, "_filtered_feature_bc_matrix.h5"))
  if (file.exists(selected_path)) {
    matrix_file <- selected_path
  } else if (file.exists(selected_h5)) {
    matrix_file <- selected_h5
  } else {
    stop(sprintf("Requested sample_id not found under dataset_root: %s", selected_prefix))
  }
} else {
  if (length(matrix_files) > 0) {
    matrix_file <- matrix_files[[1]]
  } else {
    matrix_file <- h5_files[[1]]
  }
}

is_h5 <- grepl("\\.h5$", matrix_file)
if (is_h5) {
  prefix <- sub("_filtered_feature_bc_matrix\\.h5$", "", basename(matrix_file))
} else {
  prefix <- sub("_matrix\\.mtx\\.gz$", "", basename(matrix_file))
}

coords_file <- file.path(dataset_root, paste0(prefix, "_tissue_positions.csv.gz"))
if (!file.exists(coords_file)) {
  coords_file <- file.path(dataset_root, paste0(prefix, "_tissue_positions_list.csv.gz"))
}
if (!file.exists(coords_file)) {
  coords_file <- file.path(dataset_root, paste0(prefix, "_tissue_positions_list.csv"))
}
if (!file.exists(coords_file)) {
  coords_file <- file.path(dataset_root, paste0(prefix, "_tissue_positions.csv"))
}
if (!file.exists(coords_file)) {
  stop("Missing tissue_positions file for selected sample")
}

if (is_h5) {
  counts <- read_10x_h5_counts(matrix_file)
  coords <- read.csv(coords_file, stringsAsFactors = FALSE, header = FALSE)
} else {
  barcodes_file <- file.path(dataset_root, paste0(prefix, "_barcodes.tsv.gz"))
  features_file <- file.path(dataset_root, paste0(prefix, "_features.tsv.gz"))
  if (!file.exists(barcodes_file)) barcodes_file <- file.path(dataset_root, paste0(prefix, "_barcodes.tsv"))
  if (!file.exists(features_file)) features_file <- file.path(dataset_root, paste0(prefix, "_features.tsv"))
  if (!file.exists(barcodes_file) || !file.exists(features_file)) {
    stop("Missing paired barcodes/features files for selected matrix")
  }
  counts <- readMM(gzfile(matrix_file))
  barcodes <- read.delim(gzfile(barcodes_file), header = FALSE, stringsAsFactors = FALSE)
  features <- read.delim(gzfile(features_file), header = FALSE, stringsAsFactors = FALSE)
  rownames(counts) <- make.unique(features[[2]])
  colnames(counts) <- barcodes[[1]]
  coords <- read.csv(coords_file, stringsAsFactors = FALSE)
}

if (!("barcode" %in% colnames(coords))) {
  colnames(coords) <- c(
    "barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"
  )
}

coords <- coords[coords$barcode %in% colnames(counts), , drop = FALSE]
coords <- coords[match(colnames(counts), coords$barcode), , drop = FALSE]
in_tissue <- coords$in_tissue == 1
if (sum(in_tissue) < 50) {
  stop("Too few in-tissue spots for BayesSpace run")
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

rows <- list()
for (q in k_values) {
  if (q < 2) {
    next
  }
  labels_by_seed <- list()
  spatial_scores <- c()
  marker_scores <- c()
  run_times <- c()
  for (seed in seed_values) {
    cluster_args <- list(
      sce = sce_pre,
      q = q,
      platform = "Visium",
      d = min(15, ncol(reducedDim(sce_pre, "PCA"))),
      init.method = "mclust",
      model = "t",
      gamma = gamma_val,
      nrep = nrep,
      save.chain = FALSE
    )
    if ("burn.in" %in% names(formals(BayesSpace::spatialCluster))) {
      cluster_args[["burn.in"]] <- min(100L, as.integer(cluster_args$nrep) - 1L)
    }
    if ("verbose" %in% names(formals(BayesSpace::spatialCluster))) {
      cluster_args$verbose <- FALSE
    }
    run_start <- proc.time()[["elapsed"]]
    set.seed(seed)
    sce_q <- do.call(BayesSpace::spatialCluster, cluster_args)
    run_elapsed <- proc.time()[["elapsed"]] - run_start
    labels <- as.integer(colData(sce_q)$spatial.cluster)

    if (!is.null(args$output_domain_map_tsv) && nzchar(args$output_domain_map_tsv)) {
      write_seed <- TRUE
      if (!is.null(args$domain_map_seed) && nzchar(args$domain_map_seed)) {
        write_seed <- as.integer(args$domain_map_seed) == seed
      }
      if (write_seed) {
        map_out <- data.frame(
          dataset_id = dataset_id,
          sample_id = prefix,
          method_id = "BayesSpace",
          K = q,
          seed = seed,
          barcode = colnames(sce_q),
          x = as.numeric(colData(sce_q)$imagecol),
          y = as.numeric(colData(sce_q)$imagerow),
          domain_label = labels,
          notes = args$note,
          stringsAsFactors = FALSE
        )
        out_path <- args$output_domain_map_tsv
        append_mode <- file.exists(out_path)
        write.table(
          map_out,
          file = out_path,
          sep = "\t",
          quote = FALSE,
          row.names = FALSE,
          col.names = !append_mode,
          append = append_mode
        )
      }
    }

    labels_by_seed[[as.character(seed)]] <- labels
    spatial_scores <- c(spatial_scores, neighbor_agreement(sce_q, labels))
    marker_scores <- c(marker_scores, marker_separation_score(sce_q, labels))
    run_times <- c(run_times, run_elapsed)
  }

  ari_values <- c()
  if (length(labels_by_seed) > 1) {
    seed_keys <- names(labels_by_seed)
    for (i in seq_len(length(seed_keys) - 1L)) {
      for (j in seq((i + 1L), length(seed_keys))) {
        ari_values <- c(
          ari_values,
          mclust::adjustedRandIndex(labels_by_seed[[seed_keys[[i]]]], labels_by_seed[[seed_keys[[j]]]])
        )
      }
    }
  }

  ari_median <- if (length(ari_values) > 0) median(ari_values) else NA_real_
  ari_iqr <- if (length(ari_values) > 0) IQR(ari_values) else NA_real_
  spatial_median <- if (length(spatial_scores) > 0) median(spatial_scores, na.rm = TRUE) else NA_real_
  spatial_iqr <- if (length(spatial_scores) > 1) IQR(spatial_scores, na.rm = TRUE) else 0
  marker_median <- if (length(marker_scores) > 0) median(marker_scores, na.rm = TRUE) else NA_real_
  marker_iqr <- if (length(marker_scores) > 1) IQR(marker_scores, na.rm = TRUE) else 0
  runtime_median <- if (length(run_times) > 0) median(run_times) else NA_real_

  row <- data.frame(
    dataset_id = dataset_id,
    sample_id = prefix,
    method_id = "BayesSpace",
    method_family = "bayesian_spatial",
    preprocessing_id = "bayesspace_default",
    param_set_id = paste0("K", q),
    K = q,
    seed_count = length(seed_values),
    stability_ari_median = as.numeric(ari_median),
    stability_ari_iqr = as.numeric(ari_iqr),
    spatial_coherence_median = as.numeric(spatial_median),
    spatial_coherence_iqr = as.numeric(spatial_iqr),
    marker_coherence_median = as.numeric(marker_median),
    marker_coherence_iqr = as.numeric(marker_iqr),
    wall_time_sec_median = as.numeric(runtime_median),
    peak_rss_mb_median = NA,
    failure_rate = 0,
    notes = args$note,
    stringsAsFactors = FALSE
  )
  rows[[length(rows) + 1L]] <- row
}

if (length(rows) == 0) {
  stop("No BayesSpace rows generated")
}

output <- do.call(rbind, rows)
write.table(
  output,
  file = args$output_tsv,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
cat(sprintf("Wrote %d BayesSpace rows to %s\n", nrow(output), args$output_tsv))
