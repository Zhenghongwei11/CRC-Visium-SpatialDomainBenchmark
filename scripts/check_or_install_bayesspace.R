#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
install_enabled <- TRUE
if (length(args) > 0 && args[[1]] == "--no-install") {
  install_enabled <- FALSE
}

required_pkgs <- c("BayesSpace", "SingleCellExperiment", "Matrix", "mclust", "hdf5r")
missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing_pkgs) > 0 && install_enabled) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
  }
  suppressMessages(
    BiocManager::install(
      missing_pkgs,
      ask = FALSE,
      update = FALSE
    )
  )
}

missing_after <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_after) > 0) {
  stop(sprintf("Missing required packages: %s", paste(missing_after, collapse = ", ")))
}

cat("BayesSpace environment ready\n")
