#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

bootstrap_ci <- function(x, statistic_fn, n_boot = 5000L, seed = 11L) {
  x <- x[is.finite(x)]
  if (length(x) < 2) {
    return(c(ci_lower = NA_real_, ci_upper = NA_real_))
  }
  set.seed(seed)
  n <- length(x)
  stats <- replicate(n_boot, {
    idx <- sample.int(n, size = n, replace = TRUE)
    statistic_fn(x[idx])
  })
  as.numeric(stats::quantile(stats, probs = c(0.025, 0.975), na.rm = TRUE))
}

input_tsv <- "results/benchmarks/histology_edge_alignment.tsv"
output_tsv <- "results/benchmarks/histology_edge_alignment_summary.tsv"

if (!file.exists(input_tsv)) {
  stop(sprintf("Missing %s (run scripts/build_histology_edge_alignment.py first)", input_tsv))
}

x <- read.delim(input_tsv, sep = "\t", header = TRUE, check.names = FALSE)
needed <- c("dataset_id", "sample_id", "method_id", "K", "delta_grad_boundary_minus_within", "status")
missing <- setdiff(needed, names(x))
if (length(missing) > 0) stop(sprintf("Missing required columns: %s", paste(missing, collapse = ", ")))

x <- subset(x, status == "success")
x$K <- suppressWarnings(as.integer(x$K))
x$delta_grad_boundary_minus_within <- suppressWarnings(as.numeric(x$delta_grad_boundary_minus_within))

bayes <- subset(x, method_id == "BayesSpace")
m5 <- subset(x, method_id == "M5_stagate")
if (nrow(bayes) == 0 || nrow(m5) == 0) stop("Need both BayesSpace and M5_stagate rows")

d <- merge(
  bayes[, c("dataset_id", "sample_id", "K", "delta_grad_boundary_minus_within")],
  m5[, c("dataset_id", "sample_id", "K", "delta_grad_boundary_minus_within")],
  by = c("dataset_id", "sample_id", "K"),
  suffixes = c("_bayes", "_m5")
)
if (nrow(d) == 0) stop("No overlapping sample×K units between BayesSpace and M5")

d$delta <- d$delta_grad_boundary_minus_within_bayes - d$delta_grad_boundary_minus_within_m5

# Aggregate within each sample across K={4,6} using equal weights (mean).
ds <- aggregate(delta ~ dataset_id + sample_id, data = d, FUN = function(v) mean(v, na.rm = TRUE))
deltas <- ds$delta

pval <- tryCatch(wilcox.test(deltas, mu = 0, alternative = "two.sided", exact = FALSE)$p.value, error = function(e) NA_real_)
effect <- stats::median(deltas, na.rm = TRUE)
ci <- bootstrap_ci(deltas, statistic_fn = function(z) stats::median(z, na.rm = TRUE), n_boot = 5000L, seed = 11L)
direction <- mean(deltas > 0, na.rm = TRUE)

out <- data.frame(
  claim_id = "C4_histology_edge_alignment",
  comparison_id = "BayesSpace_vs_M5_stagate_K4K6",
  metric_id = "delta_grad_boundary_minus_within",
  analysis_unit = "sample",
  test_name = "wilcoxon_signed_rank_two_sided",
  pvalue = pval,
  effect_size_name = "paired_median_delta",
  effect_size_value = effect,
  effect_size_ci_lower = ci[[1]],
  effect_size_ci_upper = ci[[2]],
  direction_consistency_value = direction,
  notes = sprintf("n_samples=%d; anchor=histology_edge_alignment", nrow(ds)),
  stringsAsFactors = FALSE
)

write.table(out, file = output_tsv, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("Wrote %s\n", output_tsv))

