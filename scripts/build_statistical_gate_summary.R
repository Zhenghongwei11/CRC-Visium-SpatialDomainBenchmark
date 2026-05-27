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

input_tsv <- "results/benchmarks/method_benchmark.tsv"
output_tsv <- "results/benchmarks/statistical_gate_summary.tsv"

x <- read.delim(input_tsv, sep = "\t", header = TRUE, check.names = FALSE)

num_cols <- c(
  "K", "seed_count", "stability_ari_median", "stability_ari_iqr",
  "spatial_coherence_median", "spatial_coherence_iqr", "marker_coherence_median",
  "marker_coherence_iqr", "wall_time_sec_median", "peak_rss_mb_median", "failure_rate"
)
for (col in intersect(num_cols, names(x))) x[[col]] <- suppressWarnings(as.numeric(x[[col]]))

dedupe_max_seed_count <- function(df, key_cols) {
  if (nrow(df) == 0) return(df)
  # Prefer rows with the largest seed_count; break ties by keeping the later row order.
  df[["..row_index"]] <- seq_len(nrow(df))
  df <- df[order(df[["seed_count"]], df[["..row_index"]], decreasing = TRUE), , drop = FALSE]
  # Keep the first row per key after sorting.
  key <- do.call(paste, c(df[key_cols], sep = "\t"))
  df <- df[!duplicated(key), , drop = FALSE]
  df[["..row_index"]] <- NULL
  df
}

bayes <- subset(x, method_id == "BayesSpace" & grepl("^rigor-backfill", notes))
bayes <- dedupe_max_seed_count(bayes, c("dataset_id", "sample_id", "K"))

base <- subset(
  x,
  method_id %in% c("M0_expr_kmeans", "M1_spatial_concat_kmeans", "M2_spatial_ward", "M3_spatial_leiden", "M4_spagcn", "M5_stagate") &
    grepl("^stage3[a-z]-full-replication", notes)
)
base <- dedupe_max_seed_count(base, c("dataset_id", "sample_id", "method_id", "K"))

if (nrow(bayes) == 0) stop("No BayesSpace rigor rows found")
if (nrow(base) == 0) stop("No baseline rows found")

keys <- unique(bayes[, c("dataset_id", "sample_id", "K")])
base <- merge(base, keys, by = c("dataset_id", "sample_id", "K"))

agg_mean <- function(df, value_col, by_cols) {
  out <- aggregate(df[[value_col]], by = df[by_cols], FUN = function(v) mean(v, na.rm = TRUE))
  names(out)[ncol(out)] <- value_col
  out
}

bayes_sp <- agg_mean(bayes, "spatial_coherence_median", c("dataset_id", "sample_id", "K"))
bayes_mk <- agg_mean(bayes, "marker_coherence_median", c("dataset_id", "sample_id", "K"))
base_sp <- agg_mean(base, "spatial_coherence_median", c("dataset_id", "sample_id", "K", "method_id"))
base_mk <- agg_mean(base, "marker_coherence_median", c("dataset_id", "sample_id", "K", "method_id"))

n_target_samples <- length(unique(bayes$sample_id))

calc_c1_row <- function(baseline_method_id, metric_id, bayes_metric_df, base_metric_df) {
  base_m <- subset(base_metric_df, method_id == baseline_method_id)
  d <- merge(bayes_metric_df, base_m, by = c("dataset_id", "sample_id", "K"), suffixes = c("_bayes", "_base"))
  if (nrow(d) == 0) {
    return(list(
      claim_id = "C1_domain_quality",
      comparison_id = paste0("BayesSpace_vs_", baseline_method_id, "_K4K6"),
      metric_id = metric_id,
      analysis_unit = "sample",
      test_name = "wilcoxon_signed_rank_two_sided",
      pvalue = NA_real_,
      alpha_family = "F1",
      multiplicity_method = "BH_FDR",
      effect_size_name = "paired_median_delta",
      effect_size_value = NA_real_,
      effect_size_ci_lower = NA_real_,
      effect_size_ci_upper = NA_real_,
      effect_size_threshold = 0.02,
      effect_size_pass = FALSE,
      direction_consistency_value = NA_real_,
      direction_consistency_threshold = 0.70,
      direction_consistency_pass = FALSE,
      stability_gate_value = 0,
      stability_gate_threshold = 0.80,
      stability_gate_pass = FALSE,
      notes = sprintf("n_samples=0 (target=%d); missing baseline rows", n_target_samples)
    ))
  }

  d$delta <- d[[paste0(metric_id, "_bayes")]] - d[[paste0(metric_id, "_base")]]

  ds <- aggregate(delta ~ dataset_id + sample_id, data = d, FUN = function(v) mean(v, na.rm = TRUE))
  deltas <- ds$delta

  pval <- tryCatch(wilcox.test(deltas, mu = 0, alternative = "two.sided", exact = FALSE)$p.value, error = function(e) NA_real_)
  effect <- stats::median(deltas, na.rm = TRUE)
  ci <- bootstrap_ci(deltas, statistic_fn = function(z) stats::median(z, na.rm = TRUE), n_boot = 5000L, seed = 11L)
  direction <- mean(deltas > 0, na.rm = TRUE)
  coverage <- nrow(ds) / n_target_samples

  effect_pass <- is.finite(effect) && effect >= 0.02
  direction_pass <- is.finite(direction) && direction >= 0.70
  coverage_pass <- is.finite(coverage) && coverage >= 0.80

  list(
    claim_id = "C1_domain_quality",
    comparison_id = paste0("BayesSpace_vs_", baseline_method_id, "_K4K6"),
    metric_id = metric_id,
    analysis_unit = "sample",
    test_name = "wilcoxon_signed_rank_two_sided",
    pvalue = pval,
    alpha_family = "F1",
    multiplicity_method = "BH_FDR",
    effect_size_name = "paired_median_delta",
    effect_size_value = effect,
    effect_size_ci_lower = ci[[1]],
    effect_size_ci_upper = ci[[2]],
    effect_size_threshold = 0.02,
    effect_size_pass = effect_pass,
    direction_consistency_value = direction,
    direction_consistency_threshold = 0.70,
    direction_consistency_pass = direction_pass,
    stability_gate_value = coverage,
    stability_gate_threshold = 0.80,
    stability_gate_pass = coverage_pass,
    notes = sprintf("n_samples=%d (target=%d)", nrow(ds), n_target_samples)
  )
}

# Primary family: a modern, strong baseline (to avoid strawman comparisons).
# Treat BayesSpace vs STAGATE (M5) as the primary C1 domain-quality endpoints.
c1_primary_rows <- list(
  calc_c1_row("M5_stagate", "spatial_coherence_median", bayes_sp, base_sp),
  calc_c1_row("M5_stagate", "marker_coherence_median", bayes_mk, base_mk)
)

# Secondary family: minimal baselines for calibration (reported, but not used to
# upgrade/claim primary superiority over modern methods).
c1_secondary_rows <- list(
  calc_c1_row("M0_expr_kmeans", "spatial_coherence_median", bayes_sp, base_sp),
  calc_c1_row("M0_expr_kmeans", "marker_coherence_median", bayes_mk, base_mk),
  calc_c1_row("M1_spatial_concat_kmeans", "spatial_coherence_median", bayes_sp, base_sp),
  calc_c1_row("M1_spatial_concat_kmeans", "marker_coherence_median", bayes_mk, base_mk)
)
for (i in seq_along(c1_secondary_rows)) c1_secondary_rows[[i]]$alpha_family <- "F1_secondary"

# Extension family (exploratory additional baselines; reported as context only).
c1_ext_rows <- list(
  calc_c1_row("M2_spatial_ward", "spatial_coherence_median", bayes_sp, base_sp),
  calc_c1_row("M2_spatial_ward", "marker_coherence_median", bayes_mk, base_mk),
  calc_c1_row("M3_spatial_leiden", "spatial_coherence_median", bayes_sp, base_sp),
  calc_c1_row("M3_spatial_leiden", "marker_coherence_median", bayes_mk, base_mk),
  calc_c1_row("M4_spagcn", "spatial_coherence_median", bayes_sp, base_sp),
  calc_c1_row("M4_spagcn", "marker_coherence_median", bayes_mk, base_mk)
)

# FDR: primary family only (2 endpoints).
c1_primary_pvals <- sapply(c1_primary_rows, function(r) r$pvalue)
c1_primary_fdr <- p.adjust(c1_primary_pvals, method = "BH")

for (i in seq_along(c1_primary_rows)) {
  c1_primary_rows[[i]]$fdr <- c1_primary_fdr[[i]]
  pass <- is.finite(c1_primary_rows[[i]]$fdr) && c1_primary_rows[[i]]$fdr < 0.05 &&
    isTRUE(c1_primary_rows[[i]]$effect_size_pass) &&
    isTRUE(c1_primary_rows[[i]]$direction_consistency_pass) &&
    isTRUE(c1_primary_rows[[i]]$stability_gate_pass)
  c1_primary_rows[[i]]$overall_gate_status <- if (pass) "pass" else "fail"
  c1_primary_rows[[i]]$support_tier <- if (pass) "supported" else if (isTRUE(c1_primary_rows[[i]]$effect_size_pass) && isTRUE(c1_primary_rows[[i]]$direction_consistency_pass)) "suggestive" else "not supported"
}

# FDR: secondary family (4 endpoints).
if (length(c1_secondary_rows) > 0) {
  pvals <- sapply(c1_secondary_rows, function(r) r$pvalue)
  fdrs <- p.adjust(pvals, method = "BH")
  for (i in seq_along(c1_secondary_rows)) {
    c1_secondary_rows[[i]]$fdr <- fdrs[[i]]
    pass <- is.finite(c1_secondary_rows[[i]]$fdr) && c1_secondary_rows[[i]]$fdr < 0.05 &&
      isTRUE(c1_secondary_rows[[i]]$effect_size_pass) &&
      isTRUE(c1_secondary_rows[[i]]$direction_consistency_pass) &&
      isTRUE(c1_secondary_rows[[i]]$stability_gate_pass)
    c1_secondary_rows[[i]]$overall_gate_status <- if (pass) "pass" else "fail"
    c1_secondary_rows[[i]]$support_tier <- if (pass) "supported" else if (isTRUE(c1_secondary_rows[[i]]$effect_size_pass) && isTRUE(c1_secondary_rows[[i]]$direction_consistency_pass)) "suggestive" else "not supported"
  }
}

if (length(c1_ext_rows) > 0) {
  for (i in seq_along(c1_ext_rows)) {
    # Separate exploratory families to avoid mixing different baseline classes.
    if (grepl("_ward", c1_ext_rows[[i]]$comparison_id)) {
      c1_ext_rows[[i]]$alpha_family <- "F1_ext"
    } else if (grepl("_leiden", c1_ext_rows[[i]]$comparison_id)) {
      c1_ext_rows[[i]]$alpha_family <- "F1_ext2"
    } else if (grepl("_spagcn", c1_ext_rows[[i]]$comparison_id)) {
      c1_ext_rows[[i]]$alpha_family <- "F1_ext3"
    } else {
      c1_ext_rows[[i]]$alpha_family <- "F1_ext3"
    }
  }
  for (fam in unique(sapply(c1_ext_rows, function(r) r$alpha_family))) {
    idx <- which(sapply(c1_ext_rows, function(r) r$alpha_family) == fam)
    pvals <- sapply(c1_ext_rows[idx], function(r) r$pvalue)
    fdrs <- p.adjust(pvals, method = "BH")
    for (j in seq_along(idx)) {
      i <- idx[[j]]
      c1_ext_rows[[i]]$fdr <- fdrs[[j]]
      pass <- is.finite(c1_ext_rows[[i]]$fdr) && c1_ext_rows[[i]]$fdr < 0.05 &&
        isTRUE(c1_ext_rows[[i]]$effect_size_pass) &&
        isTRUE(c1_ext_rows[[i]]$direction_consistency_pass) &&
        isTRUE(c1_ext_rows[[i]]$stability_gate_pass)
      c1_ext_rows[[i]]$overall_gate_status <- if (pass) "pass" else "fail"
      c1_ext_rows[[i]]$support_tier <- if (pass) "supported" else if (isTRUE(c1_ext_rows[[i]]$effect_size_pass) && isTRUE(c1_ext_rows[[i]]$direction_consistency_pass)) "suggestive" else "not supported"
    }
  }
}

c1_rows <- c(c1_primary_rows, c1_secondary_rows, c1_ext_rows)

# C2: stability gate
c2_ari <- bayes$stability_ari_median
c2_iqr <- bayes$stability_ari_iqr
c2_n <- sum(!is.na(c2_ari))
c2_p <- tryCatch(wilcox.test(c2_ari, mu = 0.60, alternative = "greater", exact = FALSE)$p.value, error = function(e) NA_real_)
c2_effect <- median(c2_ari, na.rm = TRUE)
c2_ci <- bootstrap_ci(c2_ari, statistic_fn = function(z) stats::median(z, na.rm = TRUE), n_boot = 5000L, seed = 11L)
c2_direction <- mean(c2_ari >= 0.60, na.rm = TRUE)
c2_stability <- mean(c2_iqr <= 0.15, na.rm = TRUE)

c2_effect_pass <- is.finite(c2_effect) && c2_effect >= 0.60
c2_direction_pass <- is.finite(c2_direction) && c2_direction >= 0.70
c2_stability_pass <- is.finite(c2_stability) && c2_stability >= 0.80
c2_pass <- is.finite(c2_p) && c2_p < 0.10 && c2_effect_pass && c2_direction_pass && c2_stability_pass

c2_row <- list(
  claim_id = "C2_sensitivity",
  comparison_id = "BayesSpace_stability_gate_K4K6",
  metric_id = "stability_ari_median",
  analysis_unit = "sample_k",
  test_name = "wilcoxon_one_sample_greater_than_0.60",
  pvalue = c2_p,
  fdr = c2_p,
  alpha_family = "F2",
  multiplicity_method = "BH_FDR",
  effect_size_name = "median_ari",
  effect_size_value = c2_effect,
  effect_size_ci_lower = c2_ci[[1]],
  effect_size_ci_upper = c2_ci[[2]],
  effect_size_threshold = 0.60,
  effect_size_pass = c2_effect_pass,
  direction_consistency_value = c2_direction,
  direction_consistency_threshold = 0.70,
  direction_consistency_pass = c2_direction_pass,
  stability_gate_value = c2_stability,
  stability_gate_threshold = 0.80,
  stability_gate_pass = c2_stability_pass,
  overall_gate_status = if (c2_pass) "pass" else "fail",
  support_tier = if (c2_pass) "supported" else if (c2_effect_pass && c2_direction_pass) "suggestive" else "not supported",
  notes = sprintf("n_sample_k=%d", c2_n)
)

# C3: descriptive local feasibility
runtime <- bayes$wall_time_sec_median
failure <- bayes$failure_rate
c3_n <- sum(!is.na(runtime))
c3_median_runtime <- median(runtime, na.rm = TRUE)
c3_ci <- bootstrap_ci(runtime, statistic_fn = function(z) stats::median(z, na.rm = TRUE), n_boot = 5000L, seed = 11L)
c3_local_fraction <- mean(runtime <= 1800, na.rm = TRUE)
c3_failure_free <- mean(failure == 0, na.rm = TRUE)

c3_effect_pass <- is.finite(c3_median_runtime) && c3_median_runtime <= 1800
c3_direction_pass <- is.finite(c3_local_fraction) && c3_local_fraction >= 0.80
c3_stability_pass <- is.finite(c3_failure_free) && c3_failure_free >= 1.00
c3_gate_pass <- c3_effect_pass && c3_direction_pass && c3_stability_pass

c3_row <- list(
  claim_id = "C3_compute_feasibility",
  comparison_id = "BayesSpace_local_feasibility_K4K6",
  metric_id = "wall_time_sec_median",
  analysis_unit = "sample_k",
  test_name = "descriptive_gate_check",
  pvalue = NA_real_,
  fdr = NA_real_,
  alpha_family = "F3",
  multiplicity_method = "descriptive",
  effect_size_name = "median_runtime_sec",
  effect_size_value = c3_median_runtime,
  effect_size_ci_lower = c3_ci[[1]],
  effect_size_ci_upper = c3_ci[[2]],
  effect_size_threshold = 1800,
  effect_size_pass = c3_effect_pass,
  direction_consistency_value = c3_local_fraction,
  direction_consistency_threshold = 0.80,
  direction_consistency_pass = c3_direction_pass,
  stability_gate_value = c3_failure_free,
  stability_gate_threshold = 1.00,
  stability_gate_pass = c3_stability_pass,
  overall_gate_status = if (c3_gate_pass) "pass" else "fail",
  support_tier = if (c3_gate_pass) "suggestive" else "not supported",
  notes = sprintf("n_sample_k=%d; scope=local-first-only", c3_n)
)

rows <- c(c1_rows, list(c2_row, c3_row))

order_cols <- c(
  "claim_id", "comparison_id", "metric_id", "analysis_unit",
  "test_name", "pvalue", "fdr", "alpha_family", "multiplicity_method",
  "effect_size_name", "effect_size_value", "effect_size_ci_lower", "effect_size_ci_upper",
  "effect_size_threshold", "effect_size_pass",
  "direction_consistency_value", "direction_consistency_threshold", "direction_consistency_pass",
  "stability_gate_value", "stability_gate_threshold", "stability_gate_pass",
  "overall_gate_status", "support_tier", "notes"
)

out <- do.call(rbind, lapply(rows, function(r) as.data.frame(r[order_cols], stringsAsFactors = FALSE)))
logic_cols <- c("effect_size_pass", "direction_consistency_pass", "stability_gate_pass")
for (col in logic_cols) out[[col]] <- ifelse(is.na(out[[col]]), NA, ifelse(as.logical(out[[col]]), "true", "false"))

write.table(out, file = output_tsv, sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
cat(sprintf("Wrote %d rows to %s\n", nrow(out), output_tsv))
