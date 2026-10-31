#!/usr/bin/env Rscript
# Official pinned spacexr only. No fitting fallback, parameter search or relaxed retry.
args <- commandArgs(trailingOnly = TRUE)
script_arg <- grep('^--file=', commandArgs(), value = TRUE)
script_file <- if (length(script_arg)) sub('^--file=', '', script_arg[[1]]) else 'scripts/run_rctd.R'
script_file <- normalizePath(script_file, mustWork = TRUE)
root <- normalizePath(file.path(dirname(script_file), '..'), mustWork = TRUE)
mode <- 'pilot'; requested_section <- NULL; preflight_only <- FALSE; output_root <- file.path(root, 'results/rctd')
if (length(args)) {
  i <- 1L
  while (i <= length(args)) {
    if (args[[i]] == '--mode') { i <- i + 1L; mode <- args[[i]]
    } else if (args[[i]] == '--section') { i <- i + 1L; requested_section <- args[[i]]
    } else if (args[[i]] == '--output') { i <- i + 1L; output_root <- args[[i]]
    } else if (args[[i]] == '--preflight') { preflight_only <- TRUE
    } else stop(paste('Unknown argument:', args[[i]]))
    i <- i + 1L
  }
}
stopifnot(mode %in% c('pilot', 'all'))
setwd(root)
Sys.setenv(OMP_NUM_THREADS = '1', OPENBLAS_NUM_THREADS = '1', MKL_NUM_THREADS = '1')
# Use the caller-provided R library path or the standard R library.
# PSOCK workers must resolve the same real package namespaces as the parent process.
Sys.setenv(R_LIBS_USER = paste(.libPaths(), collapse = .Platform$path.sep))
output_root <- normalizePath(output_root, mustWork = FALSE)
base_out <- file.path(output_root, mode)
dir.create(base_out, recursive = TRUE, showWarnings = FALSE)
sha <- function(path) {
  x <- system2('/usr/bin/shasum', c('-a', '256', shQuote(normalizePath(path, mustWork = TRUE))), stdout = TRUE)
  if (!is.null(attr(x, 'status')) && attr(x, 'status') != 0) stop('shasum failed')
  strsplit(x[[1]], '[[:space:]]+')[[1]][[1]]
}
now <- function() format(Sys.time(), '%Y-%m-%dT%H:%M:%OS3Z', tz = 'UTC')
# A dependency failure must leave an actual machine-readable failure, never mock results.
bootstrap_failure <- function(e) {
  target <- file.path(base_out, if (is.null(requested_section)) 'RUNNER_FAILURE.json' else paste0(requested_section, '_RUNNER_FAILURE.json'))
  writeLines(paste0('{"status":"failed_software_or_protocol_preflight","fit_started":false,"error":', encodeString(conditionMessage(e), quote = '"'), ',"time_utc":', encodeString(now(), quote = '"'), '}'), target)
  message(conditionMessage(e)); quit(save = 'no', status = 1L)
}
tryCatch({
  if (!requireNamespace('jsonlite', quietly = TRUE)) stop('Real jsonlite namespace unavailable')
  if (!requireNamespace('Matrix', quietly = TRUE)) stop('Real Matrix namespace unavailable')
}, error = bootstrap_failure)
read_json <- function(path) jsonlite::fromJSON(path, simplifyVector = FALSE)
write_json <- function(x, path) jsonlite::write_json(x, path, auto_unbox = TRUE, pretty = TRUE, null = 'null', na = 'null', digits = NA)
write_tsv <- function(x, name) {
  con <- if (grepl('\\.gz$', name)) gzfile(name, 'wt') else file(name, 'wt')
  on.exit(close(con)); write.table(x, con, sep = '\t', quote = FALSE, row.names = FALSE, na = 'NA')
}
read_tsv <- function(path, header = TRUE) {
  con <- gzfile(path, 'rt'); on.exit(close(con))
  read.delim(con, header = header, sep = '\t', quote = '', check.names = FALSE, stringsAsFactors = FALSE)
}
read_mtx <- function(path) {
  con <- gzfile(path, 'rt'); on.exit(close(con)); as(Matrix::readMM(con), 'CsparseMatrix')
}
write_matrix <- function(x, path, first = 'gene') {
  out <- data.frame(rownames(x), as.matrix(x), check.names = FALSE); names(out)[[1]] <- first
  write_tsv(out, path)
}
protocol_file <- file.path(root, 'config/rctd_model.json')
lock_file <- file.path(root, 'config/rctd_model_lock.json')
protocol <- tryCatch(read_json(protocol_file), error = bootstrap_failure)
protocol_sha <- sha(protocol_file)
runner_sha <- sha(script_file)
tryCatch(stopifnot(identical(protocol_sha, read_json(lock_file)$sha256)), error = bootstrap_failure)
settings <- protocol$settings
stopifnot(identical(settings$max_cores, 2L) || identical(settings$max_cores, 2), isFALSE(settings$test_mode), settings$doublet_mode == 'full')
ref_dir <- file.path(root, 'data/reference_inputs/single_cell')
ref_mtx <- file.path(ref_dir, 'GSE132465_real_cell_subset.mtx.gz')
contract_dir <- file.path(root, 'third_party/spacexr/R')
if (mode == 'pilot') sections <- unlist(protocol$pilot_sections) else sections <- list.dirs(file.path(output_root, 'inputs/all'), recursive = FALSE, full.names = FALSE)
if (!is.null(requested_section)) {
  stopifnot(requested_section %in% sections); sections <- requested_section
}
if (!length(sections)) stop('No requested input sections')

pinned_function <- function(name, file) {
  for (e in parse(file)) {
    if (is.call(e) && identical(e[[1]], as.name('<-')) && identical(e[[2]], as.name(name))) return(eval(e[[3]], envir = new.env(parent = baseenv())))
  }
  stop(paste('Pinned source function absent:', name))
}
software_state <- function() {
  package_manifest <- read_json(file.path(root, 'config/spacexr_source.json'))
  stopifnot(identical(package_manifest$commit, protocol$upstream_commit), identical(package_manifest$version, '2.2.1'))
  available <- requireNamespace('spacexr', quietly = TRUE)
  x <- list(time_utc = now(), available = available, R = R.version.string, lib_paths = .libPaths(), worker_R_LIBS_USER = Sys.getenv('R_LIBS_USER'), upstream_commit = protocol$upstream_commit,
            package_source_manifest_sha256 = sha(file.path(root, 'config/spacexr_source.json')))
  if (!available) return(x)
  x$version <- as.character(utils::packageVersion('spacexr')); x$package_path <- find.package('spacexr')
  x$pinned_native_function_checks <- list()
  for (n in c('create.RCTD', 'run.RCTD', 'Reference', 'SpatialRNA')) {
    source <- file.path(contract_dir, if (n %in% c('create.RCTD', 'run.RCTD')) 'spacexr.R' else paste0(n, '.R'))
    expected <- pinned_function(n, source); actual <- getFromNamespace(n, 'spacexr')
    x$pinned_native_function_checks[[n]] <- isTRUE(all.equal(formals(expected), formals(actual), check.attributes = FALSE)) && isTRUE(all.equal(body(expected), body(actual), check.attributes = FALSE))
  }
  x$required_Q_tables <- lapply(c(paste0('Q_mat_', 1:5, '.rds'), 'X_vals.rds'), function(n) {
    p <- system.file('extdata', 'Qmat', n, package = 'spacexr')
    list(name = n, path = p, exists = nzchar(p) && file.exists(p), bytes = if (nzchar(p) && file.exists(p)) unname(file.info(p)$size) else NULL)
  })
  x
}
compact_checkpoint <- function(object, out, stage, spatial_manifest) {
  vars <- object@internal_vars
  q_description <- list(Q_mat_dim = dim(vars$Q_mat), X_vals_length = length(vars$X_vals), source = 'installed pinned spacexr extdata/Qmat; not duplicated in checkpoint')
  vars$Q_mat <- NULL; vars$X_vals <- NULL
  value <- list(schema = 'rctd-compact-summary-v1', stage = stage, time_utc = now(), protocol_sha256 = protocol_sha,
                input_manifest = spatial_manifest, config = object@config, cell_type_info = object@cell_type_info, internal_vars = vars,
                results = object@results, spatial_barcodes = colnames(object@spatialRNA@counts), full_feature_nUMI = object@spatialRNA@nUMI,
                coordinates = object@spatialRNA@coords, native_probability_table_description = q_description,
                note = 'Compact native fit summary, not an independently restorable RCTD object. Counts remain only in original input files; no reference or spatial count copies serialized here.')
  saveRDS(value, file.path(out, 'compact_checkpoint.rds'), compress = 'gzip')
}

run_section <- function(sid) {
  out <- file.path(base_out, sid); dir.create(out, recursive = TRUE, showWarnings = FALSE)
  input <- file.path(output_root, 'inputs', mode, sid)
  manifest_file <- file.path(input, 'INPUT_MANIFEST.json')
  status_file <- file.path(out, 'RUN_STATUS.json')
  if (file.exists(status_file)) {
    prior <- read_json(status_file)
    if (identical(prior$status, 'fit_completed')) {
      stopifnot(identical(prior$protocol_sha256, protocol_sha), identical(prior$runner_sha256, runner_sha), identical(prior$input_manifest_sha256, sha(manifest_file)))
      for (x in prior$outputs) stopifnot(file.exists(file.path(out, x$file)), identical(sha(file.path(out, x$file)), x$sha256))
      message('Reusing completed exact-protocol checkpoint: ', sid); return(TRUE)
    }
  }
  record <- list(section = sid, mode = mode, status = 'starting', phase = 'software_preflight', fit_started = FALSE,
                 start_utc = now(), protocol_sha256 = protocol_sha, runner_sha256 = runner_sha, excluded_genes = protocol$excluded_genes,
                 settings_requested = settings, interpretation = protocol$interpretation)
  warnings <- list(); native <- NULL; started <- Sys.time(); log_path <- file.path(out, paste0('native_fit_', format(Sys.time(), '%Y%m%dT%H%M%SZ', tz = 'UTC'), '.log'))
  log_con <- file(log_path, 'wt'); sink(log_con); sink(log_con, type = 'message')
  on.exit({sink(type = 'message'); sink(); close(log_con)}, add = TRUE)
  status <- function(phase) { record$phase <<- phase; record$updated_utc <<- now(); write_json(record, status_file) }
  finish_record <- function() {
    record$warnings <<- warnings; record$elapsed_seconds <<- as.numeric(difftime(Sys.time(), started, units = 'secs')); record$native_log <<- basename(log_path)
    record$end_utc <<- now(); write_json(record, status_file)
    cat(jsonlite::toJSON(record, auto_unbox = TRUE, null = 'null', na = 'null', digits = NA), '\n', file = file.path(out, 'ATTEMPTS.jsonl'), append = TRUE)
  }
  tryCatch(withCallingHandlers({
    software <- software_state(); record$software <- software; write_json(software, file.path(out, 'SOFTWARE_STATE.json'))
    stopifnot(isTRUE(software$available), identical(software$version, '2.2.1'), all(unlist(software$pinned_native_function_checks)), all(vapply(software$required_Q_tables, function(x) isTRUE(x$exists), logical(1))))
    writeLines(capture.output(utils::sessionInfo()), file.path(out, 'SESSION_INFO.txt'))
    status('input_integrity')
    input_manifest <- read_json(manifest_file); record$input_manifest_sha256 <- sha(manifest_file)
    stopifnot(identical(input_manifest$protocol_sha256, protocol_sha), identical(input_manifest$reference_sha256, protocol$reference_input_sha256), identical(sha(ref_mtx), protocol$reference_input_sha256))
    for (name in names(input_manifest$files)) stopifnot(identical(sha(file.path(input, name)), input_manifest$files[[name]]$sha256))
    genes <- read_tsv(file.path(input, 'genes.tsv.gz'))$gene
    spots <- read_tsv(file.path(input, 'spots.tsv.gz')); counts <- read_mtx(file.path(input, 'counts.mtx.gz'))
    stopifnot(nrow(counts) == length(genes), ncol(counts) == nrow(spots), !anyDuplicated(genes), !anyDuplicated(spots$barcode), !anyNA(genes), !anyNA(spots$barcode))
    dimnames(counts) <- list(genes, spots$barcode)
    stopifnot(all(is.finite(counts@x)), all(counts@x >= 0), all(counts@x == round(counts@x)), all(Matrix::colSums(counts) == spots$fit_gene_UMIs), !any(genes %in% unlist(protocol$excluded_genes)))
    stopifnot(all(is.finite(spots$nUMI_full_feature)), all(spots$nUMI_full_feature >= Matrix::colSums(counts)), all(spots$nUMI_full_feature == round(spots$nUMI_full_feature)))
    ref_counts <- read_mtx(ref_mtx)
    ref_genes <- read_tsv(file.path(ref_dir, 'GSE132465_real_cell_subset_genes.tsv.gz'), FALSE)[[1]]
    ref_ids <- read_tsv(file.path(ref_dir, 'GSE132465_real_cell_subset_barcodes.tsv.gz'), FALSE)[[1]]
    ref_labels <- read_tsv(file.path(ref_dir, 'GSE132465_real_cell_subset_labels.tsv.gz'))
    stopifnot(identical(ref_ids, ref_labels$Index), nrow(ref_counts) == length(ref_genes), ncol(ref_counts) == length(ref_ids), !anyDuplicated(ref_genes), !anyDuplicated(ref_ids), all(Matrix::colSums(ref_counts) == ref_labels$total_UMI))
    dimnames(ref_counts) <- list(ref_genes, ref_ids)
    stopifnot(all(is.finite(ref_counts@x)), all(ref_counts@x >= 0), all(ref_counts@x == round(ref_counts@x)))
    taxonomy <- read_json(file.path(root, 'config/reference_groups.json'))
    stopifnot(identical(sha(file.path(root, 'config/reference_groups.json')), protocol$taxonomy_sha256))
    groups <- names(taxonomy$common_reference_groups)
    biological_map <- setNames(rep(groups, lengths(taxonomy$common_reference_groups)), unlist(taxonomy$common_reference_groups, use.names = FALSE))
    stopifnot(identical(unname(biological_map[ref_labels$Cell_subtype]), ref_labels$stage2_reference_group))
    ref_types <- factor(ref_labels$stage2_reference_group, levels = groups); names(ref_types) <- ref_ids
    stopifnot(!anyNA(ref_types), !any(ref_labels$Cell_subtype == 'Unknown'), all(table(ref_types) >= settings$CELL_MIN_INSTANCE))
    shared <- genes[genes %in% ref_genes & !genes %in% unlist(protocol$excluded_genes)]
    stopifnot(length(shared) == length(genes), length(shared) >= 10)
    ref_counts <- ref_counts[shared, , drop = FALSE]; counts <- counts[shared, , drop = FALSE]
    ref_umi <- as.numeric(ref_labels$total_UMI); names(ref_umi) <- ref_ids
    spot_umi <- as.numeric(spots$nUMI_full_feature); names(spot_umi) <- spots$barcode
    coords <- data.frame(x = spots$x, y = spots$y, row.names = spots$barcode)
    stopifnot(all(is.finite(as.matrix(coords))))
    write_tsv(data.frame(barcode = ref_ids, group = as.character(ref_types), patient = ref_labels$Patient, sample = ref_labels$Sample, tissue_class = ref_labels$Class, original_subtype = ref_labels$Cell_subtype, nUMI_full_feature = ref_umi, common_fit_counts = Matrix::colSums(ref_counts)), file.path(out, 'REFERENCE_CELLS.tsv.gz'))
    write_tsv(data.frame(gene = shared), file.path(out, 'COMMON_INPUT_GENES.tsv.gz'))
    record$requested_spots <- nrow(spots); record$reference_cells <- length(ref_ids); record$reference_group_counts <- as.list(table(ref_types)); record$common_input_genes <- length(shared)
    record$deliberate_offset_counts_difference <- list(spatial = 'full original spatial feature nUMI retained; counts restricted to shared genes excluding7heldout', reference = 'full original source33,694gene nUMI retained; counts restricted to same shared genes excluding7heldout')
    if (preflight_only) { record$status <- 'preflight_completed_no_fit'; status('preflight_complete'); finish_record(); return(TRUE) }
    set.seed(20261026)
    # Constructors retain true complete-library offsets; their mismatch warnings are expected and saved.
    reference <- spacexr::Reference(ref_counts, ref_types, nUMI = ref_umi)
    puck <- spacexr::SpatialRNA(coords, counts, nUMI = spot_umi)
    status('create_RCTD')
    create_settings <- settings[setdiff(names(settings), 'doublet_mode')]
    native <- do.call(spacexr::create.RCTD, c(list(spatialRNA = puck, reference = reference), create_settings))
    rm(reference, puck, ref_counts, counts); gc(verbose = FALSE)
    record$native_config <- native@config
    gene_reg <- native@internal_vars$gene_list_reg; gene_bulk <- native@internal_vars$gene_list_bulk
    stopifnot(!any(c(gene_reg, gene_bulk) %in% unlist(protocol$excluded_genes)))
    write_tsv(data.frame(gene = gene_reg), file.path(out, 'NATIVE_REGRESSION_GENES.tsv.gz'))
    write_tsv(data.frame(gene = gene_bulk), file.path(out, 'NATIVE_BULK_GENES.tsv.gz'))
    write_matrix(stats::cor(as.matrix(native@cell_type_info$info[[1]])), file.path(out, 'REFERENCE_PROFILE_CORRELATION_COMMON.tsv'), 'group')
    write_matrix(stats::cor(as.matrix(native@cell_type_info$info[[1]][gene_reg, , drop = FALSE])), file.path(out, 'REFERENCE_PROFILE_CORRELATION_REGRESSION.tsv'), 'group')
    retention <- spots
    retention$retained_initial_native_UMI_counts_filter <- retention$barcode %in% colnames(native@originalSpatialRNA@counts)
    retention$retained_native_bulk_filter <- retention$barcode %in% colnames(native@spatialRNA@counts)
    bulk_totals <- Matrix::colSums(native@originalSpatialRNA@counts[gene_bulk, , drop = FALSE])
    retention$native_bulk_gene_counts <- as.numeric(bulk_totals[retention$barcode])
    retention$filter_reason <- ifelse(retention$retained_native_bulk_filter, 'retained', ifelse(!retention$retained_initial_native_UMI_counts_filter, 'native_initial_UMI_or_common_counts_filter', 'native_bulk_gene_counts_filter'))
    write_tsv(retention, file.path(out, 'REQUESTED_RETAINED_BARCODES.tsv.gz'))
    record$retained_spots <- ncol(native@spatialRNA@counts); record$retained_fraction <- record$retained_spots / record$requested_spots
    compact_checkpoint(native, out, 'created_before_fit', input_manifest)
    record$fit_started <- TRUE; status('run_RCTD_full')
    native <- spacexr::run.RCTD(native, doublet_mode = settings$doublet_mode)
    status('postfit_diagnostics')
    native_weights <- as.matrix(native@results$weights)
    stopifnot(setequal(colnames(native_weights), groups), identical(rownames(native_weights), colnames(native@spatialRNA@counts)))
    native_weights <- native_weights[, groups, drop = FALSE]
    row_totals <- rowSums(native_weights)
    valid <- all(is.finite(native_weights)) && all(native_weights >= 0) && all(is.finite(row_totals)) && all(row_totals > 0)
    write_matrix(native_weights, file.path(out, 'NATIVE_UNCONSTRAINED_WEIGHTS.tsv.gz'), 'barcode')
    write_tsv(data.frame(barcode = names(row_totals), native_weight_sum = row_totals), file.path(out, 'NATIVE_WEIGHT_ROW_TOTALS.tsv.gz'))
    if (!valid) stop('Native weights violate finite/nonnegative/positive-row-total gate; saved without clipping or replacing')
    normalized <- native_weights / row_totals
    stopifnot(all(abs(rowSums(normalized) - 1) <= 1e-8))
    write_matrix(normalized, file.path(out, 'INFERRED_NORMALIZED_MIXTURE_WEIGHTS.tsv.gz'), 'barcode')
    write_matrix(stats::cor(as.matrix(native@cell_type_info$renorm[[1]][gene_reg, , drop = FALSE])), file.path(out, 'REFERENCE_PROFILE_CORRELATION_RENORMALIZED_REGRESSION.tsv'), 'group')
    bulk_weights <- native@internal_vars$proportions
    write_tsv(data.frame(group = native@cell_type_info$info[[2]], native_bulk_weight = as.numeric(bulk_weights)), file.path(out, 'NATIVE_BULK_WEIGHTS.tsv'))
    profiles <- as.matrix(native@cell_type_info$renorm[[1]][gene_reg, groups, drop = FALSE])
    obs <- as.matrix(native@spatialRNA@counts[gene_reg, rownames(native_weights), drop = FALSE])
    mu <- sweep(profiles %*% t(native_weights), 2, native@spatialRNA@nUMI[rownames(native_weights)], '*')
    stopifnot(all(is.finite(mu)), all(mu >= 0))
    native_nll <- getFromNamespace('calc_log_l_vec', 'spacexr')
    recon <- lapply(seq_len(ncol(obs)), function(j) {
      y <- obs[, j]; p <- mu[, j]; safe <- pmax(p, 1e-12)
      term <- ifelse(y > 0, y * log(pmax(y, 1e-12) / safe) - (y - safe), safe)
      data.frame(barcode = colnames(obs)[j], stratum = spots$stratum[match(colnames(obs)[j], spots$barcode)], n_genes = nrow(obs), observed_regression_UMIs = sum(y), predicted_regression_UMIs = sum(p), mean_absolute_count_error = mean(abs(y - p)), root_mean_square_count_error = sqrt(mean((y - p)^2)), descriptive_Poisson_deviance = 2 * sum(term), native_Poisson_lognormal_negative_log_likelihood = native_nll(p, y))
    })
    write_tsv(do.call(rbind, recon), file.path(out, 'NATIVE_FIT_RECONSTRUCTION_BY_SPOT.tsv.gz'))
    write_tsv(data.frame(gene = gene_reg, observed_bulk_count = rowSums(obs), predicted_bulk_count = rowSums(mu), mean_absolute_spot_count_error = rowMeans(abs(obs - mu))), file.path(out, 'NATIVE_FIT_RECONSTRUCTION_BY_GENE.tsv.gz'))
    diagnostic <- list(sigma = native@internal_vars$sigma, native_bulk_weight_sum = sum(bulk_weights), n_regression_genes = length(gene_reg), n_bulk_genes = length(gene_bulk), weight_semantics = 'Official full mode uses unconstrained native weights. Native values retained; normalized mixture weights are native weights divided by their own positive row sum. Neither is measured physical cell fraction.', reconstruction = 'native renormalized profiles times native weights times full-feature nUMI, on native regression genes', native_likelihood = 'spacexr internal calc_log_l_vec evaluated after run.RCTD likelihood setup; uses package native count cap/interpolation, not descriptive Poisson deviance', optimizer_convergence_codes = 'Unavailable: upstream full-mode results preserve weights only. Completion is return-without-error, not proof of global optimizer convergence.', counts_retained_in_checkpoint = FALSE)
    write_json(diagnostic, file.path(out, 'NATIVE_FIT_DIAGNOSTICS.json'))
    compact_checkpoint(native, out, 'native_full_fit_completed', input_manifest)
    record$gates <- list(official_pinned_namespace_loaded = TRUE, seven_groups_min25 = TRUE, native_run_completed = TRUE, requested_spot_retention_ge90percent = record$retained_fraction >= 0.90, finite_nonnegative_weights_positive_rows = valid, normalized_row_sums_valid = TRUE, reference_profile_ambiguity_review = 'not_assessed', systematic_fit_error_review = 'not_assessed')
    record$status <- 'fit_completed'; record$phase <- 'complete'
    outputs <- setdiff(list.files(out, full.names = TRUE), c(status_file, file.path(out, 'ATTEMPTS.jsonl'), log_path))
    record$outputs <- lapply(outputs, function(p) list(file = basename(p), bytes = unname(file.info(p)$size), sha256 = sha(p)))
    finish_record(); TRUE
  }, warning = function(w) {
    warnings[[length(warnings) + 1L]] <<- list(message = conditionMessage(w), call = paste(deparse(conditionCall(w)), collapse = ' '), phase = record$phase)
    cat('NATIVE WARNING [', record$phase, ']: ', conditionMessage(w), '\n', sep = '')
    # Also leave the original warning unsuppressed; exact text is saved immediately above and in JSON.
  }), error = function(e) {
    record$status <- 'failed'; record$error <- list(message = conditionMessage(e), call = paste(deparse(conditionCall(e)), collapse = ' '), calls = lapply(sys.calls(), function(x) paste(deparse(x), collapse = ' ')))
    if (!is.null(native)) try(compact_checkpoint(native, out, paste0('failure_last_available_', record$phase), if (exists('input_manifest')) input_manifest else NULL), silent = TRUE)
    finish_record(); message('ACTUAL RCTD FAILURE: ', conditionMessage(e)); FALSE
  })
}
for (sid in sections) {
  ok <- run_section(sid)
  if (!ok) quit(save = 'no', status = 1L)
}
