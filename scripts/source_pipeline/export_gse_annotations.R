#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% c(2,3)) {
  stop("usage: export_gse_annotations.R metadata.rds output_dir [raw_metadata.rds]")
}

metadata_path <- args[[1]]
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

metadata <- readRDS(metadata_path)
required <- c("Barcode", "Layer1", "Layer2", "Layer3")
missing <- setdiff(required, names(metadata))
if (length(missing) > 0) stop("missing metadata columns: ", paste(missing, collapse = ","))

metadata$Barcode <- as.character(metadata$Barcode)
metadata$sample_id <- sub("~.*$", "", metadata$Barcode)
metadata$spot_barcode <- sub("^.*~", "", metadata$Barcode)

sample_map <- data.frame(
  sample_id = c("M-ST-13", "M-ST-15", "M-ST-29", "M-ST-30", "M-ST-31", "M-ST-32", "M-ST-33", "M-ST-34"),
  patient_id = c("Patient1", "Patient6", "Patient11", "Patient11", "Patient9", "Patient9", "Patient10", "Patient10"),
  stringsAsFactors = FALSE
)

combined <- list()
for (sample_id in sample_map$sample_id) {
  rows <- metadata[metadata$sample_id == sample_id, c("spot_barcode", "Layer1", "Layer2", "Layer3")]
  if (nrow(rows) == 0) stop("sample not present in metadata: ", sample_id)
  if (anyDuplicated(rows$spot_barcode)) stop("duplicate label barcodes within sample: ", sample_id)
  names(rows)[1] <- "Barcode"
  names(rows)[4] <- "Pathologist_Layer3"
  write.csv(rows[c("Barcode", "Pathologist_Layer3")],
            file.path(output_dir, paste0("Pathologist_Annotations_", sample_id, ".csv")),
            row.names = FALSE, quote = TRUE, na = "")
  rows$sample_id <- sample_id
  rows$patient_id <- sample_map$patient_id[match(sample_id, sample_map$sample_id)]
  combined[[sample_id]] <- rows
}
write.table(do.call(rbind,combined), file.path(output_dir,"GSE294385_label_layers.tsv"),
            sep="\t",row.names=FALSE,quote=FALSE,na="")
if (length(args)==3) {
  raw <- readRDS(args[[3]])
  stopifnot(all(required %in% names(raw)))
  raw$sample_id <- sub("~.*$","",raw$Barcode)
  raw$Barcode <- sub("^.*~","",raw$Barcode)
  raw <- raw[raw$sample_id %in% sample_map$sample_id,c("sample_id","Barcode","Layer1","Layer2","Layer3")]
  stopifnot(!anyDuplicated(paste(raw$sample_id,raw$Barcode)))
  write.table(raw,file.path(output_dir,"GSE294385_raw_label_layers.tsv"),
              sep="\t",row.names=FALSE,quote=FALSE,na="")
}

write.table(sample_map, file.path(output_dir, "GSE294385_patient_map.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
