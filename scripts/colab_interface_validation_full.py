#!/usr/bin/env python3
"""Run the locked morphology-anchored interface validation in Colab.

The raw Valdeolivas packages are downloaded and processed in the remote
runtime. The script runs the two official neural implementations used for the
technical pilot, four declared non-neural baselines, and the frozen
pathology-anchor versus computational-boundary contrasts. The strict estimator
requires a computational tumor-stroma adjacency; the anchor-retention
estimator removes that adjacency gate while keeping the pathology-defined
near/far sets fixed. No expression-level outcome is read before this script
starts.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("JBCB_INTERFACE_ROOT", "/content/jbcb_interface_validation_full"))
STAGE = ROOT / "input"
OUT = ROOT / "output"
OFFICIAL_RUNNER = Path("/content/colab_official_sensitivity.py")
DATASET_MODE = os.environ.get("JBCB_INTERFACE_DATASET", "VALDEOLIVAS").strip().upper()
if DATASET_MODE not in {"VALDEOLIVAS", "GSE294385"}:
    raise ValueError(f"Unsupported JBCB_INTERFACE_DATASET: {DATASET_MODE}")
DATASET_ID = "VALDEOLIVAS"
RUN_BAYESPACE = os.environ.get("JBCB_RUN_BAYESPACE", "0").strip().lower() in {"1", "true", "yes"}
ALLOW_OFFICIAL_REUSE = os.environ.get("JBCB_REUSE_OFFICIAL", "0").strip().lower() in {"1", "true", "yes"}
try:
    BAYESPACE_NREP = max(100, int(os.environ.get("JBCB_BAYESPACE_NREP", "1000")))
except ValueError:
    BAYESPACE_NREP = 1000
ANNOTATION_URL = "https://zenodo.org/records/7760264/files/Pathology_SpotAnnotations.zip?download=1"
ZENODO_PACKAGE_TEMPLATE = "https://zenodo.org/api/records/7760264/files/{sample_id}.zip/content"
ZENODO_PACKAGE_DIRECT_TEMPLATE = "https://zenodo.org/records/7760264/files/{sample_id}.zip?download=1"
ZENODO_PACKAGE_PLAIN_TEMPLATE = "https://zenodo.org/records/7760264/files/{sample_id}.zip"
ZENODO_PACKAGE_RECORD_TEMPLATE = "https://zenodo.org/record/7760264/files/{sample_id}.zip?download=1"
ZENODO_PACKAGE_MIRROR_TEMPLATES = (
    "https://zenodo.org/records/7739700/files/{sample_id}.zip?download=1",
    "https://zenodo.org/records/7553463/files/{sample_id}.zip?download=1",
    "https://zenodo.org/record/7739700/files/{sample_id}.zip?download=1",
    "https://zenodo.org/record/7553463/files/{sample_id}.zip?download=1",
)
EXPECTED_PACKAGE_MD5 = {
    "SN048_A121573_Rep1": "608a39f21da059024121407967a76b8a",
    "SN048_A121573_Rep2": "a9f516ca415ad68be283ef0b431c35ce",
    "SN048_A416371_Rep1": "8aec27998074f672fcdaba6faa609da0",
    "SN048_A416371_Rep2": "9a49067a4a6d89521601b562011dc02e",
    "SN123_A551763_Rep1": "b26940f8bf3b3e9855b0a116825e638c",
    "SN123_A595688_Rep1": "b5aa2ee18977b0be67a72ff2b966ab26",
    "SN123_A798015_Rep1": "0e589a2c96546fff5ca21cb18d597791",
    "SN123_A938797_Rep1_X": "08e3c7fe308db62b658cacc6b4be4e1c",
    "SN124_A551763_Rep2": "adf0ea575a09f473fa4ceb465a6ec66e",
    "SN124_A595688_Rep2": "808ed49a24eea7d41c575d8c238caa76",
    "SN124_A798015_Rep2": "db421ccd3e1be463573be562c3266aef",
    "SN124_A938797_Rep2": "ee145e0f97146586baa8ef95edae05b0",
    "SN84_A120838_Rep1": "4a185c5e1bf88995ee0c03dbb4e8b63c",
    "SN84_A120838_Rep2": "d59ffee02162ba7b900199786e536a30",
}
PROTOCOL_VERSION = "1.1"
NOMINAL_SPOT_DIAMETER_UM = 55.0
SEEDS = [11, 23, 37]
K_VALUES = [4, 6]
PRIMARY_GENES = ["TGFB1", "CXCL12", "ACTA2", "TAGLN"]
SCORE_DEFINITIONS = {
    "primary_barrier": {
        "label": "primary stromal barrier-associated score",
        "genes": ["TGFB1", "CXCL12", "ACTA2", "TAGLN"],
        "role": "primary",
    },
    "CAF_FAP": {
        "label": "CAF/FAP-associated score",
        "genes": ["FAP", "COL1A1", "DCN", "COL3A1", "LUM"],
        "role": "secondary",
    },
    "SPP1_myeloid": {
        "label": "SPP1-myeloid-associated score",
        "genes": ["SPP1", "LST1", "TYROBP", "C1QA", "C1QB", "C1QC"],
        "role": "secondary",
    },
    "T_cell": {
        "label": "T-cell-associated score",
        "genes": ["TRAC", "CD3D", "CD3E", "CD247"],
        "role": "secondary",
    },
    "TGFb_CXCL12_barrier": {
        "label": "broader TGF-beta/CXCL12 barrier score",
        "genes": ["TGFB1", "CXCL12", "ACTA2", "TAGLN", "FAP", "COL1A1"],
        "role": "secondary",
    },
    "immune_exclusion": {
        "label": "immune-exclusion contrast",
        "genes": [],
        "role": "secondary",
    },
}
BOOTSTRAP_REPLICATES = 1000
MIN_GROUP_SPOTS = 20
MIN_BLOCKS = 3
ESTIMATOR_LABELS = {
    "strict_computational_interface": "strict computational interface",
    "anchor_retention": "morphology-anchor retention",
}

SAMPLES = [
    {"sample_id": "SN048_A121573_Rep1", "section_id": "S5_Rec", "patient_id": "A121573", "replicate": "Rep1"},
    {"sample_id": "SN048_A121573_Rep2", "section_id": "S5_Rec", "patient_id": "A121573", "replicate": "Rep2"},
    {"sample_id": "SN048_A416371_Rep1", "section_id": "S3_Col_R", "patient_id": "A416371", "replicate": "Rep1"},
    {"sample_id": "SN048_A416371_Rep2", "section_id": "S3_Col_R", "patient_id": "A416371", "replicate": "Rep2"},
    {"sample_id": "SN123_A551763_Rep1", "section_id": "S1_Cec", "patient_id": "A551763", "replicate": "Rep1"},
    {"sample_id": "SN123_A595688_Rep1", "section_id": "S2_Col_R", "patient_id": "A595688", "replicate": "Rep1"},
    {"sample_id": "SN123_A798015_Rep1", "section_id": "S7_Rec/Sig", "patient_id": "A798015", "replicate": "Rep1"},
    {"sample_id": "SN123_A938797_Rep1_X", "section_id": "S6_Rec", "patient_id": "A938797", "replicate": "Rep1_X"},
    {"sample_id": "SN124_A551763_Rep2", "section_id": "S1_Cec", "patient_id": "A551763", "replicate": "Rep2"},
    {"sample_id": "SN124_A595688_Rep2", "section_id": "S2_Col_R", "patient_id": "A595688", "replicate": "Rep2"},
    {"sample_id": "SN124_A798015_Rep2", "section_id": "S7_Rec/Sig", "patient_id": "A798015", "replicate": "Rep2"},
    {"sample_id": "SN124_A938797_Rep2", "section_id": "S6_Rec", "patient_id": "A938797", "replicate": "Rep2"},
    {"sample_id": "SN84_A120838_Rep1", "section_id": "S4_Col_Sig", "patient_id": "A120838", "replicate": "Rep1"},
    {"sample_id": "SN84_A120838_Rep2", "section_id": "S4_Col_Sig", "patient_id": "A120838", "replicate": "Rep2"},
]
if DATASET_MODE == "GSE294385":
    DATASET_ID = "GSE294385_CRC_METASTASIS_PATHOLOGY"
    SAMPLES = [
        {"sample_id": "M-ST-13", "section_id": "Primary Colon M-ST-13", "patient_id": "Patient1", "replicate": "primary"},
        {"sample_id": "M-ST-15", "section_id": "Primary Colon M-ST-15", "patient_id": "Patient6", "replicate": "primary"},
        {"sample_id": "M-ST-29", "section_id": "Primary Colon M-ST-29", "patient_id": "Patient11", "replicate": "primary"},
        {"sample_id": "M-ST-30", "section_id": "Primary Colon M-ST-30", "patient_id": "Patient11", "replicate": "primary"},
        {"sample_id": "M-ST-31", "section_id": "Primary Colon M-ST-31", "patient_id": "Patient9", "replicate": "primary"},
        {"sample_id": "M-ST-32", "section_id": "Primary Colon M-ST-32", "patient_id": "Patient9", "replicate": "primary"},
        {"sample_id": "M-ST-33", "section_id": "Primary Colon M-ST-33", "patient_id": "Patient10", "replicate": "primary"},
        {"sample_id": "M-ST-34", "section_id": "Primary Colon M-ST-34", "patient_id": "Patient10", "replicate": "primary"},
    ]
SAMPLE_META = {row["sample_id"]: row for row in SAMPLES}
GSE294385_PACKAGE_URLS = {
    "M-ST-13": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8903nnn/GSM8903668/suppl/GSM8903668_M-ST-13.tar.gz",
    "M-ST-15": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8903nnn/GSM8903670/suppl/GSM8903670_M-ST-15.tar.gz",
    "M-ST-29": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8903nnn/GSM8903684/suppl/GSM8903684_M-ST-29.tar.gz",
    "M-ST-30": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8903nnn/GSM8903685/suppl/GSM8903685_M-ST-30.tar.gz",
    "M-ST-31": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8903nnn/GSM8903686/suppl/GSM8903686_M-ST-31.tar.gz",
    "M-ST-32": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8903nnn/GSM8903687/suppl/GSM8903687_M-ST-32.tar.gz",
    "M-ST-33": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8903nnn/GSM8903688/suppl/GSM8903688_M-ST-33.tar.gz",
    "M-ST-34": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8903nnn/GSM8903689/suppl/GSM8903689_M-ST-34.tar.gz",
}
GSE294385_PACKAGE_BYTES = {
    "M-ST-13": 162904031,
    "M-ST-15": 171494310,
    "M-ST-29": 312268838,
    "M-ST-30": 189212232,
    "M-ST-31": 130266896,
    "M-ST-32": 156279360,
    "M-ST-33": 278967080,
    "M-ST-34": 192681799,
}

METHOD_LABELS = {
    "M0_expr_kmeans": "expression-only k-means",
    "M1_spatial_concat_kmeans": "coordinate-augmented k-means",
    "M2_spatial_ward": "spatial Ward",
    "M3_spatial_leiden": "spatial Leiden",
    "M4_spagcn_official": "official SpaGCN v1.2.7",
    "M5_stagate_official": "official STAGATE_pyG",
    "M6_bayesspace_official": f"official BayesSpace (nrep={BAYESPACE_NREP})",
}
BASE_METHOD_IDS = (
    "M0_expr_kmeans",
    "M1_spatial_concat_kmeans",
    "M2_spatial_ward",
    "M3_spatial_leiden",
    "M4_spagcn_official",
    "M5_stagate_official",
)


def log(rows: list[dict[str, object]], **values: object) -> None:
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    row.update(values)
    rows.append(row)
    print(json.dumps(row, sort_keys=True), flush=True)


def run(command: list[str], *, timeout: int | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("[cmd]", " ".join(command), flush=True)
    return subprocess.run(command, text=True, timeout=timeout, check=check)


def download(url: str, path: Path, *, attempts: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        path.unlink(missing_ok=True)
        request = Request(url, headers={"User-Agent": "JBCB-interface-validation/1.0"})
        try:
            with urlopen(request, timeout=300) as response, path.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = 5 * (2 ** (attempt - 1))
            print(f"[download-retry] attempt={attempt} delay={delay}s url={url}", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"download failed after {attempts} attempts: {url}: {last_error}")


def download_package(sample_id: str, path: Path) -> str:
    """Try equivalent Zenodo file URLs before failing the registered sample."""

    if DATASET_MODE == "GSE294385":
        url = GSE294385_PACKAGE_URLS[sample_id]
        download(url, path)
        expected_size = GSE294385_PACKAGE_BYTES[sample_id]
        observed_size = path.stat().st_size
        if observed_size != expected_size:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"GEO package size mismatch for {sample_id}: expected {expected_size}, observed {observed_size}")
        return url

    urls = [
        *(template.format(sample_id=sample_id) for template in ZENODO_PACKAGE_MIRROR_TEMPLATES),
        ZENODO_PACKAGE_TEMPLATE.format(sample_id=sample_id),
        ZENODO_PACKAGE_DIRECT_TEMPLATE.format(sample_id=sample_id),
        ZENODO_PACKAGE_PLAIN_TEMPLATE.format(sample_id=sample_id),
        ZENODO_PACKAGE_RECORD_TEMPLATE.format(sample_id=sample_id),
    ]
    errors: list[str] = []
    for url in urls:
        try:
            download(url, path)
            expected_md5 = EXPECTED_PACKAGE_MD5[sample_id]
            observed_md5 = md5_file(path)
            if observed_md5 != expected_md5:
                path.unlink(missing_ok=True)
                raise RuntimeError(f"MD5 mismatch: expected {expected_md5}, observed {observed_md5}")
            return url
        except RuntimeError as exc:
            errors.append(str(exc))
            print(f"[download-fallback] url failed: {url}", flush=True)
    raise RuntimeError(f"all Zenodo URLs failed for {sample_id}: {' | '.join(errors)}")


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_bytes(data: bytes) -> bytes:
    return gzip.compress(data, compresslevel=6)


def write_flat_member(target_root: Path, sample_id: str, member_name: str, data: bytes) -> Path | None:
    name = Path(member_name).name
    supported = {
        "filtered_feature_bc_matrix.h5",
        "matrix.mtx",
        "barcodes.tsv",
        "features.tsv",
        "genes.tsv",
        "tissue_positions.csv",
        "tissue_positions_list.csv",
        "scalefactors_json.json",
        "scalefactors.json",
        "tissue_lowres_image.png",
        "tissue_hires_image.png",
        "detected_tissue_image.jpg",
    }
    if name.endswith(".gz"):
        plain_name = name[:-3]
        if plain_name not in supported:
            return None
        target = target_root / f"{sample_id}_{name}"
        target.write_bytes(data)
        return target
    if name not in supported:
        return None
    if name == "filtered_feature_bc_matrix.h5":
        target = target_root / f"{sample_id}_{name}"
        target.write_bytes(data)
        return target
    target = target_root / f"{sample_id}_{name}.gz"
    target.write_bytes(gzip_bytes(data))
    return target


def unpack_sample(package: Path, sample_id: str, target_root: Path) -> list[str]:
    target_root.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    if tarfile.is_tarfile(package):
        with tarfile.open(package, "r:*") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                source = archive.extractfile(member)
                if source is None:
                    continue
                path = write_flat_member(target_root, sample_id, member.name, source.read())
                if path is not None:
                    written.append(path.name)
    else:
        with zipfile.ZipFile(package) as archive:
            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                path = write_flat_member(target_root, sample_id, member, archive.read(member))
                if path is not None:
                    written.append(path.name)
    written_set = set(written)
    matrix_options = [
        {f"{sample_id}_filtered_feature_bc_matrix.h5"},
        {
            f"{sample_id}_matrix.mtx.gz",
            f"{sample_id}_barcodes.tsv.gz",
            f"{sample_id}_features.tsv.gz",
        },
    ]
    missing = [] if any(option.issubset(written_set) for option in matrix_options) else [
        f"{sample_id}_filtered_feature_bc_matrix.h5 or Matrix Market trio"
    ]
    if f"{sample_id}_scalefactors_json.json.gz" not in written_set:
        missing.append(f"{sample_id}_scalefactors_json.json.gz")
    if not any(name in written_set for name in (f"{sample_id}_tissue_positions_list.csv.gz", f"{sample_id}_tissue_positions.csv.gz")):
        missing.append(f"{sample_id}_tissue_positions.csv(.gz)")
    if missing:
        raise RuntimeError(f"{sample_id} missing required package members: {missing}")
    return written


def annotation_paths(annotation_zip: Path, target_root: Path) -> dict[str, Path]:
    target_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    with zipfile.ZipFile(annotation_zip) as archive:
        for sample in SAMPLES:
            sample_id = str(sample["sample_id"])
            expected = f"Pathologist_Annotations_{sample_id}.csv"
            matches = [name for name in archive.namelist() if Path(name).name == expected]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one annotation CSV for {sample_id}, found {matches}")
            path = target_root / expected
            path.write_bytes(archive.read(matches[0]))
            result[sample_id] = path
    return result


def annotation_paths_from_tar(annotation_tar: Path, target_root: Path) -> dict[str, Path]:
    target_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    with tarfile.open(annotation_tar, "r:*") as archive:
        for sample in SAMPLES:
            sample_id = str(sample["sample_id"])
            expected = f"Pathologist_Annotations_{sample_id}.csv"
            matches = [member for member in archive.getmembers() if Path(member.name).name == expected and member.isfile()]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one annotation CSV for {sample_id}, found {len(matches)}")
            source = archive.extractfile(matches[0])
            if source is None:
                raise RuntimeError(f"Cannot read annotation CSV for {sample_id}")
            path = target_root / expected
            path.write_bytes(source.read())
            result[sample_id] = path
    return result


def has_position_file(data_root: Path, sample_id: str) -> bool:
    return any(
        (data_root / f"{sample_id}_{suffix}").exists()
        for suffix in ("tissue_positions_list.csv.gz", "tissue_positions.csv.gz")
    )


def prepare_input(rows: list[dict[str, object]]) -> tuple[dict[str, Path], dict[str, Path]]:
    data_root = STAGE / "data" / DATASET_ID / "extracted"
    annotation_root = STAGE / "annotations"
    reusable = (STAGE / "sample_manifest.tsv").exists() and all(
        (data_root / f"{sample['sample_id']}_filtered_feature_bc_matrix.h5").exists()
        and has_position_file(data_root, str(sample["sample_id"]))
        and (data_root / f"{sample['sample_id']}_scalefactors_json.json.gz").exists()
        and (annotation_root / f"Pathologist_Annotations_{sample['sample_id']}.csv").exists()
        for sample in SAMPLES
    )
    if reusable:
        log(rows, step="reuse_prepared_input", status="success", n_samples=len(SAMPLES))
        annotations = {
            str(sample["sample_id"]): annotation_root / f"Pathologist_Annotations_{sample['sample_id']}.csv"
            for sample in SAMPLES
        }
        data_roots = {str(sample["sample_id"]): data_root for sample in SAMPLES}
        return annotations, data_roots
    if DATASET_MODE == "GSE294385":
        annotation_archive = ROOT / "GSE294385_annotations.tar.gz"
        uploaded_annotation_archive = Path("/content/GSE294385_annotations.tar.gz")
        if not uploaded_annotation_archive.exists():
            raise RuntimeError("GSE294385 annotation archive must be uploaded before execution")
        shutil.copy2(uploaded_annotation_archive, annotation_archive)
        annotations = annotation_paths_from_tar(annotation_archive, annotation_root)
        annotation_source = "uploaded_gse294385_annotation_archive"
    else:
        annotation_archive = ROOT / "Pathology_SpotAnnotations.zip"
        uploaded_annotation_archive = Path("/content/Pathology_SpotAnnotations.zip")
        if uploaded_annotation_archive.exists():
            shutil.copy2(uploaded_annotation_archive, annotation_archive)
            annotation_source = "uploaded_local_archive"
        else:
            download(ANNOTATION_URL, annotation_archive)
            annotation_source = "zenodo_url"
        annotations = annotation_paths(annotation_archive, annotation_root)
    log(rows, step="prepare_annotations", status="success", source=annotation_source)
    annotation_hash, annotation_size = sha256_file(annotation_archive)
    package_manifest: list[dict[str, object]] = []
    package_dir = ROOT / "packages"
    for sample in SAMPLES:
        sample_id = str(sample["sample_id"])
        package_path = package_dir / f"{sample_id}.tar.gz" if DATASET_MODE == "GSE294385" else package_dir / f"{sample_id}.zip"
        source_url = download_package(sample_id, package_path)
        package_hash, package_size = sha256_file(package_path)
        written = unpack_sample(package_path, sample_id, data_root)
        package_path.unlink()
        package_manifest.append(
            {
                **sample,
                "dataset_id": DATASET_ID,
                "source_url": source_url,
                "sha256": package_hash,
                "size_bytes": package_size,
                "files_written": ";".join(sorted(written)),
            }
        )
        log(rows, step="prepare_sample", sample_id=sample_id, files=written, sha256=package_hash)
    pd.DataFrame(package_manifest).to_csv(OUT / "package_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(
        [{"source": annotation_archive.name, "source_url": ANNOTATION_URL if DATASET_MODE != "GSE294385" else "https://raw.githubusercontent.com/yliuup/CRC_micromets_ST/main/Meta_data/visium_meta_after_qc.rds", "sha256": annotation_hash, "size_bytes": annotation_size}]
    ).to_csv(OUT / "annotation_manifest.tsv", sep="\t", index=False)
    manifest = pd.DataFrame(
        [
            {
                "dataset_id": DATASET_ID,
                "sample_id": sample["sample_id"],
                "series_id": DATASET_ID,
                "bundle_root": f"data/{DATASET_ID}/extracted",
                "rationale": "morphology_anchored_external_crc_validation",
            }
            for sample in SAMPLES
        ]
    )
    manifest.to_csv(STAGE / "sample_manifest.tsv", sep="\t", index=False)
    return annotations, {sample["sample_id"]: data_root for sample in SAMPLES}


def build_input_tar() -> Path:
    input_tar = ROOT / "interface_validation_input.tar.gz"
    with tarfile.open(input_tar, "w:gz") as archive:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(STAGE))
    return input_tar


def install_stack(rows: list[dict[str, object]]) -> None:
    packages = [
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas==2.2.3",
        "numcodecs==0.15.1",
        "zarr==2.18.7",
        "anndata==0.11.4",
        "scanpy==1.10.4",
        "opencv-python-headless",
        "Pillow",
        "python-igraph",
        "leidenalg",
        "torch-geometric",
    ]
    run([sys.executable, "-m", "pip", "install", "-q", *packages], timeout=3600)
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "SpaGCN==1.2.7"], timeout=900)
    import torch

    cuda = torch.version.cuda
    tag = "cpu" if cuda is None else "cu" + str(cuda).replace(".", "")
    wheel_url = f"https://data.pyg.org/whl/torch-{torch.__version__.split('+')[0]}+{tag}.html"
    run(
        [sys.executable, "-m", "pip", "install", "-q", "torch-scatter", "torch-sparse", "-f", wheel_url],
        timeout=1800,
    )
    log(rows, step="install_stack", status="success", torch=str(torch.__version__), cuda=cuda, pyg_wheel=wheel_url)


def read_positions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, compression="infer")
    if str(frame.iloc[0, 0]).lower() == "barcode":
        frame = pd.read_csv(path, compression="infer")
    else:
        frame.columns = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ]
    frame["barcode"] = frame["barcode"].astype(str)
    return frame


def read_scalefactors(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def read_annotation(path: Path) -> pd.DataFrame:
    ann = pd.read_csv(path, compression="infer")
    ann.columns = [str(column).strip() for column in ann.columns]
    barcode_col = next(column for column in ann.columns if column.lower() == "barcode")
    label_cols = [column for column in ann.columns if column != barcode_col]
    if len(label_cols) != 1:
        raise RuntimeError(f"Expected one pathology label column in {path}, found {label_cols}")
    ann = ann.rename(columns={barcode_col: "barcode", label_cols[0]: "raw_label"})
    ann["barcode"] = ann["barcode"].astype(str)
    ann["raw_label"] = ann["raw_label"].fillna("").astype(str).str.strip()
    return ann[["barcode", "raw_label"]]


def map_label(raw_label: str) -> str:
    normalized = " ".join(str(raw_label).replace("\u00a0", " ").lower().split())
    if not normalized:
        return "unresolved"
    if "exclude" in normalized:
        return "exclude"
    if "tumor stroma" in normalized and "tumor cells" not in normalized:
        return "stroma"
    if "tumor cells" in normalized and "tumor stroma" not in normalized:
        return "tumor"
    has_tumor = "tumor" in normalized
    has_stroma = "stroma" in normalized or "fibroblastic" in normalized or "desmoplastic" in normalized
    if has_tumor and has_stroma:
        return "mixed"
    if has_tumor:
        return "tumor"
    if has_stroma:
        if " or " in normalized or ("muscularis" in normalized and "stroma" in normalized):
            return "unresolved"
        return "stroma"
    if "non neo epithelium" in normalized or "non-neoplastic" in normalized or "non neoplastic" in normalized:
        return "normal epithelium"
    if any(token in normalized for token in ("epithelium", "submucosa", "lamina propria", "glandular", "muscularis", "connective tissue")):
        return "other"
    return "unresolved"


def load_expression(root: Path, sample_id: str, annotation_path: Path) -> dict[str, object]:
    import anndata as ad
    import scanpy as sc
    from scipy import sparse
    from scipy.io import mmread
    from sklearn.decomposition import PCA

    h5 = root / f"{sample_id}_filtered_feature_bc_matrix.h5"
    matrix_path = root / f"{sample_id}_matrix.mtx.gz"
    barcodes_path = root / f"{sample_id}_barcodes.tsv.gz"
    features_path = root / f"{sample_id}_features.tsv.gz"
    positions_path = root / f"{sample_id}_tissue_positions_list.csv.gz"
    if not positions_path.exists():
        positions_path = root / f"{sample_id}_tissue_positions.csv.gz"
    scale_path = root / f"{sample_id}_scalefactors_json.json.gz"
    if not scale_path.exists():
        scale_path = root / f"{sample_id}_scalefactors.json.gz"
    if h5.exists():
        adata = sc.read_10x_h5(str(h5), gex_only=True)
    elif matrix_path.exists() and barcodes_path.exists() and features_path.exists():
        counts = mmread(matrix_path).tocsr().transpose().tocsr()
        with gzip.open(barcodes_path, "rt", encoding="utf-8") as handle:
            barcodes = [line.strip() for line in handle if line.strip()]
        features = pd.read_csv(features_path, sep="\t", header=None, compression="infer")
        if counts.shape[0] != len(barcodes) or counts.shape[1] != len(features):
            raise RuntimeError(
                f"Matrix Market dimensions do not match metadata for {sample_id}: "
                f"counts={counts.shape}, barcodes={len(barcodes)}, features={len(features)}"
            )
        adata = ad.AnnData(X=counts)
        adata.obs_names = pd.Index(barcodes, dtype="string")
        gene_names = features.iloc[:, 1 if features.shape[1] >= 2 else 0].astype(str).tolist()
        adata.var_names = pd.Index(gene_names, dtype="string")
        if features.shape[1] >= 1:
            adata.var["gene_ids"] = features.iloc[:, 0].astype(str).to_numpy()
        if features.shape[1] >= 3:
            adata.var["feature_types"] = features.iloc[:, 2].astype(str).to_numpy()
    else:
        raise FileNotFoundError(f"Missing 10x H5 or Matrix Market trio for {sample_id} under {root}")
    adata.var_names_make_unique()
    matrix_barcodes = pd.Index(adata.obs_names.astype(str), dtype="string")
    positions = read_positions(positions_path)
    if positions["barcode"].duplicated().any():
        raise RuntimeError(f"Duplicate tissue-position barcodes for {sample_id}")
    position_barcodes = pd.Index(positions["barcode"].astype(str), dtype="string")
    missing_position_barcodes = sorted(set(matrix_barcodes) - set(position_barcodes))
    if missing_position_barcodes:
        raise RuntimeError(
            f"Tissue-position file is missing {len(missing_position_barcodes)} matrix barcodes for {sample_id}"
        )
    positions = positions[positions["barcode"].isin(adata.obs_names)].copy()
    positions["barcode"] = pd.Categorical(positions["barcode"], categories=adata.obs_names, ordered=True)
    positions = positions.sort_values("barcode").reset_index(drop=True)
    in_tissue = positions["in_tissue"].astype(int).to_numpy() == 1
    positions = positions.loc[in_tissue].reset_index(drop=True)
    adata = adata[positions["barcode"].astype(str).tolist()].copy()
    if adata.n_obs != len(positions):
        raise RuntimeError(f"Expression/position mismatch for {sample_id}")
    annotation = read_annotation(annotation_path)
    if annotation["barcode"].duplicated().any():
        raise RuntimeError(f"Duplicate pathology annotation barcodes for {sample_id}")
    annotation_barcodes = pd.Index(annotation["barcode"].astype(str), dtype="string")
    annotation_outside_matrix = sorted(set(annotation_barcodes) - set(matrix_barcodes))
    if annotation_outside_matrix:
        raise RuntimeError(
            f"Pathology annotation contains {len(annotation_outside_matrix)} barcodes absent from the matrix for {sample_id}"
        )
    in_tissue_barcodes = pd.Index(adata.obs_names.astype(str), dtype="string")
    annotation_in_tissue = annotation_barcodes.intersection(in_tissue_barcodes)
    registration = {
        "matrix_barcode_count": int(len(matrix_barcodes)),
        "position_barcode_count": int(len(position_barcodes)),
        "annotation_row_count": int(len(annotation)),
        "annotation_matrix_intersection": int(len(annotation_barcodes.intersection(matrix_barcodes))),
        "annotation_in_tissue_count": int(len(annotation_in_tissue)),
        "annotation_outside_matrix_count": int(len(annotation_outside_matrix)),
        "matrix_in_tissue_count": int(len(in_tissue_barcodes)),
        "in_tissue_unlabelled_count": int(len(set(in_tissue_barcodes) - set(annotation_barcodes))),
    }
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    gene_values: dict[str, np.ndarray] = {}
    for gene in sorted({gene for definition in SCORE_DEFINITIONS.values() for gene in definition["genes"]}):
        if gene not in adata.var_names:
            continue
        values = adata[:, gene].X
        gene_values[gene] = values.toarray().ravel() if sparse.issparse(values) else np.asarray(values).ravel()
    if not set(PRIMARY_GENES).issubset(gene_values):
        missing = sorted(set(PRIMARY_GENES) - set(gene_values))
        raise RuntimeError(f"{sample_id} missing primary genes after normalization: {missing}")
    primary_score = np.mean(np.vstack([gene_values[gene] for gene in PRIMARY_GENES]), axis=0).astype(float)
    secondary_scores: dict[str, np.ndarray | None] = {}
    score_status: dict[str, dict[str, object]] = {"primary_barrier": {"status": "success", "missing_genes": ""}}
    for score_id, definition in SCORE_DEFINITIONS.items():
        if score_id == "primary_barrier":
            continue
        genes = [str(gene) for gene in definition["genes"]]
        if score_id == "immune_exclusion":
            component_ids = ["CAF_FAP", "SPP1_myeloid", "T_cell"]
            component_genes = [str(gene) for component_id in component_ids for gene in SCORE_DEFINITIONS[component_id]["genes"]]
            missing = sorted(set(component_genes) - set(gene_values))
            if missing:
                secondary_scores[score_id] = None
                score_status[score_id] = {"status": "not_evaluable", "missing_genes": ";".join(missing)}
            else:
                caf = np.mean(np.vstack([gene_values[gene] for gene in SCORE_DEFINITIONS["CAF_FAP"]["genes"]]), axis=0)
                myeloid = np.mean(np.vstack([gene_values[gene] for gene in SCORE_DEFINITIONS["SPP1_myeloid"]["genes"]]), axis=0)
                t_cell = np.mean(np.vstack([gene_values[gene] for gene in SCORE_DEFINITIONS["T_cell"]["genes"]]), axis=0)
                secondary_scores[score_id] = (caf + myeloid) - t_cell
                score_status[score_id] = {"status": "success", "missing_genes": ""}
            continue
        missing = sorted(set(genes) - set(gene_values))
        if missing:
            secondary_scores[score_id] = None
            score_status[score_id] = {"status": "not_evaluable", "missing_genes": ";".join(missing)}
        else:
            secondary_scores[score_id] = np.mean(np.vstack([gene_values[gene] for gene in genes]), axis=0).astype(float)
            score_status[score_id] = {"status": "success", "missing_genes": ""}
    x = adata.X
    means = np.asarray(x.mean(axis=0)).ravel() if sparse.issparse(x) else np.asarray(x).mean(axis=0)
    if sparse.issparse(x):
        sq_means = np.asarray(x.power(2).mean(axis=0)).ravel()
    else:
        sq_means = np.asarray(x, dtype=float).mean(axis=0) ** 2
    variances = np.maximum(sq_means - means**2, 0.0)
    n_features = min(2000, max(2, x.shape[1] - 1))
    top_idx = np.argsort(variances)[::-1][:n_features]
    dense = x[:, top_idx].toarray().astype(np.float32) if sparse.issparse(x) else np.asarray(x[:, top_idx], dtype=np.float32)
    n_components = max(2, min(20, dense.shape[0] - 1, dense.shape[1] - 1))
    pcs = PCA(n_components=n_components, random_state=0).fit_transform(dense).astype(np.float32)
    coords = positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)
    ann = annotation.set_index("barcode")
    raw_labels = np.asarray([ann["raw_label"].get(barcode, "") for barcode in adata.obs_names], dtype=object)
    coarse_labels = np.asarray([map_label(value) for value in raw_labels], dtype=object)
    scale = read_scalefactors(scale_path)
    spot_diameter = float(scale.get("spot_diameter_fullres", float("nan")))
    if not np.isfinite(spot_diameter) or spot_diameter <= 0:
        raise RuntimeError(f"{sample_id} has no finite spot_diameter_fullres scalefactor")
    return {
        "sample_id": sample_id,
        "patient_id": SAMPLE_META[sample_id]["patient_id"],
        "section_id": SAMPLE_META[sample_id]["section_id"],
        "barcodes": np.asarray(adata.obs_names.astype(str)),
        "coords": coords,
        "array_coords": positions[["array_row", "array_col"]].to_numpy(dtype=float),
        "primary_score": primary_score,
        "score_values": {"primary_barrier": primary_score, **secondary_scores},
        "score_status": score_status,
        "pcs": pcs,
        "raw_labels": raw_labels,
        "coarse_labels": coarse_labels,
        "microns_per_pixel": NOMINAL_SPOT_DIAMETER_UM / spot_diameter,
        "spot_diameter_fullres": spot_diameter,
        "registration": registration,
    }


def nearest_indices(coords: np.ndarray, neighbors: int = 6) -> np.ndarray:
    """Return deterministic nearest non-self neighbors.

    Visium lattice coordinates contain many equal-distance neighbors. The
    ordering returned by a kNN backend for those ties can vary by library
    version and execution backend, which changes connected components and
    boundary flags. Distances are therefore ranked explicitly, with the
    fixed input row order as the tie-breaker. The input row order is locked
    by the barcode-aligned section contract.
    """
    coords = np.asarray(coords, dtype=float)
    n_spots = coords.shape[0]
    if n_spots <= 1:
        return np.empty((n_spots, 0), dtype=int)
    n_neighbors = min(n_spots - 1, int(neighbors))
    if n_neighbors <= 0:
        return np.empty((n_spots, 0), dtype=int)
    indices = np.arange(n_spots, dtype=int)
    result = np.empty((n_spots, n_neighbors), dtype=int)
    for row in range(n_spots):
        distance_squared = np.sum((coords - coords[row]) ** 2, axis=1)
        distance_squared[row] = np.inf
        # lexsort uses the last key as primary: distance first, then index.
        result[row] = np.lexsort((indices, distance_squared))[:n_neighbors]
    return result


def block_ids(coords: np.ndarray, grid: int = 6) -> np.ndarray:
    x_edges = np.linspace(float(coords[:, 0].min()), float(coords[:, 0].max()), grid + 1)
    y_edges = np.linspace(float(coords[:, 1].min()), float(coords[:, 1].max()), grid + 1)
    x_bins = np.clip(np.digitize(coords[:, 0], x_edges[1:-1], right=False), 0, grid - 1)
    y_bins = np.clip(np.digitize(coords[:, 1], y_edges[1:-1], right=False), 0, grid - 1)
    return np.asarray([f"x{int(x)}_y{int(y)}" for x, y in zip(x_bins, y_bins, strict=True)], dtype=object)


def connected_stroma_components(stroma_mask: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    components = np.full(stroma_mask.shape[0], -1, dtype=int)
    adjacency: list[list[int]] = [[] for _ in range(len(stroma_mask))]
    for source, row in enumerate(neighbors):
        if not stroma_mask[source]:
            continue
        for target in row:
            target = int(target)
            if stroma_mask[target]:
                adjacency[source].append(target)
                adjacency[target].append(source)
    component_id = 0
    for start in np.flatnonzero(stroma_mask):
        if components[start] >= 0:
            continue
        queue = [int(start)]
        components[start] = component_id
        while queue:
            current = queue.pop()
            for neighbor in adjacency[current]:
                if components[neighbor] < 0:
                    components[neighbor] = component_id
                    queue.append(neighbor)
        component_id += 1
    return components


def anchor_sets(sample: dict[str, object]) -> dict[str, object]:
    from scipy.spatial import cKDTree

    coords = np.asarray(sample["coords"], dtype=float)
    coarse = np.asarray(sample["coarse_labels"], dtype=object)
    tumor_mask = coarse == "tumor"
    stroma_mask = coarse == "stroma"
    if not np.any(tumor_mask) or not np.any(stroma_mask):
        return {"status": "not_evaluable", "reason": "missing_pure_tumor_or_stroma_label", "near": np.zeros(len(coords), dtype=bool), "far": np.zeros(len(coords), dtype=bool), "distance_um": np.full(len(coords), np.nan), "components": np.full(len(coords), -1, dtype=int)}
    distance_px = cKDTree(coords[tumor_mask]).query(coords, k=1)[0]
    distance_um = distance_px * float(sample["microns_per_pixel"])
    neighbors = nearest_indices(np.asarray(sample["array_coords"], dtype=float), 6)
    components = connected_stroma_components(stroma_mask, neighbors)
    near = stroma_mask & (distance_um <= 100.0)
    far = stroma_mask & (distance_um >= 200.0) & (distance_um <= 400.0)
    near_components = set(components[near].tolist()) - {-1}
    far &= np.asarray([component in near_components for component in components], dtype=bool)
    return {
        "status": "success",
        "reason": "",
        "near": near,
        "far": far,
        "distance_um": distance_um,
        "components": components,
        "neighbors": neighbors,
        "tumor_mask": tumor_mask,
        "stroma_mask": stroma_mask,
    }


def nonempty_blocks(group: np.ndarray, blocks: np.ndarray) -> list[str]:
    return sorted({str(block) for block in np.asarray(blocks)[np.asarray(group, dtype=bool)]})


def block_resample_medians(
    values: np.ndarray,
    group: np.ndarray,
    blocks: np.ndarray,
    sampled_blocks: np.ndarray,
    available_blocks: list[str],
) -> np.ndarray:
    """Pool the sampled spatial blocks and return one median per replicate."""
    block_values = [
        np.asarray(values[(blocks == block) & group], dtype=float)
        for block in available_blocks
    ]
    max_size = max(len(block) for block in block_values)
    padded = np.full((len(block_values), max_size), np.nan, dtype=float)
    for index, block in enumerate(block_values):
        padded[index, : len(block)] = block
    block_index = {block: index for index, block in enumerate(available_blocks)}
    sampled_indices = np.vectorize(block_index.__getitem__, otypes=[int])(sampled_blocks)
    return np.nanmedian(padded[sampled_indices], axis=(1, 2))


def bootstrap_delta(values: np.ndarray, group_a: np.ndarray, group_b: np.ndarray, blocks: np.ndarray, key: str) -> tuple[float, float, float, int]:
    blocks_a = nonempty_blocks(group_a, blocks)
    blocks_b = nonempty_blocks(group_b, blocks)
    if len(blocks_a) < MIN_BLOCKS or len(blocks_b) < MIN_BLOCKS:
        return float("nan"), float("nan"), float("nan"), min(len(blocks_a), len(blocks_b))
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    sampled_a = rng.choice(blocks_a, size=(BOOTSTRAP_REPLICATES, len(blocks_a)), replace=True)
    sampled_b = rng.choice(blocks_b, size=(BOOTSTRAP_REPLICATES, len(blocks_b)), replace=True)
    estimates = block_resample_medians(values, group_a, blocks, sampled_a, blocks_a) - block_resample_medians(values, group_b, blocks, sampled_b, blocks_b)
    return float(np.median(values[group_a]) - np.median(values[group_b])), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975)), min(len(blocks_a), len(blocks_b))


def contrast_resample_key(score_id: str, sample_id: str, near: np.ndarray, far: np.ndarray) -> str:
    """Give the same comparison spots the same block-resampling draw."""
    digest = hashlib.sha256()
    digest.update(np.asarray(near, dtype=np.uint8).tobytes())
    digest.update(b"|")
    digest.update(np.asarray(far, dtype=np.uint8).tobytes())
    return f"{score_id}|{sample_id}|{digest.hexdigest()}"


def bootstrap_change(values: np.ndarray, alt_a: np.ndarray, alt_b: np.ndarray, ref_a: np.ndarray, ref_b: np.ndarray, blocks: np.ndarray, key: str, diagnostics: dict[str, object] | None = None) -> tuple[float, float, float, int]:
    alt_a_blocks = nonempty_blocks(alt_a, blocks)
    alt_b_blocks = nonempty_blocks(alt_b, blocks)
    ref_a_blocks = nonempty_blocks(ref_a, blocks)
    ref_b_blocks = nonempty_blocks(ref_b, blocks)
    support = min(len(alt_a_blocks), len(alt_b_blocks), len(ref_a_blocks), len(ref_b_blocks))
    if diagnostics is not None:
        diagnostics.update(n_attempted=0, n_accepted=0, n_used=0, acceptance_rate=float("nan"), reason="")
    if support < MIN_BLOCKS:
        if diagnostics is not None:
            diagnostics["reason"] = "paired_common_block_support_below_locked_threshold"
        return float("nan"), float("nan"), float("nan"), support
    if np.array_equal(alt_a, ref_a) and np.array_equal(alt_b, ref_b):
        if diagnostics is not None:
            diagnostics.update(n_used=BOOTSTRAP_REPLICATES, reason="identical_memberships_exact_zero")
        return 0.0, 0.0, 0.0, support
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    near_pool = sorted(set(alt_a_blocks) | set(ref_a_blocks))
    far_pool = sorted(set(alt_b_blocks) | set(ref_b_blocks))
    estimates_parts: list[np.ndarray] = []
    retained = 0
    attempted = 0
    for _ in range(100):
        if retained >= BOOTSTRAP_REPLICATES:
            break
        # A spatial block receives one draw in both partitions. Near and far
        # bands remain separately stratified; empty-group draws are rejected.
        sampled_near = rng.choice(near_pool, size=(128, len(near_pool)), replace=True)
        sampled_far = rng.choice(far_pool, size=(128, len(far_pool)), replace=True)
        attempted += 128
        valid = (
            np.isin(sampled_near, alt_a_blocks).any(axis=1)
            & np.isin(sampled_near, ref_a_blocks).any(axis=1)
            & np.isin(sampled_far, alt_b_blocks).any(axis=1)
            & np.isin(sampled_far, ref_b_blocks).any(axis=1)
        )
        if not np.any(valid):
            continue
        near_draws = sampled_near[valid]
        far_draws = sampled_far[valid]
        alt_a_medians = block_resample_medians(values, alt_a, blocks, near_draws, near_pool)
        alt_b_medians = block_resample_medians(values, alt_b, blocks, far_draws, far_pool)
        ref_a_medians = block_resample_medians(values, ref_a, blocks, near_draws, near_pool)
        ref_b_medians = block_resample_medians(values, ref_b, blocks, far_draws, far_pool)
        estimates_parts.append((alt_a_medians - alt_b_medians) - (ref_a_medians - ref_b_medians))
        retained += len(estimates_parts[-1])
    if diagnostics is not None:
        diagnostics.update(
            n_attempted=attempted,
            n_accepted=retained,
            n_used=min(retained, BOOTSTRAP_REPLICATES),
            acceptance_rate=(retained / attempted if attempted else float("nan")),
        )
    if retained < BOOTSTRAP_REPLICATES:
        if diagnostics is not None:
            diagnostics["reason"] = "insufficient_accepted_paired_draws"
        return float("nan"), float("nan"), float("nan"), support
    estimates = np.concatenate(estimates_parts)[:BOOTSTRAP_REPLICATES]
    point = (float(np.median(values[alt_a])) - float(np.median(values[alt_b]))) - (float(np.median(values[ref_a])) - float(np.median(values[ref_b])))
    return point, float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975)), support


def domain_selection(labels: np.ndarray, coarse: np.ndarray) -> tuple[dict[str, object], list[dict[str, object]]]:
    stats: list[dict[str, object]] = []
    for label in sorted(np.unique(labels).tolist()):
        in_domain = labels == int(label)
        n_spots = int(in_domain.sum())
        tumor_count = int((in_domain & (coarse == "tumor")).sum())
        stroma_count = int((in_domain & (coarse == "stroma")).sum())
        stats.append(
            {
                "domain_label": int(label),
                "domain_n": n_spots,
                "tumor_n": tumor_count,
                "stroma_n": stroma_count,
                "tumor_fraction": tumor_count / n_spots if n_spots else float("nan"),
                "stroma_fraction": stroma_count / n_spots if n_spots else float("nan"),
            }
        )
    if not any(row["tumor_n"] > 0 for row in stats) or not any(row["stroma_n"] > 0 for row in stats):
        return {"status": "not_evaluable", "reason": "pathology_anchor_class_absent"}, stats
    tumor = sorted(stats, key=lambda row: (-float(row["tumor_fraction"]), int(row["domain_label"])),)[0]
    stroma_candidates = [row for row in stats if row["domain_label"] != tumor["domain_label"]]
    stroma_candidates = [row for row in stroma_candidates if row["stroma_n"] > 0]
    if not stroma_candidates:
        return {"status": "not_evaluable", "reason": "no_distinct_stroma_domain"}, stats
    stroma = sorted(stroma_candidates, key=lambda row: (-float(row["stroma_fraction"]), int(row["domain_label"])),)[0]
    selected = {
        "status": "success",
        "reason": "",
        "tumor_domain": int(tumor["domain_label"]),
        "stroma_domain": int(stroma["domain_label"]),
        "tumor_domain_n": int(tumor["domain_n"]),
        "stroma_domain_n": int(stroma["domain_n"]),
        "tumor_fraction": float(tumor["tumor_fraction"]),
        "stroma_fraction": float(stroma["stroma_fraction"]),
    }
    return selected, stats


def computational_groups(
    anchor_near: np.ndarray,
    anchor_far: np.ndarray,
    labels: np.ndarray,
    tumor_domain: int,
    stroma_domain: int,
    neighbors: np.ndarray,
    estimator_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-anchor groups retained by one computational partition."""

    stroma_side = labels == int(stroma_domain)
    if estimator_id == "strict_computational_interface":
        computational_near = anchor_near & stroma_side & np.any(labels[neighbors] == int(tumor_domain), axis=1)
    elif estimator_id == "anchor_retention":
        computational_near = anchor_near & stroma_side
    else:
        raise ValueError(f"Unknown estimator_id: {estimator_id}")
    computational_far = anchor_far & stroma_side
    return computational_near, computational_far


def supported_conclusion(delta: float, low: float, high: float, scale: float) -> str:
    if not all(np.isfinite(value) for value in (delta, low, high, scale)) or scale <= 0:
        return "uncertain"
    if abs(delta) < 0.50 * scale:
        return "no_supported_difference"
    if low > 0:
        return "near_enrichment"
    if high < 0:
        return "near_depletion"
    return "no_supported_difference"


def conclusion_transition(anchor_conclusion: str, computational_conclusion: str) -> str:
    supported = {"near_enrichment", "near_depletion"}
    if anchor_conclusion in supported and computational_conclusion in supported:
        if anchor_conclusion == computational_conclusion:
            return "conclusion_unchanged"
        return "conclusion_reversed"
    if anchor_conclusion in supported and computational_conclusion == "no_supported_difference":
        return "conclusion_lost"
    if anchor_conclusion in supported and computational_conclusion == "uncertain":
        return "not_evaluable"
    if anchor_conclusion == "no_supported_difference" and computational_conclusion in supported:
        return "alternative_only_signal"
    return "no_established_conclusion"


def make_map(sample: dict[str, object], labels: np.ndarray, method_id: str, k_value: int, replicate_id: str, replicate_type: str, notes: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": DATASET_ID,
            "sample_id": sample["sample_id"],
            "method_id": method_id,
            "method_label": METHOD_LABELS[method_id],
            "K": int(k_value),
            "replicate_type": replicate_type,
            "replicate_id": replicate_id,
            "config_id": f"K{int(k_value)}_{replicate_id}",
            "barcode": sample["barcodes"],
            "x": sample["coords"][:, 0],
            "y": sample["coords"][:, 1],
            "domain_label": np.asarray(labels, dtype=int) + 1,
            "status": "success",
            "notes": notes,
        }
    )


def weighted_leiden_graph(pcs: np.ndarray, coords: np.ndarray, neighbors: int = 6):
    import igraph as ig
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(len(coords), neighbors + 1))
    nn.fit(coords)
    graph = nn.kneighbors_graph(coords, mode="connectivity")
    graph = graph.maximum(graph.T).tocoo()
    source = graph.row.astype(int)
    target = graph.col.astype(int)
    keep = source < target
    source = source[keep]
    target = target[keep]
    normalized = pcs.astype(float)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized /= norms
    weights = np.maximum(0.0, (1.0 + np.sum(normalized[source] * normalized[target], axis=1)) / 2.0)
    result = ig.Graph(n=len(coords), edges=list(zip(source.tolist(), target.tolist(), strict=True)), directed=False)
    result.es["weight"] = weights.tolist()
    return result


def leiden_membership(graph, resolution: float, seed: int) -> np.ndarray:
    import leidenalg

    try:
        partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=float(resolution), seed=int(seed))
    except TypeError:
        partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=float(resolution))
    return np.asarray(partition.membership, dtype=int)


def exact_leiden(graph, target_k: int, seed: int) -> tuple[np.ndarray, float, bool]:
    candidates = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0]
    evaluations: list[tuple[float, int]] = []
    for resolution in candidates:
        labels = leiden_membership(graph, resolution, seed)
        observed = int(np.unique(labels).size)
        evaluations.append((resolution, observed))
        if observed == target_k:
            return labels, resolution, True
    below = [(r, k) for r, k in evaluations if k < target_k]
    above = [(r, k) for r, k in evaluations if k > target_k]
    if below and above:
        low = max(below, key=lambda item: item[0])[0]
        high = min(above, key=lambda item: item[0])[0]
        best = (low, dict(evaluations)[low])
        for _ in range(18):
            middle = math.sqrt(low * high)
            labels = leiden_membership(graph, middle, seed)
            observed = int(np.unique(labels).size)
            if abs(observed - target_k) < abs(best[1] - target_k):
                best = (middle, observed)
            if observed == target_k:
                return labels, middle, True
            if observed < target_k:
                low = middle
            else:
                high = middle
        return leiden_membership(graph, best[0], seed), float(best[0]), False
    best_resolution, best_k = min(evaluations, key=lambda item: (abs(item[1] - target_k), item[0]))
    return leiden_membership(graph, best_resolution, seed), float(best_resolution), False


def run_baselines(samples: dict[str, dict[str, object]], map_frames: list[pd.DataFrame], rows: list[dict[str, object]]) -> None:
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.neighbors import NearestNeighbors

    for sample_id, sample in samples.items():
        pcs = np.asarray(sample["pcs"], dtype=np.float32)
        coords = np.asarray(sample["coords"], dtype=float)
        coords_scaled = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-6)
        spatial_features = np.concatenate([pcs, 0.5 * coords_scaled], axis=1)
        for method_id, features in (("M0_expr_kmeans", pcs), ("M1_spatial_concat_kmeans", spatial_features)):
            for k_value in K_VALUES:
                for seed in SEEDS:
                    labels = KMeans(n_clusters=k_value, n_init=50, random_state=seed).fit_predict(features)
                    map_frames.append(make_map(sample, labels, method_id, k_value, f"seed_{seed}", "seed", "fixed-K baseline"))
                    log(rows, step="baseline", method=method_id, sample_id=sample_id, K=k_value, seed=seed, status="success")
        for k_value in K_VALUES:
            for neighbor_count in (4, 6, 8):
                nn = NearestNeighbors(n_neighbors=min(len(coords), neighbor_count + 1))
                nn.fit(coords)
                graph = nn.kneighbors_graph(coords, mode="connectivity")
                graph = graph.maximum(graph.T)
                labels = AgglomerativeClustering(n_clusters=k_value, linkage="ward", connectivity=graph).fit_predict(pcs)
                map_frames.append(make_map(sample, labels, "M2_spatial_ward", k_value, f"neighbors_{neighbor_count}", "graph_neighbors", "fixed-K spatial Ward graph specification"))
                log(rows, step="baseline", method="M2_spatial_ward", sample_id=sample_id, K=k_value, neighbors=neighbor_count, status="success")
        graph = weighted_leiden_graph(pcs, coords, neighbors=6)
        for k_value in K_VALUES:
            for seed in SEEDS:
                labels, resolution, exact = exact_leiden(graph, k_value, seed)
                if not exact:
                    log(rows, step="baseline", method="M3_spatial_leiden", sample_id=sample_id, K=k_value, seed=seed, status="not_evaluable", reason="no_exact_K_resolution")
                    continue
                map_frames.append(make_map(sample, labels, "M3_spatial_leiden", k_value, f"seed_{seed}", "seed", f"fixed-K spatial Leiden six-neighbor graph;resolution={resolution:.8g}"))
                log(rows, step="baseline", method="M3_spatial_leiden", sample_id=sample_id, K=k_value, seed=seed, status="success", resolution=resolution)


def normalize_official_maps(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", low_memory=False)
    table = table[table["status"].astype(str).eq("success")].copy()
    table["method_id"] = table["method_id"].replace(
        {
            "Official_SpaGCN_v1_2_7": "M4_spagcn_official",
            "Official_STAGATE_pyG": "M5_stagate_official",
            "Official_BayesSpace_nrep1000": "M6_bayesspace_official",
        }
    )
    table.loc[table["method_id"].astype(str).str.startswith("Official_BayesSpace_nrep"), "method_id"] = "M6_bayesspace_official"
    table["method_label"] = table["method_id"].map(METHOD_LABELS)
    table["replicate_id"] = table["seed"].map(lambda value: f"seed_{int(value)}")
    table["replicate_type"] = "seed"
    table["config_id"] = table.apply(lambda row: f"K{int(row['K'])}_{row['replicate_id']}", axis=1)
    keep = ["dataset_id", "sample_id", "method_id", "method_label", "K", "replicate_type", "replicate_id", "config_id", "barcode", "x", "y", "domain_label", "status", "notes"]
    return table[keep]


def score_scale(values: np.ndarray, anchors: dict[str, object]) -> float:
    values = np.asarray(values, dtype=float)[np.asarray(anchors["stroma_mask"], dtype=bool)]
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def classify(delta: float, low: float, high: float, reference_delta: float, change: float, change_low: float, change_high: float, scale: float) -> str:
    if not np.isfinite(delta) or not np.isfinite(scale) or scale <= 0:
        return "uncertain"
    material = 0.25 * scale
    supported = 0.50 * scale
    if np.isfinite(reference_delta) and np.isfinite(delta) and abs(reference_delta) >= supported and abs(delta) >= supported and reference_delta * delta < 0 and np.isfinite(change) and abs(change) >= material:
        return "opposite-direction point estimate"
    if np.isfinite(reference_delta) and np.isfinite(delta) and abs(delta) < abs(reference_delta) and abs(reference_delta - delta) >= material and abs(reference_delta) >= supported and np.isfinite(change) and abs(change) >= material:
        return "materially attenuated"
    if np.isfinite(change_low) and np.isfinite(change_high) and change_low >= -material and change_high <= material:
        return "negligible"
    if np.isfinite(low) and np.isfinite(high) and (low > 0 or high < 0) and abs(delta) >= supported:
        return "interval-supported near enrichment/depletion"
    return "uncertain"


def compute_effects(
    samples: dict[str, dict[str, object]],
    maps: pd.DataFrame,
    rows: list[dict[str, object]],
    score_id: str = "primary_barrier",
    estimator_id: str = "strict_computational_interface",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    effect_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    membership_rows: list[pd.DataFrame] = []
    anchor_rows: list[dict[str, object]] = []
    score_definition = SCORE_DEFINITIONS[score_id]
    score_label = str(score_definition["label"])
    anchor_cache: dict[str, dict[str, object]] = {}
    estimator_label = ESTIMATOR_LABELS[estimator_id]
    log(rows, step="effects_start", score_id=score_id, estimator_id=estimator_id, n_samples=len(samples))
    for sample_id, sample in samples.items():
        score_values = sample["score_values"].get(score_id)
        score_status = sample["score_status"].get(score_id, {"status": "success", "missing_genes": ""})
        if score_values is None:
            anchor_rows.append(
                {
                    "score_id": score_id,
                    "score_label": score_label,
                    "sample_id": sample_id,
                    "patient_id": sample["patient_id"],
                    "section_id": sample["section_id"],
                    "status": "not_evaluable",
                    "reason": "missing_required_score_genes",
                    "missing_genes": score_status.get("missing_genes", ""),
                    "n_anchor_near": 0,
                    "n_anchor_far": 0,
                }
            )
            continue
        score_values = np.asarray(score_values, dtype=float)
        anchors = anchor_sets(sample)
        anchor_cache[sample_id] = anchors
        if anchors["status"] != "success":
            anchor_rows.append({"score_id": score_id, "score_label": score_label, "sample_id": sample_id, "patient_id": sample["patient_id"], "section_id": sample["section_id"], "status": anchors["status"], "reason": anchors["reason"], "n_anchor_near": 0, "n_anchor_far": 0})
            continue
        blocks = block_ids(np.asarray(sample["array_coords"], dtype=float))
        anchor_near = np.asarray(anchors["near"], dtype=bool)
        anchor_far = np.asarray(anchors["far"], dtype=bool)
        anchor_delta, anchor_low, anchor_high, anchor_nblocks = bootstrap_delta(
            score_values, anchor_near, anchor_far, blocks,
            contrast_resample_key(score_id, sample_id, anchor_near, anchor_far),
        )
        scale = score_scale(score_values, anchors)
        anchor_conclusion = supported_conclusion(anchor_delta, anchor_low, anchor_high, scale)
        anchor_rows.append({"score_id": score_id, "score_label": score_label, "sample_id": sample_id, "patient_id": sample["patient_id"], "section_id": sample["section_id"], "status": "success", "reason": "", "n_anchor_near": int(anchor_near.sum()), "n_anchor_far": int(anchor_far.sum()), "anchor_delta": anchor_delta, "anchor_low": anchor_low, "anchor_high": anchor_high, "anchor_n_blocks": anchor_nblocks, "scale_mad": scale, "anchor_conclusion": anchor_conclusion})
        if int(anchor_near.sum()) < MIN_GROUP_SPOTS or int(anchor_far.sum()) < MIN_GROUP_SPOTS or anchor_nblocks < MIN_BLOCKS:
            anchors["status"] = "not_evaluable"
            anchors["reason"] = "anchor_support_below_locked_threshold"
            anchor_rows[-1]["status"] = "not_evaluable"
            anchor_rows[-1]["reason"] = anchors["reason"]
        else:
            anchors["status"] = "evaluable"
            anchor_rows[-1]["status"] = "evaluable"
        spot_base = pd.DataFrame({"dataset_id": DATASET_ID, "sample_id": sample_id, "patient_id": sample["patient_id"], "section_id": sample["section_id"], "barcode": sample["barcodes"], "coarse_label": sample["coarse_labels"], "distance_um": anchors["distance_um"], "stromal_component": anchors["components"], "morphology_near": anchor_near, "morphology_far": anchor_far})
        for (method_id, k_value), group in maps[maps["sample_id"].eq(sample_id)].groupby(["method_id", "K"], sort=True):
            reference_id = "neighbors_6" if method_id == "M2_spatial_ward" else "seed_11"
            partitions: dict[str, pd.DataFrame] = {}
            for replicate_id, replicate in group.groupby("replicate_id", sort=True):
                replicate = replicate.copy()
                replicate["barcode"] = replicate["barcode"].astype(str)
                if set(replicate["barcode"]) != set(sample["barcodes"]) or replicate["barcode"].duplicated().any():
                    log(rows, step="effect", sample_id=sample_id, method=method_id, K=k_value, replicate_id=replicate_id, status="not_evaluable", reason="domain_map_barcode_mismatch")
                    continue
                partitions[str(replicate_id)] = replicate.set_index("barcode").loc[list(sample["barcodes"])].reset_index()
            if reference_id not in partitions:
                for partition_id in sorted(partitions):
                    effect_rows.append({"score_id": score_id, "score_label": score_label, "estimator_id": estimator_id, "estimator_label": estimator_label, "dataset_id": DATASET_ID, "sample_id": sample_id, "patient_id": sample["patient_id"], "section_id": sample["section_id"], "method_id": method_id, "method_label": METHOD_LABELS[method_id], "K": int(k_value), "partition_id": partition_id, "is_reference": partition_id == reference_id, "status": "not_evaluable", "status_reason": "reference_partition_missing", "anchor_conclusion": anchor_conclusion, "computational_conclusion": "uncertain", "anchor_to_computational_transition": "not_evaluable"})
                continue
            partition_records: dict[str, dict[str, object]] = {}
            for partition_id, partition in sorted(partitions.items()):
                labels = partition["domain_label"].to_numpy(dtype=int)
                selected, domain_stats = domain_selection(labels, np.asarray(sample["coarse_labels"], dtype=object))
                for domain_stat in domain_stats:
                    mapping_rows.append({"dataset_id": DATASET_ID, "sample_id": sample_id, "patient_id": sample["patient_id"], "section_id": sample["section_id"], "method_id": method_id, "K": int(k_value), "partition_id": partition_id, **domain_stat, "selected_tumor_domain": selected.get("tumor_domain", ""), "selected_stroma_domain": selected.get("stroma_domain", ""), "selection_status": selected["status"], "selection_reason": selected["reason"]})
                record: dict[str, object] = {"status": selected["status"], "reason": selected["reason"], "partition": partition, "labels": labels}
                if selected["status"] == "success" and anchors["status"] in {"success", "evaluable"}:
                    neighbors = np.asarray(anchors["neighbors"], dtype=int)
                    tumor_domain = int(selected["tumor_domain"])
                    stroma_domain = int(selected["stroma_domain"])
                    near, far = computational_groups(anchor_near, anchor_far, labels, tumor_domain, stroma_domain, neighbors, estimator_id)
                    delta, low, high, nblocks = bootstrap_delta(
                        score_values, near, far, blocks,
                        contrast_resample_key(score_id, sample_id, near, far),
                    )
                    record.update({"near": near, "far": far, "delta": delta, "low": low, "high": high, "nblocks": nblocks, "selected": selected, "computational_conclusion": supported_conclusion(delta, low, high, scale)})
                    membership = spot_base.copy()
                    membership["estimator_id"] = estimator_id
                    membership["method_id"] = method_id
                    membership["K"] = int(k_value)
                    membership["partition_id"] = partition_id
                    membership["domain_label"] = labels
                    membership["selected_tumor_domain"] = tumor_domain
                    membership["selected_stroma_domain"] = stroma_domain
                    membership["computational_near"] = near
                    membership["computational_far"] = far
                    membership["computational_boundary"] = np.any(labels[:, None] != labels[neighbors], axis=1)
                    membership_rows.append(membership)
                else:
                    record.update({"near": np.zeros(len(sample["barcodes"]), dtype=bool), "far": np.zeros(len(sample["barcodes"]), dtype=bool), "delta": float("nan"), "low": float("nan"), "high": float("nan"), "nblocks": 0, "selected": selected, "computational_conclusion": "uncertain"})
                partition_records[partition_id] = record
            ref_record = partition_records.get(reference_id, {})
            ref_delta = float(ref_record.get("delta", float("nan")))
            ref_near = np.asarray(ref_record.get("near", np.zeros(len(sample["barcodes"]), dtype=bool)), dtype=bool)
            ref_far = np.asarray(ref_record.get("far", np.zeros(len(sample["barcodes"]), dtype=bool)), dtype=bool)
            reference_usable = (
                ref_record.get("status") == "success"
                and int(ref_near.sum()) >= MIN_GROUP_SPOTS
                and int(ref_far.sum()) >= MIN_GROUP_SPOTS
                and int(ref_record.get("nblocks", 0)) >= MIN_BLOCKS
            )
            for partition_id, record in sorted(partition_records.items()):
                delta = float(record.get("delta", float("nan")))
                low = float(record.get("low", float("nan")))
                high = float(record.get("high", float("nan")))
                near = np.asarray(record["near"], dtype=bool)
                far = np.asarray(record["far"], dtype=bool)
                change_diag: dict[str, object] = {}
                anchor_change_diag: dict[str, object] = {}
                if record["status"] != "success" or anchors["status"] != "evaluable":
                    status = "not_evaluable"
                    reason = str(record.get("reason", "")) if record["status"] != "success" else str(anchors["reason"])
                    change = change_low = change_high = float("nan")
                    change_blocks = 0
                    category = "uncertain"
                    anchor_change = anchor_change_low = anchor_change_high = float("nan")
                elif not reference_usable:
                    status = "not_evaluable"
                    reason = "reference_computational_near_far_support_below_locked_threshold"
                    change = change_low = change_high = float("nan")
                    change_blocks = 0
                    category = "uncertain"
                    anchor_change = anchor_change_low = anchor_change_high = float("nan")
                else:
                    status = "success" if int(near.sum()) >= MIN_GROUP_SPOTS and int(far.sum()) >= MIN_GROUP_SPOTS and int(record["nblocks"]) >= MIN_BLOCKS else "not_evaluable"
                    reason = "" if status == "success" else "computational_near_far_support_below_locked_threshold"
                    change, change_low, change_high, change_blocks = bootstrap_change(score_values, near, far, ref_near, ref_far, blocks, f"{score_id}|{sample_id}|{method_id}|K{k_value}|{partition_id}|change", change_diag) if status == "success" else (float("nan"), float("nan"), float("nan"), 0)
                    anchor_change, anchor_change_low, anchor_change_high, _anchor_common_blocks = bootstrap_change(score_values, near, far, anchor_near, anchor_far, blocks, f"{score_id}|{sample_id}|{method_id}|K{k_value}|{partition_id}|anchor_change", anchor_change_diag) if status == "success" and anchor_nblocks >= MIN_BLOCKS else (float("nan"), float("nan"), float("nan"), 0)
                    if status != "success":
                        category = "uncertain"
                    elif partition_id == reference_id:
                        category = "reference"
                    else:
                        category = classify(delta, low, high, ref_delta, change, change_low, change_high, scale)
                computational_conclusion = str(record.get("computational_conclusion", "uncertain"))
                effect_rows.append({"score_id": score_id, "score_label": score_label, "estimator_id": estimator_id, "estimator_label": estimator_label, "dataset_id": DATASET_ID, "sample_id": sample_id, "patient_id": sample["patient_id"], "section_id": sample["section_id"], "method_id": method_id, "method_label": METHOD_LABELS[method_id], "K": int(k_value), "partition_id": partition_id, "is_reference": partition_id == reference_id, "status": status, "status_reason": reason, "n_anchor_near": int(anchor_near.sum()), "n_anchor_far": int(anchor_far.sum()), "n_computational_near": int(near.sum()), "n_computational_far": int(far.sum()), "n_blocks": int(record.get("nblocks", 0)), "n_change_blocks": int(change_blocks), "change_draws_attempted": int(change_diag.get("n_attempted", 0)), "change_draws_accepted": int(change_diag.get("n_accepted", 0)), "change_draws_used": int(change_diag.get("n_used", 0)), "change_acceptance_rate": change_diag.get("acceptance_rate", float("nan")), "change_draw_status": change_diag.get("reason", ""), "anchor_change_draws_attempted": int(anchor_change_diag.get("n_attempted", 0)), "anchor_change_draws_accepted": int(anchor_change_diag.get("n_accepted", 0)), "anchor_change_draws_used": int(anchor_change_diag.get("n_used", 0)), "anchor_change_acceptance_rate": anchor_change_diag.get("acceptance_rate", float("nan")), "anchor_change_draw_status": anchor_change_diag.get("reason", ""), "primary_scale_mad": scale, "morphology_anchor_delta": anchor_delta, "morphology_anchor_low": anchor_low, "morphology_anchor_high": anchor_high, "anchor_conclusion": anchor_conclusion, "computational_delta": delta, "computational_low": low, "computational_high": high, "computational_conclusion": computational_conclusion, "anchor_to_computational_transition": conclusion_transition(anchor_conclusion, computational_conclusion), "change_vs_reference": change, "change_low": change_low, "change_high": change_high, "anchor_change": anchor_change, "anchor_change_low": anchor_change_low, "anchor_change_high": anchor_change_high, "interpretation": category})
            log(rows, step="effects_progress", score_id=score_id, estimator_id=estimator_id, sample_id=sample_id, method=method_id, K=int(k_value), status="success")
        log(rows, step="effects_sample_complete", score_id=score_id, estimator_id=estimator_id, sample_id=sample_id, status="success")
    effects = pd.DataFrame(effect_rows)
    mapping = pd.DataFrame(mapping_rows)
    membership = pd.concat(membership_rows, ignore_index=True) if membership_rows else pd.DataFrame()
    anchors = pd.DataFrame(anchor_rows)
    return effects, mapping, membership, anchors


def patient_summary(effects: pd.DataFrame) -> pd.DataFrame:
    if effects.empty:
        return pd.DataFrame()
    successful = effects[effects["status"].eq("success")].copy()
    if successful.empty:
        return pd.DataFrame()
    summary = successful.groupby(["score_id", "score_label", "estimator_id", "estimator_label", "dataset_id", "patient_id", "method_id", "method_label", "K", "partition_id", "is_reference"], as_index=False).agg(
        n_sections=("sample_id", "nunique"),
        sections=("sample_id", lambda values: ",".join(sorted(set(map(str, values))))),
        median_morphology_anchor_delta=("morphology_anchor_delta", "median"),
        median_computational_delta=("computational_delta", "median"),
        median_change_vs_reference=("change_vs_reference", "median"),
        median_anchor_change=("anchor_change", "median"),
        n_opposite_direction=("interpretation", lambda values: int(sum(str(value) == "opposite-direction point estimate" for value in values))),
        n_materially_attenuated=("interpretation", lambda values: int(sum(str(value) == "materially attenuated" for value in values))),
        n_conclusion_lost=("anchor_to_computational_transition", lambda values: int(sum(str(value) == "conclusion_lost" for value in values))),
        n_conclusion_reversed=("anchor_to_computational_transition", lambda values: int(sum(str(value) == "conclusion_reversed" for value in values))),
        n_alternative_only_signal=("anchor_to_computational_transition", lambda values: int(sum(str(value) == "alternative_only_signal" for value in values))),
    )
    return summary


def analysis_summary(effects: pd.DataFrame) -> pd.DataFrame:
    if effects.empty:
        return pd.DataFrame()
    return effects.groupby(["score_id", "score_label", "estimator_id", "estimator_label", "method_id", "method_label", "K", "interpretation"], dropna=False, as_index=False).agg(n_rows=("sample_id", "size"), n_patients=("patient_id", "nunique"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-filter",
        default="",
        help="Comma-separated sample IDs for an outcome-blind technical pilot.",
    )
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    global SAMPLES, SAMPLE_META
    if args.sample_filter.strip():
        requested = [value.strip() for value in args.sample_filter.split(",") if value.strip()]
        available = {str(sample["sample_id"]): sample for sample in SAMPLES}
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise ValueError(f"Unknown sample IDs in --sample-filter: {unknown}")
        SAMPLES = [available[sample_id] for sample_id in requested]
        SAMPLE_META = {row["sample_id"]: row for row in SAMPLES}
    ROOT.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    returncode = 1
    try:
        if args.sample_filter.strip():
            log(rows, step="sample_filter", status="success", sample_filter=",".join(str(sample["sample_id"]) for sample in SAMPLES), n_samples=len(SAMPLES))
        annotation_map, data_roots = prepare_input(rows)
        install_stack(rows)
        input_tar = build_input_tar()
        official_map_path = OUT / "official_output" / "official_domain_maps.tsv"
        reuse_official = ALLOW_OFFICIAL_REUSE and official_map_path.exists()
        if reuse_official and RUN_BAYESPACE:
            existing_methods = set(pd.read_csv(official_map_path, sep="\t", usecols=["method_id"])["method_id"].astype(str))
            expected_bayes_method = f"Official_BayesSpace_nrep{BAYESPACE_NREP}"
            reuse_official = expected_bayes_method in existing_methods or "M6_bayesspace_official" in existing_methods
        if reuse_official:
            log(rows, step="reuse_official_output", status="success", path=str(official_map_path))
            official_returncode = 0
        else:
            command = [sys.executable, str(OFFICIAL_RUNNER), "--input-tar", str(input_tar), "--output-dir", str(OUT / "official_output"), "--run-mode", "all", "--skip-python-install"]
            if RUN_BAYESPACE:
                command.extend(["--run-bayesspace", "--bayesspace-nrep", str(BAYESPACE_NREP)])
            log(rows, step="official_runner_start", command=command)
            official = subprocess.run(command, text=True, capture_output=True, timeout=21600)
            (OUT / "official_runner.stdout").write_text(official.stdout, encoding="utf-8")
            (OUT / "official_runner.stderr").write_text(official.stderr, encoding="utf-8")
            log(rows, step="official_runner_finish", returncode=official.returncode)
            official_returncode = official.returncode
            if not official_map_path.exists():
                raise RuntimeError("Official runner did not produce official_domain_maps.tsv")
            if official_returncode != 0:
                raise RuntimeError(f"Official runner failed with return code {official_returncode}")
        raw_official_methods = set(pd.read_csv(official_map_path, sep="\t", usecols=["method_id"])["method_id"].astype(str))
        required_official_methods = {"Official_SpaGCN_v1_2_7", "Official_STAGATE_pyG"}
        if RUN_BAYESPACE:
            required_official_methods.add(f"Official_BayesSpace_nrep{BAYESPACE_NREP}")
        missing_official_methods = sorted(required_official_methods - raw_official_methods)
        if missing_official_methods:
            raise RuntimeError(f"Official output is missing required methods: {missing_official_methods}")
        samples: dict[str, dict[str, object]] = {}
        registration_rows: list[dict[str, object]] = []
        for sample in SAMPLES:
            sample_id = str(sample["sample_id"])
            samples[sample_id] = load_expression(data_roots[sample_id], sample_id, annotation_map[sample_id])
            log(rows, step="load_expression", sample_id=sample_id, n_spots=len(samples[sample_id]["barcodes"]), primary_genes=PRIMARY_GENES)
            registration_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "sample_id": sample_id,
                    "patient_id": sample["patient_id"],
                    "section_id": sample["section_id"],
                    **samples[sample_id]["registration"],
                }
            )
            log(rows, step="annotation_registration", sample_id=sample_id, status="success", **samples[sample_id]["registration"])
        pd.DataFrame(registration_rows).to_csv(OUT / "registration_manifest.tsv", sep="\t", index=False)
        map_frames = [normalize_official_maps(official_map_path)]
        run_baselines(samples, map_frames, rows)
        all_maps = pd.concat(map_frames, ignore_index=True, sort=False)
        all_maps.to_csv(OUT / "all_domain_maps.tsv.gz", sep="\t", index=False, compression="gzip")
        effects, mapping, strict_membership, anchors = compute_effects(samples, all_maps, rows, "primary_barrier", "strict_computational_interface")
        retention_effects, _retention_mapping, retention_membership, _retention_anchors = compute_effects(samples, all_maps, rows, "primary_barrier", "anchor_retention")
        effects.to_csv(OUT / "section_effects.tsv", sep="\t", index=False)
        retention_effects.to_csv(OUT / "section_effects_anchor_retention.tsv", sep="\t", index=False)
        mapping.to_csv(OUT / "domain_mapping.tsv", sep="\t", index=False)
        membership = pd.concat([strict_membership, retention_membership], ignore_index=True, sort=False)
        if not membership.empty:
            membership.to_csv(OUT / "spot_membership.tsv.gz", sep="\t", index=False, compression="gzip")
        anchors.to_csv(OUT / "anchor_section_summary.tsv", sep="\t", index=False)
        patient_summary(effects).to_csv(OUT / "patient_effects.tsv", sep="\t", index=False)
        analysis_summary(effects).to_csv(OUT / "analysis_summary.tsv", sep="\t", index=False)
        patient_summary(retention_effects).to_csv(OUT / "patient_effects_anchor_retention.tsv", sep="\t", index=False)
        analysis_summary(retention_effects).to_csv(OUT / "analysis_summary_anchor_retention.tsv", sep="\t", index=False)
        secondary_ids = [score_id for score_id, definition in SCORE_DEFINITIONS.items() if definition["role"] == "secondary"]
        secondary_effects: dict[str, list[pd.DataFrame]] = {estimator_id: [] for estimator_id in ESTIMATOR_LABELS}
        secondary_anchors: list[pd.DataFrame] = []
        for score_id in secondary_ids:
            for estimator_id in ESTIMATOR_LABELS:
                secondary, _secondary_mapping, _secondary_membership, secondary_anchor = compute_effects(samples, all_maps, rows, score_id, estimator_id)
                if not secondary.empty:
                    secondary_effects[estimator_id].append(secondary)
            if not secondary_anchor.empty:
                secondary_anchors.append(secondary_anchor)
        for estimator_id, tables in secondary_effects.items():
            if tables:
                secondary_effect_table = pd.concat(tables, ignore_index=True, sort=False)
                suffix = "" if estimator_id == "strict_computational_interface" else "_anchor_retention"
                secondary_effect_table.to_csv(OUT / f"secondary_section_effects{suffix}.tsv", sep="\t", index=False)
                patient_summary(secondary_effect_table).to_csv(OUT / f"secondary_patient_effects{suffix}.tsv", sep="\t", index=False)
                analysis_summary(secondary_effect_table).to_csv(OUT / f"secondary_analysis_summary{suffix}.tsv", sep="\t", index=False)
        if secondary_anchors:
            pd.concat(secondary_anchors, ignore_index=True, sort=False).to_csv(OUT / "secondary_anchor_section_summary.tsv", sep="\t", index=False)
        score_rows = []
        for score_id, definition in SCORE_DEFINITIONS.items():
            for sample_id, sample in samples.items():
                availability = sample["score_status"].get(score_id, {"status": "success", "missing_genes": ""})
                score_rows.append(
                    {
                        "score_id": score_id,
                        "score_label": definition["label"],
                        "role": definition["role"],
                        "required_genes": ";".join(definition["genes"]),
                        "sample_id": sample_id,
                        "status": availability["status"],
                        "missing_genes": availability.get("missing_genes", ""),
                    }
                )
        pd.DataFrame(score_rows).to_csv(OUT / "score_availability.tsv", sep="\t", index=False)
        pd.DataFrame(
            [
                {
                    "score_id": score_id,
                    "score_label": definition["label"],
                    "role": definition["role"],
                    "required_genes": ";".join(definition["genes"]),
                    "definition_note": "Mean log-normalized expression; immune_exclusion is (CAF_FAP + SPP1_myeloid) - T_cell.",
                }
                for score_id, definition in SCORE_DEFINITIONS.items()
            ]
        ).to_csv(OUT / "score_manifest.tsv", sep="\t", index=False)
        pd.DataFrame(rows).to_csv(OUT / "validation_run_log.tsv", sep="\t", index=False)
        pd.DataFrame(
            [{"dataset_id": DATASET_ID, "protocol_version": PROTOCOL_VERSION, "lock_date": "2026-09-13", "primary_genes": ",".join(PRIMARY_GENES), "score_ids": ",".join(SCORE_DEFINITIONS), "estimators": ",".join(ESTIMATOR_LABELS), "bootstrap_replicates": BOOTSTRAP_REPLICATES, "min_group_spots": MIN_GROUP_SPOTS, "min_blocks": MIN_BLOCKS, "nominal_spot_diameter_um": NOMINAL_SPOT_DIAMETER_UM, "methods": ",".join((*BASE_METHOD_IDS, "M6_bayesspace_official") if RUN_BAYESPACE else BASE_METHOD_IDS)}]
        ).to_csv(OUT / "analysis_manifest.tsv", sep="\t", index=False)
        returncode = 0 if int(official_returncode) == 0 else int(official_returncode)
    except Exception as exc:
        log(rows, step="full_validation_failure", status="failed", error=str(exc), traceback=traceback.format_exc()[-6000:])
    finally:
        pd.DataFrame(rows).to_csv(OUT / "validation_run_log.tsv", sep="\t", index=False)
        archive_path = ROOT / "colab_interface_validation_full_output.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            for path in sorted(OUT.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=path.relative_to(OUT))
        chunks = ROOT / "archive_chunks"
        chunks.mkdir(parents=True, exist_ok=True)
        for old in chunks.glob("part-*"):
            old.unlink()
        chunk_size = 5 * 1024 * 1024
        with archive_path.open("rb") as source:
            index = 0
            while True:
                data = source.read(chunk_size)
                if not data:
                    break
                (chunks / f"part-{index:03d}").write_bytes(data)
                index += 1
        print(f"{archive_path} archive_parts={index}", flush=True)
    return returncode


if __name__ == "__main__":
    # Colab executes this file inside an IPython kernel; avoid SystemExit,
    # which would terminate the runtime before the output archive is fetched.
    status = main()
    print(f"validation_returncode={status}", flush=True)
