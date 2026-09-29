#!/usr/bin/env python3
"""Validate canonical R4 section inputs before any downstream calculation.

This module has no expression normalization, PCA, or partition-fitting path.
It checks the compact artifacts produced from the retained source packages and
saved domain maps; it does not change or repair mismatched inputs silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SCHEMA_VERSION = "r4_downstream_input_v1"
SPOT_COLUMNS = (
    "dataset_id", "sample_id", "patient_id", "barcode", "raw_label",
    "coarse_label", "x_fullres", "y_fullres", "array_row", "array_col",
    "spot_diameter_fullres",
)
SCORE_COLUMNS = (
    "dataset_id", "sample_id", "barcode", "score_id", "score_value",
    "status", "missing_genes",
)
MAP_COLUMNS = (
    "dataset_id", "sample_id", "method_id", "K", "replicate_id",
    "barcode", "domain_label", "status",
)
PARTITION_KEY = ("dataset_id", "sample_id", "method_id", "K", "replicate_id")
DECLARED_METHODS = {
    "M0_expr_kmeans", "M1_spatial_concat_kmeans", "M2_spatial_ward",
    "M3_spatial_leiden", "M4_spagcn_official", "M5_stagate_official",
    "M6_bayesspace_official",
}


class InputContractError(ValueError):
    """Input artifacts cannot be joined without changing the declared analysis."""


@dataclass(frozen=True)
class SectionIdentity:
    dataset_id: str
    sample_id: str
    patient_id: str


def _require_columns(frame: pd.DataFrame, fields: tuple[str, ...], name: str) -> None:
    missing = sorted(set(fields) - set(frame.columns))
    if missing:
        raise InputContractError(f"{name}: missing columns {missing}")


def _no_missing(frame: pd.DataFrame, fields: tuple[str, ...], name: str) -> None:
    for field in fields:
        values = frame[field]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise InputContractError(f"{name}: missing {field}")


def _unique(frame: pd.DataFrame, fields: tuple[str, ...], name: str) -> None:
    if frame.duplicated(list(fields)).any():
        raise InputContractError(f"{name}: duplicate key {fields}")


def _same_barcodes(values: pd.Series, expected: set[str], name: str) -> None:
    observed = set(values.astype(str))
    if observed != expected:
        raise InputContractError(
            f"{name}: barcode mismatch; missing={len(expected - observed)}, "
            f"extra={len(observed - expected)}"
        )


def validate_section(
    identity: SectionIdentity,
    spots: pd.DataFrame,
    scores: pd.DataFrame,
    maps: pd.DataFrame,
) -> dict[str, object]:
    """Validate one section; return a count report, never a filtered table.

    Map rows with non-success status may describe failed partitions and need not
    contain spot labels. Each *successful* partition must cover every registered
    in-tissue barcode exactly once. Score rows must cover every barcode for each
    fixed score, including a uniform not-evaluable status when genes are absent.
    """
    for name, frame, fields in (
        ("spots", spots, SPOT_COLUMNS),
        ("scores", scores, SCORE_COLUMNS),
        ("maps", maps, MAP_COLUMNS),
    ):
        _require_columns(frame, fields, name)
        if frame.empty:
            raise InputContractError(f"{name}: empty")
        if not frame["dataset_id"].astype(str).eq(identity.dataset_id).all():
            raise InputContractError(f"{name}: dataset_id mismatch")
        if not frame["sample_id"].astype(str).eq(identity.sample_id).all():
            raise InputContractError(f"{name}: sample_id mismatch")

    # Unannotated in-tissue spots remain present for exact map alignment. Their
    # raw label may be empty, but the coarse label must state their category.
    _no_missing(spots, tuple(field for field in SPOT_COLUMNS if field != "raw_label"), "spots")
    if spots["raw_label"].isna().any():
        raise InputContractError("spots: null raw_label")
    _unique(spots, ("dataset_id", "sample_id", "barcode"), "spots")
    if not spots["patient_id"].astype(str).eq(identity.patient_id).all():
        raise InputContractError("spots: patient_id mismatch")
    for field in ("x_fullres", "y_fullres", "array_row", "array_col", "spot_diameter_fullres"):
        values = pd.to_numeric(spots[field], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise InputContractError(f"spots: nonfinite {field}")
        if field == "spot_diameter_fullres" and not (values > 0).all():
            raise InputContractError("spots: nonpositive spot diameter")
    diameters = pd.to_numeric(spots["spot_diameter_fullres"]).to_numpy(dtype=float)
    if not np.allclose(diameters, diameters[0], rtol=0, atol=1e-9):
        raise InputContractError("spots: multiple scalefactors in one section")
    expected = set(spots["barcode"].astype(str))

    _no_missing(scores, ("dataset_id", "sample_id", "barcode", "score_id", "status"), "scores")
    _unique(scores, ("dataset_id", "sample_id", "barcode", "score_id"), "scores")
    score_count = 0
    for score_id, group in scores.groupby("score_id", sort=True):
        _same_barcodes(group["barcode"], expected, f"scores/{score_id}")
        statuses = set(group["status"].astype(str))
        if len(statuses) != 1 or not statuses <= {"success", "not_evaluable"}:
            raise InputContractError(f"scores/{score_id}: inconsistent status")
        values = pd.to_numeric(group["score_value"], errors="coerce").to_numpy(dtype=float)
        if statuses == {"success"} and not np.isfinite(values).all():
            raise InputContractError(f"scores/{score_id}: missing successful values")
        if statuses == {"not_evaluable"} and np.isfinite(values).any():
            raise InputContractError(f"scores/{score_id}: values despite missing genes")
        score_count += 1
    if "primary_barrier" not in set(scores["score_id"].astype(str)):
        raise InputContractError("scores: primary_barrier absent")

    _no_missing(maps, PARTITION_KEY + ("status",), "maps")
    if not set(maps["method_id"].astype(str)).issubset(DECLARED_METHODS):
        raise InputContractError("maps: undeclared method")
    kvals = pd.to_numeric(maps["K"], errors="coerce")
    if kvals.isna().any() or not kvals.isin((4, 6)).all():
        raise InputContractError("maps: K outside locked 4/6 grid")
    for method_id, group in maps.groupby("method_id", sort=True):
        allowed = {"neighbors_4", "neighbors_6", "neighbors_8"} if method_id == "M2_spatial_ward" else {"seed_11", "seed_23", "seed_37"}
        if not set(group["replicate_id"].astype(str)).issubset(allowed):
            raise InputContractError(f"maps/{method_id}: undeclared replicate")
    successes = maps[maps["status"].astype(str).eq("success")]
    if successes.empty:
        raise InputContractError("maps: no successful saved partitions")
    _no_missing(successes, ("barcode", "domain_label"), "successful maps")
    _unique(successes, PARTITION_KEY + ("barcode",), "successful maps")
    partition_count = 0
    for key, group in successes.groupby(list(PARTITION_KEY), sort=True):
        _same_barcodes(group["barcode"], expected, f"maps/{key}")
        labels = pd.to_numeric(group["domain_label"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
            raise InputContractError(f"maps/{key}: invalid domain label")
        partition_count += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": identity.dataset_id,
        "sample_id": identity.sample_id,
        "patient_id": identity.patient_id,
        "n_spots": len(expected),
        "n_scores": score_count,
        "n_successful_partitions": partition_count,
        "n_failed_map_rows": int(len(maps) - len(successes)),
        "spot_diameter_fullres": float(diameters[0]),
    }
