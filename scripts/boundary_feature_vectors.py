#!/usr/bin/env python3
"""Construct gene and program-level feature vectors for boundary analyses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from build_figS3_boundary_signatures import Signature, _lognorm_sparse


@dataclass(frozen=True)
class FeatureVector:
    feature_type: str
    feature_id: str
    feature_label: str
    genes_used: str
    values: np.ndarray


def feature_vectors(counts, gene_names: list[str]) -> list[FeatureVector]:
    signatures = [
        Signature("SIG_CAF_FAP", "CAF/FAP-associated fibroblast program", ["FAP", "COL1A1", "DCN", "COL3A1", "LUM"]),
        Signature("SIG_MY_SPP1", "SPP1-myeloid-associated program", ["SPP1", "LST1", "TYROBP", "C1QA", "C1QB", "C1QC"]),
        Signature("SIG_TCELL", "T-cell-associated program", ["TRAC", "CD3D", "CD3E", "CD247"]),
        Signature(
            "SIG_EXCLUSION_TGFB_CXCL12",
            "TGF-beta/CXCL12 myofibroblast-barrier program",
            ["TGFB1", "CXCL12", "ACTA2", "TAGLN", "FAP", "COL1A1"],
        ),
        Signature("SIG_CYTOTOX", "Cytotoxic lymphocyte-associated program", ["NKG7", "GNLY", "GZMB", "PRF1", "IFNG"]),
    ]
    targets = ["FAP", "SPP1", "EPCAM", "COL1A1", "PTPRC"] + sorted(
        {gene for signature in signatures for gene in signature.genes}
    )
    gene_to_indices: dict[str, list[int]] = {}
    for index, gene in enumerate(gene_names):
        gene_to_indices.setdefault(str(gene), []).append(index)

    expression: dict[str, np.ndarray] = {}
    for gene in targets:
        indices = gene_to_indices.get(gene)
        if not indices:
            continue
        matrix = _lognorm_sparse(counts, indices)
        values = matrix[:, 0] if matrix.shape[1] == 1 else matrix.sum(axis=1)
        expression[gene] = values.astype(np.float64, copy=False)

    features: list[FeatureVector] = []
    for gene in ["FAP", "SPP1", "EPCAM", "COL1A1", "PTPRC"]:
        if gene in expression:
            features.append(FeatureVector("gene", gene, f"{gene} log-normalized expression", gene, expression[gene]))

    for signature in signatures:
        used = [gene for gene in signature.genes if gene in expression]
        if used:
            matrix = np.vstack([expression[gene] for gene in used]).T
            features.append(
                FeatureVector(
                    "signature",
                    signature.signature_id,
                    f"{signature.label} mean log-normalized expression",
                    ",".join(used),
                    np.mean(matrix, axis=1),
                )
            )

    signature_map = {signature.signature_id: signature for signature in signatures}
    caf = [gene for gene in signature_map["SIG_CAF_FAP"].genes if gene in expression]
    myeloid = [gene for gene in signature_map["SIG_MY_SPP1"].genes if gene in expression]
    tcell = [gene for gene in signature_map["SIG_TCELL"].genes if gene in expression]
    if caf and myeloid and tcell:
        caf_values = np.mean(np.vstack([expression[gene] for gene in caf]).T, axis=1)
        myeloid_values = np.mean(np.vstack([expression[gene] for gene in myeloid]).T, axis=1)
        tcell_values = np.mean(np.vstack([expression[gene] for gene in tcell]).T, axis=1)
        features.append(
            FeatureVector(
                "signature",
                "SIG_EXCLUSION_CONTRAST",
                "CAF plus SPP1-myeloid minus T-cell contrast score",
                f"CAF:{','.join(caf)};MY:{','.join(myeloid)};TCELL:{','.join(tcell)}",
                caf_values + myeloid_values - tcell_values,
            )
        )
    return features
