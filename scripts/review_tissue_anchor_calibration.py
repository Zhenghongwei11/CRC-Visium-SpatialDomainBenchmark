#!/usr/bin/env python3
"""Export a bounded scientific review of a completed frozen calibration run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from audit_tissue_anchor_simulation import audit


def review(root: Path) -> dict:
    checked = audit(root)
    if not checked["audit_pass"] or checked["mode"] != "full" or checked["run_state"] != "complete":
        raise ValueError("a complete audited full run is required")
    data = pd.read_csv(root / "simulation_cell_metrics.tsv", sep="\t", low_memory=False)
    summary = pd.read_csv(root / "simulation_summary.tsv", sep="\t")
    status = json.loads((root / "STATUS.json").read_text())
    unique = data.drop_duplicates(["case_id", "estimator", "map_name"])
    records = {}
    for target, group in summary.groupby("target"):
        good = group[group["evaluable"].gt(0)]
        records[target] = {
            "condition_groups": len(group), "evaluable_groups": len(good),
            "gate_counts": {str(k): int(v) for k, v in group["screening_gate"].value_counts().items()},
            "coverage_min": float(good["coverage"].min()),
            "coverage_median": float(good["coverage"].median()),
            "coverage_max": float(good["coverage"].max()),
            "max_absolute_bias_noise_sd": float(good["bias"].abs().max()),
            "coverage_upper_below_0p90": int(good["coverage_high"].lt(0.9).sum()),
            "coverage_upper_below_0p95_pointwise": int(good["coverage_high"].lt(0.95).sum()),
        }
    reference = unique[unique["map_name"].eq("reference_shift") & unique["anchor_truth"].abs().gt(1e-10)].copy()
    reference["relative_attenuation"] = -reference["selection_distortion"] / reference["anchor_truth"]
    example = unique[unique["case_id"].eq("case_051") & unique["estimator"].eq("anchor_retention")]
    examples = []
    for row in example.itertuples():
        examples.append({"map": row.map_name, "latent_anchor_truth": row.latent_anchor_truth,
            "measured_anchor_truth": row.anchor_truth, "map_truth": row.map_truth,
            "global_ari": row.global_ari, "near_retention": row.near_retention,
            "far_retention": row.far_retention,
            "relative_selection_attenuation": -row.selection_distortion / row.anchor_truth})
    return {
        "input_sha256": status["output_sha256"], "audit": checked,
        "calibration": records,
        "reference_maps_nonzero_signal": {
            "global_ari_range": [float(reference.global_ari.min()), float(reference.global_ari.max())],
            "near_retention_range": [float(reference.near_retention.min()), float(reference.near_retention.max())],
            "far_retention_range": [float(reference.far_retention.min()), float(reference.far_retention.max())],
            "relative_attenuation_range": [float(reference.relative_attenuation.min()), float(reference.relative_attenuation.max())],
        },
        "prespecified_illustration_case_051": examples,
        "decisions": {
            "all_conditions_pass_calibration": bool(summary[summary["target"].ne("paired") | summary["map_name"].ne("reference_shift")]["screening_gate"].eq("pass").all()),
            "scope": "Fixed outcome-independent maps; declared Gaussian signal/noise and geometry grid only",
            "real_cohort_interval_language": "Descriptive pending transportability and patient-level review",
            "demonstrated_result": "A controlled counterexample to ordering regional fidelity by global ARI",
            "mechanisms_not_demonstrated": ["CRC biological reversal", "Potts/MCMC mode hopping", "smooth-muscle causation"],
            "publication_claim_approval": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.output_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"audit_pass": result["audit"]["audit_pass"],
                      "calibration": result["calibration"], "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    main()
