#!/usr/bin/env python3
"""Read-only numeric and checkpoint audit of known-boundary calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root: Path) -> dict:
    status = json.loads((root / "STATUS.json").read_text())
    failures = []
    for name, digest in status["output_sha256"].items():
        if sha256(root / name) != digest:
            failures.append(f"output checksum: {name}")
    checkpoints = sorted((root / "checkpoints").glob("*.tsv"))
    checkpoint_rows = 0
    for path in checkpoints:
        record = json.loads(path.with_suffix(".json").read_text())
        if sha256(path) != record["sha256"]:
            failures.append(f"checkpoint checksum: {path.name}")
        checkpoint_rows += record["rows"]
    data = pd.read_csv(root / "simulation_cell_metrics.tsv", sep="\t", low_memory=False)
    if len(data) != checkpoint_rows or len(data) != status["rows"]:
        failures.append("checkpoint/aggregate row counts differ")
    if len(checkpoints) != status["sections_attempted"]:
        failures.append("checkpoint/attempted section counts differ")
    keys = ["case_id", "replicate", "map_name", "estimator"]
    if data.duplicated(keys).any():
        failures.append("duplicate simulation result keys")
    if not data.groupby(["case_id", "replicate"]).size().eq(6).all():
        failures.append("missing map/estimator combinations")
    good = data[data["status"].eq("success")]
    paired = data[data["paired_status"].eq("success")]
    for frame, columns in ((good, ["delta", "low", "high", "map_truth"]),
                           (paired, ["paired_delta", "paired_low", "paired_high", "paired_truth"])):
        if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
            failures.append(f"nonfinite successful estimates: {columns[0]}")
    if not (good["low"] <= good["high"]).all() or not (paired["paired_low"] <= paired["paired_high"]).all():
        failures.append("interval bounds inverted")
    if not np.array_equal(good["covered"].to_numpy(),
                          ((good["low"] <= good["map_truth"]) & (good["map_truth"] <= good["high"])).to_numpy()):
        failures.append("contrast coverage flags disagree with intervals")
    if not np.array_equal(paired["paired_covered"].to_numpy(),
                          ((paired["paired_low"] <= paired["paired_truth"]) & (paired["paired_truth"] <= paired["paired_high"])).to_numpy()):
        failures.append("paired coverage flags disagree with intervals")
    identity = good[good["map_name"].eq("identity_control") & good["estimator"].eq("anchor_retention")]
    for left, right in (("delta", "anchor_delta"), ("low", "anchor_low"), ("high", "anchor_high"), ("map_truth", "anchor_truth")):
        if not np.allclose(identity[left], identity[right], atol=1e-12, rtol=0):
            failures.append(f"identity control: {left}/{right}")
    zero = paired[paired["paired_degenerate"]]
    if not zero[["paired_delta", "paired_low", "paired_high", "paired_truth"]].eq(0).all().all():
        failures.append("identical memberships do not have exactly zero change")
    if not paired["paired_used"].eq(1000).all():
        failures.append("successful paired interval has fewer than 1000 draws")
    noisy_null = good[good["signal_amplitude"].eq(0)]
    if not np.allclose(noisy_null[["latent_anchor_truth", "anchor_truth", "map_truth"]], 0, atol=1e-10):
        failures.append("zero-signal generator does not have null truth")
    thresholds = pd.read_csv(root / "simulation_threshold_sensitivity.tsv", sep="\t")
    if (thresholds["numerator"] > thresholds["denominator"]).any():
        failures.append("threshold numerator exceeds denominator")
    if (status["state"] == "complete") != (status["sections_attempted"] == status["sections_expected"]):
        failures.append("completion state differs from attempted/expected counts")
    if status.get("claim_eligible"):
        failures.append("run improperly claims automatic scientific approval")
    return {"audit_pass": not failures, "mode": status["mode"], "run_state": status["state"],
            "sections_attempted": status["sections_attempted"], "sections_expected": status["sections_expected"],
            "rows": len(data), "successful_rows": len(good), "not_evaluable_rows": len(data) - len(good),
            "identity_control_rows": len(identity), "exact_zero_paired_controls": len(zero),
            "failed_checks": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    result = audit(args.output_root)
    print(json.dumps(result, indent=2))
    if not result["audit_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
