#!/usr/bin/env python3
"""Build the tissue-anchored calibration and CRC application figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/interface_validation/reframed"
OUT = BASE / "figures"


def main() -> None:
    summary = pd.read_csv(BASE / "simulation_full_v1/simulation_summary.tsv", sep="\t")
    review = json.loads((BASE / "simulation_review_v1.json").read_text())
    cohort = pd.read_csv(BASE / "real_application_v1/cohort_summary.tsv", sep="\t")
    effects = pd.read_csv(BASE / "real_application_v1/real_cohort_effects.tsv", sep="\t", low_memory=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    blue, green, orange, grey = "#0072B2", "#009E73", "#D55E00", "#5B6570"
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), constrained_layout=True)
    ax_c, ax_d, ax_a, ax_b = axes.ravel()

    targets = [("anchor", "Tissue anchor"), ("contrast", "Map selection"),
               ("paired", "Paired change")]
    for position, (target, label) in enumerate(targets):
        data = summary.loc[(summary.target == target) & summary.evaluable.gt(0), "coverage"].to_numpy()
        ax_a.boxplot(data, positions=[position], widths=0.48, showfliers=False,
                     boxprops={"color": blue}, medianprops={"color": orange, "linewidth": 1.5},
                     whiskerprops={"color": blue}, capprops={"color": blue})
        ax_a.scatter(np.full(len(data), position) + np.random.default_rng(position).uniform(-0.18, 0.18, len(data)),
                     data, s=5, color=blue, alpha=0.18, rasterized=True)
        ax_a.text(position, 1.015, f"n={len(data)}", ha="center", fontsize=7)
    ax_a.axhline(0.95, color=grey, linestyle="--", linewidth=0.8)
    ax_a.set(xticks=range(3), xticklabels=[label for _, label in targets],
             ylim=(0.83, 1.04), ylabel="Empirical interval coverage")
    ax_a.set_title("C  Conditional calibration across simulated settings", loc="left", weight="bold")

    example = {row["map"]: row for row in review["prespecified_illustration_case_051"]}
    labels = ["Mixed-spot\nanchor", "Reference\nmap", "Alternative\nmap"]
    values = [example["identity_control"]["measured_anchor_truth"],
              example["reference_shift"]["map_truth"],
              example["alternative_shift"]["map_truth"]]
    ax_b.bar(range(3), values, color=[grey, blue, green], width=0.58)
    for position, value in enumerate(values):
        ax_b.text(position, value + 0.005, f"{value:.3f}", ha="center", fontsize=8)
    ax_b.set(xticks=range(3), xticklabels=labels, ylim=(0, 0.27),
             ylabel="Known population near-minus-far contrast")
    ax_b.set_title("D  Lower map agreement can preserve the contrast better", loc="left", weight="bold")

    rows = cohort.loc[cohort.estimator_id.eq("anchor_retention")].set_index("resource")
    resources = ["Valdeolivas", "GSE294385"]
    eligible = [int(rows.loc[r, "n_anchor_eligible_sections"]) for r in resources]
    supported = [int(rows.loc[r, "n_anchor_supported_sections"]) for r in resources]
    attempted = [int(rows.loc[r, "n_attempted_sections"]) for r in resources]
    y = np.arange(2)
    ax_c.barh(y, attempted, color="#DDE3E6", label="All sections")
    ax_c.barh(y, eligible, color=blue, label="Eligible anchor")
    ax_c.barh(y, supported, color=orange, label="Supported anchor")
    for i, (a, e, s) in enumerate(zip(attempted, eligible, supported)):
        ax_c.text(a + 0.2, i, f"{s}/{e} supported", va="center", fontsize=8)
    ax_c.set(yticks=y, yticklabels=resources, xlim=(0, 17), xlabel="Sections")
    ax_c.invert_yaxis()
    ax_c.legend(frameon=False, loc="lower right", fontsize=7)
    ax_c.set_title("A  Supported tissue contrasts were uncommon", loc="left", weight="bold")

    case = effects.loc[(effects.resource == "GSE294385") & (effects.sample_id == "M-ST-13")
                       & (effects.method_id == "M6_bayesspace_official") & effects.K.eq(4)
                       & effects.estimator_id.eq("anchor_retention")].set_index("partition_id")
    order = ["seed_11", "seed_23", "seed_37"]
    for i, key in enumerate(order):
        row = case.loc[key]
        value = float(row.computational_delta)
        low, high = float(row.computational_low), float(row.computational_high)
        ax_d.errorbar(value, i, xerr=[[value - low], [high - value]], fmt="o",
                      color=blue if i == 0 else orange, capsize=3, markersize=5)
    ax_d.axvline(float(case.iloc[0].morphology_anchor_delta), color=green,
                 linestyle="--", linewidth=1, label="Morphology anchor")
    ax_d.axvline(0, color=grey, linewidth=0.6)
    ax_d.set(yticks=range(3), yticklabels=["Reference 11", "Alternative 23", "Alternative 37"],
             xlabel="Near-minus-far four-gene score", xlim=(-0.8, 0.15))
    ax_d.invert_yaxis()
    ax_d.legend(frameon=False, loc="upper right", fontsize=7)
    ax_d.set_title("B  One patient section: selected-domain contrast", loc="left", weight="bold")

    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"figure1.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
