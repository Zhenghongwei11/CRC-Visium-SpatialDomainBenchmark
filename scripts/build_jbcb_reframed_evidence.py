#!/usr/bin/env python3
"""Derive submission-facing summaries from the frozen tissue-anchor tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "docs/submissions/JBCB_REVISION_20260929"
SI = PACKAGE / "SUPPORTING_INFORMATION"
FIGURES = PACKAGE / "FIGURES"


def support(delta: pd.Series, low: pd.Series, high: pd.Series, scale: pd.Series, margin: float) -> pd.Series:
    return (delta.abs() >= margin * scale) & (((delta < 0) & (high < 0)) | ((delta > 0) & (low > 0)))


def save(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(SI / name, sep="\t", index=False, float_format="%.8g")


def main() -> None:
    global SI, FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE)
    args = parser.parse_args()
    SI = args.output_dir / "SUPPORTING_INFORMATION"
    FIGURES = args.output_dir / "FIGURES"
    SI.mkdir(parents=True, exist_ok=True)
    effects = pd.read_csv(ROOT / "results/interface_validation/reframed/real_application_v1/real_cohort_effects.tsv",
                          sep="\t", low_memory=False)
    sim = pd.read_csv(ROOT / "results/interface_validation/reframed/simulation_full_v1/simulation_threshold_sensitivity.tsv", sep="\t")
    sim_summary = pd.read_csv(ROOT / "results/interface_validation/reframed/simulation_full_v1/simulation_summary.tsv", sep="\t")

    sections = effects.loc[effects.estimator_id.eq("anchor_retention")].drop_duplicates(["resource", "sample_id"]).copy()
    sections["anchor_d_over_mad"] = sections.morphology_anchor_delta / sections.primary_scale_mad
    sections["exclusion_reason"] = np.where(
        sections.anchor_status.eq("evaluable"), "",
        np.where(sections.anchor_reason.eq("anchor_support_below_locked_threshold"),
                 "Fewer than 20 spots in a band or fewer than three blocks", sections.anchor_reason),
    )
    section_columns = ["resource", "patient_id", "sample_id", "section_id", "anchor_status", "exclusion_reason",
                       "n_anchor_near", "n_anchor_far", "morphology_anchor_delta", "morphology_anchor_low",
                       "morphology_anchor_high", "primary_scale_mad", "anchor_d_over_mad", "anchor_supported"]
    save(sections[section_columns].sort_values(["resource", "patient_id", "sample_id"]), "anchor_section_summary.tsv")

    patient_table = (sections.groupby("resource", sort=True)
                     .agg(patients_represented=("patient_id", "nunique"))
                     .reset_index())
    eligible_patients = (sections.loc[sections.anchor_status.eq("evaluable")]
                         .groupby("resource").patient_id.nunique())
    supported_patients = (sections.loc[sections.anchor_supported]
                          .groupby("resource").patient_id.nunique())
    patient_table["patients_with_eligible_section"] = patient_table.resource.map(eligible_patients).fillna(0).astype(int)
    patient_table["patients_with_supported_section"] = patient_table.resource.map(supported_patients).fillna(0).astype(int)
    save(patient_table, "manuscript_table1.tsv")

    eligible = effects.loc[effects.anchor_status.eq("evaluable") & effects.status.eq("success")].copy()
    eligible["map_arm"] = np.where(eligible.is_reference, "reference", "alternative")
    eligible["map_minus_anchor"] = eligible.computational_delta - eligible.morphology_anchor_delta
    eligible["map_minus_anchor_over_mad"] = eligible.map_minus_anchor / eligible.primary_scale_mad
    eligible["point_sign_opposite_anchor"] = eligible.computational_delta * eligible.morphology_anchor_delta < 0
    eligible["point_moved_toward_zero"] = eligible.computational_delta.abs() < eligible.morphology_anchor_delta.abs()
    eligible["both_supported_same_direction"] = (eligible.anchor_supported & eligible.computational_conclusion.eq(eligible.anchor_conclusion))
    anchor_supported = eligible.loc[eligible.anchor_supported].copy()
    map_summary = (anchor_supported.groupby(["resource", "estimator_id", "map_arm"], sort=True)
                   .agg(n_maps=("sample_id", "size"), n_sections=("sample_id", "nunique"),
                        n_supported=("both_supported_same_direction", "sum"),
                        n_point_toward_zero=("point_moved_toward_zero", "sum"),
                        n_point_opposite_anchor=("point_sign_opposite_anchor", "sum"),
                        median_map_minus_anchor_over_mad=("map_minus_anchor_over_mad", "median"),
                        median_near_retention=("near_retention", "median"),
                        median_far_retention=("far_retention", "median"))
                   .reset_index())
    map_summary["support_fraction"] = map_summary.n_supported / map_summary.n_maps
    save(map_summary, "map_selection_summary.tsv")

    sweep_rows = []
    for margin in (0.0, 0.25, 0.5, 0.75):
        work = anchor_supported.copy()
        work["anchor_pass"] = support(work.morphology_anchor_delta, work.morphology_anchor_low,
                                      work.morphology_anchor_high, work.primary_scale_mad, margin)
        work["map_pass"] = support(work.computational_delta, work.computational_low,
                                   work.computational_high, work.primary_scale_mad, margin)
        for (resource, estimator, arm), group in work.groupby(["resource", "estimator_id", "map_arm"]):
            denominator = int(group.anchor_pass.sum())
            retained = int((group.anchor_pass & group.map_pass).sum())
            sweep_rows.append({"resource": resource, "estimator_id": estimator, "map_arm": arm,
                               "margin_mad": margin, "anchor_pass": denominator, "map_pass": retained,
                               "map_pass_fraction": retained / denominator if denominator else np.nan})
    sweep = pd.DataFrame(sweep_rows)
    save(sweep, "real_threshold_sensitivity.tsv")

    sim_calibration = (sim.groupby(["estimator", "map_name", "metric", "margin_mad"], sort=True)
                       [["numerator", "denominator"]].sum().reset_index())
    sim_calibration["rate"] = sim_calibration.numerator / sim_calibration.denominator.replace(0, np.nan)
    save(sim_calibration, "simulation_calibration_by_map.tsv")

    geometry = effects.loc[(effects.membership_change_scope.eq("disjoint_selected_stromal_domains"))
                           & effects.estimator_id.eq("anchor_retention")].copy()
    save(geometry[["resource", "patient_id", "sample_id", "section_id", "method_id", "K", "partition_id",
                   "anchor_status", "status", "stroma_domain_jaccard_vs_reference",
                   "stroma_domain_centroid_shift_um", "n_changed_stroma_domain", "n_changed_stroma_outside_anchor_bands",
                   "n_changed_near", "n_changed_far"]], "disjoint_domain_summary.tsv")

    component = pd.read_csv(ROOT / "results/interface_validation/reframed/real_application_v1/marker_sensitivity.tsv",
                            sep="\t")
    availability = component[["resource", "patient_id", "sample_id", "planned_alternative_genes",
                              "contains_single_gene_values", "can_calculate_TGFB1_CXCL12_now"]].copy()
    availability["interpretation"] = "Component-gene sensitivity unavailable from retained normalized inputs"
    save(availability, "score_component_availability.tsv")

    setting_status = (effects.groupby(["resource", "estimator_id", "status", "status_reason"],
                                      dropna=False).size().reset_index(name="n_map_records"))
    setting_status["meaning"] = np.where(
        setting_status.status_reason.eq("reference_computational_near_far_support_below_locked_threshold"),
        "Reference does not meet paired-comparison eligibility; map estimate may still exist",
        np.where(setting_status.status_reason.eq("anchor_support_below_locked_threshold"),
                 "Morphology band lacks the required spot count or spatial blocks",
                 "See per-map status and estimates in real_cohort_effects.tsv"))
    save(setting_status, "setting_status_summary.tsv")

    submission_rows = effects[[
        "resource", "patient_id", "sample_id", "section_id", "estimator_id", "method_id", "K",
        "partition_id", "is_reference", "anchor_status", "anchor_reason", "n_anchor_near",
        "n_anchor_far", "morphology_anchor_delta", "morphology_anchor_low", "morphology_anchor_high",
        "primary_scale_mad", "anchor_supported", "status", "status_reason", "computational_delta",
        "computational_low", "computational_high", "computational_conclusion", "n_computational_near",
        "n_computational_far", "near_retention", "far_retention", "selected_stroma_purity",
        "stroma_domain_jaccard_vs_reference", "stroma_domain_centroid_shift_um",
        "n_changed_stroma_domain", "n_changed_stroma_outside_anchor_bands", "change_vs_reference",
        "change_low", "change_high", "additional_support_loss", "paired_change_flag",
        "supported_reversal_vs_anchor", "supported_reversal_vs_reference",
        "opposite_point_sign_vs_reference", "map_score_feature_independence",
    ]].copy()
    submission_rows["anchor_exclusion_meaning"] = np.where(
        submission_rows.anchor_reason.eq("anchor_support_below_locked_threshold"),
        "Too few morphology-band spots or spatial blocks", "")
    submission_rows["paired_comparison_meaning"] = np.where(
        submission_rows.status_reason.eq("reference_computational_near_far_support_below_locked_threshold"),
        "Reference lacks paired-comparison eligibility; inspect available map estimate", "")
    save(submission_rows, "per_map_results.tsv")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    navy, green, orange, grey = "#24536B", "#2A806C", "#C46A30", "#65717B"
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    observed = sections.loc[sections.anchor_status.eq("evaluable")].copy()
    observed["label"] = observed.sample_id.where(observed.resource.eq("Valdeolivas"), observed.sample_id)
    observed = observed.sort_values(["resource", "anchor_d_over_mad"])
    for i, row in enumerate(observed.itertuples()):
        color = orange if row.anchor_supported else grey
        ax_a.plot([row.morphology_anchor_low / row.primary_scale_mad,
                   row.morphology_anchor_high / row.primary_scale_mad], [i, i], color=color, lw=1)
        ax_a.scatter(row.anchor_d_over_mad, i, color=color, s=23, zorder=3)
    ax_a.axvline(0, color="#B7C0C5", lw=0.8)
    ax_a.set(yticks=range(len(observed)), yticklabels=observed.label, xlabel="Near minus far score / section MAD")
    ax_a.tick_params(axis="y", labelsize=6.8)
    ax_a.invert_yaxis()
    ax_a.set_title("A  Morphology-defined contrasts", loc="left", weight="bold")

    plot_rows = anchor_supported.loc[anchor_supported.estimator_id.eq("anchor_retention")].copy()
    names = [("Valdeolivas", "S6_Rec"), ("GSE294385", "Primary Colon M-ST-13"),
             ("GSE294385", "Primary Colon M-ST-15"), ("GSE294385", "Primary Colon M-ST-34")]
    for i, (resource, section) in enumerate(names):
        part = plot_rows.loc[plot_rows.resource.eq(resource) & plot_rows.section_id.eq(section)]
        anchor = float(part.anchor_standardized.iloc[0])
        ax_b.scatter(i - .23, anchor, marker="D", color=green, s=31, zorder=4)
        for j, arm in enumerate(("reference", "alternative")):
            values = part.loc[part.map_arm.eq(arm), "map_standardized"].dropna().to_numpy()
            if len(values):
                x = i + (0 if j == 0 else .23)
                ax_b.scatter(np.full(len(values), x), values, s=9, alpha=.4,
                             color=navy if j == 0 else orange, zorder=2)
                ax_b.scatter(x, np.median(values), marker="_", s=110,
                             color=navy if j == 0 else orange, zorder=4)
    ax_b.axhline(0, color="#B7C0C5", lw=.8)
    ax_b.set(xticks=range(4), xticklabels=["Valdeolivas\nS6_Rec", "M-ST-13", "M-ST-15", "M-ST-34"],
             ylabel="Near minus far score / section MAD")
    ax_b.set_title("B  Anchor and map-selected estimates", loc="left", weight="bold")
    ax_b.scatter([], [], marker="D", color=green, label="Morphology anchor")
    ax_b.scatter([], [], marker="o", color=navy, label="Reference maps")
    ax_b.scatter([], [], marker="o", color=orange, label="Alternative maps")
    ax_b.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(.02, .01))

    plotted = sweep.loc[sweep.estimator_id.eq("anchor_retention")].groupby(["margin_mad", "map_arm"])
    sweep_plot = plotted[["anchor_pass", "map_pass"]].sum().reset_index()
    for arm, color in (("reference", navy), ("alternative", orange)):
        part = sweep_plot.loc[sweep_plot.map_arm.eq(arm)]
        ax_c.plot(part.margin_mad, part.map_pass / part.anchor_pass, marker="o", color=color, label=arm.title())
    ax_c.set(xlabel="Support margin (section MAD)", ylabel="Fraction retaining support", ylim=(0, 1), xticks=[0, .25, .5, .75])
    ax_c.legend(frameon=False, fontsize=7)
    ax_c.set_title("C  Dependence on the support margin", loc="left", weight="bold")

    calibration = sim_calibration.loc[sim_calibration.metric.eq("false_loss") & sim_calibration.margin_mad.eq(.5)
                                       & ~sim_calibration.map_name.eq("identity_control")].copy()
    calibration["label"] = calibration.estimator.map({"anchor_retention": "Retention", "strict_computational_interface": "Adjacency"})
    calibration["x"] = calibration.label.map({"Retention": 0, "Adjacency": 1}) + calibration.map_name.map({"reference_shift": -.16, "alternative_shift": .16})
    for name, color in (("reference_shift", navy), ("alternative_shift", orange)):
        part = calibration.loc[calibration.map_name.eq(name)]
        ax_d.bar(part.x, part.rate, width=.28, color=color, label=name.replace("_", " ").title())
        for row in part.itertuples():
            ax_d.text(row.x, row.rate + .012, f"{row.numerator}/{row.denominator}", ha="center", fontsize=7)
    ax_d.set(xticks=[0, 1], xticklabels=["Retention", "Adjacency"], ylim=(0, .55),
             ylabel="Conditional false-loss fraction")
    ax_d.legend(frameon=False, fontsize=7)
    ax_d.set_title("D  Known-boundary simulation", loc="left", weight="bold")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "figure1.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "figure1.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = sim_summary.loc[(sim_summary.target.eq("contrast")) & (sim_summary.map_name.ne("identity_control"))]
    print(f"Wrote nine SI tables and Figure 1; shifted-map coverage min={summary.coverage.min():.2f}")


if __name__ == "__main__":
    main()
