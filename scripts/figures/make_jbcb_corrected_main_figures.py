#!/usr/bin/env python3
"""Plot regional expression comparisons and known-signal interval results.

Reads the completed scientific tables and writes the plotted rows and display
transformations beside each figure. Configurations reuse the same patients;
simulation distributions summarize conditions with known regional targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from jbcb_presentation_paths import read
from jbcb_presentation_paths import default_package, default_source_root, portable_label


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = default_package(__file__)
RESOURCES = ("GSE294385", "Valdeolivas")
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#6E7276"
RESOURCE_STYLE = {"GSE294385": (BLUE, "o"), "Valdeolivas": (GREEN, "s")}
ESTIMATOR_STYLE = {
    "anchor_retention": (BLUE, "o", "Stromal retention"),
    "strict_computational_interface": (ORANGE, "^", "Stricter adjacency"),
}
MAP_LABELS = {
    "identity_control": "True segmentation",
    "reference_shift": "Reference map displacement",
    "alternative_shift": "Alternative map displacement",
}
PURITY_THRESHOLDS = (0.0, 0.50, 0.75)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure() -> None:
    # Preserve the existing figure set's sans-serif font and embed TrueType text.
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def save_figure(fig: plt.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", metadata={"Creator": "Scientific figure"})
    fig.savefig(out / f"{stem}.png", dpi=300)
    fig.savefig(out / f"{stem}.tiff", dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def export_rows(frame: pd.DataFrame, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / f"{name}.tsv", sep="\t", index=False, na_rep="NA")


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", weight="bold", pad=12)


def patient_order(frame: pd.DataFrame, resource: str) -> pd.DataFrame:
    data = frame.loc[frame.resource.eq(resource)].copy()
    if resource == "GSE294385":
        data["order"] = data.patient_id.str.extract(r"(\d+)")[0].astype(int)
        data = data.sort_values("order").drop(columns="order")
    else:
        data = data.sort_values("patient_id")
    return data.reset_index(drop=True)


def patient_panel(ax: plt.Axes, frame: pd.DataFrame, resource: str,
                  letter: str, sources: Path, stem: str) -> None:
    data = patient_order(frame, resource)
    eligible = data.n_evaluable_maps.gt(0)
    data["display_status"] = np.where(eligible, "evaluable", "not_evaluable")
    data["display_order"] = np.arange(len(data))
    data["axis_label"] = [
        patient.replace("Patient", "Patient ") + ("*" if not has_maps else "")
        for patient, has_maps in zip(data.patient_id, eligible, strict=True)
    ]
    export_rows(data, sources, f"{stem}_panel_{letter}")
    count = "count_only_excess_abs_deviation_mad"
    block = "block_matched_excess_abs_deviation_mad"
    for i, row in data.iterrows():
        if row.n_evaluable_maps > 0:
            ax.plot([row[count], row[block]], [i - .035, i + .035], color="#ABB2B8", lw=1.25, zorder=1)
            ax.scatter(row[count], i - .035, color=BLUE, marker="o", s=40,
                       edgecolors="white", linewidths=.45, zorder=3)
            ax.scatter(row[block], i + .035, color=ORANGE, marker="s", s=38,
                       edgecolors="white", linewidths=.45, zorder=3)
    ax.set(yticks=np.arange(len(data)), yticklabels=data.axis_label, xlim=(min(-.15, data[[count, block]].min().min() - .08), max(1.08, data[[count, block]].max().max() + .08)),
           ylim=(len(data) - .5, -.5), xlabel="Excess absolute departure\n(MAD units)")
    ax.axvline(0, color=GREY, lw=.8, ls="--", zorder=0)
    ax.grid(axis="x", color="#E8ECEF", lw=.5)
    panel_title(ax, letter, f"{resource} ({int(eligible.sum())}/{len(data)} patients)")


def diagnostic_figure(patient: pd.DataFrame, purity: pd.DataFrame,
                       estimator: str, figures: Path, sources: Path, stem: str) -> None:
    data = patient.loc[patient.estimator_id.eq(estimator)].copy()
    pure = purity.loc[
        purity.estimator_id.eq(estimator) & purity.purity_min.isin(PURITY_THRESHOLDS)
    ].copy()
    if set(pure.purity_min.astype(float)) != set(PURITY_THRESHOLDS):
        raise ValueError(f"Purity summaries must contain exactly {PURITY_THRESHOLDS} for {estimator}")
    no_maps = pure.n_retained_maps.eq(0)
    pure["map_count_display_status"] = np.where(no_maps, "observed_zero", "observed_count")
    pure["deviation_display_status"] = np.where(no_maps, "not_evaluable", "evaluable")
    fig = plt.figure(figsize=(8.0, 8.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1),
                           left=.13, right=.97, bottom=.12, top=.85,
                           wspace=.39, hspace=.53)
    axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(2)]
    for ax, resource, letter in zip(axes[:2], RESOURCES, "AB", strict=True):
        patient_panel(ax, data, resource, letter, sources, stem)
    ax_c, ax_d = axes[2:]
    export_rows(pure.sort_values(["resource", "purity_min"]), sources, f"{stem}_panel_C")
    export_rows(pure.sort_values(["resource", "purity_min"]), sources, f"{stem}_panel_D")
    for resource in RESOURCES:
        rows = pure.loc[pure.resource.eq(resource)].sort_values("purity_min")
        color, marker = RESOURCE_STYLE[resource]
        x = rows.purity_min.to_numpy()
        counts = rows.n_retained_maps.to_numpy(dtype=float)
        ax_c.plot(x, counts, color=color, marker=marker, lw=1.35,
                  markersize=5, label=resource)
        for xx, nn in zip(x, counts, strict=True):
            if nn == 0:
                offset, alignment, vertical_offset = (-5, 7), "right", "bottom"
            elif nn <= 15:
                offset, alignment, vertical_offset = (5, 18), "left", "bottom"
            else:
                offset, alignment, vertical_offset = (4, 6), "left", "bottom"
            ax_c.annotate(str(int(nn)), (xx, nn), xytext=offset,
                          textcoords="offset points", ha=alignment, va=vertical_offset,
                          color=color, fontsize=9)
        medians = rows.median_block_matched_excess_abs_deviation_mad.to_numpy()
        medians = np.where(counts > 0, medians, np.nan)
        ax_d.scatter(x, medians, color=color, marker=marker, s=28, zorder=3)
        if stem == "figure2" and resource == "GSE294385":
            for xx, nn in zip(x, counts, strict=True):
                if nn == 0:
                    ax_d.annotate("No eligible\ncomparison", (xx, .07),
                                  xycoords=ax_d.get_xaxis_transform(),
                                  ha="center", va="bottom", color=color, fontsize=9)
    for ax in axes[2:]:
        ax.set(xticks=PURITY_THRESHOLDS, xticklabels=("0", "0.50", "0.75"),
               xlim=(-.045, .81), xlabel="Minimum whole-domain\nstromal fraction")
        ax.grid(axis="y", color="#E8ECEF", lw=.5)
    ax_c.set(ylim=(0, max(280, pure.n_retained_maps.max() * 1.12)), ylabel="Retained map settings (count)")
    ax_c.set_yticks((0, 50, 100, 150, 200, 250))
    ax_c.legend(frameon=False, loc="upper right")
    ax_d.set(ylim=(min(-.015, pure.median_block_matched_excess_abs_deviation_mad.min() - .025), max(.37, pure.median_block_matched_excess_abs_deviation_mad.max() + .04)), ylabel="Map-median excess departure\nafter location matching (MAD units)")
    ax_d.axhline(0, color=GREY, lw=.8, ls="--")
    panel_title(ax_c, "C", "Maps retained")
    panel_title(ax_d, "D", "Location-matched excess")
    estimator_label = "Stromal selection and regional contrast" if estimator == "anchor_retention" else "Stricter domain adjacency"
    fig.suptitle(estimator_label, x=.13, y=.975, ha="left", weight="bold", fontsize=12)
    fig.legend(handles=[
        Line2D([], [], color=BLUE, marker="o", ls="", markersize=5, label="Size-matched reference"),
        Line2D([], [], color=ORANGE, marker="s", ls="", markersize=5, label="Location-matched reference"),
    ], loc="upper left", bbox_to_anchor=(.12, .945), ncol=2, frameon=False, columnspacing=2)
    save_figure(fig, figures, stem)


def conditional_box(ax: plt.Axes, values: np.ndarray, position: float,
                    color: str, width: float = .45) -> None:
    ax.boxplot(values, positions=[position], widths=width, whis=(0, 100),
               patch_artist=True, showfliers=False,
               boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1},
               medianprops={"color": color, "linewidth": 1.4},
               whiskerprops={"color": color, "linewidth": .8},
               capprops={"color": color, "linewidth": .8})


def simulation_figure(summary: pd.DataFrame, thresholds: pd.DataFrame,
                      review: dict, figures: Path, sources: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.0, 5.4))
    fig.subplots_adjust(left=.13, right=.97, top=.81, bottom=.18, wspace=.34)
    targets = (("anchor", "Full bands"), ("contrast", "Selected domain"), ("paired", "Paired change"))
    source_a = []
    for position, (target, label) in enumerate(targets):
        all_rows = summary.loc[summary.target.eq(target)].copy()
        good = all_rows.loc[all_rows.evaluable.gt(0)].copy()
        cal = review["calibration"][target]
        if len(good) != cal["evaluable_groups"]:
            raise ValueError(f"Coverage condition count differs from frozen review: {target}")
        values = good.coverage.to_numpy()
        conditional_box(ax_a, values, position, BLUE)
        jitter = np.random.default_rng(73 + position).uniform(-.18, .18, len(good))
        good["display_x"] = position + jitter
        all_rows["display_category"] = label
        all_rows = all_rows.merge(good[["case_id", "estimator", "map_name", "display_x"]],
                                  on=["case_id", "estimator", "map_name"], how="left", validate="one_to_one")
        for gate, color, marker in (("pass", BLUE, "o"), ("fail", ORANGE, "x")):
            mask = good.screening_gate.eq(gate)
            ax_a.scatter(good.loc[mask, "display_x"], good.loc[mask, "coverage"],
                         color=color, marker=marker, s=11, alpha=.55, linewidths=.65, zorder=3)
        all_rows["screening_criteria_status"] = all_rows.screening_gate.map({
            "pass": "criteria_met", "fail": "criteria_not_met", "not_evaluable": "not_evaluable",
        }).fillna("not_evaluable")
        all_rows["screening_criteria_not_met_n"] = int(cal["gate_counts"].get("fail", 0))
        all_rows["screening_criteria_evaluable_n"] = int(cal["evaluable_groups"])
        all_rows["screening_criteria_attempted_n"] = int(cal["condition_groups"])
        all_rows["screening_criteria_not_evaluable_n"] = int(cal["gate_counts"].get(
            "not_evaluable", cal["condition_groups"] - cal["evaluable_groups"]))
        degenerate = all_rows.paired_degenerate_n.eq(all_rows.attempted) if target == "paired" else pd.Series(False, index=all_rows.index)
        relevant_n = int((~degenerate).sum())
        all_rows["coverage_denominator_status"] = np.where(degenerate, "reference_vs_itself", "nondegenerate_target")
        all_rows["coverage_relevant_groups_n"] = relevant_n
        all_rows["coverage_excluded_degenerate_groups_n"] = int(degenerate.sum())
        ax_a.text(position, .73, f"n={len(good)}/{relevant_n}",
                  ha="center", va="top", fontsize=10, color=GREY)
        source_a.append(all_rows)
    export_rows(pd.concat(source_a, ignore_index=True), sources, "figure3_panel_A")
    ax_a.axhline(.95, color=GREY, ls="--", lw=.85, zorder=0)
    ax_a.set(xticks=np.arange(3), xticklabels=["Full\nbands", "Selected\ndomain", "Paired\nchange"],
             xlim=(-.5, 2.5), ylim=(-.025, 1.055), ylabel="95% interval coverage\n(fraction)")
    ax_a.set_yticks((0, .25, .5, .75, 1.0))
    ax_a.grid(axis="y", color="#E8ECEF", lw=.5)
    panel_title(ax_a, "A", "Interval coverage")
    ax_a.legend(handles=[
        Line2D([], [], color=BLUE, marker="o", ls="", markersize=4, label="All statistical criteria met"),
        Line2D([], [], color=ORANGE, marker="x", ls="", markersize=4, label="At least one criterion not met"),
        Line2D([], [], color=GREY, ls="--", lw=.85, label="Nominal 0.95"),
    ], loc="lower left", frameon=False, fontsize=9.5)

    # The protocol's false_loss conditions additionally require both known
    # population contrasts to retain the same direction and exceed the margin.
    # Ordinary loss lacks that truth gate and is a different frozen statistic.
    loss = thresholds.loc[thresholds.metric.eq("false_loss") & thresholds.margin_mad.eq(.5)].copy()
    loss["display_status"] = np.where(loss.denominator.gt(0), "evaluable", "not_evaluable")
    loss["display_map_label"] = loss.map_name.map(MAP_LABELS)
    loss["display_x"] = np.nan
    for position, map_name in enumerate(MAP_LABELS):
        annotations = []
        for side, estimator in enumerate(ESTIMATOR_STYLE):
            color, marker, label = ESTIMATOR_STYLE[estimator]
            selected = loss.map_name.eq(map_name) & loss.estimator.eq(estimator)
            good = loss.loc[selected & loss.denominator.gt(0)]
            center = position + (-.17 if side == 0 else .17)
            conditional_box(ax_b, good.rate.to_numpy(), center, color, width=.24)
            jitter = np.random.default_rng(111 + 2 * position + side).uniform(-.075, .075, len(good))
            loss.loc[good.index, "display_x"] = center + jitter
            ax_b.scatter(center + jitter, good.rate, color=color, marker=marker,
                         s=13, alpha=.6, edgecolors="white", linewidths=.25, zorder=3)
            annotations.append(str(len(good)))
        ax_b.text(position, 1.042, " / ".join(annotations), ha="center", va="bottom", fontsize=9, color=GREY)
    export_rows(loss.sort_values(["map_name", "estimator", "case_id"]), sources, "figure3_panel_B")
    ax_b.set(xticks=np.arange(3), xticklabels=["True\nsegmentation", "Reference\nshift", "Alternative\nshift"],
             xlim=(-.5, 2.5), ylim=(-.045, 1.11), ylabel="Conditional support loss\n(fraction)")
    ax_b.tick_params(axis="x", labelsize=10, pad=5)
    ax_b.set_yticks((0, .25, .5, .75, 1.0))
    ax_b.grid(axis="y", color="#E8ECEF", lw=.5)
    panel_title(ax_b, "B", "Map construction and support")
    fig.legend(handles=[Line2D([], [], color=color, marker=marker, ls="", markersize=4, label=label)
                        for color, marker, label in ESTIMATOR_STYLE.values()],
               loc="upper center", bbox_to_anchor=(.55, .93), ncol=2, frameon=False)
    fig.suptitle("Interval calibration and support across conditions", x=.10, y=.985,
                 ha="left", fontsize=12, weight="bold")
    save_figure(fig, figures, "figure3")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--source-root", type=Path, help="Workspace containing the recalculated scientific outputs")
    args = parser.parse_args()
    pack_dir = args.pack_dir.resolve()
    figures = pack_dir / "figures"
    sources = pack_dir / "tables/figure_source_data"
    configure()
    patient_path = pack_dir / "tables/primary_patient_summary.tsv"
    purity_path = pack_dir / "tables/primary_stromal_fraction_summary.tsv"
    source_root = args.source_root or default_source_root(pack_dir, __file__)
    simulation_path = pack_dir / "tables"
    calibration = read(simulation_path / "simulation_summary.tsv", sep="\t")
    review = {"calibration": {target: {"evaluable_groups": int(((calibration.target == target) & (calibration.evaluable > 0)).sum()),
                                       "gate_counts": calibration.loc[calibration.target.eq(target),"screening_gate"].value_counts().to_dict(),
                                       "condition_groups": int(calibration.target.eq(target).sum())}
                              for target in ["anchor", "contrast", "paired"]}}
    patient = read(patient_path, sep="\t")
    purity = read(purity_path, sep="\t")
    summary = read(simulation_path / "simulation_summary.tsv", sep="\t")
    thresholds = read(simulation_path / "simulation_threshold_sensitivity.tsv", sep="\t")
    for estimator, stem in (("anchor_retention", "figure2"),
                             ("strict_computational_interface", "figureS2")):
        diagnostic_figure(patient, purity, estimator, figures, sources, stem)
    simulation_figure(summary, thresholds, review, figures, sources)
    # Scientific captions are maintained with the corrected manuscript.
    manifest = {
        "script": "scripts/" + Path(__file__).name,
        "input_sha256": {portable_label(path, pack_dir, source_root): digest(path)
                         for path in (patient_path, purity_path,
                                      simulation_path / "simulation_summary.tsv",
                                      simulation_path / "simulation_threshold_sensitivity.tsv")},
        "figures": {
            "figure2": {"A": "GSE294385 patient stromal-retention paired controls",
                        "B": "Valdeolivas patient stromal-retention paired controls",
                        "C": "Purity-retained map counts", "D": "Purity-specific map-median block excess"},
            "figureS2": {"A": "GSE294385 stricter adjacency patient paired controls",
                         "B": "Valdeolivas stricter adjacency patient paired controls",
                         "C": "Purity-retained map counts", "D": "Purity-specific map-median block excess"},
            "figure3": {"A": "Condition-specific coverage and combined statistical criteria",
                        "B": "Conditional support loss at 0.50 MAD by map construction"},
        },
        "export": {"pdf": "vector, TrueType font embedded", "png_dpi": 300, "tiff_dpi": 600,
                   "width_inches": 8.0, "axis_label_points": 11, "tick_label_points": 10,
                   "minimum_annotation_points": 9, "manuscript_width_inches": 6.3,
                   "printed_axis_label_points": 8.6625, "printed_tick_label_points": 7.875,
                   "printed_minimum_annotation_points": 7.0875},
        "inference": "Regional contrasts and matched-subset reference summaries; simulation estimates are condition-specific.",
        "source_data_sha256": {p.name: digest(p) for stem in ("figure2", "figure3", "figureS2")
                               for p in sorted(sources.glob(stem + "_panel_*.tsv"))},
    }
    (sources / "figure_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved figures 2, 3 and S2 with per-panel source data in {pack_dir}")


if __name__ == "__main__":
    main()
