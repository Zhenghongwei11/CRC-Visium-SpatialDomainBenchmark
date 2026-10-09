#!/usr/bin/env python3
"""Draw feature incidence, stromal overlap and paired patient measurements."""
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

BLUE = "#0072B2"
ORANGE = "#D55E00"
GENES = ["TGFB1", "CXCL12", "ACTA2", "TAGLN"]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--figures-only", action="store_true",
                   help="Render the existing plotted tables without replacing any scientific inputs")
    args = p.parse_args(); result = args.results; pack = args.package
    source = pack / "tables/figure_source_data"
    if args.figures_only:
        incidence = read(source / "figureS8_A_feature_incidence.tsv", sep="\t")
        overlaps = read(source / "figureS8_B_patient_overlap.tsv", sep="\t")
        patients = read(source / "figureS8_C_D_patient_excess.tsv", sep="\t")
    else:
        if result is None:
            p.error("--results is required unless --figures-only is specified")
        if json.loads((result / "independent_verification.json").read_text())["status"] != "pass":
            raise ValueError("Complete independent verification is required before presentation")
        pairs = read(result / "paired_map_results.tsv", sep="\t", float_precision="round_trip")
        patients = read(result / "patient_summary.tsv", sep="\t", float_precision="round_trip")
        patients = patients[patients.frame.eq("common_configurations")]
        incidence = []; input_files = [result / n for n in ["paired_map_results.tsv", "patient_summary.tsv", "independent_verification.json"]]
        for path in sorted((result / "sections").glob("*/*/feature_universe.tsv.gz")):
            input_files.append(path); f = read(path, sep="\t", float_precision="round_trip")
            for gene in GENES:
                incidence.append(dict(resource=path.parents[1].name, sample_id=path.parent.name,
                                      gene_symbol=gene, allowed_PCA_feature=bool(f.loc[f.gene_symbol.eq(gene), "allowed_PCA_rank"].notna().any()),
                                      withheld_PCA_feature=bool(f.loc[f.gene_symbol.eq(gene), "withheld_PCA_rank"].notna().any())))
        incidence = pd.DataFrame(incidence)
        overlap = pairs[pairs.stromal_domain_Jaccard.notna()]
        cells = overlap.groupby(["resource", "patient_id", "sample_id", "method_id", "K"]).stromal_domain_Jaccard.median()
        sections = cells.groupby(["resource", "patient_id", "sample_id"]).median()
        overlaps = sections.groupby(["resource", "patient_id"]).median().rename("stromal_domain_Jaccard").reset_index()
        counts = overlap.groupby(["resource", "patient_id"]).size().rename("n_paired_maps").reset_index()
        overlaps = overlaps.merge(counts)
        source = pack / "tables/figure_source_data"; source.mkdir(parents=True, exist_ok=True)
        incidence.to_csv(source / "figureS8_A_feature_incidence.tsv", sep="\t", index=False)
        overlaps.to_csv(source / "figureS8_B_patient_overlap.tsv", sep="\t", index=False)
        patients.to_csv(source / "figureS8_C_D_patient_excess.tsv", sep="\t", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    # Use the manuscript's final width, so all explicit text sizes remain
    # at least 8 pt after insertion. Extra height separates patient labels
    # and the shared legend without removing any attempted patients.
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 6.6), gridspec_kw={"height_ratios": [1, 1.2]})
    ax = axes[0, 0]
    for offset, resource, color in [(-.18, "Valdeolivas", BLUE), (.18, "GSE294385", ORANGE)]:
        subset = incidence[incidence.resource.eq(resource)]
        total = subset.sample_id.nunique()
        counts = subset.groupby("gene_symbol").allowed_PCA_feature.sum().reindex(GENES)
        x = np.arange(4)+offset; y = 100*counts.to_numpy()/total
        ax.bar(x, y, width=.32, color=color, label=resource)
        for px, py, n in zip(x, y, counts):
            ax.annotate(f"{int(n)}/{total}", (px, py),
                        xytext=(0, 3 if offset < 0 else 12), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(4), GENES); ax.set_ylim(0, 135)
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("Sections with gene\nin PCA (%)")
    ax.set_title("A  Readout features in\n    the allowed fits", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -.20))
    ax = axes[0, 1]
    for x, resource, color in [(0, "Valdeolivas", BLUE), (1, "GSE294385", ORANGE)]:
        subset = overlaps[overlaps.resource.eq(resource)]
        y = subset.stromal_domain_Jaccard.to_numpy()
        jitter = np.linspace(-.11, .11, len(y))
        ax.scatter(x+jitter, y, color=color, s=25, zorder=3)
        if len(y):
            ax.plot([x, x], [y.min(), y.max()], color=".35", lw=1)
            ax.plot([x-.15, x+.15], [np.median(y)]*2, color="black", lw=2)
    ax.set_xticks([0, 1], [f"Valdeolivas\n({overlaps.resource.eq('Valdeolivas').sum()} patients)",
                          f"GSE294385\n({overlaps.resource.eq('GSE294385').sum()} patients)"])
    ax.set_ylim(-.03, 1.06); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_ylabel("Retained-stroma\nJaccard overlap")
    ax.set_title("B  Stromal overlap across fits", loc="left", fontsize=9)
    yvalues = patients[["count_only_excess_mad", "block_matched_excess_mad"]].to_numpy().ravel()
    yvalues = yvalues[np.isfinite(yvalues)]
    limits = (min(-.05, yvalues.min()-.08), max(.3, yvalues.max()+.13))
    for ax, resource, label in [(axes[1, 0], "Valdeolivas", "C"), (axes[1, 1], "GSE294385", "D")]:
        subset = patients[patients.resource.eq(resource)]
        identifiers = sorted(subset.patient_id.unique(), key=lambda n: int(n[7:]) if n.startswith("Patient") else n)
        ticks = []
        for i, patient in enumerate(identifiers):
            rows = subset[subset.patient_id.eq(patient)].set_index("arm")
            n = int(rows.n_evaluable_maps.iloc[0]); ticks.append(patient.replace("Patient", "Patient ")+f"\n(n={n})")
            for shift, metric, color in [(-.14, "count_only_excess_mad", BLUE), (.14, "block_matched_excess_mad", ORANGE)]:
                y = [rows.loc[arm, metric] for arm in ["allowed", "withheld"]]
                x = [i+shift-.055, i+shift+.055]
                if np.isfinite(y).all():
                    ax.plot(x, y, color=color, lw=.8, alpha=.7)
                    for px, py, marker in zip(x, y, ["o", "s"]):
                        ax.scatter(px, py, marker=marker, s=29, facecolor="white", edgecolor=color, linewidth=.95, zorder=4)
            if n == 0:
                ax.text(i, limits[0]+.04, "*", ha="center", color=".35")
        ax.axhline(0, color=".5", lw=.7, ls="--"); ax.set_ylim(*limits)
        ax.set_xticks(np.arange(len(identifiers)), ticks)
        plt.setp(ax.get_xticklabels(), rotation=55 if resource == "Valdeolivas" else 45,
                 ha="right", rotation_mode="anchor")
        ax.set_ylabel("Excess absolute departure\n(MAD units)")
        ax.set_title(label + "  " + resource + " paired selections", loc="left", fontsize=9)
    handles = [Line2D([], [], color=color, marker=marker, markerfacecolor="white", linestyle="none", label=label)
               for color, marker, label in [(BLUE, "o", "Size-matched, allowed"), (BLUE, "s", "Size-matched, withheld"),
                                             (ORANGE, "o", "Location-matched, allowed"), (ORANGE, "s", "Location-matched, withheld")]]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(.55, .035), columnspacing=1.4, handletextpad=.5)
    fig.subplots_adjust(left=.12, right=.985, bottom=.23, top=.94, wspace=.50, hspace=.60)
    figures = pack / "figures"; figures.mkdir(exist_ok=True)
    fig.savefig(figures / "figureS8.pdf")
    fig.savefig(figures / "figureS8.png", dpi=600)
    fig.savefig(figures / "figureS8.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    if args.figures_only:
        print("Rendered Figure S8; scientific source tables retained")
        return
    metadata = dict(generator_sha256=sha(__file__), inputs={str(f.relative_to(result)): sha(f) for f in input_files},
                    plotted_files={f.name: sha(f) for f in sorted(source.glob("figureS8_*.tsv"))},
                    figures={f.name: sha(f) for f in sorted(figures.glob("figureS8.*"))})
    (source / "figureS8_source_manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print("Figure S8 and complete plotted source tables exported")


if __name__ == "__main__":
    main()
