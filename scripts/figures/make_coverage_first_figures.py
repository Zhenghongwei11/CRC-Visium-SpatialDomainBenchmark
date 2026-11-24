#!/usr/bin/env python3
"""Render coverage and measurement figures from archived or regenerated tables."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import file_digest, sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd
from PIL import Image

from make_jbcb_corrected_tissue_figures import scalebar

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "supplementary_data/SUPPLEMENTARY_SOURCE_DATA.zip"
IMAGE = ROOT / "data/tissue_images/main/tissue_lowres_image.png"
NEAR, FAR, TUMOR = "#D47A25", "#2878A8", "#746A83"
RESOURCES = {"Valdeolivas": ("#0072B2", "o"), "GSE294385": ("#D55E00", "s")}
SOURCES = {
    "coverage": "pure_domain_selection/coverage.tsv",
    "baseline": "primary_stromal_fraction_summary.tsv",
    "baseline90": "pure_domain_selection/original_purity90_denominators.tsv",
    "spots": "figure_source_data/figure2_sn124_spots.tsv",
    "scale": "figure_source_data/figure2_sn124_scale.json",
    "gse": "figure_source_data/figure2_gse294385_patient_excess.tsv",
    "val": "figure_source_data/figure2_valdeolivas_patient_excess.tsv",
}
REPRODUCED = {
    "coverage": "pure_domains/coverage.tsv",
    "baseline": "baseline/tables/primary_stromal_fraction_summary.tsv",
    "baseline90": "original_purity90_denominators.tsv",
    "spots": "baseline/tables/figure_source_data/figure1_spots.tsv",
    "scale": "baseline/tables/figure_source_data/figure1_scale.json",
    "gse": "baseline/tables/figure_source_data/figure2_panel_A.tsv",
    "val": "baseline/tables/figure_source_data/figure2_panel_B.tsv",
}


def hash_file(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def load_inputs(archive, supporting, reproduction):
    if archive:
        with ZipFile(archive) as zipped:
            entries = {key: f"SUPPORTING_INFORMATION/{path}" for key, path in SOURCES.items()}
            contents = {key: zipped.read(path) for key, path in entries.items()}
    else:
        paths = REPRODUCED if reproduction else SOURCES
        folder = reproduction or supporting
        entries = {key: str(folder / path) for key, path in paths.items()}
        contents = {key: Path(path).read_bytes() for key, path in entries.items()}
    extracted = {
        key: json.loads(blob) if key == "scale" else pd.read_csv(BytesIO(blob), sep="\t")
        for key, blob in contents.items()
    }
    identity = {key: {"path": entries[key], "sha256": sha256(blob).hexdigest()}
                for key, blob in contents.items()}
    return extracted, identity


def save(fig, output, name):
    paths = [output / f"{name}.{ext}" for ext in ("pdf", "png", "tiff")]
    fig.savefig(paths[0], dpi=450, metadata={"Creator": "Scientific figure", "CreationDate": None})
    fig.savefig(paths[1], dpi=300)
    fig.savefig(paths[2], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return {path.name: hash_file(path) for path in paths}


def coverage_figure(data, output):
    coverage = data["coverage"]
    pure = coverage.loc[coverage.whole_domain_stroma_fraction.ge(.9)]
    passing = pure.loc[pure.primary_evaluable]
    if passing.empty or not passing.anatomical_evaluable.all():
        raise ValueError("Coverage-qualified selections must have eligible complete bands")
    per_resource = Counter(passing.resource)
    absent_complete_bands = int((~pure.anatomical_evaluable).sum())
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.45),
                                      gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=.068, right=.972, bottom=.25, top=.79, wspace=.36)

    labels = ["Fine-clustering settings", "≥90% labeled stroma", "Adequate coverage in both bands"]
    counts = [len(coverage), len(pure), len(passing)]
    colors = ["#BBC8D5", "#7395AE", "#176B88"]
    for y, (label, count, color) in enumerate(zip(labels, counts, colors)):
        left.barh(y, count, height=.28, color=color)
        left.text(0, y-.23, label, ha="left", va="bottom")
        left.annotate(str(count), (count, y), xytext=(5, 0), textcoords="offset points",
                      va="center", fontweight="bold")
    left.set_title("A   Fine clustering and merging", loc="left", fontweight="bold", pad=22)
    left.set(xlabel="Section-by-setting count", xlim=(0, max(counts)*1.16),
             ylim=(2.35, -.46), yticks=[])
    left.spines["left"].set_visible(False)
    left.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    left.text(0, -.28, f"{per_resource['Valdeolivas']} Valdeolivas, {per_resource['GSE294385']} GSE294385\n"
                       f"Complete bands fell short in {absent_complete_bands} of {len(pure)} high-agreement settings.",
              transform=left.transAxes, ha="left", va="top", fontsize=7.3, linespacing=1.5)

    baseline = data["baseline"]
    baseline = baseline.loc[
        baseline.graph_id.eq("historical_raw_6nn")
        & baseline.estimator_id.isin(["stromal_retention", "anchor_retention"])
        & baseline.setting_id.eq("baseline")
        & baseline.eligibility_frame.eq("independent_eligible")
        & baseline.purity_min.isin([0, .5, .75])
    ]
    baseline90 = data["baseline90"].set_index("resource")
    for resource, (color, marker) in RESOURCES.items():
        series = baseline.loc[baseline.resource.eq(resource)].set_index("purity_min")
        values = [int(series.at[threshold, "n_retained_patients"]) for threshold in (0, .5, .75)]
        values.append(int(baseline90.at[resource, "n_retained_patients"]))
        right.plot([0, 50, 75, 90], values, color=color, marker=marker, label=resource, lw=1.6,
                   linestyle="-" if resource == "Valdeolivas" else "--",
                   markerfacecolor="white" if resource == "GSE294385" else color)
    right.set_title("B   Baseline maps", loc="left", fontweight="bold", pad=22)
    right.set(xlabel="Minimum percentage labeled stroma", ylabel="Patients with a comparison",
              xlim=(-4, 95), ylim=(-.25, 5.55), xticks=[0, 50, 75, 90], yticks=range(6))
    right.grid(axis="y", alpha=.20)
    right.set_axisbelow(True)
    right.legend(loc="lower left", fontsize=7.3, handlelength=2, frameon=False)
    return save(fig, output, "figure1")


def tissue_panel(ax, image, spots, scale, retained):
    width, height = image.size
    ax.imshow(image)
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis("off")
    tumor = spots.coarse_label.eq("tumor")
    ax.scatter(spots.loc[tumor, "x_display"], spots.loc[tumor, "y_display"], s=8,
               color=TUMOR, edgecolors="white", linewidths=.24)
    for band, color in (("near", NEAR), ("far", FAR)):
        full = spots[f"morphology_{band}"]
        kept = spots[f"retained_{band}"]
        if (kept & ~full).any():
            raise ValueError("Retained spot outside its complete band")
        chosen = kept if retained else full
        if retained:
            missing = full & ~kept
            ax.scatter(spots.loc[missing, "x_display"], spots.loc[missing, "y_display"], s=10,
                       facecolors="none", edgecolors=color, linewidths=.60)
        ax.scatter(spots.loc[chosen, "x_display"], spots.loc[chosen, "y_display"], s=9.5,
                   color=color, edgecolors="white", linewidths=.24)
    prefix = "retained" if retained else "morphology"
    label = "B   Retained in the domain" if retained else "A   Complete stromal bands"
    n_near, n_far = (int(spots[f"{prefix}_{band}"].sum()) for band in ("near", "far"))
    ax.set_title(f"{label}\n{n_near} near / {n_far} far", loc="left", fontweight="bold", pad=6)
    scalebar(ax, width, height, float(scale["scale_bar_image_pixels"]),
             int(scale["nominal_scale_bar_um"]))


def patient_panel(ax, table, resource, letter, limits):
    ordered = table.sort_values("display_order")
    ax.axvline(0, color="#7F8790", lw=.8, zorder=0)
    for y, row in enumerate(ordered.itertuples()):
        if row.display_status != "evaluable":
            continue
        size = row.count_only_excess_abs_deviation_mad
        location = row.block_matched_excess_abs_deviation_mad
        ax.plot([size, location], [y, y], lw=1.2, color="#81919D", zorder=1)
        ax.plot(size, y, "o", color="#475969", markerfacecolor="white", markersize=5, zorder=2)
        ax.plot(location, y, "s", color="#0072B2", markersize=4.7, zorder=3)
    labels = [label.rstrip("*") + ("*" if status != "evaluable" else "")
              for label, status in zip(ordered.axis_label, ordered.display_status)]
    ax.set(xlim=limits, ylim=(len(labels)-.45, -.55),
           yticks=range(len(labels)), yticklabels=labels)
    ax.set_title(f"{letter}   {resource}", loc="left", fontweight="bold", pad=7)
    ax.set_xlabel("Excess departure (stromal MAD)")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="x", lw=.6, alpha=.21)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=7)


def measurement_figure(data, image_path, output):
    spots, scale = data["spots"], data["scale"]
    if hash_file(image_path) != scale["image_sha256"]:
        raise ValueError("Tissue image differs from registered plotting source")
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    fig = plt.figure(figsize=(7.2, 6.65))
    grid = fig.add_gridspec(3, 2, height_ratios=[2.65, .46, 1.75],
                           left=.115, right=.972, bottom=.122, top=.90,
                           wspace=.38, hspace=.30)
    tissue_panel(fig.add_subplot(grid[0, 0]), image, spots, scale, False)
    tissue_panel(fig.add_subplot(grid[0, 1]), image, spots, scale, True)
    legend = fig.add_subplot(grid[1, :])
    legend.axis("off")
    handles = [Line2D([], [], ls="", marker="o", color=color, label=label)
               for color, label in ((TUMOR, "Annotated tumor"), (NEAR, "Near"), (FAR, "Far"))]
    handles.append(Line2D([], [], ls="", marker="o", markerfacecolor="none",
                          markeredgecolor="#55616B", label="Unretained band spot"))
    legend.legend(handles=handles, loc="center", ncol=4, fontsize=7.5,
                  frameon=False, handletextpad=.4, columnspacing=1.3)

    patients = [data["gse"], data["val"]]
    cols = ["count_only_excess_abs_deviation_mad", "block_matched_excess_abs_deviation_mad"]
    values = pd.concat([table[cols] for table in patients]).stack()
    spread = values.max() - values.min()
    limits = min(0, float(values.min())) - spread*.065, max(0, float(values.max())) + spread*.067
    patient_panel(fig.add_subplot(grid[2, 0]), patients[0], "GSE294385", "C", limits)
    patient_panel(fig.add_subplot(grid[2, 1]), patients[1], "Valdeolivas", "D", limits)
    fig.legend(handles=[
        Line2D([], [], ls="", marker="o", color="#475969", markerfacecolor="white", label="Matched spot count"),
        Line2D([], [], ls="", marker="s", color="#0072B2", label="Matched spot count and location"),
    ], loc="lower center", bbox_to_anchor=(.53, .016), ncol=2, frameon=False, fontsize=8)
    return save(fig, output, "figure2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path, help="Release supplementary source-data ZIP")
    source.add_argument("--supporting-information", type=Path, help="Unpacked source directory")
    source.add_argument("--reproduction", type=Path, help="Existing full reproduction output directory")
    parser.add_argument("--image", type=Path, default=IMAGE)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    archive = args.archive
    if not (archive or args.supporting_information or args.reproduction):
        archive = ARCHIVE
    frames, digests = load_inputs(archive, args.supporting_information, args.reproduction)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    coverage_exports = coverage_figure(frames, args.output)
    measurement_exports = measurement_figure(frames, args.image, args.output)
    receipt = {
        "figure1": {"sources": {k: digests[k] for k in ("coverage", "baseline", "baseline90")},
                    "exports": coverage_exports},
        "figure2": {"sources": {k: digests[k] for k in ("spots", "scale", "gse", "val")},
                    "exports": measurement_exports},
    }
    receipt["figure2"]["sources"]["image"] = {"path": str(args.image), "sha256": hash_file(args.image)}
    (args.output / "MAIN_FIGURE_SOURCES.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
