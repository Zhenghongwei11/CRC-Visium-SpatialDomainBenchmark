#!/usr/bin/env python3
"""Show the registered tissue bands and one selected spot subset."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = "SN124_A938797_Rep2"
SOURCE = ROOT / "data/figure2_source"
OUT = ROOT / "docs/submissions/JBCB_REVISION_20260929/FIGURES"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    registered = pd.read_csv(SOURCE / "registered_spots.tsv", sep="\t")
    membership = pd.read_csv(SOURCE / "spot_membership.tsv.gz", sep="\t")
    chosen = membership.loc[(membership.estimator_id == "anchor_retention")
                            & (membership.method_id == "M0_expr_kmeans")
                            & (membership.K == 6)
                            & (membership.partition_id == "seed_11")].copy()
    spots = chosen.merge(registered[["barcode", "x_display", "y_display"]], on="barcode", validate="one_to_one")
    image = plt.imread(SOURCE / "tissue_lowres_image.png")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.7), constrained_layout=True)
    near, far = "#C15032", "#126F70"
    for ax in axes:
        ax.imshow(image)
        ax.set(xlim=(110, 540), ylim=(465, 45))
        ax.set_axis_off()
    for ax, selected in zip(axes, (False, True)):
        for band, color, label in (("near", near, "Near stroma"), ("far", far, "Far stroma")):
            mask = spots[f"morphology_{band}"].astype(bool)
            if selected:
                kept = mask & spots[f"computational_{band}"].astype(bool)
                excluded = mask & ~kept
                ax.scatter(spots.loc[excluded, "x_display"], spots.loc[excluded, "y_display"],
                           s=7, facecolors="none", edgecolors=color, linewidths=.55, alpha=.45)
                mask = kept
                label += " retained"
            ax.scatter(spots.loc[mask, "x_display"], spots.loc[mask, "y_display"],
                       s=12, c=color, alpha=.85, linewidths=0, label=label)
        ax.legend(loc="lower left", framealpha=.9, fontsize=8, markerscale=1.5)
    axes[0].set_title("A  Morphology-defined bands", loc="left", weight="bold")
    axes[1].set_title("B  Spots retained by one selected domain", loc="left", weight="bold")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "figure2.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_dir / "figure2.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
