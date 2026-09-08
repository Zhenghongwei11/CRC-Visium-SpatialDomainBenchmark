#!/usr/bin/env python3
"""Create cross-method boundary-reliability figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "cross_method_boundary"
OUT = ROOT / "figures" / "cross_method_boundary"

METHOD_ORDER = [
    "M0_expr_kmeans",
    "M1_spatial_concat_kmeans",
    "M2_spatial_ward",
    "M3_spatial_leiden",
    "M4_spagcn",
    "M5_stagate",
    "BayesSpace",
]
METHOD_SHORT = {
    "M0_expr_kmeans": "Expr. k-means",
    "M1_spatial_concat_kmeans": "Coord. k-means",
    "M2_spatial_ward": "Spatial Ward",
    "M3_spatial_leiden": "Spatial Leiden",
    "M4_spagcn": "SpaGCN-style",
    "M5_stagate": "STAGATE-style",
    "BayesSpace": "BayesSpace",
}
COLORS = {
    "M0_expr_kmeans": "#4E79A7",
    "M1_spatial_concat_kmeans": "#59A14F",
    "M2_spatial_ward": "#FB2",
    "M3_spatial_leiden": "#E15759",
    "M4_spagcn": "#B07AA1",
    "M5_stagate": "#76B7B2",
    "BayesSpace": "#FE75",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, weight="bold", fontsize=10)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def method_positions() -> tuple[np.ndarray, list[str], list[str]]:
    x = np.arange(len(METHOD_ORDER))
    labels = [METHOD_SHORT[m] for m in METHOD_ORDER]
    colors = [COLORS[m] for m in METHOD_ORDER]
    return x, labels, colors


def box_by_method(
    ax: plt.Axes,
    frame: pd.DataFrame,
    value: str,
    ylabel: str,
    *,
    title: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    x, labels, colors = method_positions()
    values = [
        frame.loc[frame["method_id"].eq(method), value].dropna().to_numpy()
        for method in METHOD_ORDER
    ]
    plot = ax.boxplot(
        values,
        positions=x,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202124", "linewidth": 1.2},
        whiskerprops={"color": "#6B7280"},
        capprops={"color": "#6B7280"},
    )
    for patch, color in zip(plot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#FFFFFF")
    for pos, vals in zip(x, values):
        if len(vals):
            jitter = np.linspace(-0.16, 0.16, len(vals))
            ax.scatter(np.repeat(pos, len(vals)) + jitter, vals, s=8, color="#263238", alpha=0.35, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#D1D5DB", lw=0.5, alpha=0.7)


def figure1() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    stages = [
        ("1  Global map", "Spatial coherence\nMarker separation", "#DCEAF7", "#4E79A7"),
        ("2  Repeatability", "Seeds or fixed graph\nperturbations", "#E3F1E8", "#59A14F"),
        ("3  Local disagreement", "Switching spots\nBoundary relation", "#FFF0DB", "#F28E2B"),
        ("4  Downstream\nrobustness", "Boundary/interior\nbiological summary", "#F7E2E3", "#E15759"),
    ]
    for i, (title, body, fill, edge) in enumerate(stages):
        x = 0.035 + i * 0.245
        ax.add_patch(plt.Rectangle((x, 0.47), 0.19, 0.30, facecolor=fill, edgecolor=edge, lw=1.4))
        ax.text(x + 0.095, 0.69, title, ha="center", va="center", weight="bold", fontsize=9)
        ax.text(x + 0.095, 0.56, body, ha="center", va="center", fontsize=8, linespacing=1.35)
        if i < 3:
            ax.annotate("", xy=(x + 0.235, 0.62), xytext=(x + 0.195, 0.62), arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#4B5563"})
    ax.text(0.5, 0.90, "One evidence sequence for every evaluated method", ha="center", weight="bold", fontsize=13)
    ax.text(0.5, 0.84, "13 colorectal cancer Visium sections  |  K = 4 and 6  |  failures retained", ha="center", color="#4B5563", fontsize=9)
    ax.text(0.04, 0.32, "Operational definitions", weight="bold", fontsize=9)
    definitions = [
        ("Reference partition", "seed 11, or the 6-neighbor graph for spatial Ward"),
        ("Plausible alternative", "seed 23/37, or a fixed 4-/8-neighbor graph perturbation"),
        ("Switching spot", "aligned label differs from the reference partition"),
        ("Boundary spot", "at least one differently labelled neighbor in the 6-neighbor graph"),
    ]
    positions = [(0.04, 0.25), (0.52, 0.25), (0.04, 0.17), (0.52, 0.17)]
    for (term, meaning), (x, y) in zip(definitions, positions):
        ax.text(x, y, term + ":", weight="bold", fontsize=7.4)
        ax.text(x, y - 0.035, meaning, fontsize=7.1, color="#374151")
    ax.text(
        0.5,
        0.055,
        "K = 4 and K = 6 are coarse and finer practical resolutions, not estimates of a true or optimal number of tissue domains.",
        ha="center",
        fontsize=7.5,
        color="#374151",
    )
    save(fig, "figure1")


def figure2() -> None:
    global_data = pd.read_csv(DATA / "global_characterization.tsv", sep="\t")
    stability = pd.read_csv(DATA / "stability_summary.tsv", sep="\t")
    coverage = pd.read_csv(DATA / "coverage_manifest.tsv", sep="\t")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2))
    for method in METHOD_ORDER:
        d = global_data[global_data["method_id"].eq(method)]
        axes[0, 0].scatter(
            d["spatial_coherence_median"],
            d["marker_coherence_median"],
            s=25,
            alpha=0.65,
            color=COLORS[method],
            label=METHOD_SHORT[method],
        )
    axes[0, 0].set_xlabel("Spatial coherence")
    axes[0, 0].set_ylabel("Marker separation")
    axes[0, 0].set_title("Global map characteristics", loc="left")
    axes[0, 0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4)
    axes[0, 0].grid(color="#E5E7EB", lw=0.5)
    panel(axes[0, 0], "A")

    box_by_method(axes[0, 1], stability, "median_ari", "Median ARI", title="Repeatability or perturbation stability", ylim=(0, 1.03))
    axes[0, 1].axhline(0.60, color="#6B7280", ls="--", lw=1)
    panel(axes[0, 1], "B")

    aggregate = stability.groupby("method_id").agg(n=("median_ari", "count"), below=("median_ari", lambda x: int((x < 0.60).sum()))).reindex(METHOD_ORDER)
    frac = aggregate["below"] / aggregate["n"]
    x, labels, colors = method_positions()
    axes[1, 0].bar(x, frac, color=colors, alpha=0.85)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=35, ha="right")
    axes[1, 0].set_ylabel("Fraction below ARI 0.60")
    axes[1, 0].set_ylim(0, 0.8)
    axes[1, 0].set_title("Descriptive instability screen", loc="left")
    axes[1, 0].grid(axis="y", color="#E5E7EB", lw=0.5)
    panel(axes[1, 0], "C")

    counts = coverage.assign(
        complete=coverage["status"].eq("complete").astype(int),
        partial=coverage["status"].eq("incomplete").astype(int),
        failed=coverage["status"].eq("failed").astype(int),
    ).groupby("method_id")[["complete", "partial", "failed"]].sum().reindex(METHOD_ORDER)
    bottom = np.zeros(len(counts))
    for column, color in [("complete", "#59A14F"), ("partial", "#F28E2B"), ("failed", "#E15759")]:
        axes[1, 1].bar(x, counts[column], bottom=bottom, color=color, label=column.capitalize())
        bottom += counts[column].to_numpy()
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1, 1].set_ylabel("Section-resolution settings")
    axes[1, 1].set_title("Retained computational coverage", loc="left")
    axes[1, 1].legend(frameon=False, ncol=3, loc="upper left")
    axes[1, 1].grid(axis="y", color="#E5E7EB", lw=0.5)
    panel(axes[1, 1], "D")
    fig.tight_layout(h_pad=2.1, w_pad=1.6)
    save(fig, "figure2")


def figure3() -> None:
    relation = pd.read_csv(DATA / "switching_boundary_relationship.tsv", sep="\t")
    histology = pd.read_csv(DATA / "histology_gradient_localization.tsv", sep="\t")
    switching = pd.read_csv(DATA / "switching_spots.tsv.gz", sep="\t")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.3))
    box_by_method(axes[0, 0], relation, "switching_fraction", "Switching fraction", title="Assignment changes")
    panel(axes[0, 0], "A")
    box_by_method(axes[0, 1], relation.replace([np.inf, -np.inf], np.nan), "risk_ratio", "Boundary risk ratio", title="Switching spots at inferred boundaries")
    axes[0, 1].axhline(1, color="#6B7280", ls="--", lw=1)
    axes[0, 1].set_ylim(0, min(8, np.nanquantile(relation["risk_ratio"].replace([np.inf, -np.inf], np.nan), 0.95) + 1))
    panel(axes[0, 1], "B")
    box_by_method(axes[1, 0], histology, "median_delta_switching_minus_stable", "Gradient difference", title="Image-gradient context")
    axes[1, 0].axhline(0, color="#6B7280", ls="--", lw=1)
    panel(axes[1, 0], "C")

    tr = switching[
        switching["sample_id"].str.contains("TR11_206")
        & switching["method_id"].eq("BayesSpace")
        & switching["K"].eq(4)
    ].copy()
    spot = tr.groupby(["barcode", "x", "y"], as_index=False).size()
    axes[1, 1].scatter(spot["x"], -spot["y"], c=spot["size"], cmap="magma", s=6, alpha=0.9, rasterized=True)
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_axis_off()
    axes[1, 1].set_title("TR11_206, BayesSpace, K=4\nspots switching in one or both alternatives", loc="left")
    panel(axes[1, 1], "D")
    fig.tight_layout(h_pad=2.1, w_pad=1.5)
    save(fig, "figure3")


def figure4() -> None:
    method_summary = pd.read_csv(DATA / "downstream_method_summary.tsv", sep="\t").set_index("method_id").reindex(METHOD_ORDER)
    detail = pd.read_csv(DATA / "downstream_contrasts.tsv", sep="\t")
    robust = pd.read_csv(DATA / "downstream_robustness_summary.tsv", sep="\t")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.3))
    x, labels, colors = method_positions()
    axes[0, 0].bar(x, method_summary["fraction_with_direction_reversal"], color=colors)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0, 0].set_ylabel("Fraction of feature-settings")
    axes[0, 0].set_title("True direction reversals", loc="left")
    axes[0, 0].grid(axis="y", color="#E5E7EB", lw=0.5)
    panel(axes[0, 0], "A")
    axes[0, 1].bar(x, method_summary["median_effect_range"], color=colors)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=35, ha="right")
    axes[0, 1].set_ylabel("Median max-min effect")
    axes[0, 1].set_title("Magnitude of partition sensitivity", loc="left")
    axes[0, 1].grid(axis="y", color="#E5E7EB", lw=0.5)
    panel(axes[0, 1], "B")

    select = detail[
        detail["sample_id"].str.contains("TR11_206")
        & detail["K"].eq(4)
        & detail["method_id"].isin(["BayesSpace", "M2_spatial_ward", "M4_spagcn", "M5_stagate"])
        & detail["feature_id"].isin(["EPCAM", "SIG_CAF_FAP", "SIG_EXCLUSION_CONTRAST"])
    ].copy()
    method_x = {"BayesSpace": 0, "M2_spatial_ward": 1, "M4_spagcn": 2, "M5_stagate": 3}
    markers = {"EPCAM": "o", "SIG_CAF_FAP": "s", "SIG_EXCLUSION_CONTRAST": "^"}
    feature_labels = {"EPCAM": "EPCAM", "SIG_CAF_FAP": "CAF/FAP", "SIG_EXCLUSION_CONTRAST": "Exclusion contrast"}
    for feature, marker in markers.items():
        for method, group in select[select["feature_id"].eq(feature)].groupby("method_id"):
            base_x = method_x[method]
            ordered = group.sort_values("partition_id")
            axes[1, 0].plot(
                base_x + np.linspace(-0.14, 0.14, len(ordered)),
                ordered["median_delta_boundary_minus_interior"],
                marker=marker,
                ms=4,
                lw=0.8,
                color=COLORS[method],
                alpha=0.9,
            )
    axes[1, 0].axhline(0, color="#6B7280", ls="--", lw=1)
    axes[1, 0].set_xticks(range(4))
    axes[1, 0].set_xticklabels([METHOD_SHORT[m] for m in method_x], rotation=25, ha="right")
    axes[1, 0].set_ylabel("Boundary - interior median")
    axes[1, 0].set_title("TR11_206, K=4: alternative partitions", loc="left")
    for feature, marker in markers.items():
        axes[1, 0].scatter([], [], marker=marker, color="#4B5563", label=feature_labels[feature])
    axes[1, 0].legend(frameon=False, ncol=1)
    panel(axes[1, 0], "C")

    tr = robust[robust["sample_id"].str.contains("TR11_206") & robust["any_direction_reversal_vs_reference"]]
    counts = tr.groupby("method_id").size().reindex(METHOD_ORDER, fill_value=0)
    block = tr.groupby("method_id")["any_block_direction_reversal_vs_reference"].sum().reindex(METHOD_ORDER, fill_value=0)
    width = 0.36
    axes[1, 1].bar(x - width / 2, counts, width, color=colors, alpha=0.85, label="Spot-level")
    axes[1, 1].bar(x + width / 2, block, width, color=colors, alpha=0.38, hatch="//", label="Also reversed by blocks")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1, 1].set_ylabel("Features with a reversal")
    axes[1, 1].set_title("TR11_206 across K=4 and K=6", loc="left")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(axis="y", color="#E5E7EB", lw=0.5)
    panel(axes[1, 1], "D")
    fig.tight_layout(h_pad=2.1, w_pad=1.5)
    save(fig, "figure4")


def supplementary() -> None:
    global_data = pd.read_csv(DATA / "global_characterization.tsv", sep="\t")
    stability = pd.read_csv(DATA / "stability_summary.tsv", sep="\t")
    relation = pd.read_csv(DATA / "switching_boundary_relationship.tsv", sep="\t")
    sensitivity = pd.read_csv(DATA / "computational_sensitivity_summary.tsv", sep="\t")
    switching = pd.read_csv(DATA / "switching_spots.tsv.gz", sep="\t")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    box_by_method(axes[0], global_data, "spatial_coherence_median", "Spatial coherence", title="Spatial coherence across section-resolution settings")
    box_by_method(axes[1], global_data, "marker_coherence_median", "Marker separation", title="Marker separation across section-resolution settings")
    fig.tight_layout(w_pad=1.4)
    save(fig, "figureS1")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, kval in zip(axes, [4, 6]):
        box_by_method(ax, stability[stability["K"].eq(kval)], "median_ari", "Median ARI", title=f"K={kval}")
        ax.axhline(0.60, color="#6B7280", ls="--", lw=1)
        ax.set_ylim(0, 1.03)
    fig.suptitle("Repeatability and graph-perturbation stability by practical resolution", y=1.02, fontsize=11, weight="bold")
    fig.tight_layout(w_pad=1.4)
    save(fig, "figureS2")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, kval in zip(axes, [4, 6]):
        box_by_method(ax, relation[relation["K"].eq(kval)].replace([np.inf, -np.inf], np.nan), "risk_ratio", "Boundary risk ratio", title=f"K={kval}")
        ax.axhline(1, color="#6B7280", ls="--", lw=1)
        ax.set_ylim(0, 8)
    fig.suptitle("Switching spots are enriched at inferred boundaries, with heterogeneous magnitude", y=1.02, fontsize=11, weight="bold")
    fig.tight_layout(w_pad=1.4)
    save(fig, "figureS3")

    methods = ["BayesSpace", "M2_spatial_ward", "M3_spatial_leiden", "M4_spagcn", "M5_stagate"]
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.3))
    for ax, method in zip(axes.flat, methods):
        d = switching[
            switching["sample_id"].str.contains("TR11_206")
            & switching["method_id"].eq(method)
            & switching["K"].eq(4)
        ]
        spot = d.groupby(["barcode", "x", "y"], as_index=False).size()
        ax.scatter(spot["x"], -spot["y"], c=spot["size"], cmap="magma", s=5, alpha=0.9, rasterized=True)
        ax.set_title(METHOD_SHORT[method], fontsize=8, pad=3)
        ax.set_aspect("equal")
        ax.set_axis_off()
    axes.flat[-1].set_axis_off()
    fig.suptitle("TR11_206, K=4: spots switching under each method's declared alternatives", y=0.99, fontsize=10, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.2, w_pad=0.8)
    save(fig, "figureS4")

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    factors = sensitivity.copy()
    keys = list(factors[["method_id", "sensitivity_factor"]].itertuples(index=False, name=None))
    x = np.arange(len(keys))
    ax.bar(x, factors["median_ari"], color=[COLORS[k[0]] for k in keys])
    ax.axhline(0.60, color="#6B7280", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{METHOD_SHORT[m]}\n{factor.replace('_', ' ')}" for m, factor in keys], rotation=25, ha="right")
    ax.set_ylabel("Median ARI versus reference")
    ax.set_ylim(0, 1.03)
    ax.set_title("Sensitivity to stochastic and graph specifications", loc="left")
    ax.grid(axis="y", color="#E5E7EB", lw=0.5)
    fig.tight_layout()
    save(fig, "figureS5")


def main() -> None:
    style()
    figure1()
    figure2()
    figure3()
    figure4()
    supplementary()
    print(f"Wrote cross-method figures to {OUT}")


if __name__ == "__main__":
    main()
