#!/usr/bin/env python3
"""
Publication-quality figures for the CRC Visium spatial-clustering benchmark.

Design rationale
================
* Restrained benchmark style: high information
  density, fine grid lines, Okabe–Ito palette, matched panel labelling.
* Every numeric value is read from the locked TSV tables – nothing is hard-coded.
* No "uninformative bar charts": paired dot-line for per-sample deltas, strip +
  box for distributions, forest-plot for CIs, spatial scatter for domain maps.
* Colour-blind safe (Okabe–Ito), print-safe line widths, Arial typography.

Outputs (overwrite in-place)
============================
  plots/publication/png/figure{1..4}.png  (300 dpi)
  plots/publication/pdf/figure{1..4}.pdf  (vector)
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
# Force a headless backend (required in CI / sandboxed environments).
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── colour palette (Okabe–Ito) ──────────────────────────────────────────────
OI = {
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermillion":     "#D55E00",
    "reddish_purple": "#CC79A7",
    "gray":           "#999999",
    "dark_gray":      "#4D4D4D",
    "black":          "#000000",
    "light_gray":     "#D9D9D9",
}

METHOD_COLOR = {
    "BayesSpace":                OI["blue"],
    "M0_expr_kmeans":            OI["gray"],
    "M1_spatial_concat_kmeans":  OI["vermillion"],
    "M2_spatial_ward":           OI["bluish_green"],
    "M3_spatial_leiden":         OI["reddish_purple"],
    "M4_spagcn":                 OI["orange"],
    "M5_stagate":                OI["sky_blue"],
}

METHOD_LABEL = {
    "BayesSpace":                "BayesSpace",
    "M0_expr_kmeans":            "Expr-only k-means",
    "M1_spatial_concat_kmeans":  "Spatial k-means",
    "M2_spatial_ward":           "Spatial Ward",
    "M3_spatial_leiden":         "Spatial Leiden",
    "M4_spagcn":                 "SpaGCN",
    "M5_stagate":                "STAGATE",
}

# Short sample labels for readability
def _short_sample(sid: str) -> str:
    """GSM9322957_TR11_206 → TR11_206"""
    parts = sid.split("_", 1)
    return parts[1] if len(parts) > 1 else sid


# ── global matplotlib style ─────────────────────────────────────────────────
def _set_style() -> None:
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        # Prefer Matplotlib-bundled fonts to avoid slow/fragile system font discovery in sandboxed runs.
        "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.default":   "regular",
        "axes.titlesize":     9,
        "axes.titleweight":   "bold",
        "axes.labelsize":     8,
        "xtick.labelsize":    7,
        "ytick.labelsize":    7,
        "legend.fontsize":    7,
        "legend.title_fontsize": 7.5,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.08,
        "axes.linewidth":     0.6,
        "xtick.major.width":  0.5,
        "ytick.major.width":  0.5,
        "xtick.major.size":   3,
        "ytick.major.size":   3,
        "xtick.minor.size":   1.5,
        "ytick.minor.size":   1.5,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          False,
        "pdf.fonttype":       42,     # TrueType in PDF (editable)
        "ps.fonttype":        42,
    })


def _panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.08, y: float = 1.06) -> None:
    """Bold panel label in Axes fraction coordinates."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")


def _despine(ax: mpl.axes.Axes, keep_left: bool = True, keep_bottom: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
    if not keep_bottom:
        ax.spines["bottom"].set_visible(False)


def _add_fine_grid(ax: mpl.axes.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, linewidth=0.3, color="#E0E0E0", zorder=0)
    ax.set_axisbelow(True)


def _save(fig: mpl.figure.Figure, root: Path, name: str) -> None:
    pdf_dir = root / "plots" / "publication" / "pdf"
    png_dir = root / "plots" / "publication" / "png"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    # Also save flat into plots/publication/ to match existing filenames
    fig.savefig(pdf_dir / f"{name}.pdf")
    fig.savefig(png_dir / f"{name}.png", dpi=300)
    fig.savefig(root / "plots" / "publication" / f"{name}.pdf")
    fig.savefig(root / "plots" / "publication" / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  ✓ {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1  –  Study overview + representative domain-map exemplars
# ═══════════════════════════════════════════════════════════════════════════════
def figure1(root: Path) -> None:
    ds = pd.read_csv(root / "results" / "dataset_summary.tsv", sep="\t")
    maps = pd.read_csv(root / "results" / "figures" / "fig1_domain_maps.tsv", sep="\t")

    sample_id = maps["sample_id"].iloc[0]
    k_val = int(maps["K"].iloc[0])

    fig = plt.figure(figsize=(7.2, 6.8))  # ~180 mm wide, Nature single-page
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[0.75, 1],
        wspace=0.30, hspace=0.45,
        left=0.08, right=0.96, top=0.93, bottom=0.06,
    )

    # ── Panel A: cohort coverage (horizontal lollipop, not bars) ─────────
    ax_a = fig.add_subplot(gs[0, :])
    ds_used = ds[ds["dataset_id"].isin(["GSE267401", "GSE311294", "GSE280318"])].copy()
    ds_used["total"]   = ds_used["samples_on_disk"].astype(int)
    ds_used["covered"] = ds_used["bayesspace_samples_covered"].astype(int)
    ds_used = ds_used.sort_values("total", ascending=True).reset_index(drop=True)

    y_pos = np.arange(len(ds_used))
    # background: total samples (thin line + open circle)
    ax_a.hlines(y_pos, 0, ds_used["total"], color=OI["light_gray"], linewidth=2.0, zorder=1)
    ax_a.scatter(ds_used["total"], y_pos, s=50, facecolors="white",
                 edgecolors=OI["dark_gray"], linewidths=1.0, zorder=3, label="Total samples")
    # foreground: BayesSpace covered (filled)
    ax_a.scatter(ds_used["covered"], y_pos, s=50, c=OI["blue"],
                 edgecolors="white", linewidths=0.4, zorder=4, label="BayesSpace-analysed")
    # connector lines
    for i, row in ds_used.iterrows():
        ax_a.hlines(i, row["covered"], row["total"], color=OI["sky_blue"],
                    linewidth=1.0, linestyle=":", zorder=2)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(ds_used["dataset_id"])
    ax_a.set_xlabel("Number of Visium samples")
    ax_a.set_title("CRC Visium cohorts and spatial-method coverage")
    ax_a.legend(frameon=False, loc="lower right", ncol=1, handletextpad=0.4)
    ax_a.set_xlim(-0.3, ds_used["total"].max() + 0.8)
    _add_fine_grid(ax_a, axis="x")
    _panel_label(ax_a, "A", x=-0.06)

    # ── Panels B–D: domain-map spatial scatter ───────────────────────────
    method_order = ["M0_expr_kmeans", "M1_spatial_concat_kmeans", "BayesSpace"]
    titles       = ["Expr-only k-means", "Spatial k-means", "BayesSpace"]
    labels       = ["B", "C", "D"]

    # build a unified colour map across K domains
    all_doms = sorted(maps["domain_label"].unique())
    # Use a discrete qualitative palette from Okabe–Ito extended
    dom_colors_list = [
        OI["blue"], OI["orange"], OI["bluish_green"],
        OI["vermillion"], OI["reddish_purple"], OI["sky_blue"],
        OI["yellow"], OI["dark_gray"],
    ]
    dom_cmap = {d: dom_colors_list[i % len(dom_colors_list)] for i, d in enumerate(all_doms)}

    for col_idx, (mid, ttl, lbl) in enumerate(zip(method_order, titles, labels)):
        ax = fig.add_subplot(gs[1, col_idx])
        sub = maps[maps["method_id"] == mid].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        colors = [dom_cmap[v] for v in sub["domain_label"]]
        ax.scatter(sub["x"], sub["y"], s=5.5, c=colors, linewidths=0,
                   rasterized=True, alpha=0.85)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(ttl, fontsize=8, pad=4)
        _panel_label(ax, lbl, x=-0.04, y=1.03)

    # add shared domain legend beneath panel D
    ax_last = fig.axes[-1]
    handles = [mlines.Line2D([], [], marker="o", linestyle="",
               color=dom_cmap[d], markersize=4, label=f"Domain {d}")
               for d in all_doms[:min(len(all_doms), 8)]]
    ax_last.legend(handles=handles, title="Spatial domains",
                   frameon=False, loc="upper right", ncol=2,
                   fontsize=6, title_fontsize=6.5, handletextpad=0.2,
                   columnspacing=0.6)

    _save(fig, root, "figure1")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2  –  Domain-quality benchmarking
# ═══════════════════════════════════════════════════════════════════════════════
def figure2(root: Path) -> None:
    deltas = pd.read_csv(root / "results" / "replication" /
                         "domain_quality_deltas_by_sample.tsv", sep="\t")
    gates  = pd.read_csv(root / "results" / "benchmarks" /
                         "statistical_gate_summary.tsv", sep="\t")

    # Common full-width figure limit is about 7.5 in (2250 px at 300 dpi). Because we use
    # savefig.bbox='tight', the rendered pixel width can exceed fig.width*dpi.
    # Keep Fig 2 slightly narrower to stay within the pixel constraint.
    fig = plt.figure(figsize=(6.25, 7.5))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[1, 0.8],
        wspace=0.38, hspace=0.50,
        left=0.18, right=0.95, top=0.94, bottom=0.08,
    )

    # ── Panel A: paired dot-line for Δ spatial coherence ─────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_paired_delta_panel(ax_a, deltas, "delta_spatial_coherence",
                            "Δ Spatial coherence\n(BayesSpace − baseline)")
    _panel_label(ax_a, "A", x=0.01, y=1.02)

    # ── Panel B: paired dot-line for Δ marker coherence ──────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    _plot_paired_delta_panel(ax_b, deltas, "delta_marker_coherence",
                            "Δ Marker coherence\n(BayesSpace − baseline)")
    _panel_label(ax_b, "B", x=0.01, y=1.02)

    # ── Panel C: forest-plot – claim-level effect sizes + CI ─────────────
    ax_c = fig.add_subplot(gs[1, :])
    c1 = gates[gates["claim_id"] == "C1_domain_quality"].copy()
    _plot_forest(ax_c, c1, null_value=0,
                 title="Pre-specified effect sizes (paired median Δ, 95% CI)")
    _panel_label(ax_c, "C", x=0.01, y=1.02)

    # Figure numbering is aligned to the analysis figure set.
    # Domain-quality summary appears after the workflow schematic, so it is Fig 3.
    _save(fig, root, "figure3")


def _plot_paired_delta_panel(ax: mpl.axes.Axes, deltas: pd.DataFrame,
                             col: str, xlabel: str) -> None:
    """Dot+segment plot of BayesSpace−baseline deltas per sample×K unit."""
    samples = sorted(deltas["sample_id"].unique(), key=_short_sample)
    y_map = {s: i for i, s in enumerate(samples)}
    # Visually separate K values while keeping a single tick per section.
    k_offsets = {4: -0.18, 6: 0.18}

    for baseline, marker, color in [
        ("M0_expr_kmeans",            "o", OI["gray"]),
        ("M1_spatial_concat_kmeans",  "D", OI["vermillion"]),  # diamond
        ("M2_spatial_ward",           "^", OI["bluish_green"]),
        ("M3_spatial_leiden",         "P", OI["reddish_purple"]),  # plus-filled
        ("M4_spagcn",                 "s", OI["orange"]),  # square
        ("M5_stagate",                "X", OI["sky_blue"]),
    ]:
        sub = deltas[deltas["baseline_method_id"] == baseline].copy()
        ys = [
            y_map[s] + k_offsets.get(int(k), 0.0)
            for s, k in zip(sub["sample_id"].tolist(), sub["K"].tolist())
        ]
        ax.scatter(sub[col], ys, s=22, marker=marker, c=color,
                   edgecolors="white", linewidths=0.3, alpha=0.9, zorder=4,
                   label=METHOD_LABEL.get(baseline, baseline))

    # connect baseline deltas per (sample, K) with a thin segment
    for (sid, k), grp in deltas.groupby(["sample_id", "K"]):
        vals = [float(v) for v in grp[col].values if np.isfinite(v)]
        if len(vals) >= 2:
            vals = sorted(vals)
            y = y_map[sid] + k_offsets.get(int(k), 0.0)
            ax.plot(vals, [y] * len(vals), color=OI["dark_gray"], linewidth=0.5,
                    alpha=0.45, zorder=2)

    ax.axvline(0, color=OI["black"], linewidth=0.7, linestyle="--", zorder=1)
    ax.set_yticks(range(len(samples)))
    ax.set_yticklabels([_short_sample(s) for s in samples], fontsize=6.5)
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.set_title(col.replace("delta_", "").replace("_", " ").title(),
                 fontsize=8, pad=6)
    ax.legend(frameon=False, loc="lower right", fontsize=6,
              handletextpad=0.3, markerscale=0.9)
    ax.text(
        0.02,
        0.02,
        "K=4 (lower)   K=6 (upper)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.5,
        color=OI["dark_gray"],
    )
    _add_fine_grid(ax, axis="x")
    # annotate n on top-right
    n = deltas["sample_id"].nunique()
    ax.text(0.98, 0.98, f"n = {n} samples", transform=ax.transAxes,
            ha="right", va="top", fontsize=6, color=OI["dark_gray"])


def _plot_forest(ax: mpl.axes.Axes, gate_df: pd.DataFrame,
                 null_value: float, title: str) -> None:
    """Forest plot (horizontal CI + point estimate) for prespecified comparisons."""
    gate_df = gate_df.copy().reset_index(drop=True)
    n_rows = len(gate_df)
    y = np.arange(n_rows)

    for i, row in gate_df.iterrows():
        est = float(row["effect_size_value"])
        lo  = float(row["effect_size_ci_lower"])
        hi  = float(row["effect_size_ci_upper"])
        q   = float(row["fdr"])

        # CI line
        ax.plot([lo, hi], [i, i], color=OI["dark_gray"], linewidth=1.8,
                solid_capstyle="round", zorder=2)
        # point estimate
        ax.scatter([est], [i], s=55, c=OI["blue"], edgecolors="white",
                   linewidths=0.6, zorder=4)
        # whisker caps
        cap_h = 0.15
        ax.plot([lo, lo], [i - cap_h, i + cap_h], color=OI["dark_gray"],
                linewidth=1.0, zorder=3)
        ax.plot([hi, hi], [i - cap_h, i + cap_h], color=OI["dark_gray"],
                linewidth=1.0, zorder=3)

        # annotation: effect size [CI], q
        comp = str(row.get("comparison_id", ""))
        metric = str(row["metric_id"]).replace("_median", "").replace("_", " ")
        baseline_short = comp.split("_vs_")[-1][:25] if "_vs_" in comp else comp
        label_txt = f"{metric}"
        ax.text(hi + 0.015, i + 0.05, f"Δ = {est:.3f}  [{lo:.3f}, {hi:.3f}]",
                fontsize=6, va="center", color=OI["dark_gray"])
        ax.text(hi + 0.015, i - 0.25, f"adjusted q = {q:.4f}",
                fontsize=5.5, va="center", color=OI["blue"],
                fontstyle="italic")

    ax.axvline(null_value, color=OI["black"], linewidth=0.7, linestyle="--", zorder=1)
    ax.set_yticks(y)
    y_labels = []
    for _, row in gate_df.iterrows():
        metric = str(row["metric_id"]).replace("_median", "").replace("_", " ").title()
        comp   = str(row.get("comparison_id", ""))
        base   = comp.split("_vs_")[-1] if "_vs_" in comp else ""
        # Compact comparison labels (avoid long "M5_stagate_K4K6" strings that
        # collide with the pass/fail badge column).
        base = (
            base.replace("_K4K6", "")
            .replace("_K4", "")
            .replace("_K6", "")
        )
        base   = METHOD_LABEL.get(base, base.replace("_", " "))
        y_labels.append(f"{metric}\nvs {base}")
    ax.set_yticklabels(y_labels, fontsize=6.5, linespacing=1.1)
    ax.set_xlabel("Paired median Δ (bootstrap 95% CI)", fontsize=7.5)
    ax.set_title(title, fontsize=8, pad=6)
    ax.invert_yaxis()
    _add_fine_grid(ax, axis="x")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3  –  Stability (multi-seed ARI)
# ═══════════════════════════════════════════════════════════════════════════════
def figure3(root: Path) -> None:
    stab  = pd.read_csv(root / "results" / "replication" /
                        "bayesspace_stability_by_sample.tsv", sep="\t")
    gates = pd.read_csv(root / "results" / "benchmarks" /
                        "statistical_gate_summary.tsv", sep="\t")
    c2 = gates[gates["claim_id"] == "C2_sensitivity"].iloc[0]

    fig = plt.figure(figsize=(7.2, 4.0))
    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.4, 1],
        wspace=0.40,
        # Extra bottom margin prevents x-label collisions after downscaling.
        left=0.10, right=0.95, top=0.90, bottom=0.28,
    )

    # ── Panel A: lollipop of ARI per sample×K with threshold ─────────────
    ax_a = fig.add_subplot(gs[0, 0])
    stab = stab.sort_values(["dataset_id", "sample_id", "K"]).reset_index(drop=True)

    # Plot by sample (single tick per section) with K-specific offsets to avoid label collisions.
    k_colors = {4: OI["sky_blue"], 6: OI["orange"]}
    k_offsets = {4: -0.18, 6: 0.18}

    sample_order = stab[["dataset_id", "sample_id"]].drop_duplicates().reset_index(drop=True)
    for i, row in sample_order.iterrows():
        sid = row["sample_id"]
        for k in (4, 6):
            sub = stab[(stab["sample_id"] == sid) & (stab["K"].astype(int) == k)]
            if sub.empty:
                continue
            ari = float(sub["stability_ari_median"].iloc[0])
            col = k_colors.get(k, OI["gray"])
            x = i + k_offsets.get(k, 0.0)
            ax_a.vlines(x, 0, ari, colors=col, linewidth=1.5, zorder=2)
            ax_a.scatter(x, ari, s=32, c=col, edgecolors="white",
                         linewidths=0.4, zorder=4)

    # threshold line
    ax_a.axhline(0.60, color=OI["vermillion"], linewidth=1.0, linestyle="--",
                 zorder=1)
    ax_a.set_xticks(range(len(sample_order)))
    ax_a.set_xticklabels(
        [_short_sample(s) for s in sample_order["sample_id"].tolist()],
        rotation=60,
        ha="right",
        fontsize=5.5,
        rotation_mode="anchor",
    )
    ax_a.set_ylabel("Adjusted Rand Index (median)")
    ax_a.set_ylim(0, 1.05)
    ax_a.set_title("Multi-seed stability per sample × K", fontsize=8, pad=6)
    _add_fine_grid(ax_a, axis="y")
    # n annotation
    ax_a.text(0.98, 0.02, f"n = {len(stab)} sample×K combinations",
              transform=ax_a.transAxes, ha="right", va="bottom",
              fontsize=5.5, color=OI["dark_gray"])

    # Compact legend (K colors + threshold) – avoid overlapping x labels.
    k_handles = [
        mlines.Line2D([], [], marker="o", linestyle="", color=k_colors[k], markersize=5, label=f"K = {k}")
        for k in sorted(k_colors)
    ]
    th_handle = mlines.Line2D([], [], color=OI["vermillion"], linestyle="--",
                              linewidth=1.0, label="Threshold (0.60)")
    ax_a.legend(handles=k_handles + [th_handle], frameon=False, loc="upper left",
                fontsize=6, handletextpad=0.2, title="Resolution", title_fontsize=6.5)

    _panel_label(ax_a, "A")

    # ── Panel B: forest-plot for stability gate ──────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    est = float(c2["effect_size_value"])
    lo  = float(c2["effect_size_ci_lower"])
    hi  = float(c2["effect_size_ci_upper"])
    q   = float(c2["fdr"])

    ax_b.plot([lo, hi], [0, 0], color=OI["dark_gray"], linewidth=2.0,
              solid_capstyle="round", zorder=2)
    ax_b.scatter([est], [0], s=65, c=OI["blue"], edgecolors="white",
                 linewidths=0.8, zorder=4)
    # caps
    cap = 0.12
    ax_b.plot([lo, lo], [-cap, cap], color=OI["dark_gray"], lw=1.0, zorder=3)
    ax_b.plot([hi, hi], [-cap, cap], color=OI["dark_gray"], lw=1.0, zorder=3)
    # threshold
    ax_b.axvline(0.60, color=OI["vermillion"], linewidth=1.0, linestyle="--",
                 zorder=1)
    ax_b.text(0.60, 0.32, "threshold", fontsize=5.5, color=OI["vermillion"],
              ha="center", va="bottom")
    # annotation
    ax_b.text(est, -0.25,
              f"median ARI = {est:.3f}\n[{lo:.3f}, {hi:.3f}]\nadjusted q = {q:.4f}",
              fontsize=6, ha="center", va="top", color=OI["dark_gray"],
              linespacing=1.3)

    ax_b.set_yticks([0])
    ax_b.set_yticklabels(["Median ARI\n(across sample×K)"], fontsize=6.5)
    ax_b.set_xlabel("Bootstrap 95% CI", fontsize=7.5)
    ax_b.set_title("Stability estimate (with prespecified threshold)", fontsize=8, pad=6)
    ax_b.set_ylim(-0.6, 0.6)
    _add_fine_grid(ax_b, axis="x")
    _panel_label(ax_b, "B")

    # Stability appears after domain-quality results, so it is Fig 4.
    _save(fig, root, "figure4")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4  –  Compute feasibility
# ═══════════════════════════════════════════════════════════════════════════════
def figure4(root: Path) -> None:
    rt    = pd.read_csv(root / "results" / "benchmarks" / "runtime_memory.tsv", sep="\t")
    # Keep compute summaries aligned with the primary CRC inference set.
    crc_datasets = {"GSE267401", "GSE311294", "GSE280318"}
    rt = rt[rt["dataset_id"].isin(crc_datasets)].copy()
    # Convert per-run runtimes to per sample×K medians to keep the runtime
    # distribution panel comparable across methods (and comparable to the
    # BayesSpace per-unit medians in method_benchmark.tsv).
    rt_ok = rt[(rt["status"] == "success") & np.isfinite(rt["wall_time_sec"])].copy()
    rt_unit = (
        rt_ok.groupby(["method_id", "dataset_id", "sample_id", "K"], as_index=False)
        .agg(wall_time_sec=("wall_time_sec", "median"))
    )
    mb_path = root / "results" / "benchmarks" / "method_benchmark_locked.tsv"
    if not mb_path.exists():
        mb_path = root / "results" / "benchmarks" / "method_benchmark.tsv"
    mb = pd.read_csv(mb_path, sep="\t")
    mb = mb[mb["dataset_id"].isin(crc_datasets)].copy()
    gates = pd.read_csv(root / "results" / "benchmarks" /
                        "statistical_gate_summary.tsv", sep="\t")
    c3 = gates[gates["claim_id"] == "C3_compute_feasibility"].iloc[0]

    fig = plt.figure(figsize=(7.2, 4.2))
    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.5, 1],
        wspace=0.40,
        left=0.10, right=0.95, top=0.90, bottom=0.18,
    )

    # ── Panel A: strip + box for runtime by method ───────────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    # Collect runtimes per method group
    groups = [
        ("Expr-only\nk-means",   rt_unit[rt_unit["method_id"] == "M0_expr_kmeans"]["wall_time_sec"].values,
         OI["gray"]),
        ("Spatial\nk-means",     rt_unit[rt_unit["method_id"] == "M1_spatial_concat_kmeans"]["wall_time_sec"].values,
         OI["vermillion"]),
        ("Spatial\nWard",        rt_unit[rt_unit["method_id"] == "M2_spatial_ward"]["wall_time_sec"].values,
         OI["bluish_green"]),
        ("Spatial\nLeiden",      rt_unit[rt_unit["method_id"] == "M3_spatial_leiden"]["wall_time_sec"].values,
         OI["reddish_purple"]),
        ("SpaGCN",               rt_unit[rt_unit["method_id"] == "M4_spagcn"]["wall_time_sec"].values,
         OI["orange"]),
        ("STAGATE",              rt_unit[rt_unit["method_id"] == "M5_stagate"]["wall_time_sec"].values,
         OI["sky_blue"]),
        ("BayesSpace",           mb[mb["method_id"] == "BayesSpace"]["wall_time_sec_median"].values,
         OI["blue"]),
    ]

    positions = np.arange(len(groups))
    for pos, (label, vals, color) in zip(positions, groups):
        vals = np.array([v for v in vals if np.isfinite(v)], dtype=float)
        if len(vals) == 0:
            continue
        # box (narrow, no outlier markers – points shown as strip)
        bp = ax_a.boxplot(
            [vals], positions=[pos], widths=0.35, vert=True,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=mpl.colors.to_rgba(color, 0.20),
                          edgecolor=color, linewidth=0.8),
            whiskerprops=dict(color=color, linewidth=0.7),
            capprops=dict(color=color, linewidth=0.7),
            medianprops=dict(color=OI["black"], linewidth=1.0),
        )
        # strip jitter
        rng = np.random.RandomState(42)
        jitter = (rng.rand(len(vals)) - 0.5) * 0.18
        ax_a.scatter(np.full(len(vals), pos) + jitter, vals,
                     s=8, c=color, alpha=0.55, edgecolors="none", zorder=5)
        # NOTE: Avoid per-group n annotations under the x-axis; these routinely collide
        # with multi-line tick labels after downscaling.

    ax_a.set_xticks(positions)
    ax_a.set_xticklabels([g[0] for g in groups], fontsize=7)
    ax_a.set_yscale("log")
    ax_a.set_ylabel("Wall-clock time (s, log scale)", fontsize=7.5)
    ax_a.set_title("Runtime distribution (local compute)", fontsize=8, pad=6)
    _add_fine_grid(ax_a, axis="y")
    ax_a.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:.0f}" if v >= 1 else f"{v:.2f}"))
    _panel_label(ax_a, "A")

    # ── Panel B: forest plot for compute gate ────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    est = float(c3["effect_size_value"])
    lo  = float(c3["effect_size_ci_lower"])
    hi  = float(c3["effect_size_ci_upper"])

    ax_b.plot([lo, hi], [0, 0], color=OI["dark_gray"], linewidth=2.0,
              solid_capstyle="round", zorder=2)
    ax_b.scatter([est], [0], s=65, c=OI["blue"], edgecolors="white",
                 linewidths=0.8, zorder=4)
    cap = 0.12
    ax_b.plot([lo, lo], [-cap, cap], color=OI["dark_gray"], lw=1.0, zorder=3)
    ax_b.plot([hi, hi], [-cap, cap], color=OI["dark_gray"], lw=1.0, zorder=3)

    # 30-min threshold (scope-limited context)
    ax_b.axvline(1800, color=OI["vermillion"], linewidth=1.0, linestyle="--", zorder=1)
    ax_b.text(1800, 0.32, "30-min limit", fontsize=5.5, color=OI["vermillion"],
              ha="center", va="bottom")

    ax_b.text(est, -0.25,
              f"median = {est:.1f} s\n[{lo:.1f}, {hi:.1f}]\nlocal hardware",
              fontsize=6, ha="center", va="top", color=OI["dark_gray"],
              linespacing=1.3)

    ax_b.set_yticks([0])
    ax_b.set_yticklabels(["Median runtime\n(BayesSpace)"], fontsize=6.5)
    ax_b.set_xlabel("Bootstrap 95% CI (seconds)", fontsize=7.5)
    ax_b.set_title("Compute summary (with prespecified limit)", fontsize=8, pad=6)
    ax_b.set_ylim(-0.6, 0.6)
    _add_fine_grid(ax_b, axis="x")
    _panel_label(ax_b, "B")

    # Compute feasibility appears after stability, so it is Fig 5.
    _save(fig, root, "figure5")


# ═══════════════════════════════════════════════════════════════════════════════
def main() -> int:
    root = Path(__file__).resolve().parent.parent
    _set_style()
    print("Generating publication figures …")
    figure1(root)
    figure2(root)
    figure3(root)
    figure4(root)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
