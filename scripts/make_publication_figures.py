#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "gray": "#7F7F7F",
}

METHOD_COLORS = {
    "BayesSpace": OKABE_ITO["blue"],
    "M0_expr_kmeans": OKABE_ITO["gray"],
    "M1_spatial_concat_kmeans": OKABE_ITO["vermillion"],
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def panel_label(ax: plt.Axes, label: str, x_offset: float = -0.15) -> None:
    ax.text(
        x_offset,
        1.02,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=12,
    )


def save_figure(fig: plt.Figure, repo_root: Path, figure_name: str) -> None:
    out_pdf = repo_root / "plots" / "publication" / f"{figure_name}.pdf"
    out_png = repo_root / "plots" / "publication" / f"{figure_name}.png"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def fig1(repo_root: Path) -> None:
    ds = pd.read_csv(repo_root / "results" / "dataset_summary.tsv", sep="\t")
    maps = pd.read_csv(repo_root / "results" / "figures" / "fig1_domain_maps.tsv", sep="\t")
    sample_id = maps["sample_id"].iloc[0]
    k_val = int(maps["K"].iloc[0])

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], wspace=0.25, hspace=0.30)

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    # Panel A: dataset coverage
    ds_used = ds[ds["dataset_id"].isin(["GSE267401", "GSE311294", "GSE285505"])].copy()
    ds_used["total_samples"] = ds_used["samples_on_disk"].astype(int)
    ds_used["bayesspace_samples"] = ds_used["bayesspace_samples_covered"].astype(int)
    ds_used = ds_used.sort_values("dataset_id")

    x = np.arange(len(ds_used))
    ax_a.bar(x, ds_used["total_samples"], color=OKABE_ITO["sky_blue"], label="Total samples (downloaded)")
    ax_a.bar(
        x,
        ds_used["bayesspace_samples"],
        color=OKABE_ITO["blue"],
        label="BayesSpace-covered samples",
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(ds_used["dataset_id"].tolist())
    ax_a.set_ylabel("Sample count")
    ax_a.set_title("Cohort coverage for the CRC Visium benchmark")
    ax_a.legend(frameon=False, ncol=2, loc="upper right")
    panel_label(ax_a, "A", x_offset=-0.045)

    # Panels B/C: representative domain maps (two baselines + BayesSpace)
    def plot_map(ax: plt.Axes, method_id: str, label: str, show_legend: bool = False) -> None:
        sub = maps[maps["method_id"] == method_id].copy()
        sub = sub.sort_values("domain_label")
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        # Use a stable colormap; assign colors by label id.
        uniq = sorted(sub["domain_label"].unique())
        cmap = plt.get_cmap("tab20", len(uniq))
        color_idx = {lab: i for i, lab in enumerate(uniq)}
        colors = [cmap(color_idx[v]) for v in sub["domain_label"].to_numpy()]
        ax.scatter(sub["x"], sub["y"], s=5.5, c=colors, linewidths=0)
        if show_legend:
            handles = []
            for lab in uniq[: min(8, len(uniq))]:
                handles.append(
                    mpl.lines.Line2D([0], [0], marker="o", color="w", label=f"{lab}", markerfacecolor=cmap(color_idx[lab]), markersize=5)
                )
            ax.legend(handles=handles, title="Domain", frameon=False, ncol=4, loc="upper right")

    plot_map(ax_b, "M0_expr_kmeans", "Expression-only k-means (baseline)", show_legend=False)
    plot_map(ax_c, "M1_spatial_concat_kmeans", "Expression + coordinates k-means (baseline)", show_legend=False)
    plot_map(ax_d, "BayesSpace", "BayesSpace (spatial clustering)", show_legend=False)
    panel_label(ax_b, "B")
    panel_label(ax_c, "C")
    panel_label(ax_d, "D")

    fig.suptitle(
        f"Figure 1. Dataset coverage and representative domain maps (sample {sample_id}, K={k_val})",
        y=0.99,
        fontsize=11,
    )
    save_figure(fig, repo_root, "figure1")


def fig2(repo_root: Path) -> None:
    deltas = pd.read_csv(repo_root / "results" / "replication" / "domain_quality_deltas_by_sample.tsv", sep="\t")
    gates = pd.read_csv(repo_root / "results" / "benchmarks" / "statistical_gate_summary.tsv", sep="\t")

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], wspace=0.28, hspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # Panel A: delta spatial coherence per sample/K vs baseline
    for baseline, marker, color in [
        ("M0_expr_kmeans", "o", OKABE_ITO["gray"]),
        ("M1_spatial_concat_kmeans", "s", OKABE_ITO["vermillion"]),
        ("M2_spatial_ward", "^", OKABE_ITO["bluish_green"]),
    ]:
        sub = deltas[deltas["baseline_method_id"] == baseline].copy()
        ax_a.scatter(
            sub["delta_spatial_coherence"],
            sub["sample_id"],
            s=18,
            marker=marker,
            c=color,
            alpha=0.9,
            label=baseline,
        )
    ax_a.axvline(0, color=OKABE_ITO["black"], linewidth=0.8)
    ax_a.set_xlabel("Delta spatial coherence (BayesSpace - baseline)")
    ax_a.set_ylabel("")
    ax_a.set_title("Per-sample deltas (spatial coherence)")
    ax_a.legend(frameon=False, loc="lower right")
    panel_label(ax_a, "A")

    # Panel B: delta marker coherence per sample/K vs baseline
    for baseline, marker, color in [
        ("M0_expr_kmeans", "o", OKABE_ITO["gray"]),
        ("M1_spatial_concat_kmeans", "s", OKABE_ITO["vermillion"]),
        ("M2_spatial_ward", "^", OKABE_ITO["bluish_green"]),
    ]:
        sub = deltas[deltas["baseline_method_id"] == baseline].copy()
        ax_b.scatter(
            sub["delta_marker_coherence"],
            sub["sample_id"],
            s=18,
            marker=marker,
            c=color,
            alpha=0.9,
            label=baseline,
        )
    ax_b.axvline(0, color=OKABE_ITO["black"], linewidth=0.8)
    ax_b.set_xlabel("Delta marker coherence (BayesSpace - baseline)")
    ax_b.set_ylabel("")
    ax_b.set_title("Per-sample deltas (marker coherence)")
    panel_label(ax_b, "B")

    # Panel C: claim-level summary effect sizes with bootstrap CI (from gate table)
    c1 = gates[gates["claim_id"] == "C1_domain_quality"].copy()
    c1 = c1[c1["metric_id"].isin(["spatial_coherence_median", "marker_coherence_median"])].copy()
    metric_order = ["spatial_coherence_median", "marker_coherence_median"]
    y = np.arange(len(metric_order))
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(["Spatial coherence", "Marker coherence"])
    ax_c.set_xlabel("Paired median delta (bootstrap 95% CI)")
    ax_c.set_title("Claim-level summary (prespecified gates; BH-FDR within family)")

    for idx, metric in enumerate(metric_order):
        row = c1[c1["metric_id"] == metric].iloc[0]
        est = float(row["effect_size_value"])
        lo = float(row["effect_size_ci_lower"])
        hi = float(row["effect_size_ci_upper"])
        ax_c.plot([lo, hi], [idx, idx], color=OKABE_ITO["black"], linewidth=1.2)
        ax_c.scatter([est], [idx], c=OKABE_ITO["blue"], s=40, zorder=3)
        ax_c.text(
            hi + 0.01,
            idx,
            f"q={float(row['fdr']):.3f}",
            va="center",
            fontsize=8,
            color=OKABE_ITO["black"],
        )
    ax_c.axvline(0, color=OKABE_ITO["black"], linewidth=0.8, linestyle="--")
    panel_label(ax_c, "C")

    fig.suptitle("Figure 2. Domain-quality benchmarking: per-sample heterogeneity and effect sizes", y=0.99, fontsize=11)
    save_figure(fig, repo_root, "figure2")


def fig3(repo_root: Path) -> None:
    stability = pd.read_csv(repo_root / "results" / "replication" / "bayesspace_stability_by_sample.tsv", sep="\t")
    gates = pd.read_csv(repo_root / "results" / "benchmarks" / "statistical_gate_summary.tsv", sep="\t")
    c2 = gates[gates["claim_id"] == "C2_sensitivity"].iloc[0]

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Panel A: ARI median per sample/K with threshold
    stability = stability.sort_values(["dataset_id", "sample_id", "K"])
    x = np.arange(len(stability))
    ax_a.scatter(x, stability["stability_ari_median"], c=OKABE_ITO["blue"], s=35)
    ax_a.axhline(0.60, color=OKABE_ITO["vermillion"], linewidth=1.2, linestyle="--", label="Gate threshold (ARI=0.60)")
    ax_a.set_ylim(0, 1.0)
    ax_a.set_ylabel("Adjusted Rand index (median across seed pairs)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"{r.sample_id}\nK{int(r.K)}" for r in stability.itertuples()], rotation=90, ha="center")
    ax_a.set_title("Multiseed stability under the prespecified configuration")
    ax_a.legend(frameon=False, loc="upper right")
    panel_label(ax_a, "A")

    # Panel B: claim-level summary (from gate table)
    est = float(c2["effect_size_value"])
    lo = float(c2["effect_size_ci_lower"])
    hi = float(c2["effect_size_ci_upper"])
    q = float(c2["fdr"])
    ax_b.plot([lo, hi], [0, 0], color=OKABE_ITO["black"], linewidth=1.4)
    ax_b.scatter([est], [0], c=OKABE_ITO["blue"], s=55, zorder=3)
    ax_b.axvline(0.60, color=OKABE_ITO["vermillion"], linewidth=1.2, linestyle="--")
    ax_b.set_yticks([0])
    ax_b.set_yticklabels(["Median ARI"])
    ax_b.set_xlabel("Bootstrap 95% CI")
    ax_b.set_title(f"Gate summary (q={q:.3f})")
    panel_label(ax_b, "B")

    fig.suptitle("Figure 3. BayesSpace stability (multiseed) and claim gate summary", y=0.99, fontsize=11)
    save_figure(fig, repo_root, "figure3")


def fig4(repo_root: Path) -> None:
    mb = pd.read_csv(repo_root / "results" / "benchmarks" / "method_benchmark.tsv", sep="\t")
    rt = pd.read_csv(repo_root / "results" / "benchmarks" / "runtime_memory.tsv", sep="\t")
    gates = pd.read_csv(repo_root / "results" / "benchmarks" / "statistical_gate_summary.tsv", sep="\t")
    c3 = gates[gates["claim_id"] == "C3_compute_feasibility"].iloc[0]

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Panel A: runtime distributions (baseline from runtime_memory; BayesSpace from method_benchmark)
    bayes = mb[mb["method_id"] == "BayesSpace"].copy()
    bayes_fast = bayes[bayes["notes"].astype(str).str.contains("rigor-backfill", na=False)].copy()
    if not bayes_fast.empty and "seed_count" in bayes_fast.columns:
        # Prefer the most rigorous/updated row per sample×K (typically the largest seed_count).
        bayes_fast = (
            bayes_fast.sort_values(["dataset_id", "sample_id", "K", "seed_count"])
            .drop_duplicates(subset=["dataset_id", "sample_id", "K"], keep="last")
            .copy()
        )
    bayes_other = bayes[~bayes.index.isin(bayes_fast.index)]

    baseline_rt = rt[rt["method_id"].isin(["M0_expr_kmeans", "M1_spatial_concat_kmeans"])].copy()
    baseline_rt["group"] = baseline_rt["method_id"].map({
        "M0_expr_kmeans": "Expression-only k-means",
        "M1_spatial_concat_kmeans": "Spatial-augmented k-means"
    })
    bayes_fast = bayes_fast.assign(group="BayesSpace (Primary)")
    bayes_other = bayes_other.assign(group="BayesSpace (Replicates)")

    rows = []
    for df in [baseline_rt.rename(columns={"wall_time_sec": "wall_time_sec_median"}), bayes_fast, bayes_other]:
        for r in df.itertuples():
            val = float(getattr(r, "wall_time_sec_median"))
            if math.isfinite(val):
                rows.append({"group": getattr(r, "group"), "runtime_sec": val})
    plot = pd.DataFrame(rows)

    order = ["Expression-only k-means", "Spatial-augmented k-means", "BayesSpace (Primary)", "BayesSpace (Replicates)"]
    colors = {
        "Expression-only k-means": METHOD_COLORS["M0_expr_kmeans"],
        "Spatial-augmented k-means": METHOD_COLORS["M1_spatial_concat_kmeans"],
        "BayesSpace (Primary)": METHOD_COLORS["BayesSpace"],
        "BayesSpace (Replicates)": OKABE_ITO["yellow"],
    }

    xs = np.arange(len(order))
    for i, g in enumerate(order):
        vals = plot[plot["group"] == g]["runtime_sec"].to_numpy()
        if len(vals) == 0:
            continue
        jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.18
        ax_a.scatter(np.full(len(vals), i) + jitter, vals, s=20, c=colors[g], alpha=0.8)
        ax_a.plot([i - 0.25, i + 0.25], [np.median(vals), np.median(vals)], color=OKABE_ITO["black"], linewidth=1.2)

    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(order, rotation=25, ha="right")
    ax_a.set_yscale("log")
    ax_a.set_ylabel("Runtime (sec; log scale)")
    ax_a.set_title("Runtime distributions (local-first context)")
    panel_label(ax_a, "A")

    # Panel B: claim-level compute gate summary (scope-limited)
    est = float(c3["effect_size_value"])
    lo = float(c3["effect_size_ci_lower"])
    hi = float(c3["effect_size_ci_upper"])
    ax_b.plot([lo, hi], [0, 0], color=OKABE_ITO["black"], linewidth=1.4)
    ax_b.scatter([est], [0], c=OKABE_ITO["blue"], s=55, zorder=3)
    ax_b.axvline(1800, color=OKABE_ITO["vermillion"], linewidth=1.2, linestyle="--")
    ax_b.set_yticks([0])
    ax_b.set_yticklabels(["Median runtime"])
    ax_b.set_xlabel("Bootstrap 95% CI (sec)")
    ax_b.set_title("Compute gate (scope-limited; not a universal ranking)")
    panel_label(ax_b, "B")

    fig.suptitle("Figure 4. Compute feasibility: scope-limited evidence and runtime heterogeneity", y=0.99, fontsize=11)
    save_figure(fig, repo_root, "figure4")


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    set_style()
    fig1(repo_root)
    fig2(repo_root)
    fig3(repo_root)
    fig4(repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
