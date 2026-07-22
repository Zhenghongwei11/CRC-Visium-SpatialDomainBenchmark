#!/usr/bin/env python3
"""Generate supplementary figures for the CRC spatial benchmark.

This script is intentionally lightweight and reads only existing analysis tables.
"""

from __future__ import annotations

import csv
import gzip
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


def _read_lines(path: Path) -> list[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [line.strip() for line in handle]
    return path.read_text(encoding="utf-8").splitlines()


def load_flat_visium_sample(dataset_root: Path, sample_id: str) -> tuple[sparse.csr_matrix, np.ndarray, list[str]]:
    matrix_path = dataset_root / f"{sample_id}_matrix.mtx.gz"
    barcodes_path = dataset_root / f"{sample_id}_barcodes.tsv.gz"
    coords_path = dataset_root / f"{sample_id}_tissue_positions.csv.gz"
    features_path = dataset_root / f"{sample_id}_features.tsv.gz"

    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)
    if not barcodes_path.exists():
        raise FileNotFoundError(barcodes_path)
    if not coords_path.exists():
        raise FileNotFoundError(coords_path)
    if not features_path.exists():
        raise FileNotFoundError(features_path)

    counts = mmread(matrix_path).tocsr()  # genes x spots
    barcodes = _read_lines(barcodes_path)

    coords = pd.read_csv(coords_path, compression="infer")
    if "barcode" not in coords.columns:
        raise ValueError(f"Unexpected coords columns in {coords_path}")

    if counts.shape[1] != len(barcodes):
        raise ValueError(f"Barcode mismatch: matrix spots={counts.shape[1]} barcodes={len(barcodes)}")

    coords = coords[coords["barcode"].isin(barcodes)].copy()
    coords["barcode"] = pd.Categorical(coords["barcode"], categories=barcodes, ordered=True)
    coords = coords.sort_values("barcode")
    in_tissue = coords["in_tissue"].to_numpy().astype(int) == 1

    # transpose to spots x genes
    counts = counts.transpose().tocsr()
    counts = counts[in_tissue, :]
    xy = coords.loc[in_tissue, ["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()

    features = pd.read_csv(features_path, sep="\t", header=None, compression="infer")
    if features.shape[1] < 2:
        raise ValueError(f"Unexpected features file: {features_path}")
    gene_names = [str(v) for v in features.iloc[:, 1].tolist()]

    if counts.shape[1] != len(gene_names):
        raise ValueError(f"Gene mismatch: matrix genes={counts.shape[1]} features={len(gene_names)}")

    return counts, xy, gene_names


def normalize_log1p(counts: sparse.csr_matrix) -> np.ndarray:
    counts_per_spot = np.asarray(counts.sum(axis=1)).ravel()
    counts_per_spot[counts_per_spot == 0] = 1.0
    scale = 1e4 / counts_per_spot
    normalized = counts.multiply(scale[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    return normalized.toarray().astype(np.float32)


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _save(fig: plt.Figure, root: Path, name: str) -> None:
    png_dir = root / "plots" / "publication" / "png"
    pdf_dir = root / "plots" / "publication" / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_dir / f"{name}.png", dpi=300)
    fig.savefig(pdf_dir / f"{name}.pdf")
    plt.close(fig)


def make_s1_domain_marker_heatmap(root: Path) -> Path:
    maps = pd.read_csv(root / "results" / "figures" / "fig1_domain_maps.tsv", sep="\t")
    bs_maps = maps[(maps["method_id"] == "BayesSpace")].copy()
    if bs_maps.empty:
        raise ValueError("No BayesSpace rows found in fig1_domain_maps.tsv")

    dataset_id = str(bs_maps["dataset_id"].iloc[0])
    sample_id = str(bs_maps["sample_id"].iloc[0])
    k_val = int(bs_maps["K"].iloc[0])

    ds_root = root / "data" / "raw" / dataset_id / "extracted"
    counts, xy, gene_names = load_flat_visium_sample(ds_root, sample_id)
    x = normalize_log1p(counts)

    bs_rep = bs_maps[bs_maps["sample_id"] == sample_id].copy()
    bs_rep = bs_rep[bs_rep["K"] == k_val].copy()
    coord_to_domain: dict[tuple[int, int], int] = {}
    for _, row in bs_rep.iterrows():
        coord_to_domain[(int(round(row["x"])), int(round(row["y"]))) ] = int(row["domain_label"])

    domains = np.zeros((xy.shape[0],), dtype=int)
    matched = 0
    for i, (xv, yv) in enumerate(xy):
        key = (int(round(xv)), int(round(yv)))
        lab = coord_to_domain.get(key)
        if lab is None:
            domains[i] = 0
        else:
            domains[i] = lab
            matched += 1

    if matched < int(0.95 * len(domains)):
        raise ValueError(f"Domain-label match rate too low: matched={matched} total={len(domains)}")

    keep = domains > 0
    x = x[keep, :]
    domains = domains[keep]

    marker_genes = [
        "EPCAM",
        "KRT19",
        "MKI67",
        "MUC1",
        "VIM",
        "COL1A1",
        "COL1A2",
        "DCN",
        "PECAM1",
        "VWF",
        "PTPRC",
        "LST1",
        "CD3D",
        "MS4A1",
        "FCGR3A",
        "S100A8",
        "S100A9",
        "MZB1",
    ]

    name_to_idx = {name: i for i, name in enumerate(gene_names)}
    found = [g for g in marker_genes if g in name_to_idx]
    if len(found) < 8:
        raise ValueError(f"Too few marker genes found in features: found={len(found)}")

    dom_list = sorted(int(d) for d in np.unique(domains))
    heat = np.zeros((len(found), len(dom_list)), dtype=float)
    for gi, gene in enumerate(found):
        col = name_to_idx[gene]
        for dj, dom in enumerate(dom_list):
            heat[gi, dj] = float(x[domains == dom, col].mean())

    heat_z = (heat - heat.mean(axis=1, keepdims=True)) / (heat.std(axis=1, keepdims=True) + 1e-6)

    out_rows: list[dict[str, object]] = []
    for gi, gene in enumerate(found):
        row: dict[str, object] = {"gene": gene}
        for dj, dom in enumerate(dom_list):
            row[f"domain_{dom}_mean_log1p"] = round(float(heat[gi, dj]), 6)
            row[f"domain_{dom}_z"] = round(float(heat_z[gi, dj]), 6)
        out_rows.append(row)

    out_table = root / "results" / "figures" / "figS1_domain_marker_heatmap.tsv"
    fieldnames = ["gene"] + [f"domain_{d}_mean_log1p" for d in dom_list] + [f"domain_{d}_z" for d in dom_list]
    write_tsv(out_table, out_rows, fieldnames=fieldnames)

    fig = plt.figure(figsize=(6.8, 3.6))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(heat_z, aspect="auto", cmap="RdBu_r", vmin=-2.0, vmax=2.0)
    ax.set_yticks(np.arange(len(found)))
    ax.set_yticklabels(found, fontsize=7)
    ax.set_xticks(np.arange(len(dom_list)))
    ax.set_xticklabels([str(d) for d in dom_list], fontsize=7)
    ax.set_xlabel("BayesSpace domain", fontsize=8)
    ax.set_title(
        f"Representative domain marker expression (z-scored)\n{dataset_id} {sample_id}, K={k_val}",
        fontsize=9,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("z-score (per gene)", fontsize=8)

    fig.tight_layout()
    _save(fig, root, "figureS1")

    print(f"Wrote {out_table}")
    print("Wrote plots/publication/png/figureS1.png")
    print("Wrote plots/publication/pdf/figureS1.pdf")
    return out_table


def make_s2_workflow_schematic(root: Path) -> None:
    """Benchmark workflow schematic: chevron pipeline + detail panels + gates."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # ── colours ──
    BLUE     = "#3B7DD8"
    BLUE_L   = "#EBF1FA"
    AMBER    = "#D4920B"
    AMBER_L  = "#FDF6E7"
    GREY     = "#6B7280"
    GREY_L   = "#F3F4F6"
    DARK     = "#1E293B"
    MID      = "#475569"
    LIGHT_LN = "#CBD5E1"
    WHITE    = "#FFFFFF"

    W, H = 7.5, 5.6
    fig, ax = plt.subplots(figsize=(W, H), facecolor=WHITE)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ── helpers ──
    def chevron(x: float, y: float, w: float, h: float,
                color: str, edge: str) -> float:
        tip = 0.18 * h
        verts = [(x, y), (x + w - tip, y), (x + w, y + h / 2),
                 (x + w - tip, y + h), (x, y + h), (x + tip, y + h / 2)]
        ax.add_patch(plt.Polygon(verts, closed=True, facecolor=color,
                                  edgecolor=edge, linewidth=0.7, zorder=3))
        return x + tip

    def section_bar(x: float, y: float, w: float, h: float,
                    color: str, label: str) -> None:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04",
            facecolor=color, edgecolor="none", linewidth=0, zorder=3))
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=6.5,
                fontweight="bold", color=WHITE, zorder=4)

    def bullet(x: float, y: float, text: str, color: str = MID,
               bold_prefix: str = "", desc: str = "") -> None:
        """Bullet with optional bold prefix + lighter description."""
        ax.text(x, y, "\u2022", fontsize=7, color=GREY, va="top", ha="left")
        if bold_prefix:
            t = ax.text(x + 0.12, y, bold_prefix,
                        fontsize=6.0, color=DARK, va="top", ha="left",
                        fontweight="bold")
            # measure bold portion width to place description after it
            fig.canvas.draw()
            bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
            inv = ax.transData.inverted()
            x_end = inv.transform((bb.x1, bb.y0))[0]
            if desc:
                ax.text(x_end + 0.04, y, desc,
                        fontsize=6.0, color=color, va="top", ha="left")
        else:
            ax.text(x + 0.12, y, text,
                    fontsize=6.0, color=color, va="top", ha="left")

    # ══ TOP: 4 chevrons ══
    chev_y, chev_h, chev_w, chev_gap = 4.25, 0.55, 1.52, 0.13
    x0 = 0.42

    labels = ["DATA", "PREPROCESSING", "METHODS", "OUTPUTS"]
    colors = [BLUE, BLUE, BLUE, GREY]
    edges  = ["#2E66B5", "#2E66B5", "#2E66B5", "#4B5563"]
    chev_xs: list[float] = []

    for i, (lab, col, edg) in enumerate(zip(labels, colors, edges)):
        cx = x0 + i * (chev_w + chev_gap)
        chev_xs.append(cx)
        tx = chevron(cx, chev_y, chev_w, chev_h, col, edg)
        ax.text(tx + (chev_w - 0.18 * chev_h) / 2, chev_y + chev_h / 2,
                lab, ha="center", va="center",
                fontsize=7, fontweight="bold", color=WHITE, zorder=4)

    for i, cx in enumerate(chev_xs):
        mid_x = cx + chev_w / 2
        ax.text(mid_x, chev_y + chev_h + 0.12, str(i + 1),
                ha="center", va="bottom", fontsize=7, fontweight="bold",
                color=BLUE if i < 3 else GREY,
                bbox=dict(boxstyle="circle,pad=0.15",
                          facecolor=WHITE,
                          edgecolor=BLUE if i < 3 else GREY,
                          linewidth=0.6))

    # ══ DETAIL PANELS ══
    panel_top = 4.08
    # Each item: (bold_prefix, description, color_for_desc)
    details = [
        [("GEO", "public CRC Visium", MID),
         ("GSE267401", "primary, n = 4", MID),
         ("GSE311294", "replication, n = 5", MID),
         ("GSE280318", "replication, n = 4", MID),
         ("13 samples", "in primary comparison", DARK),
         ("GSE289934", "portability demo (non-CRC), n = 2", MID)],
        [("Spot \u00d7 gene", "counts + spatial coords", MID),
         ("Log-normalisation", "", MID),
         ("HVG", "selection", MID),
         ("PCA", "\u2192 expression embedding", MID),
         ("Coordinates", "retained for spatial", MID)],
        [("M0", "expression-only k-means", MID),
         ("M1", "PC + scaled-coord k-means", MID),
         ("M2", "spatially constrained Ward", MID),
         ("M3", "spatial graph Leiden", MID),
         ("M4", "SpaGCN baseline", MID),
         ("M5", "STAGATE baseline", MID),
         ("BayesSpace", "Bayesian / Potts prior", DARK)],
        [("Fig. 1\u20135", "main results", MID),
         ("S1\u2013S5", "supplementary figures", MID),
         ("S1\u2013S21", "supporting data tables", MID),
         ("GitHub repo", "code + docs", MID),
         ("Zenodo", "archived release", MID)],
    ]
    for ci, col_items in enumerate(details):
        cx = x0 + ci * (chev_w + chev_gap) + 0.15
        mid_x = x0 + ci * (chev_w + chev_gap) + chev_w / 2
        ax.plot([mid_x, mid_x], [chev_y - 0.02, panel_top],
                color=LIGHT_LN, lw=0.5, ls=":", zorder=1)
        row_step = 0.26
        if ci in (0, 2):
            # Methods column lists more items; tighten spacing slightly.
            row_step = 0.22
        for ri, (bpfx, desc, clr) in enumerate(col_items):
            bullet(cx, panel_top - 0.06 - ri * row_step,
                   "", color=clr, bold_prefix=bpfx, desc=desc)

    # ══ BOTTOM: separator + two panels ══
    sep_y = 2.50
    ax.plot([0.42, W - 0.42], [sep_y, sep_y], color=LIGHT_LN, lw=0.6, zorder=1)

    lp_x, lp_w, lp_top = 0.42, 3.20, 2.32
    section_bar(lp_x, lp_top, lp_w, 0.28, BLUE, "EVALUATION METRICS")
    for i, (title, desc) in enumerate([
        ("Spatial coherence", "neighbour agreement within domains"),
        ("Marker coherence", "domain-level expression contrast"),
        ("Stability", "adjusted Rand index across random seeds"),
        ("Compute", "wall-clock runtime (scope-limited)"),
    ]):
        ey = lp_top - 0.14 - i * 0.38
        ax.text(lp_x + 0.12, ey, title, fontsize=6.5, fontweight="bold",
                color=DARK, va="top")
        ax.text(lp_x + 0.12, ey - 0.16, desc, fontsize=5.8, color=GREY, va="top")

    rp_x, rp_w = 3.88, 3.20
    section_bar(rp_x, lp_top, rp_w, 0.28, AMBER, "STATISTICAL CHECKS")
    for i, (title, desc) in enumerate([
        ("Paired Wilcoxon signed-rank", "method-level comparisons"),
        ("One-sided stability threshold", "planned ARI cutoff"),
        ("BH-FDR correction", "q < 0.05, family-controlled"),
        ("Effect sizes", "median \u0394 + bootstrap 95 % CI"),
        ("Negative controls", "planned coordinate shuffle"),
    ]):
        gy = lp_top - 0.14 - i * 0.34
        ax.text(rp_x + 0.12, gy, title, fontsize=6.5, fontweight="bold",
                color=DARK, va="top")
        ax.text(rp_x + 0.12, gy - 0.16, desc, fontsize=5.8, color=GREY, va="top")

    # connectors
    mx = chev_xs[2] + chev_w / 2
    ax.annotate("", xy=(lp_x + lp_w / 2, lp_top + 0.34),
                xytext=(mx, chev_y - 0.02),
                arrowprops=dict(arrowstyle="-|>", lw=0.6,
                                color=LIGHT_LN, mutation_scale=7,
                                linestyle=(0, (4, 3))))
    ox = chev_xs[3] + chev_w / 2
    ax.annotate("", xy=(rp_x + rp_w / 2, lp_top + 0.34),
                xytext=(ox, chev_y - 0.02),
                arrowprops=dict(arrowstyle="-|>", lw=0.6,
                                color=LIGHT_LN, mutation_scale=7,
                                linestyle=(0, (4, 3))))
    ax.annotate("", xy=(rp_x - 0.06, lp_top + 0.14),
                xytext=(lp_x + lp_w + 0.06, lp_top + 0.14),
                arrowprops=dict(arrowstyle="-|>", lw=0.6,
                                color=GREY, mutation_scale=7))

    # subtle background
    ax.add_patch(mpatches.FancyBboxPatch(
        (lp_x - 0.06, 0.56), lp_w + 0.12, lp_top + 0.28 - 0.56 + 0.08,
        boxstyle="round,pad=0.06",
        facecolor=BLUE_L, edgecolor="none", linewidth=0, zorder=0, alpha=0.5))
    ax.add_patch(mpatches.FancyBboxPatch(
        (rp_x - 0.06, 0.56), rp_w + 0.12, lp_top + 0.28 - 0.56 + 0.08,
        boxstyle="round,pad=0.06",
        facecolor=AMBER_L, edgecolor="none", linewidth=0, zorder=0, alpha=0.5))

    ax.text(W / 2, H - 0.22, "Study overview and evaluation framework",
            ha="center", va="top", fontsize=10, fontweight="bold", color=DARK)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.06)
    # Workflow schematic is exported as figure2 for the analysis figure set.
    _save(fig, root, "figure2")
    print("Wrote plots/publication/png/figure2.png")
    print("Wrote plots/publication/pdf/figure2.pdf")


def make_s3_instability_case_study(root: Path) -> None:
    import textwrap

    src = root / "results" / "figures" / "figS3_instability_case_study.tsv"
    if not src.exists():
        raise FileNotFoundError(
            "Missing figS3 input table. Generate it with:\n"
            "  Rscript scripts/build_figS3_instability_case_study.R"
        )

    df = pd.read_csv(src, sep="\t")
    if df.empty:
        raise RuntimeError("figS3 table is empty")

    dataset_id = str(df["dataset_id"].iloc[0])
    sample_id = str(df["sample_id"].iloc[0])
    k = int(df["K"].iloc[0])
    ari_med = float(df["stability_ari_median"].iloc[0])
    ari_iqr = float(df["stability_ari_iqr"].iloc[0])

    seeds_all = sorted(df["seed"].unique().tolist())
    show_seeds = [seeds_all[0], seeds_all[1], seeds_all[len(seeds_all)//2], seeds_all[-1]]

    # Marker names are repeated per row by the builder.
    epi_name = str(df["marker_epithelial_name"].iloc[0])
    str_name = str(df["marker_stromal_name"].iloc[0])

    # Build label alignment to the first seed so colours stay comparable.
    ref_seed = show_seeds[0]
    ref = df[df["seed"] == ref_seed].copy()
    if ref.empty:
        raise RuntimeError("Reference seed rows missing")

    ref_labels = ref["domain_label"].astype(int).to_numpy()
    ref_barcode = ref["barcode"].astype(str).to_numpy()
    ref_map = {b: int(l) for b, l in zip(ref_barcode, ref_labels)}
    label_set = sorted(set(ref_labels.tolist()))

    def _align_labels(barcodes: np.ndarray, labels: np.ndarray) -> np.ndarray:
        # Align to reference labels by maximum overlap (Hungarian if available).
        obs = np.array([ref_map.get(b, -1) for b in barcodes], dtype=int)
        keep = obs >= 0
        if keep.sum() == 0:
            return labels
        obs = obs[keep]
        lab = labels[keep]

        ref_ids = sorted(set(obs.tolist()))
        lab_ids = sorted(set(lab.tolist()))
        # Build overlap matrix (negative for minimization).
        M = np.zeros((len(ref_ids), len(lab_ids)), dtype=int)
        for i, r in enumerate(ref_ids):
            for j, c in enumerate(lab_ids):
                M[i, j] = np.sum((obs == r) & (lab == c))
        # Solve assignment
        mapping = {}
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore

            rr, cc = linear_sum_assignment(-M)
            for i, j in zip(rr, cc):
                mapping[lab_ids[j]] = ref_ids[i]
        except Exception:
            # Greedy fallback
            used_ref = set()
            used_lab = set()
            pairs = []
            for i, r in enumerate(ref_ids):
                for j, c in enumerate(lab_ids):
                    pairs.append((M[i, j], r, c))
            for _, r, c in sorted(pairs, reverse=True):
                if r in used_ref or c in used_lab:
                    continue
                mapping[c] = r
                used_ref.add(r)
                used_lab.add(c)

        aligned = labels.copy()
        for c, r in mapping.items():
            aligned[labels == c] = r
        return aligned

    # ── style ──
    BLUE = "#3B7DD8"
    DARK = "#1E293B"
    GREY = "#6B7280"
    LIGHT = "#E5E7EB"
    WHITE = "#FFFFFF"

    palette = [
        "#4477AA", "#EE6677", "#228833", "#CCBB44",
        "#66CCEE", "#AA3377", "#BBBBBB", "#000000",
    ]
    colour_map = {lab: palette[(i) % len(palette)] for i, lab in enumerate(sorted(label_set))}

    fig = plt.figure(figsize=(7.5, 6.2), facecolor=WHITE)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.9], hspace=0.18)
    gs_top = outer[0].subgridspec(1, 4, wspace=0.05)
    # Give the right colorbar a slightly wider lane so its tick labels are never clipped.
    gs_bot = outer[1].subgridspec(1, 5, width_ratios=[1.0, 0.055, 0.12, 1.0, 0.075], wspace=0.10)

    # top row: 4 seeds
    for i, seed in enumerate(show_seeds):
        ax = fig.add_subplot(gs_top[0, i])
        sub = df[df["seed"] == seed].copy()
        sub = sub.sort_values("barcode")
        barcodes = sub["barcode"].astype(str).to_numpy()
        labels = sub["domain_label"].astype(int).to_numpy()
        labels = _align_labels(barcodes, labels)
        xs = sub["x"].to_numpy()
        ys = sub["y"].to_numpy()
        colors = [colour_map.get(int(l), GREY) for l in labels]
        ax.scatter(xs, ys, s=2.0, c=colors, linewidths=0, alpha=0.95)
        ax.set_title(f"seed={seed}", fontsize=8, color=DARK)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # bottom row: epithelial/stromal marker expression (reference seed coords)
    def _marker_panel(ax, values, title):
        xs = ref["x"].to_numpy()
        ys = ref["y"].to_numpy()
        v = np.asarray(values, dtype=float)
        vmax = np.quantile(v, 0.99)
        sc = ax.scatter(xs, ys, s=2.2, c=v, cmap="viridis", vmin=0, vmax=vmax, linewidths=0)
        ax.set_title(title, fontsize=8, color=DARK)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return sc

    ax_epi = fig.add_subplot(gs_bot[0, 0])
    sc1 = _marker_panel(ax_epi, ref["expr_epithelial"], f"{epi_name} (logcounts)")
    cax1 = fig.add_subplot(gs_bot[0, 1])
    cb1 = fig.colorbar(sc1, cax=cax1)
    # Keep tick labels away from the central gutter to avoid clipping in exported figures.
    cb1.ax.yaxis.set_ticks_position("left")
    cb1.ax.tick_params(labelsize=7, pad=1)
    cb1.outline.set_linewidth(0.8)

    ax_str = fig.add_subplot(gs_bot[0, 3])
    sc2 = _marker_panel(ax_str, ref["expr_stromal"], f"{str_name} (logcounts)")
    cax2 = fig.add_subplot(gs_bot[0, 4])
    cb2 = fig.colorbar(sc2, cax=cax2)
    # Keep the right-side tick labels comfortably inside the canvas.
    cb2.ax.tick_params(labelsize=7, pad=1)
    cb2.outline.set_linewidth(0.8)

    title = f"Instability case study (BayesSpace) — {dataset_id}/{sample_id}, K={k}"
    fig.suptitle(title, y=0.98, fontsize=10, fontweight="bold", color=DARK)

    fig.subplots_adjust(left=0.03, right=0.95, top=0.92, bottom=0.05)
    _save(fig, root, "figureS2")
    print("Wrote plots/publication/png/figureS2.png")
    print("Wrote plots/publication/pdf/figureS2.pdf")


def make_s4_boundary_vs_interior_seed_sensitivity(root: Path) -> None:
    """Downstream sensitivity: boundary-vs-interior signature deltas across seeds."""
    src = root / "results" / "figures" / "figS4_boundary_vs_interior_seed_sensitivity.tsv"
    if not src.exists():
        raise FileNotFoundError(
            "Missing boundary-sensitivity table. Generate it with:\n"
            "  python3 scripts/build_figS4_boundary_vs_interior_seed_sensitivity.py "
            "--dataset-root data/raw/GSE311294/extracted"
        )

    df = pd.read_csv(src, sep="\t")
    if df.empty:
        raise RuntimeError("Boundary sensitivity table is empty")

    # Focus on signature-level readouts (more informative than sparse single-gene medians).
    keep = [
        "SIG_CAF_FAP",
        "SIG_MY_SPP1",
        "SIG_TCELL",
        "SIG_EXCLUSION_TGFB_CXCL12",
        "SIG_EXCLUSION_CONTRAST",
    ]
    sub = df[(df["feature_type"] == "signature") & (df["feature_id"].isin(keep))].copy()
    if sub.empty:
        raise RuntimeError("No expected signature rows found in boundary sensitivity table")

    feature_order = keep
    label_map = {
        "SIG_CAF_FAP": "CAF / FAP",
        "SIG_MY_SPP1": "SPP1-myeloid",
        "SIG_TCELL": "T-cell",
        "SIG_EXCLUSION_TGFB_CXCL12": "Exclusion (TGFβ/CXCL12)",
        "SIG_EXCLUSION_CONTRAST": "Exclusion contrast",
    }

    seeds = sorted({int(s) for s in sub["seed"].astype(int).unique().tolist()})
    palette = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7"]  # colorblind-friendly
    seed_to_col = {s: palette[i % len(palette)] for i, s in enumerate(seeds)}
    col_dark = "#4D4D4D"
    col_black = "#000000"

    fig = plt.figure(figsize=(7.2, 3.2))
    ax = fig.add_subplot(1, 1, 1)

    xs = np.arange(len(feature_order), dtype=float)
    jitter = np.linspace(-0.18, 0.18, num=max(2, len(seeds)))

    for si, seed in enumerate(seeds):
        ds = sub[sub["seed"].astype(int) == int(seed)].copy()
        ys = []
        for fid in feature_order:
            row = ds[ds["feature_id"] == fid]
            if row.empty:
                ys.append(np.nan)
            else:
                ys.append(float(row["median_delta_boundary_minus_interior"].iloc[0]))
        ax.scatter(xs + jitter[si], ys, s=36, color=seed_to_col[seed],
                   edgecolors="white", linewidths=0.6, zorder=3, label=f"seed={seed}")

    # Add a light summary line (median across seeds) to guide the eye.
    med = []
    for fid in feature_order:
        vals = sub[sub["feature_id"] == fid]["median_delta_boundary_minus_interior"].astype(float).to_numpy()
        med.append(float(np.nanmedian(vals)))
    ax.plot(xs, med, color=col_dark, linewidth=1.2, zorder=2)

    ax.axhline(0, color=col_black, linewidth=0.7, linestyle="--", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([label_map.get(f, f) for f in feature_order], fontsize=7)
    ax.set_ylabel("Boundary − interior (median Δ)", fontsize=8)
    ax.set_title("Downstream sensitivity: boundary-associated signature deltas across seeds",
                 fontsize=9, pad=8)

    ax.legend(frameon=False, fontsize=6, ncol=len(seeds), loc="upper right",
              handletextpad=0.3, columnspacing=0.8)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    fig.tight_layout()

    _save(fig, root, "figureS3")
    print("Wrote plots/publication/png/figureS3.png")
    print("Wrote plots/publication/pdf/figureS3.pdf")


def make_s5_histology_overlay_switching_spots(root: Path) -> None:
    """Histology overlay for the instability case study (stable vs switching spots).

    Renders a reviewer-friendly visualization showing that switching spots are
    spatially structured (interface-localized) rather than scattered noise.
    """
    import io
    import textwrap

    from PIL import Image  # type: ignore

    src = root / "results" / "figures" / "figS3_instability_case_study.tsv"
    if not src.exists():
        raise FileNotFoundError(
            "Missing figS3 input table. Generate it with:\n"
            "  Rscript scripts/build_figS3_instability_case_study.R"
        )

    df = pd.read_csv(src, sep="\t")
    if df.empty:
        raise RuntimeError("figS3 table is empty")

    dataset_id = str(df["dataset_id"].iloc[0])
    sample_id = str(df["sample_id"].iloc[0])
    k = int(df["K"].iloc[0])

    # Load the section image shipped with the GEO bundle.
    img_gz = root / "data" / "raw" / dataset_id / "extracted" / f"{sample_id}_detected_tissue_image.jpg.gz"
    if not img_gz.exists():
        raise FileNotFoundError(img_gz)
    with gzip.open(img_gz, "rb") as handle:
        img = Image.open(io.BytesIO(handle.read())).convert("RGB")
    w, h = img.size

    seeds = sorted(int(s) for s in df["seed"].astype(int).unique().tolist())
    ref_seed = seeds[0]
    ref = df[df["seed"].astype(int) == ref_seed].copy()
    if ref.empty:
        raise RuntimeError("Reference seed rows missing")
    ref = ref.sort_values("barcode")
    ref_barcodes = ref["barcode"].astype(str).to_numpy()
    ref_labels = ref["domain_label"].astype(int).to_numpy()
    ref_xy = ref[["x", "y"]].to_numpy(dtype=float)
    ref_map = {b: int(l) for b, l in zip(ref_barcodes, ref_labels)}

    def _align_labels(barcodes: np.ndarray, labels: np.ndarray) -> np.ndarray:
        obs = np.array([ref_map.get(b, -1) for b in barcodes], dtype=int)
        keep = obs >= 0
        if keep.sum() == 0:
            return labels
        obs = obs[keep]
        lab = labels[keep]

        ref_ids = sorted(set(obs.tolist()))
        lab_ids = sorted(set(lab.tolist()))
        M = np.zeros((len(ref_ids), len(lab_ids)), dtype=int)
        for i, r in enumerate(ref_ids):
            for j, c in enumerate(lab_ids):
                M[i, j] = int(np.sum((obs == r) & (lab == c)))

        mapping: dict[int, int] = {}
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore

            rr, cc = linear_sum_assignment(-M)
            for i, j in zip(rr, cc):
                mapping[int(lab_ids[j])] = int(ref_ids[i])
        except Exception:
            used_ref: set[int] = set()
            used_lab: set[int] = set()
            pairs: list[tuple[int, int, int]] = []
            for i, r in enumerate(ref_ids):
                for j, c in enumerate(lab_ids):
                    pairs.append((int(M[i, j]), int(r), int(c)))
            for _, r, c in sorted(pairs, reverse=True):
                if r in used_ref or c in used_lab:
                    continue
                mapping[c] = r
                used_ref.add(r)
                used_lab.add(c)

        aligned = labels.copy()
        for c, r in mapping.items():
            aligned[labels == c] = r
        return aligned

    # Build aligned labels per seed and classify switching barcodes.
    labs_by_barcode: dict[str, list[int]] = {str(b): [] for b in ref_barcodes.tolist()}
    for seed in seeds:
        sub = df[df["seed"].astype(int) == int(seed)].copy().sort_values("barcode")
        bar = sub["barcode"].astype(str).to_numpy()
        lab = sub["domain_label"].astype(int).to_numpy()
        lab = _align_labels(bar, lab)
        for b, l in zip(bar.tolist(), lab.tolist()):
            if b in labs_by_barcode:
                labs_by_barcode[b].append(int(l))

    switching_mask = np.array([len(set(labs_by_barcode[b])) > 1 for b in ref_barcodes], dtype=bool)

    # Choose a zoom window centered on the switching distribution.
    switch_xy = ref_xy[switching_mask]
    if len(switch_xy) < 10:
        raise RuntimeError("Too few switching spots found; expected a non-trivial switching set")
    cx = float(np.median(switch_xy[:, 0]))
    cy = float(np.median(switch_xy[:, 1]))
    half = 520.0
    x0 = max(0.0, cx - half)
    x1 = min(float(w), cx + half)
    y0 = max(0.0, cy - half)
    y1 = min(float(h), cy + half)

    # ── Plot ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.4, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.06)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    for ax in (ax0, ax1):
        ax.imshow(img, origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Full view with zoom box
    ax0.scatter(ref_xy[~switching_mask, 0], ref_xy[~switching_mask, 1], s=2.0, c="#111827", alpha=0.12, linewidths=0)
    ax0.scatter(ref_xy[switching_mask, 0], ref_xy[switching_mask, 1], s=3.6, c="#DC2626", alpha=0.75, linewidths=0)
    ax0.add_patch(
        mpatches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="#DC2626",
            linewidth=1.0,
            linestyle="--",
        )
    )
    ax0.set_title("Stable vs switching spots (full section)", fontsize=9, pad=2)

    # Zoom view
    ax1.set_xlim(x0, x1)
    ax1.set_ylim(y1, y0)  # keep origin='upper' orientation
    ax1.scatter(ref_xy[~switching_mask, 0], ref_xy[~switching_mask, 1], s=4.0, c="#111827", alpha=0.14, linewidths=0)
    ax1.scatter(ref_xy[switching_mask, 0], ref_xy[switching_mask, 1], s=7.5, c="#DC2626", alpha=0.85, linewidths=0)
    ax1.set_title("Zoom: interface-localized switching", fontsize=9, pad=2)

    # Legend
    handles = [
        mpatches.Patch(color="#111827", label="Stable spots"),
        mpatches.Patch(color="#DC2626", label="Switching spots"),
    ]
    ax1.legend(handles=handles, frameon=False, fontsize=7, loc="lower right")

    title = f"Histology overlay for the instability case study — {dataset_id}/{sample_id}, K={k}"
    subtitle = (
        "Switching spots (changed domain label across seeds after label alignment) "
        "form a structured band rather than scattered noise."
    )
    fig.suptitle(title, y=0.98, fontsize=10, fontweight="bold", color="#111827")
    fig.text(0.5, 0.945, textwrap.fill(subtitle, width=95), ha="center", va="top", fontsize=7.5, color="#4B5563")

    # Keep extra headroom so the subtitle does not collide with panel titles.
    fig.subplots_adjust(left=0.02, right=0.99, top=0.83, bottom=0.05)
    _save(fig, root, "figureS4")
    print("Wrote plots/publication/png/figureS4.png")
    print("Wrote plots/publication/pdf/figureS4.pdf")


def make_s6_histology_feature_audit(root: Path) -> None:
    """Quantify histology contrast around switching vs stable spots.

    Writes:
    - results/figures/figS5_histology_patch_features.tsv  (per-spot features)
    - plots/publication/png/figureS5.png (distribution summary)
    """
    import io

    from PIL import Image  # type: ignore

    src = root / "results" / "figures" / "figS3_instability_case_study.tsv"
    if not src.exists():
        raise FileNotFoundError(
            "Missing figS3 input table. Generate it with:\n"
            "  Rscript scripts/build_figS3_instability_case_study.R"
        )

    df = pd.read_csv(src, sep="\t")
    if df.empty:
        raise RuntimeError("figS3 table is empty")

    dataset_id = str(df["dataset_id"].iloc[0])
    sample_id = str(df["sample_id"].iloc[0])
    k = int(df["K"].iloc[0])

    seeds = sorted(int(s) for s in df["seed"].astype(int).unique().tolist())
    ref_seed = seeds[0]
    ref = df[df["seed"].astype(int) == ref_seed].copy()
    ref = ref.sort_values("barcode")
    ref_barcodes = ref["barcode"].astype(str).to_numpy()
    ref_labels = ref["domain_label"].astype(int).to_numpy()
    ref_xy = ref[["x", "y"]].to_numpy(dtype=float)
    ref_map = {b: int(l) for b, l in zip(ref_barcodes, ref_labels)}

    def _align_labels(barcodes: np.ndarray, labels: np.ndarray) -> np.ndarray:
        obs = np.array([ref_map.get(b, -1) for b in barcodes], dtype=int)
        keep = obs >= 0
        if keep.sum() == 0:
            return labels
        obs = obs[keep]
        lab = labels[keep]

        ref_ids = sorted(set(obs.tolist()))
        lab_ids = sorted(set(lab.tolist()))
        M = np.zeros((len(ref_ids), len(lab_ids)), dtype=int)
        for i, r in enumerate(ref_ids):
            for j, c in enumerate(lab_ids):
                M[i, j] = int(np.sum((obs == r) & (lab == c)))

        mapping: dict[int, int] = {}
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore

            rr, cc = linear_sum_assignment(-M)
            for i, j in zip(rr, cc):
                mapping[int(lab_ids[j])] = int(ref_ids[i])
        except Exception:
            used_ref: set[int] = set()
            used_lab: set[int] = set()
            pairs: list[tuple[int, int, int]] = []
            for i, r in enumerate(ref_ids):
                for j, c in enumerate(lab_ids):
                    pairs.append((int(M[i, j]), int(r), int(c)))
            for _, r, c in sorted(pairs, reverse=True):
                if r in used_ref or c in used_lab:
                    continue
                mapping[c] = r
                used_ref.add(r)
                used_lab.add(c)

        aligned = labels.copy()
        for c, r in mapping.items():
            aligned[labels == c] = r
        return aligned

    labs_by_barcode: dict[str, list[int]] = {str(b): [] for b in ref_barcodes.tolist()}
    for seed in seeds:
        sub = df[df["seed"].astype(int) == int(seed)].copy().sort_values("barcode")
        bar = sub["barcode"].astype(str).to_numpy()
        lab = sub["domain_label"].astype(int).to_numpy()
        lab = _align_labels(bar, lab)
        for b, l in zip(bar.tolist(), lab.tolist()):
            if b in labs_by_barcode:
                labs_by_barcode[b].append(int(l))

    switching_mask = np.array([len(set(labs_by_barcode[b])) > 1 for b in ref_barcodes], dtype=bool)

    img_gz = root / "data" / "raw" / dataset_id / "extracted" / f"{sample_id}_detected_tissue_image.jpg.gz"
    if not img_gz.exists():
        raise FileNotFoundError(img_gz)
    with gzip.open(img_gz, "rb") as handle:
        img = Image.open(io.BytesIO(handle.read())).convert("RGB")
    w, h = img.size
    arr = np.asarray(img, dtype=np.float32)
    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).astype(np.float32)

    patch_r = 24  # ~50 px window

    def _patch(x: int, y: int) -> np.ndarray:
        x0 = max(0, x - patch_r)
        x1 = min(w, x + patch_r + 1)
        y0 = max(0, y - patch_r)
        y1 = min(h, y + patch_r + 1)
        return gray[y0:y1, x0:x1]

    feat_rows: list[dict[str, object]] = []
    for (x_f, y_f), bc, is_sw in zip(ref_xy.tolist(), ref_barcodes.tolist(), switching_mask.tolist()):
        x = int(round(float(x_f)))
        y = int(round(float(y_f)))
        if not (0 <= x < w and 0 <= y < h):
            continue
        p = _patch(x, y)
        if p.size < 25:
            continue
        # simple gradient magnitude (finite differences)
        dx = p[:, 2:] - p[:, :-2] if p.shape[1] >= 3 else np.zeros_like(p)
        dy = p[2:, :] - p[:-2, :] if p.shape[0] >= 3 else np.zeros_like(p)
        if dx.size and dy.size:
            # match shapes by trimming to common interior
            dx_i = dx[1:-1, :] if dx.shape[0] > 2 else dx
            dy_i = dy[:, 1:-1] if dy.shape[1] > 2 else dy
            g = np.sqrt(dx_i**2 + dy_i**2)
        else:
            g = np.zeros((1, 1), dtype=np.float32)

        feat_rows.append(
            {
                "dataset_id": dataset_id,
                "sample_id": sample_id,
                "K": k,
                "barcode": str(bc),
                "x": float(x_f),
                "y": float(y_f),
                "is_switching": int(bool(is_sw)),
                "patch_gray_mean": float(np.mean(p)),
                "patch_gray_sd": float(np.std(p)),
                "patch_grad_mean": float(np.mean(g)),
                "patch_grad_p95": float(np.quantile(g, 0.95)),
            }
        )

    out_table = root / "results" / "figures" / "figS5_histology_patch_features.tsv"
    fieldnames = [
        "dataset_id",
        "sample_id",
        "K",
        "barcode",
        "x",
        "y",
        "is_switching",
        "patch_gray_mean",
        "patch_gray_sd",
        "patch_grad_mean",
        "patch_grad_p95",
    ]
    write_tsv(out_table, feat_rows, fieldnames=fieldnames)

    feat = pd.DataFrame(feat_rows)
    sw = feat[feat["is_switching"] == 1]
    st = feat[feat["is_switching"] == 0]

    fig = plt.figure(figsize=(7.4, 3.0))
    gs = fig.add_gridspec(1, 2, wspace=0.35)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    def _box(ax, col, title, ylab):
        vals = [st[col].to_numpy(dtype=float), sw[col].to_numpy(dtype=float)]
        ax.boxplot(
            vals,
            widths=0.5,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor="#E5E7EB", edgecolor="#111827", linewidth=0.8),
            whiskerprops=dict(color="#111827", linewidth=0.8),
            capprops=dict(color="#111827", linewidth=0.8),
            medianprops=dict(color="#DC2626", linewidth=1.2),
        )
        rng = np.random.RandomState(42)
        for i, v in enumerate(vals, start=1):
            if len(v) == 0:
                continue
            jitter = (rng.rand(len(v)) - 0.5) * 0.18
            ax.scatter(np.full(len(v), i) + jitter, v, s=8, c="#111827", alpha=0.10, linewidths=0)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Stable", "Switching"], fontsize=8)
        ax.set_title(title, fontsize=9, pad=6)
        ax.set_ylabel(ylab, fontsize=8)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)

    _box(ax0, "patch_grad_mean", "Local histology gradient", "Mean gradient (a.u.)")
    _box(ax1, "patch_gray_sd", "Local histology contrast", "Gray SD (a.u.)")

    fig.suptitle(
        f"Histology feature analysis for switching spots — {dataset_id}/{sample_id}, K={k}",
        y=0.98,
        fontsize=10,
        fontweight="bold",
        color="#111827",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.14)
    _save(fig, root, "figureS5")

    print(f"Wrote {out_table}")
    print("Wrote plots/publication/png/figureS5.png")
    print("Wrote plots/publication/pdf/figureS5.pdf")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    make_s1_domain_marker_heatmap(root)
    # NOTE: The workflow schematic is maintained as a vector diagram
    # (see `plots/diagrams/figure2.svg`) and rendered separately.
    # Keep the legacy matplotlib schematic optional to avoid overwriting that asset.
    if os.environ.get("MAKE_LEGACY_FIG2_WORKFLOW", "").strip() == "1":
        make_s2_workflow_schematic(root)
    make_s3_instability_case_study(root)
    make_s4_boundary_vs_interior_seed_sensitivity(root)
    make_s5_histology_overlay_switching_spots(root)
    make_s6_histology_feature_audit(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
