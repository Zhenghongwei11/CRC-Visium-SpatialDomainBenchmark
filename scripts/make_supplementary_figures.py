#!/usr/bin/env python3
"""Generate supplementary figures for the CRC spatial benchmark submission.

This script is intentionally lightweight and reads only existing, locked tables.
"""

from __future__ import annotations

import csv
import gzip
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
        f"S1 Fig. Representative domain marker expression (z-scored)\n{dataset_id} {sample_id}, K={k_val}",
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
         ("GSE285505", "replication, n = 4", MID),
         ("6 samples", "in primary comparison", DARK)],
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
        [("Fig. 1\u20134", "main results", MID),
         ("S1\u2013S3", "supplementary figures", MID),
         ("S1\u2013S11", "supporting data tables", MID),
         ("GitHub repo", "+ review bundle", MID),
         ("Checksum", "manifest", MID)],
    ]
    for ci, col_items in enumerate(details):
        cx = x0 + ci * (chev_w + chev_gap) + 0.15
        mid_x = x0 + ci * (chev_w + chev_gap) + chev_w / 2
        ax.plot([mid_x, mid_x], [chev_y - 0.02, panel_top],
                color=LIGHT_LN, lw=0.5, ls=":", zorder=1)
        row_step = 0.26
        if ci == 2:
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
    section_bar(rp_x, lp_top, rp_w, 0.28, AMBER, "STATISTICAL GATES")
    for i, (title, desc) in enumerate([
        ("Paired Wilcoxon signed-rank", "method-level comparisons"),
        ("One-sided stability threshold", "prespecified ARI cutoff"),
        ("BH-FDR correction", "q < 0.05, family-controlled"),
        ("Effect sizes", "median \u0394 + bootstrap 95 % CI"),
        ("Negative controls", "prespecified coordinate shuffle"),
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

    ax.text(W / 2, H - 0.22, "Study overview and prespecified evaluation framework",
            ha="center", va="top", fontsize=10, fontweight="bold", color=DARK)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.06)
    _save(fig, root, "figureS2")
    print("Wrote plots/publication/png/figureS2.png")
    print("Wrote plots/publication/pdf/figureS2.pdf")


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
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.9], wspace=0.05, hspace=0.18)

    # top row: 4 seeds
    for i, seed in enumerate(show_seeds):
        ax = fig.add_subplot(gs[0, i])
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

    ax_epi = fig.add_subplot(gs[1, 0:2])
    sc1 = _marker_panel(ax_epi, ref["expr_epithelial"], f"{epi_name} (logcounts)")
    cb1 = fig.colorbar(sc1, ax=ax_epi, fraction=0.046, pad=0.02)
    cb1.ax.tick_params(labelsize=7)

    ax_str = fig.add_subplot(gs[1, 2:4])
    sc2 = _marker_panel(ax_str, ref["expr_stromal"], f"{str_name} (logcounts)")
    cb2 = fig.colorbar(sc2, ax=ax_str, fraction=0.046, pad=0.02)
    cb2.ax.tick_params(labelsize=7)

    title = f"S3: Instability case study (BayesSpace) — {dataset_id}/{sample_id}, K={k}"
    subtitle = f"Multi-seed stability: median ARI={ari_med:.3f} (IQR={ari_iqr:.3f}); labels aligned to seed={ref_seed} for visualization"
    fig.suptitle(title, y=0.98, fontsize=10, fontweight="bold", color=DARK)
    fig.text(0.5, 0.945, textwrap.fill(subtitle, width=95), ha="center", va="top", fontsize=7.5, color=GREY)

    fig.subplots_adjust(left=0.03, right=0.985, top=0.90, bottom=0.05)
    _save(fig, root, "figureS3")
    print("Wrote plots/publication/png/figureS3.png")
    print("Wrote plots/publication/pdf/figureS3.pdf")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    make_s1_domain_marker_heatmap(root)
    make_s2_workflow_schematic(root)
    make_s3_instability_case_study(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
