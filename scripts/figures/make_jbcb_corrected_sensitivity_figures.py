#!/usr/bin/env python3
"""Render patient-level regional sensitivity figures from the frozen summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from jbcb_presentation_paths import read
from jbcb_presentation_paths import default_package, portable_label


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = default_package(__file__) / "workspace/analysis/regional"
DEFAULT_PACK = default_package(__file__)
RESOURCES = ("GSE294385", "Valdeolivas")
FRAMES = ("historical_restricted", "independent_eligible")
PRIMARY_ESTIMATOR = "anchor_retention"
SETTINGS = ("baseline", "grid4", "grid8", "near150", "far500")
SETTING_LABELS = {
    "baseline": "Baseline\n6 x 6",
    "grid4": "Grid\n4 x 4",
    "grid8": "Grid\n8 x 8",
    "near150": "Near\n150 um",
    "far500": "Far\n500 um",
}
READOUTS = (
    "primary_barrier", "TGFB1", "CXCL12", "ACTA2", "TAGLN",
    "ACTA2_TAGLN", "TGFB1_CXCL12",
)
READOUT_LABELS = {
    "primary_barrier": "Four-gene\nmean",
    "TGFB1": "TGFB1",
    "CXCL12": "CXCL12",
    "ACTA2": "ACTA2",
    "TAGLN": "TAGLN",
    "ACTA2_TAGLN": "ACTA2/\nTAGLN mean",
    "TGFB1_CXCL12": "TGFB1/\nCXCL12 mean",
}
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#6E7276"
FRAME_STYLE = {
    "historical_restricted": (BLUE, "o", -0.16, "Reference-eligible subset"),
    "independent_eligible": (ORANGE, "s", 0.16, "Own-band selections"),
}
RAW_SCORE_COLUMNS = {
    "count_only": "count_only_excess_abs_deviation_score",
    "block_matched": "block_matched_excess_abs_deviation_score",
}
MAD_COLUMN = "block_matched_excess_abs_deviation_mad"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_path(path: Path) -> str:
    return portable_label(path, DEFAULT_PACK)


def configure() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.labelsize": 10, "axes.titlesize": 10.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 10,
        "legend.fontsize": 9.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def export_rows(frame: pd.DataFrame, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / f"{name}.tsv", sep="\t", index=False, na_rep="NA")


def save_figure(fig: plt.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", metadata={"Creator": "Scientific figure"})
    fig.savefig(out / f"{stem}.png", dpi=300)
    fig.savefig(out / f"{stem}.tiff", dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(f"{letter}  {title}", loc="left", weight="bold", pad=10)


def patient_sort_key(patient: str, resource: str) -> tuple[object, ...]:
    if resource == "GSE294385":
        match = re.search(r"(\d+)", patient)
        return (0, int(match.group(1)) if match else 10**9, patient)
    return (0, patient)


def patient_order(frame: pd.DataFrame, resource: str) -> list[str]:
    patients = frame.patient_id.astype(str).unique().tolist()
    return sorted(patients, key=lambda patient: patient_sort_key(patient, resource))


def centered_offsets(patients: list[str], span: float) -> dict[str, float]:
    if len(patients) < 2:
        return {patient: 0.0 for patient in patients}
    offsets = np.linspace(-span / 2, span / 2, len(patients))
    return dict(zip(patients, offsets, strict=True))


def read_summary(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Regional sensitivity patient summary not found: {path}")
    data = read(path, sep="\t", low_memory=False)
    data = data.loc[data.graph_id.eq("historical_raw_6nn")].copy()
    required = {
        "eligibility_frame", "resource", "patient_id", "estimator_id",
        "setting_id", "readout_id", "n_attempted_sections",
        "n_evaluable_sections", "n_declared_maps", "n_evaluable_maps",
        "n_standardized_maps", *RAW_SCORE_COLUMNS.values(), MAD_COLUMN,
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError("Patient summary lacks required columns: " + ", ".join(sorted(missing)))
    identity = ["eligibility_frame", "resource", "patient_id", "estimator_id", "setting_id", "readout_id"]
    if data.duplicated(identity).any():
        raise ValueError("Patient summary has duplicated patient/frame/setting/readout rows")
    if set(data.eligibility_frame.astype(str)) != set(FRAMES):
        raise ValueError("Patient summary must contain both eligibility frames")
    if set(data.resource.astype(str)) != set(RESOURCES):
        raise ValueError("Patient summary must contain both registered resources")
    for column in ("n_declared_maps", "n_evaluable_maps", "n_standardized_maps",
                   "n_attempted_sections", "n_evaluable_sections"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    if (data.n_evaluable_maps < 0).any() or (data.n_declared_maps < data.n_evaluable_maps).any():
        raise ValueError("Map counts must satisfy 0 <= evaluable <= declared")
    if (data.n_standardized_maps < 0).any() or (data.n_standardized_maps > data.n_evaluable_maps).any():
        raise ValueError("Standardized map counts must be a subset of evaluable maps")
    return data


def primary_rows(data: pd.DataFrame) -> pd.DataFrame:
    return data.loc[data.estimator_id.astype(str).eq(PRIMARY_ESTIMATOR)].copy()


def validate_s3_rows(data: pd.DataFrame) -> pd.DataFrame:
    selected = data.loc[
        data.eligibility_frame.astype(str).eq("independent_eligible")
        & data.setting_id.astype(str).eq("baseline")
        & data.readout_id.astype(str).isin(READOUTS)
    ].copy()
    for resource in RESOURCES:
        frame = selected.loc[selected.resource.astype(str).eq(resource)]
        if set(frame.readout_id.astype(str)) != set(READOUTS):
            raise ValueError(f"Figure S3 readouts are incomplete for {resource}")
        patient_sets = frame.groupby("readout_id").patient_id.apply(lambda values: frozenset(values.astype(str)))
        if patient_sets.nunique() != 1:
            raise ValueError(f"Figure S3 attempted-patient identities differ by readout for {resource}")
    return selected


def s3_figure(data: pd.DataFrame, figures: Path, sources: Path, *, export_sources: bool = True) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.25))
    fig.subplots_adjust(left=0.12, right=0.985, top=0.82, bottom=0.12, hspace=0.70)
    plot_values = []
    source_panels = {}

    for ax, resource, letter in zip(axes, RESOURCES, "AB", strict=True):
        panel = data.loc[data.resource.astype(str).eq(resource)].copy()
        patients = patient_order(panel, resource)
        patient_x = centered_offsets(patients, 0.24)
        patient_rank = {patient: index for index, patient in enumerate(patients)}
        panel["readout_order"] = panel.readout_id.map({key: i for i, key in enumerate(READOUTS)})
        panel["readout_display_label"] = panel.readout_id.map(READOUT_LABELS)
        panel["patient_order"] = panel.patient_id.astype(str).map(patient_rank)
        panel["n_zero_mad_maps"] = (panel.n_evaluable_maps - panel.n_standardized_maps).astype(int)
        panel["has_zero_mad_maps"] = panel.n_zero_mad_maps.gt(0)
        panel["display_x"] = np.nan
        panel["display_status"] = "not_evaluable"
        panel["n_attempted_patients"] = 0
        panel["n_evaluable_patients"] = 0
        panel["resource_median_count_only_excess_abs_deviation_score"] = np.nan
        panel["resource_median_block_matched_excess_abs_deviation_score"] = np.nan
        tick_labels = []

        for position, readout in enumerate(READOUTS):
            rows = panel.loc[panel.readout_id.astype(str).eq(readout)].copy()
            if rows.empty:
                raise ValueError(f"Figure S3 has no rows for {resource}/{readout}")
            count_values = pd.to_numeric(rows[RAW_SCORE_COLUMNS["count_only"]], errors="coerce")
            block_values = pd.to_numeric(rows[RAW_SCORE_COLUMNS["block_matched"]], errors="coerce")
            geometry_evaluable = rows.n_evaluable_maps.gt(0)
            finite_scores = np.isfinite(count_values.to_numpy(float)) & np.isfinite(block_values.to_numpy(float))
            if (geometry_evaluable.to_numpy() & ~finite_scores).any():
                raise ValueError(f"Raw score-unit summary is missing for an evaluable patient: {resource}/{readout}")
            evaluable = rows.loc[geometry_evaluable].copy()
            if len(evaluable):
                med_count = float(evaluable[RAW_SCORE_COLUMNS["count_only"]].median())
                med_block = float(evaluable[RAW_SCORE_COLUMNS["block_matched"]].median())
                for row in evaluable.itertuples():
                    x = position + patient_x[str(row.patient_id)]
                    y_count = float(getattr(row, RAW_SCORE_COLUMNS["count_only"]))
                    y_block = float(getattr(row, RAW_SCORE_COLUMNS["block_matched"]))
                    ax.plot([x, x], [y_count, y_block], color=GREY, lw=0.7, alpha=0.35, zorder=1)
                    ax.scatter(x, y_count, color=BLUE, marker="o", s=29, alpha=0.42,
                               edgecolors="white", linewidths=0.35, zorder=3)
                    ax.scatter(x, y_block, color=ORANGE, marker="s", s=28, alpha=0.42,
                               edgecolors="white", linewidths=0.35, zorder=3)
                    plot_values.extend((y_count, y_block))
                median_x = position + 0.24
                ax.plot([median_x, median_x], [med_count, med_block], color=GREY,
                        lw=1.6, alpha=0.9, zorder=4)
                ax.scatter(median_x, med_count, color=BLUE, marker="o", s=82,
                           edgecolors="#25282B", linewidths=0.8, zorder=5)
                ax.scatter(median_x, med_block, color=ORANGE, marker="s", s=78,
                           edgecolors="#25282B", linewidths=0.8, zorder=5)
                plot_values.extend((med_count, med_block))
            else:
                med_count = med_block = np.nan

            attempted = int(rows.patient_id.nunique())
            n_evaluable = int(geometry_evaluable.sum())
            indices = rows.index
            panel.loc[indices, "display_x"] = [position + patient_x[str(p)] for p in rows.patient_id]
            panel.loc[indices, "display_status"] = np.where(geometry_evaluable, "evaluable", "not_evaluable")
            panel.loc[indices, "n_attempted_patients"] = attempted
            panel.loc[indices, "n_evaluable_patients"] = n_evaluable
            panel.loc[indices, "resource_median_count_only_excess_abs_deviation_score"] = med_count
            panel.loc[indices, "resource_median_block_matched_excess_abs_deviation_score"] = med_block
            tick_labels.append(f"{READOUT_LABELS[readout]}\nn={n_evaluable}/{attempted}")

        panel_title(ax, letter, resource)
        ax.set_xticks(np.arange(len(READOUTS)), tick_labels)
        ax.set_xlim(-0.48, len(READOUTS) - 0.28)
        ax.set_ylabel("Excess absolute departure\n(log-normalized expression units)")
        ax.axhline(0, color=GREY, lw=0.8, ls="--", zorder=0)
        ax.grid(axis="y", color="#E8ECEF", lw=0.5)
        source_panels[letter] = panel.sort_values(["readout_order", "patient_order"]).reset_index(drop=True)

    low = min(plot_values, default=0.0)
    high = max(plot_values, default=0.0)
    padding = max((high - low) * 0.12, 0.05)
    for ax in axes:
        ax.set_ylim(min(0, low) - padding, max(0, high) + padding)
    handles = [
        Line2D([], [], color=BLUE, marker="o", ls="", markersize=4.5,
               label="Size-matched"),
        Line2D([], [], color=ORANGE, marker="s", ls="", markersize=4.5,
               label="Location-matched"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.94),
               ncol=2, frameon=False, handletextpad=0.5, columnspacing=1.1)
    fig.suptitle("Component expression and regional selection", x=0.12, y=0.985,
                 ha="left", weight="bold", fontsize=12)
    save_figure(fig, figures, "figureS3")
    if export_sources:
        for letter, panel in source_panels.items():
            export_rows(panel, sources, f"figureS3_panel_{letter}")


def mad_status(frame: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(frame[MAD_COLUMN], errors="coerce")
    evaluable = frame.n_evaluable_maps.gt(0)
    standardized = frame.n_standardized_maps.gt(0)
    finite = np.isfinite(values.to_numpy(float))
    if (evaluable & standardized & ~finite).any():
        raise ValueError("Standardized patient summary is missing despite standardized maps")
    if ((~standardized) & finite).any():
        raise ValueError("MAD patient summary is present without standardized maps")
    return pd.Series(np.select(
        [evaluable & standardized & finite, evaluable & ~standardized],
        ["evaluable", "zero_mad_only"], default="no_evaluable_maps"), index=frame.index)


def denominator_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["resource", "eligibility_frame", "setting_id"]
    for identity, group in frame.groupby(keys, sort=False):
        resource, eligibility_frame, setting_id = identity
        map_evaluable = group.n_evaluable_maps.gt(0)
        mad_evaluable = group.mad_display_status.eq("evaluable")
        attempted_ids = patient_order(group, str(resource))
        map_patient_ids = patient_order(group.loc[map_evaluable], str(resource))
        mad_patient_ids = patient_order(group.loc[mad_evaluable], str(resource))
        rows.append({
            "resource": resource,
            "eligibility_frame": eligibility_frame,
            "setting_id": setting_id,
            "n_attempted_patients": int(group.patient_id.nunique()),
            "n_evaluable_patients": int(map_evaluable.sum()),
            "n_evaluable_mad_patients": int(mad_evaluable.sum()),
            "n_attempted_sections": int(group.n_attempted_sections.sum()),
            "n_evaluable_sections": int(group.n_evaluable_sections.sum()),
            "n_declared_maps": int(group.n_declared_maps.sum()),
            "n_evaluable_maps": int(group.n_evaluable_maps.sum()),
            "n_standardized_maps": int(group.n_standardized_maps.sum()),
            "attempted_patient_ids": ";".join(attempted_ids),
            "evaluable_map_patient_ids": ";".join(map_patient_ids),
            "evaluable_mad_patient_ids": ";".join(mad_patient_ids),
            "resource_median_block_matched_excess_abs_deviation_mad": (
                float(group.loc[mad_evaluable, MAD_COLUMN].median()) if mad_evaluable.any() else np.nan
            ),
        })
    return pd.DataFrame(rows)


def s4_figure(data: pd.DataFrame, figures: Path, sources: Path, *, export_sources: bool = True) -> None:
    selected = data.loc[
        data.readout_id.astype(str).eq("primary_barrier")
        & data.setting_id.astype(str).isin(SETTINGS)
        & data.eligibility_frame.astype(str).isin(FRAMES)
    ].copy()
    for resource in RESOURCES:
        for eligibility_frame in FRAMES:
            rows = selected.loc[
                selected.resource.astype(str).eq(resource)
                & selected.eligibility_frame.astype(str).eq(eligibility_frame)
            ]
            if set(rows.setting_id.astype(str)) != set(SETTINGS):
                raise ValueError(f"Figure S4 settings are incomplete for {resource}/{eligibility_frame}")

    selected["mad_display_status"] = mad_status(selected)
    denominators = denominator_summary(selected)
    selected = selected.merge(denominators, on=["resource", "eligibility_frame", "setting_id"],
                              how="left", validate="many_to_one", suffixes=("", "_resource"))
    selected["setting_order"] = selected.setting_id.map({key: i for i, key in enumerate(SETTINGS)})
    selected["setting_display_label"] = selected.setting_id.map(SETTING_LABELS)
    selected["display_x"] = np.nan
    selected["resource_median_display_x"] = np.nan
    left_panels = {}
    for resource in RESOURCES:
        panel = selected.loc[selected.resource.astype(str).eq(resource)].copy()
        patients = patient_order(panel, resource)
        patient_rank = {patient: index for index, patient in enumerate(patients)}
        panel["patient_order"] = panel.patient_id.astype(str).map(patient_rank)
        jitter = centered_offsets(patients, 0.08)
        for setting_order, setting in enumerate(SETTINGS):
            for eligibility_frame, (_, _, frame_offset, _) in FRAME_STYLE.items():
                mask = panel.eligibility_frame.astype(str).eq(eligibility_frame) & panel.setting_id.astype(str).eq(setting)
                rows = panel.loc[mask]
                panel.loc[rows.index, "display_x"] = [
                    setting_order + frame_offset + jitter[str(patient)] for patient in rows.patient_id
                ]
                median_offset = -0.09 if frame_offset < 0 else 0.09
                panel.loc[rows.index, "resource_median_display_x"] = setting_order + frame_offset + median_offset
        left_panels[resource] = panel.sort_values(
            ["setting_order", "eligibility_frame", "patient_order"]
        ).reset_index(drop=True)

    count_panels = denominators.copy()
    count_panels["setting_order"] = count_panels.setting_id.map({key: i for i, key in enumerate(SETTINGS)})
    count_panels["setting_display_label"] = count_panels.setting_id.map(SETTING_LABELS)
    count_panels["display_x"] = [
        float(row.setting_order) + FRAME_STYLE[str(row.eligibility_frame)][2]
        for row in count_panels.itertuples()
    ]
    count_panels["display_y"] = count_panels.n_evaluable_maps.astype(int)
    count_panels["display_label"] = [
        str(int(row.n_evaluable_maps)) for row in count_panels.itertuples()
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.8))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.83, bottom=0.15, wspace=0.40, hspace=0.67)
    plot_mad_values = []
    letters = {"GSE294385": ("A", "B"), "Valdeolivas": ("C", "D")}
    for row_index, resource in enumerate(RESOURCES):
        ax_mad, ax_counts = axes[row_index]
        panel = left_panels[resource]
        for eligibility_frame, (color, marker, frame_offset, label) in FRAME_STYLE.items():
            med_x, med_y = [], []
            for setting_order, setting in enumerate(SETTINGS):
                group = panel.loc[
                    panel.eligibility_frame.astype(str).eq(eligibility_frame)
                    & panel.setting_id.astype(str).eq(setting)
                ]
                evaluable = group.loc[group.mad_display_status.eq("evaluable")]
                if len(evaluable):
                    values = evaluable[MAD_COLUMN].to_numpy(float)
                    xs = evaluable.display_x.to_numpy(float)
                    ax_mad.scatter(xs, values, color=color, marker=marker, s=24,
                                   alpha=0.38, edgecolors="white", linewidths=0.35, zorder=2)
                    median = float(np.median(values))
                    median_x = setting_order + frame_offset + (-0.09 if frame_offset < 0 else 0.09)
                    med_x.append(median_x)
                    med_y.append(median)
                    plot_mad_values.extend(values.tolist())
            if med_x:
                ax_mad.scatter(med_x, med_y, color=color, marker=marker, s=76,
                               edgecolors="#25282B", linewidths=0.8, zorder=4)
        ax_mad.set_xticks(np.arange(len(SETTINGS)), [SETTING_LABELS[key] for key in SETTINGS])
        ax_mad.set_xlim(-0.55, len(SETTINGS) - 0.45)
        ax_mad.set_ylabel("Excess absolute departure\n(MAD units)")
        ax_mad.axhline(0, color=GREY, lw=0.8, ls="--", zorder=0)
        ax_mad.grid(axis="y", color="#E8ECEF", lw=0.5)
        panel_title(ax_mad, letters[resource][0], f"{resource}: location-matched excess")

        count_data = count_panels.loc[count_panels.resource.astype(str).eq(resource)]
        for frame_index, (eligibility_frame, (color, marker, _, label)) in enumerate(FRAME_STYLE.items()):
            rows = count_data.loc[count_data.eligibility_frame.astype(str).eq(eligibility_frame)].sort_values("setting_order")
            xs = rows.display_x.to_numpy(float)
            ys = rows.display_y.to_numpy(float)
            ax_counts.scatter(xs, ys, color=color, marker=marker,
                              s=30, label=label, zorder=2)
            for point, row in enumerate(rows.itertuples()):
                ax_counts.annotate(row.display_label, (xs[point], ys[point]),
                                   xytext=(0, -14 if frame_index == 0 else 9), textcoords="offset points",
                                   ha="center", color=color, fontsize=9.5)
        ax_counts.set_xticks(np.arange(len(SETTINGS)), [SETTING_LABELS[key] for key in SETTINGS])
        ax_counts.set_xlim(-0.72, len(SETTINGS) - 0.28)
        ax_counts.set_ylabel("Evaluable primary maps (count)")
        ax_counts.set_ylim(bottom=0)
        ax_counts.grid(axis="y", color="#E8ECEF", lw=0.5)
        panel_title(ax_counts, letters[resource][1], f"{resource}: primary map counts")

    limit = max((abs(value) for value in plot_mad_values), default=0.0)
    limit = max(limit * 1.15, 0.25)
    for row in range(2):
        axes[row, 0].set_ylim(-limit, limit)
    max_map_count = max(count_panels.n_evaluable_maps.max(), 1)
    for row in range(2):
        axes[row, 1].set_ylim(0, max_map_count * 1.27)
    frame_handles = [
        Line2D([], [], color=color, marker=marker, ls="", markersize=5,
               label=label)
        for color, marker, _, label in FRAME_STYLE.values()
    ]
    fig.legend(handles=frame_handles, loc="upper center", bbox_to_anchor=(0.57, 0.94),
               ncol=2, frameon=False, columnspacing=2)
    fig.suptitle("Four-gene mean sensitivity across spatial settings", x=0.12, y=0.985,
                 ha="left", weight="bold", fontsize=12)
    save_figure(fig, figures, "figureS4")

    if not export_sources:
        return

    for resource, letter in (("GSE294385", "A"), ("Valdeolivas", "C")):
        export_rows(left_panels[resource], sources, f"figureS4_panel_{letter}")
    for resource, letter in (("GSE294385", "B"), ("Valdeolivas", "D")):
        panel = count_panels.loc[count_panels.resource.astype(str).eq(resource)].sort_values(
            ["setting_order", "eligibility_frame"]
        ).reset_index(drop=True)
        export_rows(panel, sources, f"figureS4_panel_{letter}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--only", nargs="+", choices=("figureS3", "figureS4"),
                        default=("figureS3", "figureS4"))
    parser.add_argument("--figures-only", action="store_true",
                        help="Render frozen summaries without replacing source tables or manifests")
    args = parser.parse_args()
    input_path = args.pack_dir / "tables/regional_patient_sensitivity.tsv"
    data = read_summary(input_path)
    primary = primary_rows(data)
    s3_data = validate_s3_rows(primary)
    figures = args.pack_dir / "figures"
    sources = args.pack_dir / "tables/figure_source_data"
    configure()
    if "figureS3" in args.only:
        s3_figure(s3_data, figures, sources, export_sources=not args.figures_only)
    if "figureS4" in args.only:
        s4_figure(primary, figures, sources, export_sources=not args.figures_only)
    if args.figures_only:
        print("Rendered " + ", ".join(args.only) + "; source tables retained")
        return

    source_names = [f"figureS3_panel_{letter}.tsv" for letter in "AB"]
    source_names.extend(f"figureS4_panel_{letter}.tsv" for letter in "ABCD")
    manifest = {
        "script": "scripts/" + Path(__file__).name,
        "input_sha256": {portable_label(input_path, args.pack_dir): digest(input_path)},
        "primary_estimator": PRIMARY_ESTIMATOR,
        "figureS3": {
            "eligibility_frame": "independent_eligible", "setting_id": "baseline",
            "readouts": list(READOUTS),
            "summary_level": "patient medians; raw score-unit count-only and block-matched excess deviations",
            "zero_mad_policy": "retain all geometry-evaluable patient score-unit summaries",
        },
        "figureS4": {
            "eligibility_frames": list(FRAMES), "settings": list(SETTINGS),
            "summary_level": "patient-level block-matched MAD summaries and resource-level eligible map counts",
            "denominators": "panel source tables include attempted/evaluable patients, sections and maps",
        },
        "figures_sha256": {
            f"{stem}.{extension}": digest(args.pack_dir / "figures" / f"{stem}.{extension}")
            for stem in ("figureS3", "figureS4") for extension in ("pdf", "png", "tiff")
        },
        "source_data_sha256": {name: digest(sources / name) for name in source_names},
        "export": {"pdf": "vector, TrueType font embedded", "png_dpi": 300, "tiff_dpi": 600},
    }
    sources.mkdir(parents=True, exist_ok=True)
    manifest_path = sources / "regional_sensitivity_figure_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved Figures S3 and S4 with panel source data in {args.pack_dir}")


if __name__ == "__main__":
    main()
