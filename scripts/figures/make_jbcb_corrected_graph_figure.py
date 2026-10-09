#!/usr/bin/env python3
"""Preview paired patient summaries across the two corrected graph definitions.

This figure is descriptive: it plots the patient-level value for each graph
without pooling patients, imputing missing values, or adding inferential tests.
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
from jbcb_presentation_paths import default_package


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = default_package(__file__)
DEFAULT_INPUT = DEFAULT_PACK / "tables/regional_patient_sensitivity.tsv"
DEFAULT_OUTPUT = ROOT / "results"

RESOURCES = ("Valdeolivas", "GSE294385")
CONTROLS = {
    "count_only": "Size-matched reference",
    "block_matched": "Location-matched reference",
}
GRAPHS = ("historical_raw_6nn", "nominal_honeycomb_exact")
GRAPH_LABELS = {
    "historical_raw_6nn": "Array-coordinate 6NN",
    "nominal_honeycomb_exact": "Exact honeycomb",
}
GRAPH_TICK_LABELS = {
    "historical_raw_6nn": "Array-coordinate\n6NN",
    "nominal_honeycomb_exact": "Exact\nhoneycomb",
}
FRAME = "independent_eligible"
ESTIMATOR = "anchor_retention"
SETTING = "baseline"
READOUT = "primary_barrier"
EXPECTED_PATIENTS_BY_RESOURCE = {"Valdeolivas": 7, "GSE294385": 5}
EXPECTED_TOTAL_PATIENTS = 12
METRIC_COLUMNS = {
    "count_only": "count_only_excess_abs_deviation_score",
    "block_matched": "block_matched_excess_abs_deviation_score",
}
EXPECTED_SOURCE_ROWS = EXPECTED_TOTAL_PATIENTS * len(GRAPHS) * len(METRIC_COLUMNS)

GRAPH_STYLE = {
    "historical_raw_6nn": ("#0072B2", "o"),
    "nominal_honeycomb_exact": ("#D55E00", "s"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 10,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def load_rows(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Patient summary not found: {path}")
    data = read(path, sep="\t", low_memory=False, na_values=["NA", ""])
    data["input_row_number"] = np.arange(2, len(data) + 2, dtype=int)
    required = {
        "eligibility_frame", "resource", "patient_id", "graph_id", "estimator_id",
        "setting_id", "readout_id", "n_attempted_sections", "n_evaluable_sections",
        "n_declared_maps", "n_evaluable_maps", *METRIC_COLUMNS.values(),
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "The corrected patient table is not ready for this figure; missing columns: "
            + ", ".join(sorted(missing))
        )

    keep = (
        data["eligibility_frame"].astype(str).eq(FRAME)
        & data["estimator_id"].astype(str).eq(ESTIMATOR)
        & data["setting_id"].astype(str).eq(SETTING)
        & data["readout_id"].astype(str).eq(READOUT)
        & data["resource"].astype(str).isin(RESOURCES)
        & data["graph_id"].astype(str).isin(GRAPHS)
    )
    selected = data.loc[keep].copy()
    if selected.empty:
        raise ValueError("No rows match the frozen Figure S6 selection contract")

    key = ["resource", "patient_id", "graph_id"]
    if selected.duplicated(key).any():
        duplicates = selected.loc[selected.duplicated(key, keep=False), key]
        raise ValueError("Duplicate selected patient/graph rows:\n" + duplicates.to_string(index=False))

    seen_resources = set(selected.resource.astype(str))
    if seen_resources != set(RESOURCES):
        raise ValueError(f"Expected resources {RESOURCES}; observed {sorted(seen_resources)}")
    seen_graphs = set(selected.graph_id.astype(str))
    if seen_graphs != set(GRAPHS):
        raise ValueError(f"Expected graph IDs {GRAPHS}; observed {sorted(seen_graphs)}")

    for resource in RESOURCES:
        part = selected.loc[selected.resource.astype(str).eq(resource)]
        patient_sets = {
            graph: frozenset(part.loc[part.graph_id.astype(str).eq(graph), "patient_id"].astype(str))
            for graph in GRAPHS
        }
        if patient_sets[GRAPHS[0]] != patient_sets[GRAPHS[1]]:
            raise ValueError(f"Graph patient IDs differ for {resource}; do not plot unmatched sets")
        expected_count = EXPECTED_PATIENTS_BY_RESOURCE[resource]
        if len(patient_sets[GRAPHS[0]]) != expected_count:
            raise ValueError(
                f"Expected {expected_count} attempted patients for {resource}; "
                f"found {len(patient_sets[GRAPHS[0]])}"
            )
    if sum(EXPECTED_PATIENTS_BY_RESOURCE.values()) != EXPECTED_TOTAL_PATIENTS:
        raise ValueError("Resource patient denominators do not sum to the declared total")

    selected["patient_id"] = selected.patient_id.astype(str)
    for column in [*METRIC_COLUMNS.values(), "n_attempted_sections", "n_evaluable_sections",
                   "n_declared_maps", "n_evaluable_maps"]:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    if (selected["n_evaluable_maps"].dropna() < 0).any():
        raise ValueError("n_evaluable_maps cannot be negative")
    if (selected["n_declared_maps"].dropna() < selected["n_evaluable_maps"].dropna()).any():
        raise ValueError("Expected 0 <= n_evaluable_maps <= n_declared_maps")
    return selected


def patient_order(rows: pd.DataFrame) -> list[str]:
    return sorted(rows.patient_id.astype(str).unique(), key=lambda item: (int(item.removeprefix("Patient")) if item.removeprefix("Patient").isdigit() else 10**9, item))


def make_source_table(data: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for resource in RESOURCES:
        for control, metric in METRIC_COLUMNS.items():
            part = data.loc[data.resource.astype(str).eq(resource)].copy()
            part["metric_name"] = metric
            part["source_metric_value"] = part[metric]
            part["map_evaluable"] = part.n_evaluable_maps.fillna(0).gt(0)
            part["metric_value"] = part.source_metric_value.where(part.map_evaluable)
            part["control_id"] = control
            part["control_label"] = CONTROLS[control]
            part["metric_missing"] = part.metric_value.isna()
            frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    columns = [
        "eligibility_frame", "resource", "patient_id", "graph_id", "estimator_id",
        "setting_id", "readout_id", "control_id", "control_label", "metric_name",
        "source_metric_value", "metric_value", "metric_missing", "map_evaluable", "n_attempted_sections",
        "n_evaluable_sections", "n_declared_maps", "n_evaluable_maps", "input_row_number",
    ]
    out = out.loc[:, columns].sort_values(["resource", "control_id", "patient_id", "graph_id"]).reset_index(drop=True)
    if len(out) != EXPECTED_SOURCE_ROWS:
        raise ValueError(f"Expected {EXPECTED_SOURCE_ROWS} source rows (7 + 5 patients × 2 graphs × 2 controls); found {len(out)}")
    return out


def graph_tick_labels(panel: pd.DataFrame) -> list[str]:
    labels = []
    for graph in GRAPHS:
        values = panel.loc[panel.graph_id.astype(str).eq(graph), "metric_value"]
        labels.append(GRAPH_TICK_LABELS[graph] + ("*" if values.isna().any() else ""))
    return labels


def plot_figure(source: pd.DataFrame, figures_dir: Path, input_path: Path) -> dict[str, object]:
    configure()
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.0), sharex=False)
    fig.subplots_adjust(left=0.12, right=0.975, top=0.80, bottom=0.16, hspace=0.55, wspace=0.34)
    letters = ("A", "B", "C", "D")
    panel_records = []

    for ax, (resource, control), letter in zip(
        axes.flat,
        [(r, c) for r in RESOURCES for c in METRIC_COLUMNS],
        letters,
        strict=True,
    ):
        panel = source.loc[
            source.resource.astype(str).eq(resource) & source.control_id.astype(str).eq(control)
        ].copy()
        patient_ids = patient_order(panel)
        jitter = {patient: (i - (len(patient_ids) - 1) / 2) * 0.014 for i, patient in enumerate(patient_ids)}

        missing_by_graph = {}
        for graph in GRAPHS:
            missing_by_graph[graph] = int(panel.loc[panel.graph_id.astype(str).eq(graph), "metric_value"].isna().sum())

        for patient in patient_ids:
            values = {}
            for graph in GRAPHS:
                row = panel.loc[panel.patient_id.astype(str).eq(patient) & panel.graph_id.astype(str).eq(graph)]
                if len(row) != 1:
                    raise ValueError(f"Expected one row for {resource}/{control}/{patient}/{graph}")
                value = row.metric_value.iloc[0]
                values[graph] = float(value) if pd.notna(value) else np.nan
            xvals = np.arange(2, dtype=float) + jitter[patient]
            yvals = np.array([values[g] for g in GRAPHS], dtype=float)
            finite = np.isfinite(yvals)
            if finite.all():
                ax.plot(xvals, yvals, color="#78858D", alpha=0.55, lw=0.75, zorder=1)
            for i, graph in enumerate(GRAPHS):
                if np.isfinite(yvals[i]):
                    color, marker = GRAPH_STYLE[graph]
                    ax.scatter([xvals[i]], [yvals[i]], s=28, marker=marker,
                               c=[color], edgecolors="white", linewidths=0.35, zorder=2)

        control_label = CONTROLS[control].capitalize()
        ax.set_title(f"{letter}  {resource} · {control_label}", loc="left", weight="bold", pad=8)
        ax.set_xticks([0, 1], graph_tick_labels(panel))
        ax.set_xlim(-0.35, 1.35)
        ax.tick_params(axis="x", labelrotation=0, labelsize=9, pad=5)
        ax.axhline(0, color="#AEB8BE", lw=0.65, zorder=0)
        ax.grid(axis="y", color="#E7EBEE", lw=0.5)
        ax.set_axisbelow(True)
        ax.set_ylabel("Excess absolute departure\n(log-normalized expression units)")
        ax.margins(y=0.20)
        ax.text(0.99, 0.98, f"{len(patient_ids)} attempted patients", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="#52616B")
        panel_records.append({
            "resource": resource,
            "control_id": control,
            "metric_column": METRIC_COLUMNS[control],
            "attempted_patients": len(patient_ids),
            "missing_by_graph": missing_by_graph,
        })

    graph_handles = [
        Line2D([], [], color=GRAPH_STYLE[g][0], marker=GRAPH_STYLE[g][1], ls="",
               markersize=5, label=GRAPH_LABELS[g])
        for g in GRAPHS
    ]
    fig.legend(handles=graph_handles, loc="upper center", bbox_to_anchor=(0.56, 0.91),
               ncol=2, frameon=False, columnspacing=1.8, handletextpad=0.45)
    fig.suptitle("Patient-level summaries across graph definitions", x=0.12, y=0.975,
                 ha="left", weight="bold", fontsize=11)
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = "figureS6"
    fig.savefig(figures_dir / f"{stem}.pdf", metadata={"Creator": "make_jbcb_corrected_graph_figure.py"})
    fig.savefig(figures_dir / f"{stem}.png", dpi=300)
    fig.savefig(figures_dir / f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return {
        "figure_stem": stem,
        "panels": panel_records,
        "input_path": "tables/" + input_path.name,
        "input_sha256": sha256(input_path),
        "export": {"pdf_fonttype": 42, "ps_fonttype": 42, "png_dpi": 300, "tiff_dpi": 600},
        "figure_sha256": {f"{stem}.{ext}": sha256(figures_dir / f"{stem}.{ext}") for ext in ("pdf", "png", "tiff")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    args.input = args.input or args.pack_dir / "tables/regional_patient_sensitivity.tsv"
    args.output_dir = args.output_dir or args.pack_dir
    data = load_rows(args.input)
    plotted_source = make_source_table(data)
    figures_dir = args.output_dir / "figures"
    source_dir = args.output_dir / "tables/figure_source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    manifest = plot_figure(plotted_source, figures_dir, args.input)
    source_path = source_dir / "figureS6_panel_source.tsv"
    plotted_source.to_csv(source_path, sep="\t", index=False, na_rep="NA")
    manifest["source_tsv"] = source_path.name
    manifest["source_tsv_sha256"] = sha256(source_path)
    manifest["selection"] = {
        "eligibility_frame": FRAME,
        "setting_id": SETTING,
        "readout_id": READOUT,
        "estimator_id": ESTIMATOR,
        "graph_ids": list(GRAPHS),
        "resources": list(RESOURCES),
        "controls_and_metrics": METRIC_COLUMNS,
        "expected_attempted_patients_by_resource": EXPECTED_PATIENTS_BY_RESOURCE,
        "expected_attempted_patients_total": EXPECTED_TOTAL_PATIENTS,
        "expected_source_rows": EXPECTED_SOURCE_ROWS,
        "missing_policy": "preserve NA; summaries with zero evaluable maps are not plotted; never impute; add an asterisk to a graph tick in a panel when any patient value is missing/unavailable",
    }
    manifest_path = source_dir / "figureS6_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved preview to {args.output_dir}")


if __name__ == "__main__":
    main()
