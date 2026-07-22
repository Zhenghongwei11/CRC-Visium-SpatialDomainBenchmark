#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def fnum(x: str) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


RE_CFG_STAGATE = re.compile(r"^sensitivity_stagate_latent(\d+)_epochs(\d+)$")
RE_CFG_SPAGCN = re.compile(r"^sensitivity_spagcn_p([0-9]*\\.?[0-9]+)$")


def parse_cfg(cfg: str) -> dict[str, str]:
    cfg = (cfg or "").strip()
    m = RE_CFG_STAGATE.match(cfg)
    if m:
        latent, epochs = m.groups()
        return {
            "config_family": "stagate",
            "stagate_latent_dim": latent,
            "stagate_max_epochs": epochs,
            "spagcn_p": "",
        }
    m = RE_CFG_SPAGCN.match(cfg)
    if m:
        return {
            "config_family": "spagcn",
            "stagate_latent_dim": "",
            "stagate_max_epochs": "",
            "spagcn_p": m.group(1),
        }
    return {
        "config_family": "unknown",
        "stagate_latent_dim": "",
        "stagate_max_epochs": "",
        "spagcn_p": "",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize baseline hyperparameter sensitivity runs for STAGATE/SpaGCN.")
    ap.add_argument(
        "--input-dir",
        default="results/sensitivity_runs",
        help="Directory containing per-config subfolders (each with method_benchmark.tsv).",
    )
    ap.add_argument(
        "--output-tsv",
        default="results/benchmarks/baseline_hyperparam_sensitivity.tsv",
        help="Per-config table.",
    )
    ap.add_argument(
        "--summary-tsv",
        default="results/benchmarks/baseline_hyperparam_sensitivity_summary.tsv",
        help="Aggregate summary table.",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    frames: list[pd.DataFrame] = []
    for cfg_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        mb = cfg_dir / "method_benchmark.tsv"
        if not mb.exists():
            continue
        df = read_tsv(mb)
        df["config_id"] = cfg_dir.name
        frames.append(df)

    if not frames:
        raise SystemExit(f"No per-config method_benchmark.tsv found under {input_dir}")

    df = pd.concat(frames, ignore_index=True)

    needed = {"dataset_id", "sample_id", "method_id", "K", "config_id"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in concatenated sensitivity input: {', '.join(sorted(missing))}")

    df = df[df["method_id"].isin(["M4_spagcn", "M5_stagate"])].copy()
    if df.empty:
        raise SystemExit("No M4_spagcn/M5_stagate rows found in sensitivity inputs")

    df["seed_count_num"] = df["seed_count"].map(lambda x: fnum(x) or 0.0)
    df["_row"] = range(len(df))
    df = (
        df.sort_values(["dataset_id", "sample_id", "method_id", "K", "config_id", "seed_count_num", "_row"])
        .drop_duplicates(["dataset_id", "sample_id", "method_id", "K", "config_id"], keep="last")
        .copy()
    )

    out_rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        cfg = str(r.get("config_id", "")).strip()
        cfg_parsed = parse_cfg(cfg)

        out_rows.append(
            {
                "dataset_id": r.get("dataset_id", ""),
                "sample_id": r.get("sample_id", ""),
                "method_id": r.get("method_id", ""),
                "K": r.get("K", ""),
                "seed_count": r.get("seed_count", ""),
                "config_id": cfg,
                "config_family": cfg_parsed["config_family"],
                "spagcn_p": cfg_parsed["spagcn_p"],
                "stagate_latent_dim": cfg_parsed["stagate_latent_dim"],
                "stagate_max_epochs": cfg_parsed["stagate_max_epochs"],
                "spatial_coherence_median": r.get("spatial_coherence_median", ""),
                "marker_coherence_median": r.get("marker_coherence_median", ""),
                "stability_ari_median": r.get("stability_ari_median", ""),
                "wall_time_sec_median": r.get("wall_time_sec_median", ""),
                "failure_rate": r.get("failure_rate", ""),
                "notes": r.get("notes", ""),
            }
        )

    out_fields = [
        "dataset_id",
        "sample_id",
        "method_id",
        "K",
        "seed_count",
        "config_id",
        "config_family",
        "spagcn_p",
        "stagate_latent_dim",
        "stagate_max_epochs",
        "spatial_coherence_median",
        "marker_coherence_median",
        "stability_ari_median",
        "wall_time_sec_median",
        "failure_rate",
        "notes",
    ]
    write_tsv(Path(args.output_tsv), out_rows, fieldnames=out_fields)

    # Aggregate: for each method×metric, summarize spread across the hyperparameter grid within each section×K,
    # then report medians across the two sections (so this doesn't masquerade as a cohort-level benchmark).
    df_out = pd.DataFrame(out_rows)
    for col in ["spatial_coherence_median", "marker_coherence_median", "stability_ari_median", "wall_time_sec_median"]:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

    summary_rows: list[dict[str, Any]] = []
    metrics = ["spatial_coherence_median", "marker_coherence_median", "stability_ari_median", "wall_time_sec_median"]
    for (method_id, k), sub in df_out.groupby(["method_id", "K"], dropna=False):
        for metric in metrics:
            vals_by_unit = []
            for (_, __), unit in sub.groupby(["dataset_id", "sample_id"], dropna=False):
                v = unit[metric].dropna().to_numpy()
                if v.size == 0:
                    continue
                vals_by_unit.append(
                    {
                        "unit_median": float(pd.Series(v).median()),
                        "unit_min": float(pd.Series(v).min()),
                        "unit_max": float(pd.Series(v).max()),
                        "unit_iqr": float(pd.Series(v).quantile(0.75) - pd.Series(v).quantile(0.25)),
                    }
                )
            if not vals_by_unit:
                continue
            u = pd.DataFrame(vals_by_unit)
            summary_rows.append(
                {
                    "method_id": method_id,
                    "K": str(k),
                    "metric_id": metric,
                    "n_units": int(len(vals_by_unit)),
                    "median_of_unit_medians": float(u["unit_median"].median()),
                    "median_unit_range": float((u["unit_max"] - u["unit_min"]).median()),
                    "median_unit_iqr": float(u["unit_iqr"].median()),
                    "notes": "2-section hyperparameter sweep; interpret as sensitivity, not performance ranking",
                }
            )

    write_tsv(
        Path(args.summary_tsv),
        summary_rows,
        fieldnames=[
            "method_id",
            "K",
            "metric_id",
            "n_units",
            "median_of_unit_medians",
            "median_unit_range",
            "median_unit_iqr",
            "notes",
        ],
    )

    print(f"Wrote {args.output_tsv}")
    print(f"Wrote {args.summary_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
