#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def ffloat(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize BayesSpace nrep sensitivity (nrep=100 vs nrep=1000) for a small subset of sections."
    )
    parser.add_argument(
        "--method-benchmark",
        default="results/benchmarks/method_benchmark.tsv",
        help="Main benchmark TSV (nrep=100 BayesSpace rows are used as baseline).",
    )
    parser.add_argument(
        "--nrep1000",
        default="results/benchmarks/bayesspace_nrep1000_sensitivity.tsv",
        help="BayesSpace reruns with nrep=1000 (output of stage3g runner).",
    )
    parser.add_argument(
        "--out",
        default="results/benchmarks/bayesspace_nrep_sensitivity_summary.tsv",
        help="Output TSV path.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    method_benchmark_path = (repo_root / args.method_benchmark).resolve()
    nrep1000_path = (repo_root / args.nrep1000).resolve()
    out_path = (repo_root / args.out).resolve()

    mb = read_tsv(method_benchmark_path)
    nrep1000 = read_tsv(nrep1000_path)

    # Filter to BayesSpace nrep=100 (main benchmark) for the same sample×K units
    mb_bayes = {
        (r["dataset_id"], r["sample_id"], r["K"]): r
        for r in mb
        if r.get("method_id") == "BayesSpace"
        and r.get("dataset_id")
        and r.get("sample_id")
        and r.get("K")
    }
    mb_stagate = {
        (r["dataset_id"], r["sample_id"], r["K"]): r
        for r in mb
        if r.get("method_id") == "M5_stagate"
        and r.get("dataset_id")
        and r.get("sample_id")
        and r.get("K")
    }

    rows: list[dict[str, str]] = []
    for r1000 in nrep1000:
        key = (r1000.get("dataset_id", ""), r1000.get("sample_id", ""), r1000.get("K", ""))
        if key not in mb_bayes:
            continue
        r100 = mb_bayes[key]
        r_m5 = mb_stagate.get(key, {})

        spatial_100 = ffloat(r100.get("spatial_coherence_median", ""))
        marker_100 = ffloat(r100.get("marker_coherence_median", ""))
        ari_100 = ffloat(r100.get("stability_ari_median", ""))
        rt_100 = ffloat(r100.get("wall_time_sec_median", ""))

        spatial_1000 = ffloat(r1000.get("spatial_coherence_median", ""))
        marker_1000 = ffloat(r1000.get("marker_coherence_median", ""))
        ari_1000 = ffloat(r1000.get("stability_ari_median", ""))
        rt_1000 = ffloat(r1000.get("wall_time_sec_median", ""))

        m5_spatial = ffloat(r_m5.get("spatial_coherence_median", ""))
        m5_marker = ffloat(r_m5.get("marker_coherence_median", ""))

        rows.append(
            {
                "dataset_id": key[0],
                "sample_id": key[1],
                "K": key[2],
                "seeds": "11,23",
                "nrep_main": "100",
                "nrep_sensitivity": "1000",
                "bayes_spatial_main": r100.get("spatial_coherence_median", ""),
                "bayes_spatial_nrep1000": r1000.get("spatial_coherence_median", ""),
                "bayes_spatial_delta": f"{spatial_1000 - spatial_100:.6f}",
                "bayes_marker_main": r100.get("marker_coherence_median", ""),
                "bayes_marker_nrep1000": r1000.get("marker_coherence_median", ""),
                "bayes_marker_delta": f"{marker_1000 - marker_100:.6f}",
                "bayes_ari_main": r100.get("stability_ari_median", ""),
                "bayes_ari_nrep1000": r1000.get("stability_ari_median", ""),
                "bayes_ari_delta": f"{ari_1000 - ari_100:.6f}",
                "bayes_runtime_sec_main": r100.get("wall_time_sec_median", ""),
                "bayes_runtime_sec_nrep1000": r1000.get("wall_time_sec_median", ""),
                "bayes_runtime_ratio": "" if rt_100 != rt_100 or rt_100 == 0 else f"{rt_1000 / rt_100:.3f}",
                "stagate_spatial_main": r_m5.get("spatial_coherence_median", ""),
                "stagate_marker_main": r_m5.get("marker_coherence_median", ""),
                "delta_spatial_vs_stagate_main": "" if m5_spatial != m5_spatial else f"{spatial_100 - m5_spatial:.6f}",
                "delta_spatial_vs_stagate_nrep1000": "" if m5_spatial != m5_spatial else f"{spatial_1000 - m5_spatial:.6f}",
                "delta_marker_vs_stagate_main": "" if m5_marker != m5_marker else f"{marker_100 - m5_marker:.6f}",
                "delta_marker_vs_stagate_nrep1000": "" if m5_marker != m5_marker else f"{marker_1000 - m5_marker:.6f}",
                "notes": "BayesSpace nrep sensitivity on two sections (1 stable, 1 unstable in main runs).",
            }
        )

    rows.sort(key=lambda r: (r["dataset_id"], r["sample_id"], int(r["K"])))
    write_tsv(
        out_path,
        rows,
        fieldnames=[
            "dataset_id",
            "sample_id",
            "K",
            "seeds",
            "nrep_main",
            "nrep_sensitivity",
            "bayes_spatial_main",
            "bayes_spatial_nrep1000",
            "bayes_spatial_delta",
            "bayes_marker_main",
            "bayes_marker_nrep1000",
            "bayes_marker_delta",
            "bayes_ari_main",
            "bayes_ari_nrep1000",
            "bayes_ari_delta",
            "bayes_runtime_sec_main",
            "bayes_runtime_sec_nrep1000",
            "bayes_runtime_ratio",
            "stagate_spatial_main",
            "stagate_marker_main",
            "delta_spatial_vs_stagate_main",
            "delta_spatial_vs_stagate_nrep1000",
            "delta_marker_vs_stagate_main",
            "delta_marker_vs_stagate_nrep1000",
            "notes",
        ],
    )
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
