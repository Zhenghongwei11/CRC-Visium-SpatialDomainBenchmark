#!/usr/bin/env python3

from __future__ import annotations

import csv
import itertools
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


RE_INT = re.compile(r"(-?\d+)")
RE_STAGE3_BASELINE = re.compile(r"^stage3[a-z]-full-replication")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_int_maybe(value: str) -> int | None:
    if value is None:
        return None
    match = RE_INT.search(str(value))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def count_extracted_samples(dataset_root: Path) -> int:
    if not dataset_root.exists():
        return 0
    # Flat mtx format
    prefixes = set()
    for path in dataset_root.glob("*_matrix.mtx.gz"):
        prefixes.add(path.name.replace("_matrix.mtx.gz", ""))
    if prefixes:
        return len(prefixes)
    # 10x h5 format
    h5s = list(dataset_root.glob("*_filtered_feature_bc_matrix.h5"))
    if h5s:
        return len(h5s)
    return 0


def read_tsv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        line = handle.readline().rstrip("\n")
    return line.split("\t") if line else []


@dataclass(frozen=True)
class MethodKey:
    dataset_id: str
    sample_id: str
    method_id: str
    K: str


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    docs_dir = repo_root / "docs"
    results_dir = repo_root / "results"

    dataset_landscape_path = docs_dir / "DATASET_LANDSCAPE.tsv"
    method_benchmark_path = results_dir / "benchmarks" / "method_benchmark.tsv"
    gate_summary_path = results_dir / "benchmarks" / "statistical_gate_summary.tsv"

    if not dataset_landscape_path.exists():
        raise FileNotFoundError(dataset_landscape_path)
    if not method_benchmark_path.exists():
        raise FileNotFoundError(method_benchmark_path)
    if not gate_summary_path.exists():
        raise FileNotFoundError(gate_summary_path)

    dataset_landscape = read_tsv(dataset_landscape_path)
    method_benchmark_header = read_tsv_header(method_benchmark_path)
    method_benchmark = read_tsv(method_benchmark_path)
    gate_summary = read_tsv(gate_summary_path)

    # -------------------------
    # results/dataset_summary.tsv
    # -------------------------
    mb_by_dataset_samples: dict[str, set[str]] = defaultdict(set)
    mb_by_dataset_methods: dict[str, set[str]] = defaultdict(set)
    mb_by_dataset_bayespace_samples: dict[str, set[str]] = defaultdict(set)
    mb_by_dataset_bayespace_rows: dict[str, int] = defaultdict(int)

    for row in method_benchmark:
        ds = row.get("dataset_id", "").strip()
        sid = row.get("sample_id", "").strip()
        mid = row.get("method_id", "").strip()
        if not ds or not sid or not mid:
            continue
        mb_by_dataset_samples[ds].add(sid)
        mb_by_dataset_methods[ds].add(mid)
        if mid == "BayesSpace":
            mb_by_dataset_bayespace_samples[ds].add(sid)
            mb_by_dataset_bayespace_rows[ds] += 1

    ds_rows: list[dict[str, Any]] = []
    for row in dataset_landscape:
        ds = (row.get("dataset_id_or_name") or "").strip()
        if not ds.startswith("GSE"):
            continue
        extracted_root = repo_root / "data" / "raw" / ds / "extracted"
        samples_on_disk = count_extracted_samples(extracted_root)

        role = (row.get("intended_role") or "").strip()
        modality = (row.get("modality") or "").strip()
        sample_size_declared = parse_int_maybe(row.get("sample_size") or "")
        methods_covered = sorted(mb_by_dataset_methods.get(ds, set()))
        bayespace_samples = sorted(mb_by_dataset_bayespace_samples.get(ds, set()))

        has_flat_mtx = bool(list(extracted_root.glob("*_matrix.mtx.gz")))
        has_h5 = bool(list(extracted_root.glob("*_filtered_feature_bc_matrix.h5")))
        format_hint = "mtx" if has_flat_mtx else ("h5" if has_h5 else "unknown")

        ds_rows.append(
            {
                "dataset_id": ds,
                "intended_role": role,
                "modality": modality,
                "sample_size_declared": sample_size_declared if sample_size_declared is not None else "",
                "samples_on_disk": samples_on_disk,
                "format_hint": format_hint,
                "samples_in_method_benchmark": len(mb_by_dataset_samples.get(ds, set())),
                "methods_covered": ",".join(methods_covered),
                "bayesspace_samples_covered": len(bayespace_samples),
                "bayesspace_rows": mb_by_dataset_bayespace_rows.get(ds, 0),
                "notes": (row.get("notes") or "").strip(),
            }
        )

    write_tsv(
        results_dir / "dataset_summary.tsv",
        ds_rows,
        fieldnames=[
            "dataset_id",
            "intended_role",
            "modality",
            "sample_size_declared",
            "samples_on_disk",
            "format_hint",
            "samples_in_method_benchmark",
            "methods_covered",
            "bayesspace_samples_covered",
            "bayesspace_rows",
            "notes",
        ],
    )

    # -------------------------
    # results/effect_sizes/claim_effects.tsv
    # -------------------------
    effect_rows: list[dict[str, Any]] = []
    for row in gate_summary:
        note = (row.get("notes") or "").strip()
        n_val = None
        for key in ["n_samples", "n_sample_k", "n"]:
            match = re.search(rf"{key}=(\d+)", note)
            if match:
                n_val = int(match.group(1))
                break
        effect_rows.append(
            {
                "claim_id": row.get("claim_id", ""),
                "dataset_id": "meta",
                "outcome": row.get("metric_id", ""),
                "model": row.get("comparison_id", ""),
                "analysis_unit": row.get("analysis_unit", ""),
                "test_name": row.get("test_name", ""),
                "effect_type": row.get("effect_size_name", ""),
                "effect": row.get("effect_size_value", ""),
                "ci_lower": row.get("effect_size_ci_lower", ""),
                "ci_upper": row.get("effect_size_ci_upper", ""),
                "pvalue": row.get("pvalue", ""),
                "fdr": row.get("fdr", ""),
                "n": n_val if n_val is not None else "",
                "overall_gate_status": row.get("overall_gate_status", ""),
                "support_tier": row.get("support_tier", ""),
                "notes": note,
            }
        )

    write_tsv(
        results_dir / "effect_sizes" / "claim_effects.tsv",
        effect_rows,
        fieldnames=[
            "claim_id",
            "dataset_id",
            "outcome",
            "model",
            "analysis_unit",
            "test_name",
            "effect_type",
            "effect",
            "ci_lower",
            "ci_upper",
            "pvalue",
            "fdr",
            "n",
            "overall_gate_status",
            "support_tier",
            "notes",
        ],
    )

    # -------------------------
    # results/replication/* (lightweight tables)
    # -------------------------
    baseline_methods = {
        "M0_expr_kmeans",
        "M1_spatial_concat_kmeans",
        "M2_spatial_ward",
        "M3_spatial_leiden",
        "M4_spagcn",
        "M5_stagate",
    }

    # Lock the benchmark table to the rows used for claims/figures:
    # - BayesSpace: rigor-backfill rows only (dedup by max seed_count per sample×K)
    # - Baselines: stage-3a/3b full-replication rows (including -m2/-m4 variants; same prefix)
    locked_candidates: list[tuple[int, dict[str, str]]] = []
    for idx, row in enumerate(method_benchmark):
        mid = (row.get("method_id") or "").strip()
        note = (row.get("notes") or "").strip()
        if mid == "BayesSpace" and note.startswith("rigor-backfill"):
            locked_candidates.append((idx, row))
            continue
        if mid in baseline_methods and RE_STAGE3_BASELINE.match(note):
            locked_candidates.append((idx, row))

    def seed_count(row: dict[str, str]) -> int:
        try:
            return int(float(row.get("seed_count") or "0"))
        except Exception:
            return 0

    locked_by_key: dict[MethodKey, tuple[int, int, dict[str, str]]] = {}
    for idx, row in locked_candidates:
        key = MethodKey(
            dataset_id=(row.get("dataset_id") or "").strip(),
            sample_id=(row.get("sample_id") or "").strip(),
            method_id=(row.get("method_id") or "").strip(),
            K=(row.get("K") or "").strip(),
        )
        if not key.dataset_id or not key.sample_id or not key.method_id or not key.K:
            continue
        sc = seed_count(row)
        prev = locked_by_key.get(key)
        if prev is None:
            locked_by_key[key] = (sc, idx, row)
            continue
        prev_sc, prev_idx, _ = prev
        if sc > prev_sc or (sc == prev_sc and idx > prev_idx):
            locked_by_key[key] = (sc, idx, row)

    locked_rows = [r for _, __, r in sorted(locked_by_key.values(), key=lambda t: t[1])]
    if method_benchmark_header:
        write_tsv(
            results_dir / "benchmarks" / "method_benchmark_locked.tsv",
            locked_rows,
            fieldnames=method_benchmark_header,
        )

    index: dict[MethodKey, dict[str, str]] = {}
    for row in locked_rows:
        key = MethodKey(
            dataset_id=row.get("dataset_id", ""),
            sample_id=row.get("sample_id", ""),
            method_id=row.get("method_id", ""),
            K=row.get("K", ""),
        )
        if not key.dataset_id or not key.sample_id or not key.method_id or not key.K:
            continue
        index[key] = row

    bayespace_keys = [k for k in index.keys() if k.method_id == "BayesSpace"]
    deltas: list[dict[str, Any]] = []
    for k in bayespace_keys:
        bs = index[k]
        for baseline in sorted(baseline_methods):
            base_key = MethodKey(k.dataset_id, k.sample_id, baseline, k.K)
            if base_key not in index:
                continue
            base = index[base_key]
            def fnum(val: str) -> float | None:
                try:
                    return float(val)
                except Exception:
                    return None

            bs_sp = fnum(bs.get("spatial_coherence_median", ""))
            bs_mk = fnum(bs.get("marker_coherence_median", ""))
            base_sp = fnum(base.get("spatial_coherence_median", ""))
            base_mk = fnum(base.get("marker_coherence_median", ""))
            if bs_sp is None or bs_mk is None or base_sp is None or base_mk is None:
                continue
            deltas.append(
                {
                    "dataset_id": k.dataset_id,
                    "sample_id": k.sample_id,
                    "K": k.K,
                    "baseline_method_id": baseline,
                    "delta_spatial_coherence": bs_sp - base_sp,
                    "delta_marker_coherence": bs_mk - base_mk,
                    "bayesspace_note": bs.get("notes", ""),
                    "baseline_note": base.get("notes", ""),
                }
            )

    write_tsv(
        results_dir / "replication" / "domain_quality_deltas_by_sample.tsv",
        deltas,
        fieldnames=[
            "dataset_id",
            "sample_id",
            "K",
            "baseline_method_id",
            "delta_spatial_coherence",
            "delta_marker_coherence",
            "bayesspace_note",
            "baseline_note",
        ],
    )

    # BayesSpace stability summaries per sample/K (for replication reporting)
    stability_rows: list[dict[str, Any]] = []
    for k in bayespace_keys:
        bs = index[k]
        stability_rows.append(
            {
                "dataset_id": k.dataset_id,
                "sample_id": k.sample_id,
                "K": k.K,
                "seed_count": bs.get("seed_count", ""),
                "stability_ari_median": bs.get("stability_ari_median", ""),
                "stability_ari_iqr": bs.get("stability_ari_iqr", ""),
                "notes": bs.get("notes", ""),
            }
        )

    write_tsv(
        results_dir / "replication" / "bayesspace_stability_by_sample.tsv",
        stability_rows,
        fieldnames=[
            "dataset_id",
            "sample_id",
            "K",
            "seed_count",
            "stability_ari_median",
            "stability_ari_iqr",
            "notes",
        ],
    )

    # Simple per-dataset/method/K median summaries (for replication overview)
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in locked_rows:
        ds = row.get("dataset_id", "")
        mid = row.get("method_id", "")
        kval = row.get("K", "")
        if not ds or not mid or not kval:
            continue
        grouped[(ds, mid, kval)].append(row)

    def median(values: list[float]) -> float | None:
        if not values:
            return None
        values = sorted(values)
        m = len(values) // 2
        if len(values) % 2 == 1:
            return values[m]
        return 0.5 * (values[m - 1] + values[m])

    summary_rows: list[dict[str, Any]] = []
    for (ds, mid, kval), rows in sorted(grouped.items()):
        def collect(col: str) -> list[float]:
            out = []
            for r in rows:
                try:
                    out.append(float(r.get(col, "")))
                except Exception:
                    continue
            return out

        summary_rows.append(
            {
                "dataset_id": ds,
                "method_id": mid,
                "K": kval,
                "n_samples": len({r.get("sample_id", "") for r in rows if r.get("sample_id", "")}),
                "spatial_coherence_median_of_samples": median(collect("spatial_coherence_median")) or "",
                "marker_coherence_median_of_samples": median(collect("marker_coherence_median")) or "",
                "stability_ari_median_of_samples": median(collect("stability_ari_median")) or "",
                "wall_time_sec_median_of_samples": median(collect("wall_time_sec_median")) or "",
                "failure_rate_median_of_samples": median(collect("failure_rate")) or "",
            }
        )

    write_tsv(
        results_dir / "replication" / "method_benchmark_dataset_level_summary.tsv",
        summary_rows,
        fieldnames=[
            "dataset_id",
            "method_id",
            "K",
            "n_samples",
            "spatial_coherence_median_of_samples",
            "marker_coherence_median_of_samples",
            "stability_ari_median_of_samples",
            "wall_time_sec_median_of_samples",
            "failure_rate_median_of_samples",
        ],
    )

    # -------------------------
    # results/figures/figS3_instability_case_study_summary.tsv
    # -------------------------
    case_tsv = results_dir / "figures" / "figS3_instability_case_study.tsv"
    if case_tsv.exists():
        case_rows_all = read_tsv(case_tsv)
        case_rows = [
            row
            for row in case_rows_all
            if (row.get("notes") or "").strip() == "figS3-bayesspace-instability-case-study"
        ]
        if case_rows:
            def to_int(value: str) -> int:
                return int(float(str(value).strip()))

            seeds = sorted({to_int(r.get("seed", "0")) for r in case_rows if (r.get("seed") or "").strip()})
            ref_seed = min(seeds) if seeds else 0

            ref_labels: dict[str, int] = {}
            ref_expr: dict[str, dict[str, float]] = {}
            for row in case_rows:
                if to_int(row.get("seed", "0")) != ref_seed:
                    continue
                barcode = (row.get("barcode") or "").strip()
                if not barcode:
                    continue
                ref_labels[barcode] = to_int(row.get("domain_label", "0"))
                def fnum(v: str) -> float:
                    try:
                        return float(v)
                    except Exception:
                        return float("nan")
                ref_expr[barcode] = {
                    "expr_epithelial": fnum(row.get("expr_epithelial", "")),
                    "expr_stromal": fnum(row.get("expr_stromal", "")),
                    "expr_immune": fnum(row.get("expr_immune", "")),
                }

            n_spots = len(ref_labels)
            if n_spots:
                first = case_rows[0]
                dataset_id = (first.get("dataset_id") or "").strip()
                sample_id = (first.get("sample_id") or "").strip()
                method_id = (first.get("method_id") or "").strip()
                K = to_int(first.get("K", "0"))
                label_set = sorted({to_int(r.get("domain_label", "0")) for r in case_rows if (r.get("domain_label") or "").strip()})
                if K and len(label_set) != K:
                    label_set = list(range(1, K + 1))

                unstable: set[str] = set()
                diff_rates: list[float] = []

                for seed in seeds:
                    if seed == ref_seed:
                        continue
                    seed_rows = [r for r in case_rows if to_int(r.get("seed", "0")) == seed]
                    # contingency[current_label][ref_label] = count
                    contingency: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
                    by_barcode: dict[str, int] = {}
                    for row in seed_rows:
                        barcode = (row.get("barcode") or "").strip()
                        if not barcode or barcode not in ref_labels:
                            continue
                        cur = to_int(row.get("domain_label", "0"))
                        by_barcode[barcode] = cur
                        contingency[cur][ref_labels[barcode]] += 1

                    best_mapping: dict[int, int] | None = None
                    best_score = -1
                    for perm in itertools.permutations(label_set):
                        mapping = dict(zip(label_set, perm))
                        score = 0
                        for cur in label_set:
                            mapped = mapping[cur]
                            score += contingency.get(cur, {}).get(mapped, 0)
                        if score > best_score:
                            best_score = score
                            best_mapping = mapping

                    if not best_mapping:
                        continue

                    mismatches = 0
                    for barcode, cur in by_barcode.items():
                        aligned = best_mapping.get(cur, cur)
                        if aligned != ref_labels[barcode]:
                            unstable.add(barcode)
                            mismatches += 1
                    diff_rates.append(mismatches / n_spots)

                stable = set(ref_labels.keys()) - unstable
                def collect(values: dict[str, dict[str, float]], keys: set[str], col: str) -> list[float]:
                    out: list[float] = []
                    for k in keys:
                        v = values.get(k, {}).get(col)
                        if v is None:
                            continue
                        if v != v:  # NaN
                            continue
                        out.append(float(v))
                    return out

                stable_epi = collect(ref_expr, stable, "expr_epithelial")
                unstable_epi = collect(ref_expr, unstable, "expr_epithelial")
                stable_stroma = collect(ref_expr, stable, "expr_stromal")
                unstable_stroma = collect(ref_expr, unstable, "expr_stromal")
                stable_imm = collect(ref_expr, stable, "expr_immune")
                unstable_imm = collect(ref_expr, unstable, "expr_immune")

                def collect_total(values: dict[str, dict[str, float]], keys: set[str]) -> list[float]:
                    out: list[float] = []
                    for k in keys:
                        row = values.get(k)
                        if not row:
                            continue
                        a = row.get("expr_epithelial")
                        b = row.get("expr_stromal")
                        c = row.get("expr_immune")
                        if a is None or b is None or c is None:
                            continue
                        if a != a or b != b or c != c:  # NaN
                            continue
                        out.append(float(a + b + c))
                    return out

                stable_total = collect_total(ref_expr, stable)
                unstable_total = collect_total(ref_expr, unstable)

                summary_row = {
                    "dataset_id": dataset_id,
                    "sample_id": sample_id,
                    "method_id": method_id,
                    "K": K,
                    "seeds": ",".join(str(s) for s in seeds),
                    "reference_seed": ref_seed,
                    "n_spots": n_spots,
                    "n_switch_spots_any_vs_ref": len(unstable),
                    "switch_rate_any_vs_ref": (len(unstable) / n_spots) if n_spots else "",
                    "mean_diff_rate_vs_ref_across_seeds": (sum(diff_rates) / len(diff_rates)) if diff_rates else "",
                    "marker_epithelial_name": (first.get("marker_epithelial_name") or "").strip(),
                    "marker_stromal_name": (first.get("marker_stromal_name") or "").strip(),
                    "marker_immune_name": (first.get("marker_immune_name") or "").strip(),
                    "n_stable_spots": len(stable),
                    "n_unstable_spots": len(unstable),
                    "median_expr_epithelial_stable": median(stable_epi) if stable_epi else "",
                    "median_expr_epithelial_unstable": median(unstable_epi) if unstable_epi else "",
                    "median_expr_stromal_stable": median(stable_stroma) if stable_stroma else "",
                    "median_expr_stromal_unstable": median(unstable_stroma) if unstable_stroma else "",
                    "median_expr_immune_stable": median(stable_imm) if stable_imm else "",
                    "median_expr_immune_unstable": median(unstable_imm) if unstable_imm else "",
                    "median_total_marker_signal_stable": median(stable_total) if stable_total else "",
                    "median_total_marker_signal_unstable": median(unstable_total) if unstable_total else "",
                }

                write_tsv(
                    results_dir / "figures" / "figS3_instability_case_study_summary.tsv",
                    [summary_row],
                    fieldnames=[
                        "dataset_id",
                        "sample_id",
                        "method_id",
                        "K",
                        "seeds",
                        "reference_seed",
                        "n_spots",
                        "n_switch_spots_any_vs_ref",
                        "switch_rate_any_vs_ref",
                        "mean_diff_rate_vs_ref_across_seeds",
                        "marker_epithelial_name",
                        "marker_stromal_name",
                        "marker_immune_name",
                        "n_stable_spots",
                        "n_unstable_spots",
                        "median_expr_epithelial_stable",
                        "median_expr_epithelial_unstable",
                        "median_expr_stromal_stable",
                        "median_expr_stromal_unstable",
                        "median_expr_immune_stable",
                        "median_expr_immune_unstable",
                        "median_total_marker_signal_stable",
                        "median_total_marker_signal_unstable",
                    ],
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
