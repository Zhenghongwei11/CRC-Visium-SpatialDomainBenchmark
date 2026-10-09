"""Known-boundary spatial simulations and condition summaries."""
from __future__ import annotations

import argparse

import hashlib

import itertools

import json

import math

import platform

import time

from datetime import datetime, timezone

from pathlib import Path

import numpy as np

import pandas as pd

import scipy

import sklearn

from scipy.ndimage import gaussian_filter, gaussian_filter1d

from scipy.optimize import brentq

from scipy.special import ndtr

from sklearn.metrics import adjusted_rand_score

import regional_helpers as historical

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    # Exact keys prevent an unrecorded default from entering an outcome run.
    keys = {
        "core_grid": "rows mixing_max boundary_width_um signal_amplitude correlation_length_um",
        "geometry": "columns spot_pitch_um spot_diameter_um boundary_column boundary_offset_um boundary_wave_amplitude_um boundary_wave_period_rows near_max_um far_min_um far_max_um block_grid",
        "signal": "noise_sd nugget_variance_fraction tangential_heterogeneity_amplitude tangential_period_rows mixing_decay_um tumor_mean baseline_mean",
        "maps": "core_graph_neighbors core_smoothing_steps reference_max_shift_um reference_shift_period_rows alternative_wave_amplitude_um alternative_wave_period_rows alternative_phase_radians map_names",
        "estimation": "estimators bootstrap_replicates min_group_spots min_blocks legacy_support_margins_mad paired_change_margins_mad population_root_tolerance numeric_tolerance",
        "replication": "seed_start independent_replicates_per_case smoke_case_indices smoke_replicate_indices",
        "calibration_gates": "nominal_coverage minimum_wilson_coverage_lower_95 maximum_absolute_bias_noise_sd maximum_wilson_null_rejection_upper_95 minimum_evaluable_fraction minimum_independent_replicates familywise_claims",
        "resources": "workers max_wall_seconds_per_invocation max_output_megabytes cloud_training",
    }
    required = set(keys) | {"schema_version", "analysis_role", "topology_grid"}
    if set(manifest) != required or manifest["schema_version"] != "tissue_anchor_simulation_v1":
        raise ValueError("invalid manifest sections or schema")
    for section, names in keys.items():
        if set(manifest[section]) != set(names.split()):
            raise ValueError(f"manifest field mismatch in {section}")
    topology = manifest["topology_grid"]
    if set(topology) != {"base", "graph_neighbors", "smoothing_steps"}:
        raise ValueError("invalid topology grid")
    if set(topology["base"]) != set(manifest["core_grid"]):
        raise ValueError("topology base incomplete")
    g, e, s = manifest["geometry"], manifest["estimation"], manifest["signal"]
    if (g["near_max_um"], g["far_min_um"], g["far_max_um"]) != (100, 200, 400):
        raise ValueError("historical anchor bands must remain 100/200/400")
    if (e["bootstrap_replicates"], e["min_group_spots"], e["min_blocks"]) != (
        historical.BOOTSTRAP_REPLICATES, historical.MIN_GROUP_SPOTS, historical.MIN_BLOCKS
    ):
        raise ValueError("manifest differs from the historical estimator")
    if e["estimators"] != ["anchor_retention", "strict_computational_interface"]:
        raise ValueError("both historical estimators are required")
    if manifest["maps"]["map_names"] != ["identity_control", "reference_shift", "alternative_shift"]:
        raise ValueError("map controls incomplete")
    if not 0 <= s["nugget_variance_fraction"] <= 1 or s["noise_sd"] <= 0:
        raise ValueError("invalid noise")
    if not all(0 <= x <= 1 for x in manifest["core_grid"]["mixing_max"]):
        raise ValueError("mixture fractions must be in [0, 1]")
    if manifest["resources"]["workers"] != 1 or manifest["resources"]["cloud_training"]:
        raise ValueError("only bounded scalar CPU calibration is allowed")
    if manifest["calibration_gates"]["nominal_coverage"] != 0.95:
        raise ValueError("historical percentile intervals are 95%")
    return manifest

def enumerate_cases(manifest: dict) -> list[dict]:
    grid = manifest["core_grid"]
    cases = []
    for values in itertools.product(*grid.values()):
        cases.append(dict(zip(grid, values, strict=True)) | {
            "experiment": "core", "graph_neighbors": manifest["maps"]["core_graph_neighbors"],
            "smoothing_steps": manifest["maps"]["core_smoothing_steps"],
        })
    topology = manifest["topology_grid"]
    for degree, steps in itertools.product(topology["graph_neighbors"], topology["smoothing_steps"]):
        cases.append(topology["base"] | {
            "experiment": "topology", "graph_neighbors": degree, "smoothing_steps": steps,
        })
    return [case | {"case_id": f"case_{i:03d}"} for i, case in enumerate(cases)]

def population_median(means: np.ndarray, sd: float, tolerance: float) -> float:
    means = np.asarray(means, dtype=float)
    if not len(means) or not np.isfinite(means).all() or sd <= 0:
        raise ValueError("empty/nonfinite population or invalid standard deviation")
    return float(brentq(lambda value: float(ndtr((value - means) / sd).mean()) - 0.5,
                        float(means.min() - 12 * sd), float(means.max() + 12 * sd),
                        xtol=tolerance))

def population_contrast(means: np.ndarray, near: np.ndarray, far: np.ndarray, manifest: dict) -> float:
    e, s = manifest["estimation"], manifest["signal"]
    return (population_median(means[near], s["noise_sd"], e["population_root_tolerance"])
            - population_median(means[far], s["noise_sd"], e["population_root_tolerance"]))

def noise_filter(shape: tuple[int, int], case: dict, manifest: dict) -> tuple[tuple[float, float], np.ndarray]:
    pitch = manifest["geometry"]["spot_pitch_um"]
    sigma = (case["correlation_length_um"] / (pitch * math.sqrt(3) / 2),
             case["correlation_length_um"] / pitch)
    # Kernel variances are deterministic; realization-wise normalization is biased.
    variances = []
    for size, width in zip(shape, sigma, strict=True):
        kernel = gaussian_filter1d(np.eye(size), width, axis=0, mode="reflect") if width else np.eye(size)
        variances.append(np.sum(kernel ** 2, axis=1))
    return sigma, np.sqrt(variances[0][:, None] * variances[1][None, :])

def generate_noise(structure: dict, seed: int, manifest: dict) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.normal(size=structure["shape"])
    correlated = gaussian_filter(white, structure["sigma"], mode="reflect") / structure["filter_sd"]
    fraction = manifest["signal"]["nugget_variance_fraction"]
    field = (math.sqrt(1 - fraction) * correlated
             + math.sqrt(fraction) * rng.normal(size=structure["shape"]))
    return field.ravel() * manifest["signal"]["noise_sd"]

def support_reason(near: np.ndarray, far: np.ndarray, blocks: np.ndarray, manifest: dict) -> str:
    e = manifest["estimation"]
    if min(int(near.sum()), int(far.sum())) < e["min_group_spots"]:
        return "insufficient_group_spots"
    if min(len(historical.nonempty_blocks(near, blocks)), len(historical.nonempty_blocks(far, blocks))) < e["min_blocks"]:
        return "insufficient_spatial_blocks"
    return ""

def build_structure(case: dict, manifest: dict) -> dict:
    g, s, maps = manifest["geometry"], manifest["signal"], manifest["maps"]
    shape = (int(case["rows"]), int(g["columns"]))
    row, column = np.indices(shape)
    array_col = 2 * column + row % 2
    pitch = g["spot_pitch_um"]
    coords = np.column_stack([(array_col * pitch / 2).ravel(), (row * pitch * math.sqrt(3) / 2).ravel()])
    array_coords = np.column_stack([row.ravel(), array_col.ravel()]).astype(float)
    boundary = (g["boundary_column"] * pitch + g["boundary_offset_um"]
                + g["boundary_wave_amplitude_um"] * np.sin(2 * np.pi * row / g["boundary_wave_period_rows"]))
    signed_distance = (coords[:, 0] - boundary.ravel())
    tumor = signed_distance < 0
    coarse = np.where(tumor, "tumor", "stroma")
    anchors = historical.anchor_sets({"coords": coords, "array_coords": array_coords,
                                      "coarse_labels": coarse, "microns_per_pixel": 1.0})
    if anchors["status"] != "success":
        raise ValueError("synthetic geometry lacks tissue support")
    blocks = historical.block_ids(array_coords, grid=int(g["block_grid"]))
    tangent = 1 + s["tangential_heterogeneity_amplitude"] * np.sin(2 * np.pi * row.ravel() / s["tangential_period_rows"])
    latent = s["baseline_mean"] + case["signal_amplitude"] * np.exp(-np.maximum(signed_distance, 0) / case["boundary_width_um"]) * tangent
    latent[tumor] = s["tumor_mean"]
    mixture = case["mixing_max"] * np.exp(-np.maximum(signed_distance, 0) / s["mixing_decay_um"])
    mixture[tumor] = 1.0
    observed = (1 - mixture) * latent + mixture * s["tumor_mean"]
    reference_shift = maps["reference_max_shift_um"] * (1 + np.sin(2 * np.pi * row.ravel() / maps["reference_shift_period_rows"])) / 2
    alternative_shift = reference_shift + maps["alternative_wave_amplitude_um"] * np.sin(
        2 * np.pi * row.ravel() / maps["alternative_wave_period_rows"] + maps["alternative_phase_radians"])
    graph = historical.nearest_indices(array_coords, int(case["graph_neighbors"]))
    labels_by_map = {"identity_control": np.where(tumor, 1, 2),
                     "reference_shift": np.where(signed_distance < reference_shift, 1, 2),
                     "alternative_shift": np.where(signed_distance < alternative_shift, 1, 2)}
    groups = {}
    for name, labels in labels_by_map.items():
        if name != "identity_control":
            for _ in range(int(case["smoothing_steps"])):
                votes = np.sum(labels[graph] == 1, axis=1)
                labels = np.where(2 * votes > graph.shape[1], 1,
                                  np.where(2 * votes < graph.shape[1], 2, labels))
        selected, _ = historical.domain_selection(labels, coarse)
        if selected["status"] != "success":
            raise ValueError("synthetic map lacks distinct domains")
        for estimator in manifest["estimation"]["estimators"]:
            near, far = historical.computational_groups(
                anchors["near"], anchors["far"], labels, int(selected["tumor_domain"]),
                int(selected["stroma_domain"]), anchors["neighbors"], estimator)
            groups[(estimator, name)] = {
                "near": near, "far": far, "labels": labels,
                "reason": support_reason(near, far, blocks, manifest),
                "truth": population_contrast(observed, near, far, manifest) if near.any() and far.any() else float("nan"),
                "global_ari": adjusted_rand_score(coarse, labels),
            }
    sigma, filter_sd = noise_filter(shape, case, manifest)
    return {"coords": coords, "array_coords": array_coords, "shape": shape,
            "sigma": sigma, "filter_sd": filter_sd, "anchors": anchors, "blocks": blocks,
            "latent_means": latent, "means": observed, "mixing_fraction": mixture,
            "anchor_truth": population_contrast(observed, anchors["near"], anchors["far"], manifest),
            "latent_truth": population_contrast(latent, anchors["near"], anchors["far"], manifest),
            "anchor_reason": support_reason(anchors["near"], anchors["far"], blocks, manifest),
            "groups": groups}

def hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str((contiguous.shape, contiguous.dtype.str)).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()

def conclusion(point: float, low: float, high: float, margin: float) -> int:
    if not np.isfinite([point, low, high, margin]).all():
        return 0
    if abs(point) < margin:
        return 0
    return 1 if low > 0 else (-1 if high < 0 else 0)

def evaluate_replicate(case: dict, structure: dict, replicate: int, manifest: dict) -> list[dict]:
    seed = int(np.random.SeedSequence([manifest["replication"]["seed_start"],
                                      int(case["case_id"].split("_")[-1]), replicate]).generate_state(1)[0])
    values = structure["means"] + generate_noise(structure, seed, manifest)
    anchors, blocks = structure["anchors"], structure["blocks"]
    sample_hash = hash_arrays(structure["coords"], structure["means"], values, anchors["near"], anchors["far"])
    key = f"{case['case_id']}|replicate_{replicate}"
    anchor_key = hash_arrays(anchors["near"], anchors["far"])
    anchor = historical.bootstrap_delta(values, anchors["near"], anchors["far"], blocks, key + "|" + anchor_key) if not structure["anchor_reason"] else (float("nan"),) * 3 + (0,)
    scale = historical.score_scale(values, anchors)
    cache = {anchor_key: anchor}
    for group in structure["groups"].values():
        mask_key = hash_arrays(group["near"], group["far"])
        if mask_key not in cache:
            cache[mask_key] = (historical.bootstrap_delta(values, group["near"], group["far"], blocks, key + "|" + mask_key)
                               if not group["reason"] else (float("nan"),) * 3 + (0,))
    rows = []
    for (estimator, name), group in structure["groups"].items():
        ref = structure["groups"][(estimator, "reference_shift")]
        delta, low, high, support = cache[hash_arrays(group["near"], group["far"])]
        ref_delta, ref_low, ref_high, _ = cache[hash_arrays(ref["near"], ref["far"])]
        reason = structure["anchor_reason"] or group["reason"]
        diag = {}
        pair_reason = group["reason"] or ref["reason"]
        paired = (historical.bootstrap_change(values, group["near"], group["far"], ref["near"], ref["far"], blocks,
                                              key + "|paired|" + estimator + "|" + name, diag)
                  if not pair_reason else (float("nan"),) * 3 + (0,))
        pair_reason = pair_reason or diag.get("reason", "")
        if pair_reason == "identical_memberships_exact_zero":
            pair_reason = ""
        truth, ref_truth = group["truth"], ref["truth"]
        pair_truth = truth - ref_truth
        degenerate = np.array_equal(group["near"], ref["near"]) and np.array_equal(group["far"], ref["far"])
        row = case | {
            "replicate": replicate, "seed": seed, "map_name": name, "estimator": estimator,
            "input_sha256": sample_hash, "status": "not_evaluable" if reason else "success", "reason": reason,
            "near_n": int(group["near"].sum()), "far_n": int(group["far"].sum()), "block_support": support,
            "near_retention": float(group["near"].sum() / anchors["near"].sum()),
            "far_retention": float(group["far"].sum() / anchors["far"].sum()), "global_ari": group["global_ari"],
            "anchor_truth": structure["anchor_truth"], "latent_anchor_truth": structure["latent_truth"],
            "measurement_distortion": structure["anchor_truth"] - structure["latent_truth"],
            "map_truth": truth, "selection_distortion": truth - structure["anchor_truth"],
            "anchor_delta": anchor[0], "anchor_low": anchor[1], "anchor_high": anchor[2],
            "anchor_covered": bool(anchor[1] <= structure["anchor_truth"] <= anchor[2]),
            "anchor_status": "not_evaluable" if structure["anchor_reason"] else "success",
            "anchor_error": anchor[0] - structure["anchor_truth"],
            "anchor_null": bool(abs(structure["anchor_truth"]) <= manifest["estimation"]["numeric_tolerance"]),
            "anchor_null_rejected": bool(np.isfinite(anchor[:3]).all() and (anchor[1] > 0 or anchor[2] < 0)),
            "delta": delta, "low": low, "high": high, "estimation_error": delta - truth,
            "covered": bool(low <= truth <= high),
            "paired_delta": paired[0], "paired_low": paired[1], "paired_high": paired[2], "paired_truth": pair_truth,
            "paired_error": paired[0] - pair_truth, "paired_covered": bool(paired[1] <= pair_truth <= paired[2]),
            "paired_status": "not_evaluable" if pair_reason else "success", "paired_reason": pair_reason,
            "paired_degenerate": degenerate, "paired_attempted": diag.get("n_attempted", 0),
            "paired_accepted": diag.get("n_accepted", 0), "paired_used": diag.get("n_used", 0),
            "paired_acceptance_rate": diag.get("acceptance_rate", float("nan")), "score_mad": scale,
        }
        tol = manifest["estimation"]["numeric_tolerance"]
        row["null_contrast"] = bool(abs(truth) <= tol)
        row["null_rejected"] = bool(np.isfinite([low, high]).all() and (low > 0 or high < 0))
        row["paired_null"] = bool(abs(pair_truth) <= tol)
        row["paired_null_rejected"] = bool(np.isfinite(paired[:3]).all() and (paired[1] > 0 or paired[2] < 0))
        for margin in manifest["estimation"]["legacy_support_margins_mad"]:
            suffix = str(margin).replace(".", "p")
            bound = margin * scale
            ac, mc, rc = conclusion(*anchor[:3], bound), conclusion(delta, low, high, bound), conclusion(ref_delta, ref_low, ref_high, bound)
            truth_same = structure["anchor_truth"] * truth > 0 and min(abs(structure["anchor_truth"]), abs(truth)) >= bound
            row[f"loss_den_{suffix}"] = bool(ac)
            row[f"loss_{suffix}"] = bool(ac and mc == 0)
            row[f"false_loss_den_{suffix}"] = bool(ac and truth_same)
            row[f"false_loss_{suffix}"] = bool(ac and mc == 0 and truth_same)
            row[f"reversal_den_{suffix}"] = bool(ac)
            row[f"reversal_{suffix}"] = bool(ac * mc == -1)
            row[f"additional_loss_den_{suffix}"] = bool(ac and rc == ac and name == "alternative_shift" and not pair_reason)
            row[f"additional_loss_{suffix}"] = bool(row[f"additional_loss_den_{suffix}"] and mc == 0)
        for margin in manifest["estimation"]["paired_change_margins_mad"]:
            suffix = str(margin).replace(".", "p")
            row[f"paired_flag_{suffix}"] = bool(not pair_reason and abs(paired[0]) >= margin * scale
                                                 and (paired[1] > 0 or paired[2] < 0))
        rows.append(row)
    return rows

def wilson(successes: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return p, max(0.0, center - half), min(1.0, center + half)

def summarize(table: pd.DataFrame, manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, rates = [], []
    gates = manifest["calibration_gates"]
    for key, group in table.groupby(["case_id", "estimator", "map_name"], sort=True):
        record = dict(zip(["case_id", "estimator", "map_name"], key, strict=True))
        good = group[group["status"].eq("success")]
        paired = group[group["paired_status"].eq("success") & ~group["paired_degenerate"]]
        for target, valid, error, coverage, null, rejected in (
            ("anchor", group[group["anchor_status"].eq("success")], "anchor_error", "anchor_covered", "anchor_null", "anchor_null_rejected"),
            ("contrast", good, "estimation_error", "covered", "null_contrast", "null_rejected"),
            ("paired", paired, "paired_error", "paired_covered", "paired_null", "paired_null_rejected"),
        ):
            count = len(valid)
            rate, coverage_low, coverage_high = wilson(int(valid[coverage].sum()), count)
            null_rows = valid[valid[null]]
            null_rate, null_low, null_high = wilson(int(null_rows[rejected].sum()), len(null_rows))
            bias = float(valid[error].mean()) if count else float("nan")
            fraction = count / len(group)
            passed = (count >= gates["minimum_independent_replicates"] and fraction >= gates["minimum_evaluable_fraction"]
                      and abs(bias) / manifest["signal"]["noise_sd"] <= gates["maximum_absolute_bias_noise_sd"]
                      and coverage_low >= gates["minimum_wilson_coverage_lower_95"]
                      and (not len(null_rows) or null_high <= gates["maximum_wilson_null_rejection_upper_95"]))
            summaries.append(record | {"target": target, "attempted": len(group), "evaluable": count,
                "evaluable_fraction": fraction, "bias": bias,
                "rmse": float(np.sqrt(np.mean(valid[error] ** 2))) if count else float("nan"),
                "coverage": rate, "coverage_low": coverage_low, "coverage_high": coverage_high,
                "null_n": len(null_rows), "null_rejection": null_rate, "null_rejection_low": null_low, "null_rejection_high": null_high,
                "screening_gate": "pass" if passed else ("not_evaluable" if count == 0 else "fail"),
                "paired_degenerate_n": int(group["paired_degenerate"].sum()),
                "global_ari": float(group["global_ari"].iloc[0]), "near_retention": float(group["near_retention"].iloc[0]),
                "far_retention": float(group["far_retention"].iloc[0]), "selection_distortion": float(group["selection_distortion"].iloc[0])})
        for margin in manifest["estimation"]["legacy_support_margins_mad"]:
            suffix = str(margin).replace(".", "p")
            for metric in ("loss", "false_loss", "reversal", "additional_loss"):
                valid = good[good[f"{metric}_den_{suffix}"]]
                count = int(valid[f"{metric}_{suffix}"].sum())
                rate, lower, upper = wilson(count, len(valid))
                rates.append(record | {"metric": metric, "margin_mad": margin, "numerator": count,
                                       "denominator": len(valid), "rate": rate, "low": lower, "high": upper})
        for margin in manifest["estimation"]["paired_change_margins_mad"]:
            suffix = str(margin).replace(".", "p")
            count = int(paired[f"paired_flag_{suffix}"].sum())
            rate, lower, upper = wilson(count, len(paired))
            rates.append(record | {"metric": "paired_flag", "margin_mad": margin, "numerator": count,
                                   "denominator": len(paired), "rate": rate, "low": lower, "high": upper})
    summary = pd.DataFrame(summaries)
    anchors = summary[summary["target"].eq("anchor")].drop_duplicates("case_id").copy()
    anchors["estimator"], anchors["map_name"] = "morphology_anchor", "tissue_anchor"
    anchors["global_ari"] = 1.0
    anchors["near_retention"], anchors["far_retention"], anchors["selection_distortion"] = 1.0, 1.0, 0.0
    anchors["paired_degenerate_n"] = 0
    summary = pd.concat([anchors, summary[~summary["target"].eq("anchor")]], ignore_index=True)
    parameters = table[["case_id", "experiment", "rows", "mixing_max", "boundary_width_um",
                        "signal_amplitude", "correlation_length_um", "graph_neighbors", "smoothing_steps"]].drop_duplicates()
    return (summary.merge(parameters, on="case_id", validate="many_to_one"),
            pd.DataFrame(rates).merge(parameters, on="case_id", validate="many_to_one"))
