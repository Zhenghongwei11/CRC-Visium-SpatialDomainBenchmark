"""Spatial tissue selection and numerical helper functions."""

from __future__ import annotations

import math
import hashlib

import numpy as np

import pandas as pd

BOOTSTRAP_REPLICATES = 1000

MIN_GROUP_SPOTS = 20

MIN_BLOCKS = 3

def nearest_indices(coords: np.ndarray, neighbors: int = 6) -> np.ndarray:
    """Return deterministic nearest non-self neighbors.

    Visium lattice coordinates contain many equal-distance neighbors. The
    ordering returned by a kNN backend for those ties can vary by library
    version and execution backend, which changes connected components and
    boundary flags. Distances are therefore ranked explicitly, with the
    fixed input row order as the tie-breaker. The input row order is locked
    by the barcode-aligned section contract.
    """
    coords = np.asarray(coords, dtype=float)
    n_spots = coords.shape[0]
    if n_spots <= 1:
        return np.empty((n_spots, 0), dtype=int)
    n_neighbors = min(n_spots - 1, int(neighbors))
    if n_neighbors <= 0:
        return np.empty((n_spots, 0), dtype=int)
    indices = np.arange(n_spots, dtype=int)
    result = np.empty((n_spots, n_neighbors), dtype=int)
    for row in range(n_spots):
        distance_squared = np.sum((coords - coords[row]) ** 2, axis=1)
        distance_squared[row] = np.inf
        # lexsort uses the last key as primary: distance first, then index.
        result[row] = np.lexsort((indices, distance_squared))[:n_neighbors]
    return result

def block_ids(coords: np.ndarray, grid: int = 6) -> np.ndarray:
    x_edges = np.linspace(float(coords[:, 0].min()), float(coords[:, 0].max()), grid + 1)
    y_edges = np.linspace(float(coords[:, 1].min()), float(coords[:, 1].max()), grid + 1)
    x_bins = np.clip(np.digitize(coords[:, 0], x_edges[1:-1], right=False), 0, grid - 1)
    y_bins = np.clip(np.digitize(coords[:, 1], y_edges[1:-1], right=False), 0, grid - 1)
    return np.asarray([f"x{int(x)}_y{int(y)}" for x, y in zip(x_bins, y_bins, strict=True)], dtype=object)

def connected_stroma_components(stroma_mask: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    components = np.full(stroma_mask.shape[0], -1, dtype=int)
    adjacency: list[list[int]] = [[] for _ in range(len(stroma_mask))]
    for source, row in enumerate(neighbors):
        if not stroma_mask[source]:
            continue
        for target in row:
            target = int(target)
            if stroma_mask[target]:
                adjacency[source].append(target)
                adjacency[target].append(source)
    component_id = 0
    for start in np.flatnonzero(stroma_mask):
        if components[start] >= 0:
            continue
        queue = [int(start)]
        components[start] = component_id
        while queue:
            current = queue.pop()
            for neighbor in adjacency[current]:
                if components[neighbor] < 0:
                    components[neighbor] = component_id
                    queue.append(neighbor)
        component_id += 1
    return components

def anchor_sets(sample: dict[str, object]) -> dict[str, object]:
    from scipy.spatial import cKDTree

    coords = np.asarray(sample["coords"], dtype=float)
    coarse = np.asarray(sample["coarse_labels"], dtype=object)
    tumor_mask = coarse == "tumor"
    stroma_mask = coarse == "stroma"
    if not np.any(tumor_mask) or not np.any(stroma_mask):
        return {"status": "not_evaluable", "reason": "missing_pure_tumor_or_stroma_label", "near": np.zeros(len(coords), dtype=bool), "far": np.zeros(len(coords), dtype=bool), "distance_um": np.full(len(coords), np.nan), "components": np.full(len(coords), -1, dtype=int)}
    distance_px = cKDTree(coords[tumor_mask]).query(coords, k=1)[0]
    distance_um = distance_px * float(sample["microns_per_pixel"])
    neighbors = nearest_indices(np.asarray(sample["array_coords"], dtype=float), 6)
    components = connected_stroma_components(stroma_mask, neighbors)
    near = stroma_mask & (distance_um <= 100.0)
    far = stroma_mask & (distance_um >= 200.0) & (distance_um <= 400.0)
    near_components = set(components[near].tolist()) - {-1}
    far &= np.asarray([component in near_components for component in components], dtype=bool)
    return {
        "status": "success",
        "reason": "",
        "near": near,
        "far": far,
        "distance_um": distance_um,
        "components": components,
        "neighbors": neighbors,
        "tumor_mask": tumor_mask,
        "stroma_mask": stroma_mask,
    }

def nonempty_blocks(group: np.ndarray, blocks: np.ndarray) -> list[str]:
    return sorted({str(block) for block in np.asarray(blocks)[np.asarray(group, dtype=bool)]})

def block_resample_medians(
    values: np.ndarray,
    group: np.ndarray,
    blocks: np.ndarray,
    sampled_blocks: np.ndarray,
    available_blocks: list[str],
) -> np.ndarray:
    """Pool the sampled spatial blocks and return one median per replicate."""
    block_values = [
        np.asarray(values[(blocks == block) & group], dtype=float)
        for block in available_blocks
    ]
    max_size = max(len(block) for block in block_values)
    padded = np.full((len(block_values), max_size), np.nan, dtype=float)
    for index, block in enumerate(block_values):
        padded[index, : len(block)] = block
    block_index = {block: index for index, block in enumerate(available_blocks)}
    sampled_indices = np.vectorize(block_index.__getitem__, otypes=[int])(sampled_blocks)
    return np.nanmedian(padded[sampled_indices], axis=(1, 2))

def bootstrap_delta(values: np.ndarray, group_a: np.ndarray, group_b: np.ndarray, blocks: np.ndarray, key: str) -> tuple[float, float, float, int]:
    blocks_a = nonempty_blocks(group_a, blocks)
    blocks_b = nonempty_blocks(group_b, blocks)
    if len(blocks_a) < MIN_BLOCKS or len(blocks_b) < MIN_BLOCKS:
        return float("nan"), float("nan"), float("nan"), min(len(blocks_a), len(blocks_b))
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    sampled_a = rng.choice(blocks_a, size=(BOOTSTRAP_REPLICATES, len(blocks_a)), replace=True)
    sampled_b = rng.choice(blocks_b, size=(BOOTSTRAP_REPLICATES, len(blocks_b)), replace=True)
    estimates = block_resample_medians(values, group_a, blocks, sampled_a, blocks_a) - block_resample_medians(values, group_b, blocks, sampled_b, blocks_b)
    return float(np.median(values[group_a]) - np.median(values[group_b])), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975)), min(len(blocks_a), len(blocks_b))

def bootstrap_change(values: np.ndarray, alt_a: np.ndarray, alt_b: np.ndarray, ref_a: np.ndarray, ref_b: np.ndarray, blocks: np.ndarray, key: str, diagnostics: dict[str, object] | None = None) -> tuple[float, float, float, int]:
    alt_a_blocks = nonempty_blocks(alt_a, blocks)
    alt_b_blocks = nonempty_blocks(alt_b, blocks)
    ref_a_blocks = nonempty_blocks(ref_a, blocks)
    ref_b_blocks = nonempty_blocks(ref_b, blocks)
    support = min(len(alt_a_blocks), len(alt_b_blocks), len(ref_a_blocks), len(ref_b_blocks))
    if diagnostics is not None:
        diagnostics.update(n_attempted=0, n_accepted=0, n_used=0, acceptance_rate=float("nan"), reason="")
    if support < MIN_BLOCKS:
        if diagnostics is not None:
            diagnostics["reason"] = "paired_common_block_support_below_locked_threshold"
        return float("nan"), float("nan"), float("nan"), support
    if np.array_equal(alt_a, ref_a) and np.array_equal(alt_b, ref_b):
        if diagnostics is not None:
            diagnostics.update(n_used=BOOTSTRAP_REPLICATES, reason="identical_memberships_exact_zero")
        return 0.0, 0.0, 0.0, support
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    near_pool = sorted(set(alt_a_blocks) | set(ref_a_blocks))
    far_pool = sorted(set(alt_b_blocks) | set(ref_b_blocks))
    estimates_parts: list[np.ndarray] = []
    retained = 0
    attempted = 0
    for _ in range(100):
        if retained >= BOOTSTRAP_REPLICATES:
            break
        # A spatial block receives one draw in both partitions. Near and far
        # bands remain separately stratified; empty-group draws are rejected.
        sampled_near = rng.choice(near_pool, size=(128, len(near_pool)), replace=True)
        sampled_far = rng.choice(far_pool, size=(128, len(far_pool)), replace=True)
        attempted += 128
        valid = (
            np.isin(sampled_near, alt_a_blocks).any(axis=1)
            & np.isin(sampled_near, ref_a_blocks).any(axis=1)
            & np.isin(sampled_far, alt_b_blocks).any(axis=1)
            & np.isin(sampled_far, ref_b_blocks).any(axis=1)
        )
        if not np.any(valid):
            continue
        near_draws = sampled_near[valid]
        far_draws = sampled_far[valid]
        alt_a_medians = block_resample_medians(values, alt_a, blocks, near_draws, near_pool)
        alt_b_medians = block_resample_medians(values, alt_b, blocks, far_draws, far_pool)
        ref_a_medians = block_resample_medians(values, ref_a, blocks, near_draws, near_pool)
        ref_b_medians = block_resample_medians(values, ref_b, blocks, far_draws, far_pool)
        estimates_parts.append((alt_a_medians - alt_b_medians) - (ref_a_medians - ref_b_medians))
        retained += len(estimates_parts[-1])
    if diagnostics is not None:
        diagnostics.update(
            n_attempted=attempted,
            n_accepted=retained,
            n_used=min(retained, BOOTSTRAP_REPLICATES),
            acceptance_rate=(retained / attempted if attempted else float("nan")),
        )
    if retained < BOOTSTRAP_REPLICATES:
        if diagnostics is not None:
            diagnostics["reason"] = "insufficient_accepted_paired_draws"
        return float("nan"), float("nan"), float("nan"), support
    estimates = np.concatenate(estimates_parts)[:BOOTSTRAP_REPLICATES]
    point = (float(np.median(values[alt_a])) - float(np.median(values[alt_b]))) - (float(np.median(values[ref_a])) - float(np.median(values[ref_b])))
    return point, float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975)), support

def domain_selection(labels: np.ndarray, coarse: np.ndarray) -> tuple[dict[str, object], list[dict[str, object]]]:
    stats: list[dict[str, object]] = []
    for label in sorted(np.unique(labels).tolist()):
        in_domain = labels == int(label)
        n_spots = int(in_domain.sum())
        tumor_count = int((in_domain & (coarse == "tumor")).sum())
        stroma_count = int((in_domain & (coarse == "stroma")).sum())
        stats.append(
            {
                "domain_label": int(label),
                "domain_n": n_spots,
                "tumor_n": tumor_count,
                "stroma_n": stroma_count,
                "tumor_fraction": tumor_count / n_spots if n_spots else float("nan"),
                "stroma_fraction": stroma_count / n_spots if n_spots else float("nan"),
            }
        )
    if not any(row["tumor_n"] > 0 for row in stats) or not any(row["stroma_n"] > 0 for row in stats):
        return {"status": "not_evaluable", "reason": "pathology_anchor_class_absent"}, stats
    tumor = sorted(stats, key=lambda row: (-float(row["tumor_fraction"]), int(row["domain_label"])),)[0]
    stroma_candidates = [row for row in stats if row["domain_label"] != tumor["domain_label"]]
    stroma_candidates = [row for row in stroma_candidates if row["stroma_n"] > 0]
    if not stroma_candidates:
        return {"status": "not_evaluable", "reason": "no_distinct_stroma_domain"}, stats
    stroma = sorted(stroma_candidates, key=lambda row: (-float(row["stroma_fraction"]), int(row["domain_label"])),)[0]
    selected = {
        "status": "success",
        "reason": "",
        "tumor_domain": int(tumor["domain_label"]),
        "stroma_domain": int(stroma["domain_label"]),
        "tumor_domain_n": int(tumor["domain_n"]),
        "stroma_domain_n": int(stroma["domain_n"]),
        "tumor_fraction": float(tumor["tumor_fraction"]),
        "stroma_fraction": float(stroma["stroma_fraction"]),
    }
    return selected, stats

def computational_groups(
    anchor_near: np.ndarray,
    anchor_far: np.ndarray,
    labels: np.ndarray,
    tumor_domain: int,
    stroma_domain: int,
    neighbors: np.ndarray,
    estimator_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-anchor groups retained by one computational partition."""

    stroma_side = labels == int(stroma_domain)
    if estimator_id == "strict_computational_interface":
        computational_near = anchor_near & stroma_side & np.any(labels[neighbors] == int(tumor_domain), axis=1)
    elif estimator_id == "anchor_retention":
        computational_near = anchor_near & stroma_side
    else:
        raise ValueError(f"Unknown estimator_id: {estimator_id}")
    computational_far = anchor_far & stroma_side
    return computational_near, computational_far

def supported_conclusion(delta: float, low: float, high: float, scale: float) -> str:
    if not all(np.isfinite(value) for value in (delta, low, high, scale)) or scale <= 0:
        return "uncertain"
    if abs(delta) < 0.50 * scale:
        return "no_supported_difference"
    if low > 0:
        return "near_enrichment"
    if high < 0:
        return "near_depletion"
    return "no_supported_difference"

def score_scale(values: np.ndarray, anchors: dict[str, object]) -> float:
    values = np.asarray(values, dtype=float)[np.asarray(anchors["stroma_mask"], dtype=bool)]
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))
