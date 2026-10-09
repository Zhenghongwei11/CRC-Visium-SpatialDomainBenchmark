#!/usr/bin/env python3
"""Regional comparisons under alternative spatial definitions.

Calculate common-block intervals and a 175-micrometer near-band comparison
using the registered tissue, component measurements and fitted domain maps.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

from configuration import parse_configuration
C = parse_configuration(__doc__)
ROOT = C.workspace
PROTOCOL = C.protocol
PLAN = C.plan
OUTPUT = C.results
BASE = C.results
MASTER = 'jbcb-reader-revision-20261006-methods-v1'
REPS = 1000
METRICS = ['abs_deviation_score', 'abs_deviation_mad', 'near_retention', 'far_retention',
           'selected_stroma_purity', 'count_only_excess_abs_deviation_score',
           'block_matched_excess_abs_deviation_score', 'count_only_excess_abs_deviation_mad',
           'block_matched_excess_abs_deviation_mad']
P = json.loads(PROTOCOL.read_text())


def read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep='\t', float_precision='round_trip', low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def record(path: Path) -> dict:
    label = (path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else
             'results/' + path.relative_to(OUTPUT).as_posix() if path.is_relative_to(OUTPUT) else
             'analysis_code/' + path.name)
    return {'path': label, 'bytes': path.stat().st_size, 'sha256': sha(path)}


def rng_for(*fields) -> tuple[np.random.Generator, str]:
    key = '|'.join(map(str, (MASTER, *fields)))
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')), key


def neighbors(lattice: np.ndarray, exact: bool) -> np.ndarray:
    n = len(lattice)
    if exact:
        lookup = {tuple(x): i for i, x in enumerate(lattice)}
        out = np.repeat(np.arange(n)[:, None], 6, axis=1)
        for i, (r, c) in enumerate(lattice):
            found = sorted(lookup[(r + dr, c + dc)] for dr, dc in
                           [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                           if (r + dr, c + dc) in lookup)
            out[i, :len(found)] = found
        return out
    out = np.empty((n, min(6, n - 1)), int)
    for start in range(0, n, 128):
        distance = cdist(lattice[start:start + 128], lattice, 'sqeuclidean')
        distance[np.arange(len(distance)), np.arange(start, start + len(distance))] = np.inf
        out[start:start + len(distance)] = np.argsort(distance, axis=1, kind='stable')[:, :out.shape[1]]
    return out


def components(stroma: np.ndarray, nn: np.ndarray) -> np.ndarray:
    rows = np.repeat(np.arange(len(nn)), nn.shape[1]); cols = nn.ravel()
    keep = stroma[rows] & stroma[cols]
    graph = csr_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(len(nn), len(nn)))
    _, result = connected_components(graph, directed=False)
    result[~stroma] = -1
    return result


def block_ids(lattice: np.ndarray, grid: int) -> np.ndarray:
    axes = [np.searchsorted(np.linspace(lattice[:, i].min(), lattice[:, i].max(), grid + 1)[1:-1],
                            lattice[:, i], side='right') for i in range(2)]
    return np.asarray(['x%d_y%d' % pair for pair in zip(*axes)])


def eligibility(near: np.ndarray, far: np.ndarray, blocks: np.ndarray) -> bool:
    return bool(min(near.sum(), far.sum()) >= 20 and
                min(len(np.unique(blocks[near])), len(np.unique(blocks[far]))) >= 3)


def contrast(values: np.ndarray, near: np.ndarray, far: np.ndarray) -> float:
    return float(np.median(values[near]) - np.median(values[far])) if near.any() and far.any() else np.nan


def weighted_medians(values: np.ndarray, group: np.ndarray, blocks: np.ndarray,
                     universe: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Exact pooled-spot median under integer multiplicity of each physical block."""
    values = values[group]; group_blocks = blocks[group]
    order = np.argsort(values, kind='stable'); values = values[order]
    block_index = np.searchsorted(universe, group_blocks[order])
    result = np.empty(len(weights))
    for start in range(0, len(weights), 100):
        cumulative = np.cumsum(weights[start:start + 100, block_index], axis=1)
        total = cumulative[:, -1]
        if np.any(total <= 0):
            raise ValueError('Empty pooled band must be rejected before taking its median')
        lower = np.argmax(cumulative >= ((total - 1) // 2 + 1)[:, None], axis=1)
        upper = np.argmax(cumulative >= (total // 2 + 1)[:, None], axis=1)
        result[start:start + len(total)] = (values[lower] + values[upper]) / 2
    return result


def joint_bootstrap(values: np.ndarray, near: np.ndarray, far: np.ndarray,
                    blocks: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    universe = np.unique(blocks[near | far]); width = len(universe)
    near_occupied = np.isin(universe, blocks[near]); far_occupied = np.isin(universe, blocks[far])
    accepted = []; attempted = rejected = 0
    while sum(len(x) for x in accepted) < REPS:
        if attempted >= 100000:
            raise ValueError('Joint bootstrap did not obtain 1000 nonempty draws')
        needed = REPS - sum(len(x) for x in accepted)
        draws = rng.integers(0, width, size=(needed, width))
        weights = np.asarray([np.bincount(row, minlength=width) for row in draws])
        good = ((weights[:, near_occupied].sum(axis=1) > 0) &
                (weights[:, far_occupied].sum(axis=1) > 0))
        attempted += len(weights); rejected += int((~good).sum()); accepted.append(weights[good])
    weights = np.concatenate(accepted)
    vector = (weighted_medians(values, near, blocks, universe, weights) -
              weighted_medians(values, far, blocks, universe, weights))
    return vector, weights.astype(np.int16), {
        'joint_block_universe_n': width, 'joint_attempted': attempted,
        'joint_rejected_empty_band': rejected, 'joint_used': len(vector),
        'joint_block_universe': '|'.join(universe),
    }


def independent_bootstrap(values: np.ndarray, near: np.ndarray, far: np.ndarray,
                          blocks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for group in (near, far):
        universe = np.unique(blocks[group])
        draws = rng.integers(0, len(universe), size=(REPS, len(universe)))
        weights = np.asarray([np.bincount(row, minlength=len(universe)) for row in draws])
        parts.append(weighted_medians(values, group, blocks, universe, weights))
    return parts[0] - parts[1]


def control_vector(values: np.ndarray, blocks: np.ndarray, near: np.ndarray,
                   far: np.ndarray, selected_near: np.ndarray, selected_far: np.ndarray,
                   kind: str, rng: np.random.Generator) -> np.ndarray:
    medians = []
    for full, selected in ((near, selected_near), (far, selected_far)):
        if kind == 'count_only':
            pools = [(np.flatnonzero(full), int(selected.sum()))]
        else:
            pools = [(np.flatnonzero(full & (blocks == block)), int((selected & (blocks == block)).sum()))
                     for block in np.unique(blocks[selected])]
        indices = []
        for pool, count in pools:
            indices.append(np.broadcast_to(pool, (REPS, count)) if len(pool) == count else
                           np.stack([rng.choice(pool, count, replace=False) for _ in range(REPS)]))
        medians.append(np.median(values[np.concatenate(indices, axis=1)], axis=1))
    return medians[0] - medians[1]


def nested_summary(table: pd.DataFrame, prefix: str, output: Path) -> dict[str, pd.DataFrame]:
    keys = ['resource', 'patient_id', 'sample_id', 'graph_id', 'estimator_id', 'setting_id', 'readout_id']
    valid = table[table.independent_status.eq('success')]
    cells = valid.groupby(keys + ['method_id', 'K'])[METRICS].median().reset_index()
    sections = cells.groupby(keys)[METRICS].median().reset_index()
    counts = table.assign(included=table.independent_status.eq('success')).groupby(keys).agg(
        n_declared_maps=('included', 'size'), n_evaluable_maps=('included', 'sum')).reset_index()
    sections = counts.merge(sections, on=keys, how='left', validate='one_to_one')
    patient_keys = [x for x in keys if x != 'sample_id']
    patients = sections.groupby(patient_keys)[METRICS].median().reset_index()
    patient_counts = sections.assign(section_eligible=sections.n_evaluable_maps.gt(0)).groupby(patient_keys).agg(
        n_attempted_sections=('sample_id', 'size'), n_evaluable_sections=('section_eligible', 'sum'),
        n_declared_maps=('n_declared_maps', 'sum'), n_evaluable_maps=('n_evaluable_maps', 'sum')).reset_index()
    patients = patient_counts.merge(patients, on=patient_keys, how='left', validate='one_to_one')
    resource_keys = ['resource', 'graph_id', 'estimator_id', 'setting_id', 'readout_id']
    resources = patients.groupby(resource_keys)[METRICS].median().reset_index()
    resource_counts = patients.assign(patient_eligible=patients.n_evaluable_maps.gt(0)).groupby(resource_keys).agg(
        n_attempted_patients=('patient_id', 'size'), n_evaluable_patients=('patient_eligible', 'sum'),
        n_attempted_sections=('n_attempted_sections', 'sum'), n_evaluable_sections=('n_evaluable_sections', 'sum'),
        n_declared_maps=('n_declared_maps', 'sum'), n_evaluable_maps=('n_evaluable_maps', 'sum')).reset_index()
    resources = resource_counts.merge(resources, on=resource_keys, validate='one_to_one')
    map_medians = valid.groupby(resource_keys)[['abs_deviation_score', 'abs_deviation_mad']].median().reset_index().rename(
        columns={x: 'map_median_' + x for x in ('abs_deviation_score', 'abs_deviation_mad')})
    resources = resources.merge(map_medians, on=resource_keys, how='left', validate='one_to_one')
    result = dict(method_k=cells, section=sections, patient=patients, resource=resources)
    for name, data in result.items():
        data.to_csv(output / f'{prefix}_{name}_summary.tsv', sep='\t', index=False)
    return result


def main_frame(output: Path) -> dict:
    table = read(ROOT / P['output_dir'] / 'map_sensitivity.tsv')
    identity = ['resource', 'sample_id', 'graph_id', 'setting_id', 'estimator_id', 'method_id', 'K', 'partition_id', 'readout_id']
    if len(table) != 36960 or table.duplicated(identity).any():
        raise ValueError('Current full diagnostic grid identity mismatch')
    summaries = nested_summary(table, 'main_independent', output)
    baseline = table[table.graph_id.eq('historical_raw_6nn') & table.setting_id.eq('baseline') &
                     table.estimator_id.eq('anchor_retention') & table.readout_id.eq('primary_barrier')]
    den = []
    for resource, group in baseline.groupby('resource'):
        eligible = group[group.independent_status.eq('success')]
        den.append({'resource': resource, 'declared_maps': len(group), 'eligible_maps': len(eligible),
                    'attempted_sections': group.sample_id.nunique(), 'eligible_sections': eligible.sample_id.nunique(),
                    'attempted_patients': group.patient_id.nunique(), 'eligible_patients': eligible.patient_id.nunique(),
                    'count_envelope_outside_n': int(eligible.count_only_outside_descriptive_95_envelope.sum()),
                    'block_envelope_outside_n': int(eligible.block_matched_outside_descriptive_95_envelope.sum())})
    pd.DataFrame(den).to_csv(output / 'main_baseline_denominators.tsv', sep='\t', index=False)
    resources = summaries['resource']
    resources = resources[resources.graph_id.eq('historical_raw_6nn') & resources.setting_id.eq('baseline') &
                          resources.estimator_id.eq('anchor_retention') & resources.readout_id.eq('primary_barrier')]
    resources.to_csv(output / 'main_baseline_resource_summary.tsv', sep='\t', index=False)
    return {'denominators': den, 'resource_summary': resources.to_dict('records')}


def section_job(section: dict, output_name: str) -> dict:
    started = time.monotonic(); output = Path(output_name)
    sample = section['sample_id']; source = ROOT / section['source_root']
    section_output = output / 'sections' / section['resource'] / sample
    checkpoint=section_output/'manifest.json'
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        if saved['source_script_sha256']!=sha(__file__) or saved['plan_sha256']!=sha(PLAN):
            raise ValueError('Sensitivity checkpoint belongs to different code or settings')
        for item in saved['inputs']:
            if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Changed sensitivity input')
        for name,expected in saved.get('output_sha256',{}).items():
            if sha(section_output/name)!=expected:raise ValueError('Changed sensitivity output')
        if not saved.get('output_sha256'):raise ValueError('Sensitivity checkpoint has no output identities')
        return saved
    section_output.mkdir(parents=True, exist_ok=True)
    inputs = [source / 'spot_inputs.tsv.gz', source / 'spot_scores.tsv.gz', source / 'input_maps.tsv.gz']
    spots = read(inputs[0]).set_index('barcode')
    if spots.index.duplicated().any():
        raise ValueError('Repeated registered barcode')
    lattice = spots[['array_row', 'array_col']].to_numpy(float)
    if not np.isfinite(lattice).all() or not np.array_equal(lattice, np.rint(lattice)):
        raise ValueError('Invalid integer lattice')
    lattice = lattice.astype(np.int64)
    stroma = spots.coarse_label.eq('stroma').to_numpy(); tumor = spots.coarse_label.eq('tumor').to_numpy()
    score_table = read(inputs[1]); scores = score_table[score_table.score_id.eq('primary_barrier')].set_index('barcode')
    if set(scores.index) != set(spots.index) or scores.index.duplicated().any():
        raise ValueError('Score barcode universe mismatch')
    values = scores.loc[spots.index, 'score_value'].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite primary readout')
    scale = float(np.median(np.abs(values[stroma] - np.median(values[stroma])))) if stroma.any() else np.nan
    dnum = np.full(len(spots), np.nan)
    if tumor.any():
        for start in range(0, len(spots), 128):
            differences = lattice[start:start + 128, None, :] - lattice[tumor][None, :, :]
            dnum[start:start + len(differences)] = (3 * differences[:, :, 0]**2 + differences[:, :, 1]**2).min(axis=1)
    map_table = read(inputs[2]); maps = {}
    if not map_table.empty:
        for key, group in map_table[map_table.status.eq('success')].groupby(['method_id', 'K', 'replicate_id']):
            group = group.set_index('barcode')
            if set(group.index) != set(spots.index) or group.index.duplicated().any():
                raise ValueError('Map barcode universe mismatch')
            labels = group.loc[spots.index, 'domain_label'].to_numpy(int)
            fractions = [(int(label), float(tumor[labels == label].mean()), float(stroma[labels == label].mean()),
                          int(stroma[labels == label].sum())) for label in np.unique(labels)]
            if not tumor.any() or not stroma.any():
                selected = None
            else:
                tumor_domain = min(fractions, key=lambda x: (-x[1], x[0]))[0]
                candidates = [x for x in fractions if x[0] != tumor_domain and x[3] > 0]
                selected = (tumor_domain, min(candidates, key=lambda x: (-x[2], x[0]))) if candidates else None
            maps[(str(key[0]), int(key[1]), str(key[2]))] = (labels, selected)
    grid = read(ROOT / P['grid_path']); grid = grid[(grid.resource == section['resource']) & (grid.sample_id == sample)]
    inputs.append(ROOT / P['grid_path'])
    original_anchors_path = ROOT / P['output_dir'] / 'sections' / section['resource'] / sample / 'anchor_section_summary.tsv'
    original_anchors = read(original_anchors_path); inputs.append(original_anchors_path)
    anchor_rows = []; map_rows = []; anchor_vectors = {}; weight_vectors = {}; controls = {}; membership_vectors = {}; geometry = {}
    settings = P['spatial_settings'] + [{'id': 'near175', 'grid': 6, 'near_dnum_max': 12, 'far_dnum_min': 16, 'far_dnum_max': 64}]
    for graph in P['graph_ids']:
        nn = neighbors(lattice, graph == 'nominal_honeycomb_exact'); comp = components(stroma, nn)
        for setting in settings:
            sid = setting['id']; blocks = block_ids(lattice, setting['grid'])
            near = stroma & (dnum <= setting['near_dnum_max'])
            far = stroma & (dnum >= setting['far_dnum_min']) & (dnum <= setting['far_dnum_max']) & np.isin(comp, np.unique(comp[near]))
            if np.any(near & far):
                raise ValueError('Near and far overlap')
            eligible = eligibility(near, far, blocks); point = contrast(values, near, far)
            identity = {key: section[key] for key in ['resource', 'dataset_id', 'patient_id', 'sample_id', 'section_id']}
            identity.update(graph_id=graph, setting_id=sid)
            prefix = graph + '|' + sid
            row = {**identity, 'status': 'evaluable' if eligible else 'not_evaluable',
                   'n_registered': len(spots), 'n_near': int(near.sum()), 'n_far': int(far.sum()),
                   'n_near_blocks': len(np.unique(blocks[near])), 'n_far_blocks': len(np.unique(blocks[far])),
                   'overlapping_blocks': len(np.intersect1d(blocks[near], blocks[far])),
                   'delta': point, 'stroma_mad': scale,
                   'independent_low': np.nan, 'independent_high': np.nan, 'joint_low': np.nan, 'joint_high': np.nan,
                   'joint_used': 0, 'joint_attempted': 0, 'joint_rejected_empty_band': 0}
            if sid != 'near175':
                original = original_anchors[(original_anchors.graph_id == graph) & (original_anchors.setting_id == sid)].iloc[0]
                if (int(near.sum()) != original.n_anchor_near or int(far.sum()) != original.n_anchor_far or
                        ('evaluable' if eligible else 'not_evaluable') != original.anchor_status or
                        not np.isclose(point, original.morphology_anchor_delta, equal_nan=True, atol=0, rtol=0)):
                    raise ValueError('Fresh anchor geometry/eligibility does not reproduce the original row')
            if eligible:
                rng, key = rng_for(section['dataset_id'], sample, graph, sid, 'anchor', 'joint_multinomial')
                draws, weights, diagnostic = joint_bootstrap(values, near, far, blocks, rng)
                row.update(diagnostic); row['joint_rng_key'] = key
                row['joint_low'], row['joint_high'] = np.quantile(draws, [.025, .975])
                anchor_vectors[prefix + '|joint'] = draws; weight_vectors[prefix + '|joint_weights'] = weights
                if sid == 'near175':
                    rng, key = rng_for(section['dataset_id'], sample, graph, sid, 'anchor', 'independent_sides')
                    old_draws = independent_bootstrap(values, near, far, blocks, rng)
                    row['independent_rng_key'] = key
                    row['independent_low'], row['independent_high'] = np.quantile(old_draws, [.025, .975])
                    anchor_vectors[prefix + '|independent_sides'] = old_draws
                else:
                    row['independent_low'] = original.morphology_anchor_low
                    row['independent_high'] = original.morphology_anchor_high
            anchor_rows.append(row)
            if sid != 'near175':
                continue
            geometry[prefix + '|near'] = np.packbits(near); geometry[prefix + '|far'] = np.packbits(far)
            geometry[prefix + '|blocks'] = blocks
            for saved in grid.to_dict('records'):
                key = (str(saved['method_id']), int(saved['K']), str(saved['partition_id']))
                estimator = saved['estimator_id']; sn = np.zeros(len(spots), bool); sf = sn.copy(); purity = np.nan
                if key not in maps:
                    reason = 'saved_partition_unavailable'
                elif maps[key][1] is None:
                    reason = 'no_distinct_pathology_stroma_domain'
                else:
                    labels, (tumor_domain, selected) = maps[key]
                    sn = near & (labels == selected[0]); sf = far & (labels == selected[0]); purity = selected[2]
                    if estimator == 'strict_computational_interface':
                        sn &= (labels[nn] == tumor_domain).any(axis=1)
                    reason = ''
                ok = eligible and eligibility(sn, sf, blocks)
                if not eligible:
                    reason = 'morphology_band_support_below_threshold'
                elif not ok and not reason:
                    reason = 'selected_band_support_below_threshold'
                selected_point = contrast(values, sn, sf); departure = selected_point - point
                rec = {**identity, 'estimator_id': estimator, 'method_id': key[0], 'K': key[1], 'partition_id': key[2],
                       'readout_id': 'primary_barrier', 'historical_status': saved['status'],
                       'independent_status': 'success' if ok else 'not_evaluable', 'independent_reason': reason,
                       'n_anchor_near': int(near.sum()), 'n_anchor_far': int(far.sum()),
                       'n_selected_near': int(sn.sum()), 'n_selected_far': int(sf.sum()),
                       'n_near_blocks': len(np.unique(blocks[sn])), 'n_far_blocks': len(np.unique(blocks[sf])),
                       'near_retention': sn.sum() / near.sum() if near.any() else np.nan,
                       'far_retention': sf.sum() / far.sum() if far.any() else np.nan,
                       'selected_stroma_purity': purity, 'anchor_delta_score': point,
                       'selected_delta_score': selected_point, 'stroma_mad': scale,
                       'deviation_score': departure, 'abs_deviation_score': abs(departure),
                       'deviation_mad': departure / scale if scale > 0 else np.nan,
                       'abs_deviation_mad': abs(departure) / scale if scale > 0 else np.nan}
                uid = '|'.join(map(str, [graph, sid, estimator, *key]))
                membership_vectors[uid + '|near'] = np.packbits(sn); membership_vectors[uid + '|far'] = np.packbits(sf)
                for kind in ['count_only', 'block_matched']:
                    for column in ['excess_abs_deviation_score', 'excess_abs_deviation_mad',
                                   'control_abs_deviation_median_score', 'deviation_q025_score', 'deviation_q975_score']:
                        rec[kind + '_' + column] = np.nan
                    if ok:
                        rng, context = rng_for(section['dataset_id'], sample, graph, sid, estimator, *key, 'control', kind)
                        draws = control_vector(values, blocks, near, far, sn, sf, kind, rng)
                        controls[uid + '|' + kind] = draws
                        departures = draws - point; center = np.median(abs(departures)); excess = abs(departure) - center
                        rec.update({kind + '_rng_key': context, kind + '_control_used': len(draws),
                                    kind + '_control_abs_deviation_median_score': center,
                                    kind + '_excess_abs_deviation_score': excess,
                                    kind + '_excess_abs_deviation_mad': excess / scale if scale > 0 else np.nan,
                                    kind + '_deviation_q025_score': np.quantile(departures, .025),
                                    kind + '_deviation_q975_score': np.quantile(departures, .975)})
                map_rows.append(rec)
    pd.DataFrame(anchor_rows).to_csv(section_output / 'anchor_joint_sensitivity.tsv', sep='\t', index=False)
    pd.DataFrame(map_rows).to_csv(section_output / 'near175_map_sensitivity.tsv', sep='\t', index=False)
    np.savez_compressed(section_output / 'anchor_draws.npz', **anchor_vectors)
    np.savez_compressed(section_output / 'joint_block_weights.npz', **weight_vectors)
    np.savez_compressed(section_output / 'near175_control_draws.npz', **controls)
    np.savez_compressed(section_output / 'near175_memberships.npz', **membership_vectors)
    np.savez_compressed(section_output / 'near175_geometry.npz', **geometry)
    manifest = {'identity': section, 'inputs': [record(path) for path in inputs],
                'source_script_sha256': sha(Path(__file__).resolve()), 'plan_sha256': sha(PLAN),
                'n_anchor_rows': len(anchor_rows), 'n_eligible_anchor_rows': sum(r['status'] == 'evaluable' for r in anchor_rows),
                'n_map_rows': len(map_rows), 'n_eligible_map_rows': sum(r['independent_status'] == 'success' for r in map_rows),
                'output_sha256':{p.name:sha(p) for p in section_output.iterdir() if p.is_file() and p.name!='manifest.json'},
                'seconds': time.monotonic() - started}
    (section_output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main() -> None:
    args = C
    if not PLAN.is_file():
        raise FileNotFoundError('Write post-outcome plan before running numerical sensitivities')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = main_frame(OUTPUT)
    (OUTPUT / 'main_baseline_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if args.main_frame_only:
        return
    manifests = []
    with ProcessPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(section_job, section, str(OUTPUT)) for section in P['source_sections']]):
            result = future.result(); manifests.append(result)
            print(json.dumps({key: result[key] for key in ['identity', 'n_eligible_anchor_rows', 'n_eligible_map_rows', 'seconds']}), flush=True)
    anchor_files = sorted((OUTPUT / 'sections').glob('*/*/anchor_joint_sensitivity.tsv'))
    map_files = sorted((OUTPUT / 'sections').glob('*/*/near175_map_sensitivity.tsv'))
    anchors = pd.concat([read(path) for path in anchor_files], ignore_index=True)
    maps = pd.concat([read(path) for path in map_files], ignore_index=True)
    if len(manifests) != 22 or len(anchors) != 264 or len(maps) != 3360:
        raise ValueError('Incomplete full section/anchor/map output grid')
    anchors.to_csv(OUTPUT / 'anchor_joint_sensitivity.tsv', sep='\t', index=False)
    maps.to_csv(OUTPUT / 'near175_map_sensitivity.tsv', sep='\t', index=False)
    nested_summary(maps, 'near175_independent', OUTPUT)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'analysis_class': 'post_outcome_sensitivity',
                'plan': record(PLAN), 'source_script': record(Path(__file__).resolve()), 'protocol': record(PROTOCOL),
                'original_executed_source': {'path': 'scripts/analysis_pipeline/spatial/run_spatial_sensitivity.py', 'sha256': '4b5a4599d7008828e20ea58437aaab3e488c1dd2690550ca8f667a590def49ed'},
                'master_seed': MASTER, 'bootstrap_and_control_replicates': REPS,
                'software': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
                             'scipy': scipy.__version__, 'platform': platform.platform()},
                'n_sections': len(manifests), 'n_anchor_rows': len(anchors), 'n_map_rows': len(maps),
                'sections': manifests,
                'files': [record(path) for path in sorted(OUTPUT.rglob('*')) if path.is_file()]}
    (OUTPUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'complete': True, 'output': str(OUTPUT), 'n_sections': len(manifests)}), flush=True)


if __name__ == '__main__':
    main()
