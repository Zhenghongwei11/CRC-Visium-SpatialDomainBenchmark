#!/usr/bin/env python3
"""Full descriptive rerun on frozen nominal Visium geometry and source components.

No historical result or random draw is used as a corrected outcome. All declared
maps are retained, including missing maps and newly ineligible configurations.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import regional_helpers as interface

GENES = ['TGFB1', 'CXCL12', 'ACTA2', 'TAGLN']
KEY = ['estimator_id', 'method_id', 'K', 'partition_id']
METRICS = ['abs_deviation_score', 'abs_deviation_mad', 'near_retention', 'far_retention',
           'selected_stroma_purity', 'count_only_excess_abs_deviation_score',
           'block_matched_excess_abs_deviation_score', 'count_only_excess_abs_deviation_mad',
           'block_matched_excess_abs_deviation_mad']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda: f.read(1048576), b''):
            h.update(c)
    return h.hexdigest()


def read(path):
    try:
        return pd.read_csv(path, sep='\t', float_precision='round_trip')
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_json(path, record):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def unique(frame, label):
    if frame.barcode.astype(str).duplicated().any():
        raise ValueError('Duplicate barcode: ' + label)
    return frame.assign(barcode=frame.barcode.astype(str)).set_index('barcode')


def seed(key, endian='little'):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], endian)


def context(protocol, section, graph, setting, *parts):
    return '|'.join(map(str, [protocol['master_seed'], section['dataset_id'], section['sample_id'],
                             graph, setting, *parts]))


def exact_neighbors(lattice):
    lookup = {tuple(p): i for i, p in enumerate(lattice.astype(int))}
    if len(lookup) != len(lattice):
        raise ValueError('Duplicate lattice positions')
    result = np.broadcast_to(np.arange(len(lattice))[:, None], (len(lattice), 6)).copy()
    degrees = np.zeros(len(lattice), int)
    for i, (row, col) in enumerate(lattice.astype(int)):
        found = sorted(lookup[(row + dr, col + dc)] for dr, dc in
                       [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                       if (row + dr, col + dc) in lookup)
        degrees[i] = len(found)
        result[i, :len(found)] = found
    # Self padding preserves a rectangular array and adds no distinct edge.
    return result, degrees


def eligible(near, far, blocks, protocol):
    return (near.sum() >= protocol['min_group_spots'] and far.sum() >= protocol['min_group_spots']
            and len(np.unique(blocks[near])) >= protocol['min_blocks']
            and len(np.unique(blocks[far])) >= protocol['min_blocks'])


def median_delta(values, near, far):
    return float(np.median(values[near]) - np.median(values[far])) if near.any() and far.any() else np.nan


def pooled_medians(values, group, blocks, available, sampled):
    """Exact pooled median using block multiplicities and ordered spot values."""
    positions = np.flatnonzero(group)
    positions = positions[np.argsort(values[positions], kind='stable')]
    lookup = {str(b): i for i, b in enumerate(available)}
    spot_blocks = np.array([lookup[str(b)] for b in blocks[positions]], int)
    draw_indices = np.array([[lookup[str(b)] for b in row] for row in sampled], int)
    answer = []
    for start in range(0, len(sampled), 128):
        draw = draw_indices[start:start + 128]
        weights = np.zeros((len(draw), len(available)), np.int32)
        np.add.at(weights, (np.arange(len(draw))[:, None], draw), 1)
        cumulative = np.cumsum(weights[:, spot_blocks], axis=1)
        total = cumulative[:, -1]
        if np.any(total <= 0):
            raise ValueError('Empty bootstrap group')
        lower = (total - 1) // 2 + 1
        upper = total // 2 + 1
        ia = (cumulative >= lower[:, None]).argmax(axis=1)
        ib = (cumulative >= upper[:, None]).argmax(axis=1)
        answer.append((values[positions[ia]] + values[positions[ib]]) / 2)
    return np.concatenate(answer)


def bootstrap_contrast(values, near, far, blocks, key, protocol):
    a, b = np.unique(blocks[near]), np.unique(blocks[far])
    if min(len(a), len(b)) < protocol['min_blocks']:
        return np.array([], float), {'n_used': 0, 'reason': 'insufficient_occupied_blocks'}
    rng = np.random.default_rng(seed(key))
    reps = protocol['bootstrap_replicates']
    da = rng.choice(a, (reps, len(a)), replace=True)
    db = rng.choice(b, (reps, len(b)), replace=True)
    return pooled_medians(values, near, blocks, a, da) - pooled_medians(values, far, blocks, b, db), {
        'n_used': reps, 'reason': '', 'seed': str(seed(key))}


def bootstrap_change(values, an, af, rn, rf, blocks, key, protocol):
    groups = [np.unique(blocks[g]) for g in [an, af, rn, rf]]
    diagnostics = {'n_attempted': 0, 'n_accepted': 0, 'n_used': 0, 'reason': '', 'seed': str(seed(key))}
    if min(map(len, groups)) < protocol['min_blocks']:
        diagnostics['reason'] = 'insufficient_occupied_blocks_in_compared_groups'
        return np.array([], float), diagnostics
    reps = protocol['bootstrap_replicates']
    if np.array_equal(an, rn) and np.array_equal(af, rf):
        diagnostics.update(n_used=reps, reason='identical_memberships_exact_zero')
        return np.zeros(reps), diagnostics
    npool = np.union1d(groups[0], groups[2])
    fpool = np.union1d(groups[1], groups[3])
    rng = np.random.default_rng(seed(key))
    accepted, ndraws, fdraws = 0, [], []
    for _ in range(100):
        if accepted >= reps:
            break
        nd = rng.choice(npool, (128, len(npool)), replace=True)
        fd = rng.choice(fpool, (128, len(fpool)), replace=True)
        diagnostics['n_attempted'] += 128
        valid = (np.isin(nd, groups[0]).any(axis=1) & np.isin(nd, groups[2]).any(axis=1)
                 & np.isin(fd, groups[1]).any(axis=1) & np.isin(fd, groups[3]).any(axis=1))
        accepted += int(valid.sum())
        ndraws.append(nd[valid]); fdraws.append(fd[valid])
    diagnostics.update(n_accepted=accepted, n_used=min(accepted, reps))
    if accepted < reps:
        diagnostics['reason'] = 'insufficient_accepted_paired_draws'
        return np.array([], float), diagnostics
    nd, fd = np.concatenate(ndraws)[:reps], np.concatenate(fdraws)[:reps]
    estimates = ((pooled_medians(values, an, blocks, npool, nd) - pooled_medians(values, af, blocks, fpool, fd))
                 - (pooled_medians(values, rn, blocks, npool, nd) - pooled_medians(values, rf, blocks, fpool, fd)))
    return estimates, diagnostics


def control_indices(blocks, near, far, sn, sf, kind, key, protocol):
    rng = np.random.default_rng(seed(key, 'big'))
    bands = []
    reps = protocol['control_replicates']
    for anchor, selected in [(near, sn), (far, sf)]:
        if np.any(selected & ~anchor):
            raise ValueError('Selected spot outside morphology band')
        pools = ([(np.flatnonzero(anchor), int(selected.sum()))] if kind == 'count_only' else
                 [(np.flatnonzero(anchor & (blocks == b)), int((selected & (blocks == b)).sum()))
                  for b in np.unique(blocks[selected])])
        parts = []
        for pool, n in pools:
            if n == len(pool):
                parts.append(np.broadcast_to(pool, (reps, n)))
            else:
                parts.append(np.stack([rng.choice(pool, size=n, replace=False) for _ in range(reps)]))
        bands.append(np.concatenate(parts, axis=1))
    return bands


def vectors_for(root, section, spots, protocol):
    scores = unique(read(root / section['source_root'] / 'spot_scores.tsv.gz').query("score_id == 'primary_barrier'"), 'score')
    record = protocol['component_execution_lineage']['section_artifacts'][section['dataset_id'] + '::' + section['sample_id']]
    values = unique(read(root / record['artifacts']['spot_components.tsv.gz']['path']), 'components')
    if set(scores.index) != set(spots.index) or set(values.index) != set(spots.index):
        raise ValueError('Component/score barcode universe mismatch')
    values = values.loc[spots.index]
    baseline = scores.loc[spots.index, 'score_value'].to_numpy(float)
    gene_values = values[GENES].to_numpy(float)
    if not np.isfinite(gene_values).all() or not np.isfinite(baseline).all():
        raise ValueError('Nonfinite source expression')
    if not np.allclose(gene_values.mean(axis=1), baseline, atol=2e-7, rtol=2e-6):
        raise ValueError('Four-gene reconstruction mismatch')
    vectors = {'primary_barrier': baseline, **{g: gene_values[:, i] for i, g in enumerate(GENES)},
               'ACTA2_TAGLN': values[['ACTA2', 'TAGLN']].mean(axis=1).to_numpy(float),
               'TGFB1_CXCL12': values[['TGFB1', 'CXCL12']].mean(axis=1).to_numpy(float)}
    return vectors


def prepare_maps(source, spots):
    table = read(source / 'input_maps.tsv.gz')
    if table.empty:
        return {}
    result = {}
    for (method, k, partition), g in table[table.status.eq('success')].groupby(['method_id', 'K', 'replicate_id'], sort=True):
        g = unique(g, 'map')
        if set(g.index) != set(spots.index):
            raise ValueError('Map barcode universe mismatch')
        labels = g.loc[spots.index, 'domain_label'].to_numpy(int)
        selection, _ = interface.domain_selection(labels, spots.coarse_label.to_numpy(str))
        result[(str(method), int(k), str(partition))] = (labels, selection)
    return result


def domain_geometry(labels, selection, ref, near, far, nominal):
    if ref is None or ref[1]['status'] != 'success' or selection['status'] != 'success':
        return {'membership_change_scope': 'reference_geometry_unavailable'}
    rlabels, rsel = ref
    alt = labels == selection['stroma_domain']; original = rlabels == rsel['stroma_domain']
    change = alt ^ original; union = alt | original; overlap = alt & original
    shift = np.linalg.norm(nominal[alt].mean(axis=0) - nominal[original].mean(axis=0))
    n = int(change.sum()); outside = int((change & ~(near | far)).sum())
    return {'stroma_domain_jaccard_vs_reference': float(overlap.sum() / union.sum()),
            'stroma_domain_centroid_shift_um': float(shift), 'n_changed_stroma_domain': n,
            'n_changed_stroma_outside_anchor_bands': outside,
            'fraction_changed_stroma_outside_anchor_bands': outside / n if n else np.nan,
            'membership_change_scope': ('disjoint_selected_stromal_domains' if not overlap.any() else
                                        'stromal_domain_change_extent_reported' if n else 'identical_selected_stromal_domains')}


def interval_record(values, near, far, blocks, key, protocol, is_eligible):
    point = median_delta(values, near, far)
    if is_eligible:
        draws, diag = bootstrap_contrast(values, near, far, blocks, key, protocol)
    else:
        draws, diag = np.array([], float), {'n_used': 0, 'reason': 'support_below_threshold'}
    low, high = np.quantile(draws, [.025, .975]) if len(draws) else (np.nan, np.nan)
    return {'delta': point, 'low': float(low), 'high': float(high), **diag}, draws


def analyze_section(root_string, section, protocol, protocol_sha):
    root = Path(root_string)
    output = root / protocol['output_dir'] / 'sections' / section['resource'] / section['sample_id']
    output.mkdir(parents=True, exist_ok=True)
    done = output / 'DONE.json'
    if done.exists():
        record = json.loads(done.read_text())
        if record['protocol_sha256'] != protocol_sha:
            raise ValueError('Incompatible section checkpoint')
        for name, digest in record['files'].items():
            if sha256(output / name) != digest:
                raise ValueError('Checkpoint hash mismatch: ' + name)
        return section['sample_id'], record['n_rows'], True
    source = root / section['source_root']
    spots = unique(read(source / 'spot_inputs.tsv.gz'), 'registered spots')
    vectors = vectors_for(root, section, spots, protocol)
    stroma = spots.coarse_label.eq('stroma').to_numpy()
    lattice = spots[['array_row', 'array_col']].to_numpy(int)
    nominal = np.column_stack([lattice[:, 1] / 2, np.sqrt(3) * lattice[:, 0] / 2]) * 100
    pool = read(root / protocol['geometry_output_dir'] / 'nominal_pool_all_registered.tsv.gz')
    pool = unique(pool[pool.sample_id.eq(section['sample_id'])], 'nominal pool').loc[spots.index]
    if not np.array_equal(pool[['array_row', 'array_col']].to_numpy(int), lattice):
        raise ValueError('Frozen lattice identity mismatch')
    distance = pool.nearest_pure_tumor_dnum_delta_col2_plus_3_delta_row2.to_numpy(float)
    maps = prepare_maps(source, spots)
    grid = read(root / protocol['grid_path'])
    grid = grid[(grid.resource == section['resource']) & (grid.sample_id == section['sample_id'])]
    rows, anchors_out, effects, draws, boots, case_rows, geo_arrays, membership_arrays = [], [], [], {}, {}, [], {}, {}
    legacy = read(source / 'spot_membership.tsv.gz')
    legacy_groups = {} if legacy.empty else {key: g for key, g in legacy.groupby(KEY, sort=True)}
    reconciliation = []
    for graph in protocol['graph_ids']:
        if graph == 'historical_raw_6nn':
            neighbors = interface.nearest_indices(lattice)
            degrees = np.full(len(spots), neighbors.shape[1])
        else:
            neighbors, degrees = exact_neighbors(lattice)
        components = interface.connected_stroma_components(stroma, neighbors)
        geo_arrays[graph + '|neighbors'] = neighbors
        geo_arrays[graph + '|neighbor_count'] = degrees
        geo_arrays[graph + '|components'] = components
        for setting in protocol['spatial_settings']:
            setting_id = setting['id']
            blocks = interface.block_ids(lattice, grid=setting['grid']).astype(str)
            near = stroma & np.isfinite(distance) & (distance <= setting['near_dnum_max'])
            farraw = stroma & (distance >= setting['far_dnum_min']) & (distance <= setting['far_dnum_max'])
            far = farraw & np.isin(components, np.unique(components[near]))
            if graph == 'historical_raw_6nn' and setting_id == 'baseline':
                if not np.array_equal(near, pool.nominal_near_pure_stroma_dnum_le_4.to_numpy(bool)) or not np.array_equal(
                        far, pool.nominal_far_final_after_near_component_rule.to_numpy(bool)):
                    raise ValueError('Frozen baseline pool mismatch')
            anchor_ok = eligible(near, far, blocks, protocol)
            geo_arrays[graph + '|' + setting_id + '|near'] = near
            geo_arrays[graph + '|' + setting_id + '|far'] = far
            geo_arrays[graph + '|' + setting_id + '|blocks'] = blocks
            values = vectors['primary_barrier']
            mad = float(np.median(np.abs(values[stroma] - np.median(values[stroma])))) if stroma.any() else np.nan
            ai, adraw = interval_record(values, near, far, blocks, context(protocol, section, graph, setting_id, 'anchor'), protocol, anchor_ok)
            boots[graph + '|' + setting_id + '|anchor'] = adraw
            anchor_conclusion = interface.supported_conclusion(ai['delta'], ai['low'], ai['high'], mad)
            anchor_key = {**section, 'graph_id': graph, 'setting_id': setting_id}
            anchors_out.append({**anchor_key, 'anchor_status': 'evaluable' if anchor_ok else 'not_evaluable',
                                'n_anchor_near': int(near.sum()), 'n_anchor_far': int(far.sum()),
                                'n_near_blocks': len(np.unique(blocks[near])), 'n_far_blocks': len(np.unique(blocks[far])),
                                'morphology_anchor_delta': ai['delta'], 'morphology_anchor_low': ai['low'],
                                'morphology_anchor_high': ai['high'], 'primary_scale_mad': mad,
                                'anchor_conclusion': anchor_conclusion, 'anchor_bootstrap_used': ai['n_used']})
            groups = {}
            for mk, (labels, selection) in maps.items():
                if selection['status'] == 'success':
                    for estimator in protocol['estimators']:
                        groups[(estimator, *mk)] = interface.computational_groups(near, far, labels, selection['tumor_domain'],
                                                                                 selection['stroma_domain'], neighbors, estimator)
            for saved in grid.to_dict('records'):
                mk = (str(saved['method_id']), int(saved['K']), str(saved['partition_id']))
                estimator = str(saved['estimator_id']); gkey = (estimator, *mk)
                refmk = (mk[0], mk[1], 'neighbors_6' if mk[0] == 'M2_spatial_ward' else 'seed_11')
                rkey = (estimator, *refmk); isref = mk == refmk
                zeros = np.zeros(len(spots), bool)
                sn, sf = groups.get(gkey, (zeros, zeros))
                rn, rf = groups.get(rkey, (zeros, zeros))
                selected_ok = anchor_ok and eligible(sn, sf, blocks, protocol)
                reference_ok = anchor_ok and eligible(rn, rf, blocks, protocol)
                selection = maps[mk][1] if mk in maps else {'status': 'not_evaluable', 'reason': 'saved_partition_unavailable'}
                reason = ('saved_partition_unavailable' if mk not in maps else
                          selection.get('reason', '') if selection['status'] != 'success' else
                          'morphology_band_support_below_threshold' if not anchor_ok else
                          'selected_band_support_below_threshold' if not selected_ok else '')
                key = {k: section[k] for k in ['resource', 'dataset_id', 'patient_id', 'sample_id', 'section_id']}
                key.update(graph_id=graph, setting_id=setting_id, estimator_id=estimator, method_id=mk[0], K=mk[1], partition_id=mk[2])
                uid = '|'.join(map(str, [graph, setting_id, estimator, *mk]))
                membership_arrays[uid + '|near'] = np.packbits(sn)
                membership_arrays[uid + '|far'] = np.packbits(sf)
                identity = {**key, 'historical_status': saved['status'], 'historical_reason': str(saved.get('status_reason', '')),
                            'independent_status': 'success' if selected_ok else 'not_evaluable', 'independent_reason': reason,
                            'n_anchor_near': int(near.sum()), 'n_anchor_far': int(far.sum()),
                            'n_selected_near': int(sn.sum()), 'n_selected_far': int(sf.sum()),
                            'n_near_blocks': len(np.unique(blocks[sn])), 'n_far_blocks': len(np.unique(blocks[sf])),
                            'near_retention': sn.sum() / near.sum() if near.any() else np.nan,
                            'far_retention': sf.sum() / far.sum() if far.any() else np.nan,
                            'selected_stroma_purity': selection.get('stroma_fraction', np.nan)}
                cidx = {}
                if selected_ok:
                    for kind in protocol['control_kinds']:
                        ckey = context(protocol, section, graph, setting_id, estimator, *mk, 'control', kind)
                        cidx[kind] = control_indices(blocks, near, far, sn, sf, kind, ckey, protocol)
                readouts = vectors if setting_id == 'baseline' else {'primary_barrier': vectors['primary_barrier']}
                for readout, vector in readouts.items():
                    scale = float(np.median(np.abs(vector[stroma] - np.median(vector[stroma])))) if stroma.any() else np.nan
                    av, sv = median_delta(vector, near, far), median_delta(vector, sn, sf)
                    diff = sv - av; scaled = np.isfinite(scale) and scale > 0
                    rec = {**identity, 'readout_id': readout, 'anchor_delta_score': av, 'selected_delta_score': sv,
                           'stroma_mad': scale, 'standardized_status': 'success' if scaled else 'zero_mad',
                           'deviation_score': diff, 'abs_deviation_score': abs(diff),
                           'deviation_mad': diff / scale if scaled else np.nan,
                           'abs_deviation_mad': abs(diff) / scale if scaled else np.nan}
                    for kind, (ni, fi) in cidx.items():
                        cv = np.median(vector[ni], axis=1) - np.median(vector[fi], axis=1)
                        draws[uid + '|' + readout + '|' + kind] = cv
                        ds = cv - av; ab = np.abs(ds); med = float(np.median(ab)); lo, hi = np.quantile(ds, [.025, .975])
                        ckey = context(protocol, section, graph, setting_id, estimator, *mk, 'control', kind)
                        rec.update({kind + '_seed': str(seed(ckey, 'big')), kind + '_control_used': len(cv),
                                    kind + '_control_abs_deviation_median_score': med,
                                    kind + '_excess_abs_deviation_score': abs(diff) - med,
                                    kind + '_excess_abs_deviation_mad': (abs(diff) - med) / scale if scaled else np.nan,
                                    kind + '_deviation_q025_score': float(lo), kind + '_deviation_q975_score': float(hi),
                                    kind + '_outside_descriptive_95_envelope': bool(diff < lo or diff > hi)})
                    rows.append(rec)
                    if selected_ok and setting_id == 'baseline' and section['patient_id'] in protocol['case_patients']:
                        for band, anchor, selected in [('near', near, sn), ('far', far, sf)]:
                            for block in np.unique(blocks[anchor]):
                                full = anchor & (blocks == block); kept = selected & (blocks == block)
                                case_rows.append({**key, 'readout_id': readout, 'band': band, 'spatial_block': block,
                                                  'n_anchor': int(full.sum()), 'n_selected': int(kept.sum()),
                                                  'anchor_block_median': float(np.median(vector[full])),
                                                  'selected_block_median': float(np.median(vector[kept])) if kept.any() else np.nan})
                si, sdraw = interval_record(values, sn, sf, blocks,
                    context(protocol, section, graph, setting_id, estimator, *mk, 'selected'), protocol, selected_ok)
                boots[uid + '|selected'] = sdraw
                change_draws, change_diag = (bootstrap_change(values, sn, sf, rn, rf, blocks,
                    context(protocol, section, graph, setting_id, estimator, *mk, 'paired_reference'), protocol)
                    if selected_ok and reference_ok else (np.array([], float), {'n_used': 0, 'reason': 'comparison_support_below_threshold'}))
                boots[uid + '|paired_reference'] = change_draws
                anchor_change, anchor_diag = (bootstrap_change(values, sn, sf, near, far, blocks,
                    context(protocol, section, graph, setting_id, estimator, *mk, 'paired_anchor'), protocol)
                    if selected_ok else (np.array([], float), {'n_used': 0, 'reason': 'comparison_support_below_threshold'}))
                boots[uid + '|paired_anchor'] = anchor_change
                cl, ch = np.quantile(change_draws, [.025, .975]) if len(change_draws) else (np.nan, np.nan)
                al, ah = np.quantile(anchor_change, [.025, .975]) if len(anchor_change) else (np.nan, np.nan)
                geometry = domain_geometry(maps[mk][0], selection, maps.get(refmk), near, far, nominal) if mk in maps else {}
                er = {**identity, 'is_reference': isref, 'anchor_status': 'evaluable' if anchor_ok else 'not_evaluable',
                      'anchor_reason': '' if anchor_ok else 'morphology_band_support_below_threshold',
                      'status': 'success' if selected_ok else 'not_evaluable', 'status_reason': reason,
                      'reference_status': 'success' if reference_ok else 'not_evaluable',
                      'paired_reference_status': 'success' if len(change_draws) else 'not_evaluable',
                      'paired_anchor_status': 'success' if len(anchor_change) else 'not_evaluable',
                      'morphology_anchor_delta': ai['delta'], 'morphology_anchor_low': ai['low'], 'morphology_anchor_high': ai['high'],
                      'primary_scale_mad': mad, 'anchor_conclusion': anchor_conclusion,
                      'computational_delta': si['delta'], 'computational_low': si['low'], 'computational_high': si['high'],
                      'computational_conclusion': interface.supported_conclusion(si['delta'], si['low'], si['high'], mad),
                      'reference_delta': median_delta(values, rn, rf), 'change_vs_reference': si['delta'] - median_delta(values, rn, rf),
                      'change_low': float(cl), 'change_high': float(ch),
                      'change_vs_anchor': si['delta'] - ai['delta'], 'anchor_change_low': float(al), 'anchor_change_high': float(ah),
                      'n_computational_near': int(sn.sum()), 'n_computational_far': int(sf.sum()),
                      'selected_bootstrap_used': si['n_used'], **geometry}
                for family, diagnostic in [('paired_reference', change_diag), ('paired_anchor', anchor_diag)]:
                    er.update({family + '_' + k: v for k, v in diagnostic.items()})
                effects.append(er)
                if setting_id == 'baseline':
                    old = legacy_groups.get(gkey)
                    rc = {**key, 'historical_status': saved['status'], 'corrected_status': er['status'], 'legacy_membership_available': old is not None}
                    if old is not None:
                        old = unique(old, 'legacy membership').loc[spots.index]
                        for name, new, column in [('near', near, 'morphology_near'), ('far', far, 'morphology_far'),
                                                   ('selected_near', sn, 'computational_near'), ('selected_far', sf, 'computational_far')]:
                            orig = old[column].to_numpy(bool)
                            rc.update({name + '_old_n': int(orig.sum()), name + '_new_n': int(new.sum()),
                                       name + '_gained_n': int((new & ~orig).sum()), name + '_lost_n': int((orig & ~new).sum())})
                    for col in ['morphology_anchor_delta', 'computational_delta', 'change_vs_reference', 'primary_scale_mad']:
                        rc[col + '_old'] = saved.get(col, np.nan); rc[col + '_new'] = er[col]
                    reconciliation.append(rc)
    table = pd.DataFrame(rows)
    table.to_csv(output / 'map_sensitivity.tsv', sep='\t', index=False)
    pd.DataFrame(effects).to_csv(output / 'real_cohort_effects.tsv', sep='\t', index=False)
    pd.DataFrame(anchors_out).to_csv(output / 'anchor_section_summary.tsv', sep='\t', index=False)
    pd.DataFrame(reconciliation).to_csv(output / 'historical_reconciliation.tsv', sep='\t', index=False)
    pd.DataFrame(case_rows).to_csv(output / 'case_block_contributions.tsv', sep='\t', index=False)
    np.savez_compressed(output / 'control_draws.npz', **draws)
    np.savez_compressed(output / 'bootstrap_draws.npz', **boots)
    np.savez_compressed(output / 'geometry_arrays.npz', **geo_arrays)
    np.savez_compressed(output / 'selected_memberships.npz', **membership_arrays)
    if section['patient_id'] in protocol['case_patients']:
        case = spots.reset_index().copy()
        case['nearest_tumor_dnum'] = distance
        case['distance_um'] = 50 * np.sqrt(distance)
        for name, vector in vectors.items():
            case[name] = vector
        case.to_csv(output / 'case_spot_context.tsv.gz', sep='\t', index=False, compression='gzip')
    files = {p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name != 'DONE.json'}
    write_json(done, {'protocol_sha256': protocol_sha, 'runner_sha256': sha256(__file__), **section,
                      'n_rows': len(table), 'n_effect_rows': len(effects), 'n_available_maps': len(maps), 'files': files})
    return section['sample_id'], len(table), False


def aggregate(table, output):
    group = ['resource', 'patient_id', 'sample_id', 'graph_id', 'estimator_id', 'setting_id', 'readout_id']
    for frame_id in ['historical_restricted', 'independent_eligible']:
        included = table.independent_status.eq('success')
        if frame_id == 'historical_restricted':
            included &= table.historical_status.eq('success')
        valid = table.loc[included].copy()
        for metric in METRICS:
            if metric not in valid:
                valid[metric] = np.nan
        cells = valid.groupby(group + ['method_id', 'K'])[METRICS].median().reset_index()
        sections = cells.groupby(group)[METRICS].median().reset_index()
        counts = table.assign(included=included, standardized=included & table.standardized_status.eq('success')).groupby(group).agg(
            n_declared_maps=('included', 'size'), n_evaluable_maps=('included', 'sum'), n_standardized_maps=('standardized', 'sum')).reset_index()
        sections = counts.merge(sections, on=group, how='left', validate='one_to_one')
        patient_key = [x for x in group if x != 'sample_id']
        patients = sections.groupby(patient_key)[METRICS].median().reset_index()
        pc = sections.assign(evaluable_section=sections.n_evaluable_maps.gt(0)).groupby(patient_key).agg(
            n_attempted_sections=('sample_id', 'size'), n_evaluable_sections=('evaluable_section', 'sum'),
            n_declared_maps=('n_declared_maps', 'sum'), n_evaluable_maps=('n_evaluable_maps', 'sum'),
            n_standardized_maps=('n_standardized_maps', 'sum')).reset_index()
        patients = pc.merge(patients, on=patient_key, how='left', validate='one_to_one')
        for name, data in [('method_k', cells), ('section', sections), ('patient', patients)]:
            data.insert(0, 'eligibility_frame', frame_id)
            data.to_csv(output / f'{frame_id}_{name}_summary.tsv', sep='\t', index=False)
    purity = []
    primary = table[table.readout_id.eq('primary_barrier')]
    for keys, g in primary.groupby(['resource', 'graph_id', 'estimator_id', 'setting_id'], sort=True):
        for cutoff in [0.0, 0.5, 0.75]:
            for frame in ['historical_restricted', 'independent_eligible']:
                keep = g.independent_status.eq('success') & g.selected_stroma_purity.ge(cutoff)
                if frame == 'historical_restricted':
                    keep &= g.historical_status.eq('success')
                valid = g.loc[keep]
                rec = dict(zip(['resource', 'graph_id', 'estimator_id', 'setting_id'], keys))
                rec.update(eligibility_frame=frame, purity_threshold=cutoff, n_declared_maps=len(g),
                           n_eligible_maps=len(valid), n_attempted_sections=g.sample_id.nunique(),
                           n_eligible_sections=valid.sample_id.nunique(), n_attempted_patients=g.patient_id.nunique(),
                           n_eligible_patients=valid.patient_id.nunique())
                pkeys = ['patient_id', 'sample_id', 'method_id', 'K']
                med = valid.groupby(pkeys)[METRICS].median().groupby(['patient_id', 'sample_id']).median().groupby('patient_id').median()
                for metric in METRICS:
                    rec[metric] = med[metric].median()
                purity.append(rec)
    pd.DataFrame(purity).to_csv(output / 'purity_summary.tsv', sep='\t', index=False)


def derive_effects(effects):
    supported = {'near_enrichment', 'near_depletion'}
    effects = effects.copy()
    refs = effects[effects.is_reference].set_index(['resource', 'sample_id', 'graph_id', 'setting_id', 'estimator_id', 'method_id', 'K'])
    derived = []
    for r in effects.to_dict('records'):
        key = tuple(r[k] for k in ['resource', 'sample_id', 'graph_id', 'setting_id', 'estimator_id', 'method_id', 'K'])
        ref = refs.loc[key]
        anchor_support = r['anchor_status'] == 'evaluable' and r['anchor_conclusion'] in supported
        ok = r['status'] == 'success'
        reproduces = ref.status == 'success' and anchor_support and ref.computational_conclusion == r['anchor_conclusion']
        paired_ok = r['paired_reference_status'] == 'success' and not r['is_reference']
        excludes = paired_ok and (r['change_low'] > 0 or r['change_high'] < 0)
        magnitude = np.isfinite(r['primary_scale_mad']) and r['primary_scale_mad'] > 0 and abs(r['change_vs_reference']) >= .25 * r['primary_scale_mad']
        additional_eligible = not r['is_reference'] and ok and reproduces
        additional_loss = additional_eligible and r['computational_conclusion'] == 'no_supported_difference'
        r.update(anchor_supported=anchor_support, reference_reproduces_anchor=bool(reproduces),
                 additional_loss_eligible=bool(additional_eligible), additional_support_loss=bool(additional_loss),
                 paired_change_eligible=bool(paired_ok), paired_interval_excludes_zero=bool(excludes),
                 paired_change_flag=bool(excludes and magnitude), flagged_paired_additional_loss=bool(additional_loss and excludes and magnitude),
                 map_to_anchor_support_loss=bool(ok and anchor_support and r['computational_conclusion'] == 'no_supported_difference'),
                 supported_reversal_vs_anchor=bool(ok and anchor_support and r['computational_conclusion'] in supported and r['computational_conclusion'] != r['anchor_conclusion']),
                 supported_reversal_vs_reference=bool(ok and ref.status == 'success' and r['computational_conclusion'] in supported
                                                     and ref.computational_conclusion in supported and r['computational_conclusion'] != ref.computational_conclusion),
                 opposite_point_sign_vs_reference=bool(paired_ok and r['computational_delta'] * r['reference_delta'] < 0))
        derived.append(r)
    return pd.DataFrame(derived)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    for name, expected in protocol['bound_files'].items():
        if sha256(root / name) != expected['sha256'] or (root / name).stat().st_size != expected['bytes']:
            raise ValueError('Frozen file mismatch: ' + name)
    if protocol['bound_code_sha256']['new_statistics_runner_sha256'] != sha256(__file__):
        raise ValueError('Runner does not match protocol')
    output = root / protocol['output_dir']; output.mkdir(parents=True, exist_ok=True)
    fingerprint = {'protocol_sha256': sha256(protocol_path), 'runner_sha256': sha256(__file__)}
    start_file = output / 'START.json'
    if start_file.exists() and json.loads(start_file.read_text()) != fingerprint:
        raise ValueError('Output namespace belongs to another run')
    write_json(start_file, fingerprint)
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs = [executor.submit(analyze_section, str(root), s, protocol, fingerprint['protocol_sha256']) for s in protocol['source_sections']]
        for job in as_completed(jobs):
            sample, rows, resumed = job.result()
            print('RESUMED' if resumed else 'COMPLETED', sample, rows, 'elapsed_s', round(time.monotonic() - start), flush=True)
    sections = [output / 'sections' / s['resource'] / s['sample_id'] for s in protocol['source_sections']]
    combined = pd.concat([read(s / 'map_sensitivity.tsv') for s in sections], ignore_index=True)
    combined.to_csv(output / 'map_sensitivity.tsv', sep='\t', index=False)
    aggregate(combined, output)
    effects = derive_effects(pd.concat([read(s / 'real_cohort_effects.tsv') for s in sections], ignore_index=True))
    effects.to_csv(output / 'real_cohort_effects.tsv', sep='\t', index=False)
    for name in ['anchor_section_summary', 'historical_reconciliation', 'case_block_contributions']:
        pd.concat([read(s / (name + '.tsv')) for s in sections], ignore_index=True).to_csv(output / (name + '.tsv'), sep='\t', index=False)
    inventory = {p.relative_to(output).as_posix(): {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in output.rglob('*')
                 if p.is_file() and p.name != 'run_manifest.json'}
    write_json(output / 'run_manifest.json', {**fingerprint, 'n_sections': len(sections), 'n_rows': len(combined),
                 'n_effect_rows': len(effects), 'n_independent_success': int(combined.independent_status.eq('success').sum()),
                 'elapsed_seconds': time.monotonic() - start, 'files': inventory,
                 'interpretation': 'Post-outcome regional measurement sensitivity; no cell mechanism or biological consequence established.'})


if __name__ == '__main__':
    main()
