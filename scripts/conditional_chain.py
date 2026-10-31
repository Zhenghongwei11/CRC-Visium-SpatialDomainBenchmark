#!/usr/bin/env python3
"""Fixed symmetric-swap conditional-reference diagnostic, not causal attribution.

All14 section memberships/diagnostics are globally hash-frozen before any
component cache is opened for the new reference readouts. No raw counts are read.
--static-only and --self-check never run the production chains or new readouts.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '2'
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'config'
OLD = ROOT / 'results/residual_composition'
CACHE = ROOT / 'results/pure_domains/sections/Valdeolivas'
DEFAULT = ROOT / 'results/conditional_chain'
MAPS_RESULTS = ROOT / 'data/table_inputs/current_per_map_results.tsv'
CONTROLS = ('C_MC', 'LC_MC')
READOUT = ('four_gene_mean', 'TGFB1', 'CXCL12', 'ACTA2', 'TAGLN', 'ACTA2_TAGLN_mean', 'TGFB1_CXCL12_mean')
CASE = ['resource', 'patient_id', 'sample_id', 'method_id', 'K', 'partition_id', 'case_id']
META_COLUMNS = ['resource', 'patient_id', 'sample_id', 'method_id', 'K', 'partition_id',
                'estimator_id', 'independent_status', 'n_selected_near', 'n_selected_far']
TOL = .10
ROUND_OFF = 1e-12  # same finite-precision allowance as the preserved LP-reference experiment


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def path_key(path):
    path=Path(path).resolve()
    try:return str(path.relative_to(ROOT))
    except ValueError:return str(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def js(path, value):
    def clean(x):
        if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer): return int(x)
        if isinstance(x, np.bool_): return bool(x)
        return x
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read(path, **kwargs):
    try: return pd.read_csv(path, sep='\t', float_precision='round_trip', low_memory=False, **kwargs)
    except pd.errors.EmptyDataError: return pd.DataFrame()


def save(rows, path, columns=None):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, sep='\t', index=False, na_rep='NA')
    return frame


def rng_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little')


def load_plan():
    path = DOC / 'conditional_chain.json'
    plan = json.loads(path.read_text())
    assert sha(path) == json.loads((DOC / 'conditional_chain_lock.json').read_text())['sha256']
    stage = DOC / 'regional_profiles.json'
    taxonomy = DOC / 'reference_groups.json'
    assert sha(stage) == json.loads((DOC / 'PROTOCOL_LOCK_v1.json').read_text())['sha256']
    assert tuple(plan['controls']) == CONTROLS
    design = plan['draw_design']
    assert design['chains'] == 4 and design['burnin_proposals_each_chain_band'] == 5000
    assert design['stored_states_each_chain_band'] == 50 and design['seed_master'] == 20261026
    assert design['proposals_between_stored_states'] == 'max(100,2*n_selected_band)'
    gates = plan['quality_gates_before_readout']
    assert gates['atleast_joint_unique_states'] == 50 and gates['minimum_distinct_states_each_band'] == 30
    assert gates['minimum_accepted_proposals_per_chain_after_burnin'] == 100
    assert gates['max_composition_moment_Rhat_when_variance_positive'] == 1.2
    manifest = ROOT / json.loads(stage.read_text())['source_manifest']
    sections = [s for s in json.loads(manifest.read_text())['sections'] if s['resource'] == 'Valdeolivas']
    assert len(sections) == 14
    current = MAPS_RESULTS
    # Selection phase cannot access expression contrasts: explicitly restrict parsed columns.
    original = read(current, usecols=META_COLUMNS)
    original = original[original.resource.eq('Valdeolivas') & original.estimator_id.eq('stromal_retention')].copy()
    original['case_id'] = original.apply(lambda r: f'{r.method_id}|K{int(r.K)}|{r.partition_id}', axis=1)
    assert len(original) == 504 and not original[['sample_id', 'case_id']].duplicated().any()
    comp = ROOT / 'data/reference_inputs/cell2location/W_cell_density.csv'
    keymap = ROOT / 'data/reference_inputs/cell2location/barcode_map.tsv'
    old_manifest = json.loads((OLD / 'RUN_MANIFEST.json').read_text())
    assert sha(OLD / 'availability.tsv') == old_manifest['outputs']['availability.tsv']
    assert sha(OLD / 'contrasts.tsv') == old_manifest['outputs']['contrasts.tsv']
    cache_identity = {s['sample_id']: {'measurement_sha256': sha(CACHE/s['sample_id']/'measurement.npz'),
                      'producer_DONE_sha256': sha(CACHE/s['sample_id']/'DONE.json')} for s in sections}
    cache_digest = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()
    identity = {'component_cache_identity_sha256': cache_digest,
                'script_sha256': sha(__file__), 'chain_amendment_sha256': sha(path),
                'stage2_sha256': sha(stage), 'taxonomy_sha256': sha(taxonomy),
                'source_manifest_sha256': sha(manifest), 'current_table_sha256': sha(current),
                'composition_sha256': sha(comp), 'keymap_sha256': sha(keymap),
                'preserved_LP_manifest_sha256': sha(OLD / 'RUN_MANIFEST.json'),
                'preserved_LP_availability_sha256': sha(OLD / 'availability.tsv'),
                'preserved_LP_contrasts_sha256': sha(OLD / 'contrasts.tsv')}
    return plan, sections, original, json.loads(taxonomy.read_text())['author_model_summary_groups'], comp, keymap, identity


def source_model(groups, comp, keymap):
    m = pd.read_csv(comp).set_index('spot_id')
    m.columns = [re.sub('^mean_spot_factors', '', c) for c in m.columns]
    assert len(groups) == 8 and sorted(g for v in groups.values() for g in v) == sorted(m.columns)
    abundance = pd.DataFrame({g: m[v].sum(axis=1) for g, v in groups.items()})
    assert np.allclose(abundance.sum(axis=1), m.sum(axis=1), rtol=1e-12, atol=1e-12)
    return abundance, read(keymap)


def section_inputs(section, abundance, keys):
    source = ROOT / section['source_root']
    spots = read(source / 'spot_inputs.tsv.gz')
    assert len(spots) == section['n_spots'] and not spots.barcode.duplicated().any()
    geo = ROOT / 'data/reference_inputs/geometry/Valdeolivas' / section['sample_id'] / 'geometry_arrays.npz'
    with np.load(geo, allow_pickle=False) as z:
        near = z['historical_raw_6nn|baseline|near'].copy()
        far = z['historical_raw_6nn|baseline|far'].copy()
        blocks = z['historical_raw_6nn|baseline|blocks'].copy()
    assert near.shape == far.shape == blocks.shape == (len(spots),)
    assert not (near & far).any() and spots.coarse_label.eq('stroma').to_numpy()[near | far].all()
    mapped = keys[keys.sample_id.eq(section['sample_id'])].set_index('barcode').reindex(spots.barcode)
    X = abundance.reindex(mapped.spot_id).to_numpy(float)
    available = np.isfinite(X).all(axis=1)
    maps = read(source / 'input_maps.tsv.gz')
    mapping = read(source / 'domain_mapping.tsv.gz')
    files = {str(p.relative_to(ROOT)): sha(p) for p in
             (source / 'spot_inputs.tsv.gz', source / 'input_maps.tsv.gz', source / 'domain_mapping.tsv.gz', geo)}
    return spots, near, far, blocks, X, available, maps, mapping, files


def initial_case(row, spots, near, far, blocks, available, maps, mapping):
    reason = ''
    if row.independent_status != 'success': reason = 'original_domain_comparison_unavailable'
    if not available[near | far].all(): reason = reason or 'incomplete_author_model_full_band'
    match = mapping[(mapping.method_id == row.method_id) & (mapping.K == row.K) &
                    (mapping.partition_id == row.partition_id) & mapping.selection_status.eq('success')] if not mapping.empty else pd.DataFrame()
    if match.empty: reason = reason or 'source_domain_mapping_unavailable'
    if reason: return None, reason
    mr = maps[(maps.method_id == row.method_id) & (maps.K == row.K) & (maps.replicate_id == row.partition_id)]
    assert not mr.barcode.duplicated().any()
    selected = mr.set_index('barcode').domain_label.reindex(spots.barcode).to_numpy() == match.iloc[0].selected_stroma_domain
    assert (selected & near).sum() == row.n_selected_near and (selected & far).sum() == row.n_selected_far
    if min((selected & near).sum(), (selected & far).sum()) < 20:
        return None, 'original_selected_band_below20spots'
    if min(len(np.unique(blocks[selected & near])), len(np.unique(blocks[selected & far]))) < 3:
        return None, 'original_selected_band_below3blocks'
    return selected, ''


def rhat(moment):
    """Classical unsplit sqrt(varhat/W), 4chains by50storedstates; no RNA input."""
    moment = np.asarray(moment, float)
    m, n = moment.shape
    assert m == 4 and n >= 2 and np.isfinite(moment).all()
    W = float(np.mean(np.var(moment, axis=1, ddof=1)))
    B = float(n * np.var(np.mean(moment, axis=1), ddof=1))
    if W == 0:
        return (1., 'identical_constant', W, B) if B == 0 else (np.inf, 'different_constants_zero_within_variance', W, B)
    return float(np.sqrt(((n - 1) / n * W + B / n) / W)), 'positive_within_variance', W, B


def lag_acf(x, lag):
    x = np.asarray(x, float)
    if np.all(x == x[0]): return np.nan, 'constant'
    center = x - x.mean()
    denominator = float(center @ center)
    return float(center[:-lag] @ center[lag:] / denominator), 'finite'


def membership_acf(states, pool_N, lag):
    """Vector ACF of all per-spot inclusion indicators, centered per spot."""
    indicator = np.zeros((len(states), pool_N), float)
    indicator[np.arange(len(states))[:, None], states] = 1
    centered = indicator - indicator.mean(axis=0)
    denominator = float(np.sum(centered * centered))
    if denominator == 0: return np.nan, 'constant_membership'
    return float(np.sum(centered[:-lag] * centered[lag:]) / denominator), 'finite'


def run_chain(features, actual, blockcode, local, seed_value, burn, thin, nstates):
    """Uniform selected slot, uniform eligible outside slot; composition rejection.

    C proposal(a->b)=1/[n*(N-n)]; LC proposal(a->b)=
    1/[n*(N_block-n_block)]. Block counts are invariant, so reverse probabilities
    are identical. Rejections/no-outside moves are self transitions. This gives
    uniform stationarity only within each reachable admissible component.
    """
    N, P = features.shape
    actual = np.sort(np.asarray(actual, dtype=np.int32))
    n = len(actual)
    assert 0 < n <= N and len(np.unique(actual)) == n
    selected = actual.copy()
    inside = np.zeros(N, bool); inside[selected] = True
    outside = {int(b): np.flatnonzero((blockcode == b) & ~inside).astype(np.int32)
               for b in np.unique(blockcode)} if local else {0: np.flatnonzero(~inside).astype(np.int32)}
    target = features[actual].mean(axis=0)
    centered = features - target
    sum_deviation = centered[selected].sum(axis=0)
    assert np.max(abs(sum_deviation / n)) <= ROUND_OFF
    rng = np.random.default_rng(seed_value)
    states = np.empty((nstates, n), np.int32)
    moments = np.empty((nstates, P), float)
    state_diag = []
    counts = {k: 0 for k in ('accepted_burn', 'accepted_production', 'rejected_burn',
                             'rejected_production', 'no_outside_burn', 'no_outside_production')}
    total = burn + nstates * thin
    stored = 0
    max_running_sum_disagreement = 0.
    for proposal in range(total):
        phase = 'burn' if proposal < burn else 'production'
        slot = int(rng.integers(n)); old = int(selected[slot])
        b = int(blockcode[old]) if local else 0
        pool = outside[b]
        if len(pool) == 0:
            counts['no_outside_' + phase] += 1
        else:
            outslot = int(rng.integers(len(pool))); new = int(pool[outslot])
            candidate = sum_deviation + centered[new] - centered[old]
            if np.max(np.abs(candidate / n)) <= TOL + ROUND_OFF:
                selected[slot] = new; pool[outslot] = old; sum_deviation = candidate
                counts['accepted_' + phase] += 1
            else: counts['rejected_' + phase] += 1
        if proposal >= burn and (proposal - burn + 1) % thin == 0:
            canonical = np.sort(selected)
            exact_sum = centered[canonical].sum(axis=0)
            max_running_sum_disagreement = max(max_running_sum_disagreement, float(np.max(abs(exact_sum - sum_deviation))))
            assert np.max(abs(exact_sum - sum_deviation)) < 1e-8
            # Re-sum at a fixed storage schedule to contain harmless arithmetic drift.
            sum_deviation = exact_sum
            moment = features[canonical].mean(axis=0)
            error = moment - target
            assert np.max(abs(error)) <= TOL + ROUND_OFF
            assert len(np.unique(canonical)) == n
            if local:
                assert np.array_equal(np.bincount(blockcode[canonical], minlength=blockcode.max()+1),
                                      np.bincount(blockcode[actual], minlength=blockcode.max()+1))
            overlap = int(np.isin(canonical, actual).sum())
            states[stored] = canonical; moments[stored] = moment
            state_diag.append({'stored_state': stored, 'actual_subset': overlap == n,
                               'hamming_spot_N': 2 * (n - overlap),
                               'fraction_original_selected_replaced': (n - overlap) / n,
                               'jaccard_distance_from_actual': 1 - overlap / (2*n - overlap),
                               'mean_L1_standardized_distance': float(np.abs(error).sum()),
                               'mean_L2_standardized_distance': float(np.linalg.norm(error)),
                               'max_standardized_mean_error': float(np.max(abs(error)))})
            stored += 1
    assert stored == nstates
    diag = {**counts, 'seed': seed_value, 'n_selected': n, 'pool_N': N, 'burn_proposals': burn,
            'production_proposals': nstates*thin, 'thin_proposals': thin,
            'self_transitions_burn': counts['rejected_burn'] + counts['no_outside_burn'],
            'self_transitions_production': counts['rejected_production'] + counts['no_outside_production'],
            'unique_stored_subsets': len({r.tobytes() for r in states}),
            'actual_subset_recurrence_fraction': float(np.mean([r['actual_subset'] for r in state_diag])),
            'mean_hamming_spot_N': float(np.mean([r['hamming_spot_N'] for r in state_diag])),
            'mean_fraction_original_selected_replaced': float(np.mean([r['fraction_original_selected_replaced'] for r in state_diag])),
            'max_running_sum_disagreement': max_running_sum_disagreement}
    assert diag['accepted_burn'] + diag['self_transitions_burn'] == burn
    assert diag['accepted_production'] + diag['self_transitions_production'] == nstates*thin
    return states, moments, state_diag, diag


def prepare_section(section, originals, abundance, keys, group_names, plan, output, identity):
    folder = output / 'sections' / section['sample_id']; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'SELECTION_DONE.json').exists():
        done = json.loads((folder / 'SELECTION_DONE.json').read_text())
        assert done['identity'] == identity
        for name, digest in done['outputs'].items(): assert sha(folder / name) == digest
        return done
    spots, near, far, blocks, X, available, maps, mapping, files = section_inputs(section, abundance, keys)
    _, blockcode = np.unique(blocks, return_inverse=True)
    arrays = {'barcodes': spots.barcode.to_numpy(str)}
    availability = []; chains = []; moment_rows = []; lag_rows = []; states_rows = []
    case_rows = originals[originals.sample_id.eq(section['sample_id'])]
    design = plan['draw_design']; gates = plan['quality_gates_before_readout']
    for row in case_rows.itertuples(index=False):
        meta = {k: getattr(row, k) for k in CASE}
        selected, reason = initial_case(row, spots, near, far, blocks, available, maps, mapping)
        for control in CONTROLS:
            if reason:
                availability.append({**meta, 'control': control, 'chains_attempted': 0,
                                     'evaluable': False, 'reason': reason})
                continue
            tag = hashlib.sha256((row.case_id + '|' + control).encode()).hexdigest()[:20]
            stored_bands = []; band_unique = []; accepted_min = []; moment_rhats = []; reasons = []
            for band, mask in [('near', near), ('far', far)]:
                pool = np.flatnonzero(mask)
                actual = np.flatnonzero(selected[pool])
                standardization = np.maximum(X[pool].std(axis=0), 1e-6)
                features = X[pool] / standardization
                target = features[actual].mean(axis=0)
                thin = max(100, 2*len(actual))
                chain_states = []; chain_moments = []
                for chain in range(design['chains']):
                    # Preserve the original seed domain as bytes for numerical reproducibility.
                    seed_domain = bytes.fromhex('726576696577355f636f6e646974696f6e616c5f636861696e').decode('ascii')
                    sv = rng_seed(design['seed_master'], seed_domain, section['sample_id'], row.case_id, control, band, chain)
                    state, moment, sd, diag = run_chain(features, actual, blockcode[pool], control == 'LC_MC',
                                                       sv, design['burnin_proposals_each_chain_band'], thin,
                                                       design['stored_states_each_chain_band'])
                    chains.append({**meta, 'control': control, 'band': band, 'chain': chain, **diag})
                    accepted_min.append(diag['accepted_production'])
                    for rr in sd: states_rows.append({**meta, 'control': control, 'band': band, 'chain': chain, **rr})
                    for g, name in enumerate(group_names):
                        for lag in (1, 5):
                            value, status = lag_acf(moment[:, g], lag)
                            lag_rows.append({**meta, 'control': control, 'band': band, 'chain': chain,
                                             'diagnostic': 'composition_moment', 'variable': name, 'lag': lag,
                                             'autocorrelation': value, 'status': status})
                    for lag in (1, 5):
                        value, status = membership_acf(state, len(pool), lag)
                        lag_rows.append({**meta, 'control': control, 'band': band, 'chain': chain,
                                         'diagnostic': 'membership_indicator_vector', 'variable': 'all_source_band_spots',
                                         'lag': lag, 'autocorrelation': value, 'status': status})
                    for variable in ('fraction_original_selected_replaced', 'actual_subset', 'max_standardized_mean_error'):
                        for lag in (1, 5):
                            value, status = lag_acf([z[variable] for z in sd], lag)
                            lag_rows.append({**meta, 'control': control, 'band': band, 'chain': chain,
                                             'diagnostic': 'membership_or_balance', 'variable': variable,
                                             'lag': lag, 'autocorrelation': value, 'status': status})
                    chain_states.append(state); chain_moments.append(moment)
                state = np.stack(chain_states); moment = np.stack(chain_moments)
                global_state = pool[state].astype(np.int32)
                arrays[tag + '_' + band] = global_state
                arrays[tag + '_' + band + '_moments'] = moment
                arrays[tag + '_' + band + '_target'] = target
                arrays[tag + '_' + band + '_SD'] = standardization
                stored_bands.append(global_state.reshape(200, -1))
                unique = len({z.tobytes() for z in state.reshape(200, -1)})
                band_unique.append(unique)
                if unique < gates['minimum_distinct_states_each_band']: reasons.append(band + '_distinct_states_below30')
                for j, name in enumerate(group_names):
                    value, status, W, B = rhat(moment[:, :, j]); moment_rhats.append(value)
                    moment_rows.append({**meta, 'control': control, 'band': band, 'group': name,
                                        'Rhat': value, 'Rhat_status': status, 'within_chain_variance_W': W,
                                        'between_chain_variance_B': B, 'target': target[j],
                                        'source_band_SD_floor1e_minus6': standardization[j],
                                        'max_abs_standardized_target_error': float(np.max(abs(moment[:, :, j] - target[j])))})
            joint_unique = len({a.tobytes() + b.tobytes() for a, b in zip(*stored_bands)})
            actual_near = np.flatnonzero(selected & near); actual_far = np.flatnonzero(selected & far)
            joint_actual = np.all(stored_bands[0] == actual_near, axis=1) & np.all(stored_bands[1] == actual_far, axis=1)
            if joint_unique < gates['atleast_joint_unique_states']: reasons.append('joint_distinct_states_below50')
            if min(accepted_min) < gates['minimum_accepted_proposals_per_chain_after_burnin']: reasons.append('production_accepted_proposals_below100_in_chain')
            maximum_rhat = max(moment_rhats)
            if maximum_rhat > gates['max_composition_moment_Rhat_when_variance_positive']: reasons.append('composition_moment_Rhat_above1p2_or_incompatible_constants')
            availability.append({**meta, 'control': control, 'arrays_tag': tag, 'chains_attempted': 8,
                                 'joint_states_stored': 200, 'near_unique_states': band_unique[0],
                                 'far_unique_states': band_unique[1], 'joint_unique_states': joint_unique,
                                 'joint_actual_subset_fraction': float(joint_actual.mean()),
                                 'minimum_production_accepted_proposals': min(accepted_min),
                                 'maximum_composition_moment_Rhat': maximum_rhat,
                                 'evaluable': not reasons, 'reason': ';'.join(reasons)})
            print('CHAIN_CASE', section['sample_id'], row.case_id, control, 'unique', joint_unique,
                  'eligible', not reasons, flush=True)
    np.savez_compressed(folder / 'chain_states.npz', **arrays)
    save(availability, folder / 'availability.tsv')
    save(chains, folder / 'chain_diagnostics.tsv', columns=CASE + ['control', 'band', 'chain'] if not chains else None)
    save(moment_rows, folder / 'moment_Rhat.tsv', columns=CASE + ['control', 'band', 'group'] if not moment_rows else None)
    save(lag_rows, folder / 'lag_diagnostics.tsv.gz', columns=CASE + ['control', 'band', 'chain'] if not lag_rows else None)
    save(states_rows, folder / 'stored_state_diagnostics.tsv.gz', columns=CASE + ['control', 'band', 'chain'] if not states_rows else None)
    done = {'identity': identity, 'sample_id': section['sample_id'], 'completed_utc': stamp(),
            'source_inputs': files, 'new_readouts_computed': False,
            'outputs': {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != 'SELECTION_DONE.json'}}
    js(folder / 'SELECTION_DONE.json', done)
    return done


def freeze_selections(sections, records, output, identity):
    names = ('availability.tsv', 'chain_diagnostics.tsv', 'moment_Rhat.tsv', 'lag_diagnostics.tsv.gz', 'stored_state_diagnostics.tsv.gz')
    for name in names:
        pieces = [read(output / 'sections' / s['sample_id'] / name) for s in sections]
        save(pd.concat(pieces, ignore_index=True), output / name)
    availability = read(output / 'availability.tsv')
    assert len(availability) == 1008 and not availability[CASE + ['control']].duplicated().any()
    all_files = {str(p.relative_to(output)): sha(p) for s in sections for p in
                 (output / 'sections' / s['sample_id']).iterdir() if p.is_file()}
    all_files.update({name: sha(output / name) for name in names})
    lock = {'locked_utc': stamp(), 'identity': identity, 'new_readouts_computed': False,
            'all14_sections_prepared': True, 'attempted_setting_controls': len(availability),
            'evaluable_setting_controls': int(availability.evaluable.sum()),
            'selection_and_diagnostic_files': all_files,
            'sampling_limit': 'correlated states in reachable component; no proof of global uniform mixing'}
    js(output / 'FROZEN_SELECTIONS.json', lock)
    return lock


def verify_freeze(output, identity):
    lock = json.loads((output / 'FROZEN_SELECTIONS.json').read_text())
    assert lock['identity'] == identity and lock['all14_sections_prepared'] and not lock['new_readouts_computed']
    for name, digest in lock['selection_and_diagnostic_files'].items(): assert sha(output / name) == digest
    return lock


def hierarchy(frame, scope, basis):
    metrics = ['excess_expression_units', 'excess_mad_units', 'reference_median_absolute_departure']
    if frame.empty: return [], []
    a = frame.groupby(['resource', 'patient_id', 'sample_id', 'method_id', 'K', 'control', 'readout'], as_index=False)[metrics].median()
    b = a.groupby(['resource', 'patient_id', 'sample_id', 'control', 'readout'], as_index=False)[metrics].median()
    c = b.groupby(['resource', 'patient_id', 'control', 'readout'], as_index=False)[metrics].median()
    c['scope'] = scope; c['eligibility_basis'] = basis
    summary = []
    for (resource, control, readout), g in c.groupby(['resource', 'control', 'readout']):
        summary.append({'scope': scope, 'eligibility_basis': basis, 'resource': resource,
                        'control': control, 'readout': readout, 'case_covered_patient_N': len(g),
                        **{m: g[m].median() for m in metrics},
                        **{m + '_finite_patient_N': int(g[m].notna().sum()) for m in metrics}})
    return c.to_dict('records'), summary


def readout_phase(sections, originals, output, identity):
    lock = verify_freeze(output, identity)
    cache_now = {s['sample_id']: {'measurement_sha256': sha(CACHE/s['sample_id']/'measurement.npz'),
                 'producer_DONE_sha256': sha(CACHE/s['sample_id']/'DONE.json')} for s in sections}
    assert hashlib.sha256(json.dumps(cache_now, sort_keys=True).encode()).hexdigest() == identity['component_cache_identity_sha256']
    rows = []; source_cache = []; distributions = []
    original_values = read(MAPS_RESULTS)
    original_values = original_values[original_values.resource.eq('Valdeolivas') & original_values.estimator_id.eq('stromal_retention')]
    abundance = keys = None
    for section in sections:
        sid = section['sample_id']; folder = output / 'sections' / sid
        availability = read(folder / 'availability.tsv')
        # Cache identity is bound to the previously completed original pilot; no raw matrix path.
        measurement = CACHE / sid / 'measurement.npz'; prior_done = json.loads((CACHE / sid / 'DONE.json').read_text())
        assert sha(measurement) == prior_done['files']['measurement.npz']
        spots = read(ROOT / section['source_root'] / 'spot_inputs.tsv.gz')
        with np.load(measurement, allow_pickle=False) as z:
            assert np.array_equal(z['barcodes'], spots.barcode.to_numpy(str))
            v4 = z['values'].copy(); assert v4.shape == (len(spots), 4)
        v = np.column_stack([v4.mean(axis=1), v4, v4[:, 2:].mean(axis=1), v4[:, :2].mean(axis=1)])
        with np.load(ROOT / 'data/reference_inputs/geometry/Valdeolivas' / sid / 'geometry_arrays.npz', allow_pickle=False) as z:
            near = z['historical_raw_6nn|baseline|near']; far = z['historical_raw_6nn|baseline|far']
        maps = read(ROOT / section['source_root'] / 'input_maps.tsv.gz'); mapping = read(ROOT / section['source_root'] / 'domain_mapping.tsv.gz')
        stroma = spots.coarse_label.eq('stroma').to_numpy(); scale = np.median(abs(v[stroma] - np.median(v[stroma], axis=0)), axis=0) if stroma.any() else np.full(7, np.nan)
        reference_arrays = {}
        with np.load(folder / 'chain_states.npz', allow_pickle=False) as states:
            assert np.array_equal(states['barcodes'], spots.barcode.to_numpy(str))
            for row in availability.itertuples(index=False):
                meta = {k: getattr(row, k) for k in CASE}
                if not row.evaluable:
                    for name in READOUT: rows.append({**meta, 'control': row.control, 'readout': name, 'evaluable': False, 'reason': row.reason})
                    continue
                match = mapping[(mapping.method_id == row.method_id) & (mapping.K == row.K) & (mapping.partition_id == row.partition_id) & mapping.selection_status.eq('success')]
                mr = maps[(maps.method_id == row.method_id) & (maps.K == row.K) & (maps.replicate_id == row.partition_id)]
                selected = mr.set_index('barcode').domain_label.reindex(spots.barcode).to_numpy() == match.iloc[0].selected_stroma_domain
                full = np.median(v[near], axis=0) - np.median(v[far], axis=0)
                actual = np.median(v[near & selected], axis=0) - np.median(v[far & selected], axis=0)
                original = original_values[(original_values.sample_id == sid) & (original_values.method_id == row.method_id) & (original_values.K == row.K) & (original_values.partition_id == row.partition_id)].iloc[0]
                assert np.allclose([full[0], actual[0]], [original.full_band_contrast, original.domain_selected_contrast], rtol=2e-6, atol=2e-7)
                ns = states[row.arrays_tag + '_near'].reshape(200, -1); fs = states[row.arrays_tag + '_far'].reshape(200, -1)
                reference = np.median(v[ns], axis=1) - np.median(v[fs], axis=1)
                reference_arrays[row.arrays_tag] = reference
                departure = abs(reference - full)
                med = np.median(departure, axis=0); excess = abs(actual - full) - med
                for j, name in enumerate(READOUT):
                    rows.append({**meta, 'control': row.control, 'readout': name, 'evaluable': True,
                                 'reason': '', 'stored_correlated_joint_states': 200,
                                 'full_band_contrast': full[j], 'domain_selected_contrast': actual[j],
                                 'absolute_domain_departure': abs(actual[j] - full[j]),
                                 'reference_median_absolute_departure': med[j],
                                 'excess_expression_units': excess[j], 'section_stroma_mad': scale[j],
                                 'excess_mad_units': excess[j] / scale[j] if scale[j] > 0 else np.nan,
                                 'reference_odd_even_median_difference': abs(np.median(departure[::2, j]) - np.median(departure[1::2, j]))})
        np.savez_compressed(folder / 'readout_reference_distributions.npz', readout_names=np.asarray(READOUT), **reference_arrays)
        source_cache.append({'sample_id': sid, 'path': path_key(measurement), 'sha256': sha(measurement), 'cache_producer_manifest_sha256': sha(CACHE / sid / 'DONE.json')})
        distributions.append({'path': str((folder / 'readout_reference_distributions.npz').relative_to(output)), 'sha256': sha(folder / 'readout_reference_distributions.npz')})
    frame = save(rows, output / 'contrasts.tsv')
    assert len(frame) == 1008 * 7
    good = frame[frame.evaluable]
    old = read(OLD / 'contrasts.tsv'); old = old[old.control.isin(['S', 'L'])]
    patients = []; summaries = []; matched = []
    # Different new-control support stays separate; never subtract differently covered medians.
    for basis in CONTROLS:
        cases = good[good.control.eq(basis)][CASE].drop_duplicates()
        previous = old.merge(cases, on=CASE, validate='many_to_one')
        assert len(previous) == len(cases) * 2 * 7
        comparison = pd.concat([good[good.control.eq(basis)], previous], ignore_index=True)
        comparison['eligibility_basis'] = basis; comparison['scope'] = 'new_control_specific_identical_cases'
        matched.append(comparison)
        p, s = hierarchy(comparison, 'new_control_specific_identical_cases', basis); patients.extend(p); summaries.extend(s)
    common = good.groupby(CASE).control.nunique(); common = common[common.eq(2)].reset_index()[CASE]
    common.to_csv(output / 'all_new_control_common_cases.tsv', sep='\t', index=False)
    current = good.merge(common, on=CASE, validate='many_to_one')
    previous = old.merge(common, on=CASE, validate='many_to_one')
    combined = pd.concat([current, previous], ignore_index=True)
    combined['eligibility_basis'] = 'C_MC_and_LC_MC'; combined['scope'] = 'both_new_controls_identical_cases'
    matched.append(combined)
    p, s = hierarchy(combined, 'both_new_controls_identical_cases', 'C_MC_and_LC_MC'); patients.extend(p); summaries.extend(s)
    save(pd.concat(matched, ignore_index=True), output / 'matched_reference_comparisons.tsv')
    save(patients, output / 'patient_summaries.tsv', columns=['scope', 'eligibility_basis', 'resource', 'patient_id', 'control', 'readout', 'excess_expression_units', 'excess_mad_units', 'reference_median_absolute_departure'] if not patients else None)
    save(summaries, output / 'resource_summaries.tsv', columns=['scope', 'eligibility_basis', 'resource', 'control', 'readout', 'case_covered_patient_N', 'excess_expression_units', 'excess_mad_units', 'reference_median_absolute_departure'] if not summaries else None)
    return {'selection_lock_sha256': sha(output / 'FROZEN_SELECTIONS.json'), 'selection_locked_utc': lock['locked_utc'],
            'new_readouts_started_after_all_selections_frozen': True,
            'raw_count_matrices_opened': 0, 'component_cache_inputs': source_cache,
            'reference_distribution_files': distributions, 'n_readout_attempt_rows': len(frame),
            'n_evaluable_readout_rows': len(good)}


def transition_matrix(features, actual, blocks, local, tolerance):
    """Exhaustive finite toy-state check of the same symmetric proposal law."""
    N = len(features); n = len(actual); target = features[actual].mean(axis=0)
    counts = np.bincount(blocks[actual], minlength=blocks.max()+1)
    states = []
    for combination in itertools.combinations(range(N), n):
        s = np.asarray(combination)
        if local and not np.array_equal(np.bincount(blocks[s], minlength=len(counts)), counts): continue
        if np.max(abs(features[s].mean(axis=0)-target)) <= tolerance + ROUND_OFF: states.append(combination)
    index = {s: i for i, s in enumerate(states)}; matrix = np.zeros((len(states), len(states)))
    for i, state in enumerate(states):
        for old in state:
            outside = [x for x in range(N) if x not in state and (not local or blocks[x] == blocks[old])]
            if not outside: matrix[i, i] += 1/n; continue
            for new in outside:
                replacement = tuple(sorted((set(state)-{old})|{new}))
                destination = index.get(replacement, i)
                matrix[i, destination] += 1/(n*len(outside))
    assert np.allclose(matrix.sum(axis=1), 1, atol=1e-14)
    assert np.allclose(matrix, matrix.T, atol=1e-14)
    assert np.allclose(np.full(len(states), 1/len(states))@matrix, np.full(len(states), 1/len(states)), atol=1e-14)
    return {'admissible_toy_states': len(states), 'row_sum_max_error': float(np.max(abs(matrix.sum(axis=1)-1))),
            'symmetry_max_error': float(np.max(abs(matrix-matrix.T))),
            'positive_diagonal_states': int((np.diag(matrix)>0).sum())}


def self_check(output):
    features = np.column_stack([np.arange(6)/5, np.array([.4,.1,.2,.5,.7,.9])])
    actual = np.array([0,1,3]); blocks = np.array([0,0,1,1,2,2])
    proof = {control: transition_matrix(features, actual, blocks, control == 'LC_MC', .10) for control in CONTROLS}
    for control in CONTROLS:
        state, moments, sd, diag = run_chain(features, actual, blocks, control == 'LC_MC', 12345, 10, 5, 20)
        assert state.shape == (20, 3) and np.max(abs(moments-features[actual].mean(axis=0))) <= TOL+1e-10
        assert all(len(set(s))==3 for s in state)
    assert rhat(np.ones((4,50)))[0] == 1 and rhat(np.ones((4,50)))[1] == 'identical_constant'
    assert np.isinf(rhat(np.arange(4)[:,None]*np.ones((4,50)))[0])
    example = np.tile(np.arange(50),(4,1))+np.arange(4)[:,None]
    rr, _, W, B = rhat(example)
    assert np.isclose(rr, np.sqrt((49/50*W+B/50)/W))
    output.mkdir(parents=True, exist_ok=True)
    js(output / 'MATHEMATICAL_CHECK.json', {'checked_utc': stamp(), 'script_sha256': sha(__file__),
       'status': 'finite_toy_kernel_symmetry_stationarity_constraints_and_Rhat_checks_passed',
       'original_count_matrices_opened': 0, 'production_chains_run': 0, 'new_reference_RNA_readouts_computed': False,
       'toy_transition_checks': proof, 'interpretation': 'mathematical/code check only; no claim about real-chain mixing'})
    print('MATHEMATICAL_CHECK_PASS', json.dumps(proof), flush=True)


def static_check(plan, sections, originals, identity, output):
    old = read(OLD / 'availability.tsv'); eligible = old[old.control.eq('S') & old.evaluable]
    meta = eligible[CASE].merge(originals, on=CASE, validate='one_to_one')
    n = meta[['n_selected_near', 'n_selected_far']].to_numpy(int)
    proposals = (5000+50*np.maximum(100,2*n))*4*2
    cache_bytes = sum((CACHE/s['sample_id']/'measurement.npz').stat().st_size for s in sections)
    output.mkdir(parents=True, exist_ok=True)
    js(output/'STATIC_CHECK.json', {'checked_utc': stamp(), 'identity': identity,
       'status': 'static_plan_and_budget_only_no_production_chains_or_new_readouts',
       'source_sections': len(sections), 'all_attempted_case_control_rows': len(originals)*2,
       'previous_model_complete_qualified_cases': len(meta), 'maximum_production_case_controls': len(meta)*2,
       'planned_chain_band_runs': len(meta)*2*2*4, 'planned_swap_proposals': int(proposals.sum()),
       'uncompressed_index_array_bytes': int(n.sum()*2*200*4), 'all14_component_cache_bytes': cache_bytes,
       'expected_memory': 'small source composition table, one section metadata/maps and int32 chain states; no original matrix; reserve512MiB-1GiB, runtime measurement pending',
       'expected_time': '17620800 sequential proposals; no scientific-data timing run yet; approximately minutes, depends on CPU contention',
       'reads_before_freeze': ['locked JSON/TSV identities', 'source spots/maps/domain mapping', 'raw6nn saved geometry', 'author36type abundance means and barcode keymap'],
       'reads_after_freeze': ['previously saved four-component spot cache, about0.50MB total', 'preserved S/L contrasts on identical new eligible cases'],
       'original_count_matrices_opened': 0, 'production_chains_run': 0,
       'new_reference_RNA_readouts_computed': False,
       'Rhat_definition': 'classical unsplit4chain50state sqrt(varhat/W); exactzero W,B constant rules; no positive-variance threshold tuning',
       'acceptance_roundoff': ROUND_OFF, 'lag_definition': 'within-chain centered ACF at stored-state lags1and5; all8compositionmoments, vectorper-spotmembershipindicators and scalarturnover/actualrecurrence/balance; diagnostics only',
       'phase_boundary': 'global FROZEN_SELECTIONS hashes all14section indices and all diagnostic/eligibility tables before readout_phase opens any component cache'})
    print('STATIC_CHECK_PASS', int(proposals.sum()), 'planned proposals; no production sampling', flush=True)


def main():
    global OLD, CACHE, MAPS_RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--residual', type=Path, default=OLD)
    parser.add_argument('--domains', type=Path, default=ROOT / 'results/pure_domains')
    parser.add_argument('--maps-results', type=Path, default=MAPS_RESULTS)
    parser.add_argument('--static-only', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='Sample/diagnose/freeze all sections; no readout')
    parser.add_argument('--readout-only', action='store_true', help='Verify existing global freeze before reading lightweight component caches')
    args = parser.parse_args(); OLD=args.residual.resolve(); CACHE=args.domains.resolve()/'sections/Valdeolivas'; MAPS_RESULTS=args.maps_results.resolve(); assert sum([args.static_only,args.self_check,args.prepare_only,args.readout_only]) <= 1
    if args.self_check: self_check(args.output); return
    plan, sections, originals, groups, comp, keymap, identity = load_plan()
    if args.static_only: static_check(plan, sections, originals, identity, args.output); return
    args.output.mkdir(parents=True, exist_ok=True); started = stamp(); tic = time.monotonic()
    js(args.output/'STARTED.json', {'started_utc': started, 'identity': identity,
       'phase': 'readout_only' if args.readout_only else 'chain_preparation', 'raw_count_matrices_opened': 0})
    if not args.readout_only:
        if (args.output/'FROZEN_SELECTIONS.json').exists(): verify_freeze(args.output, identity)
        else:
            abundance, keys = source_model(groups, comp, keymap)
            records = [prepare_section(s, originals, abundance, keys, list(groups), plan, args.output, identity) for s in sections]
            freeze_selections(sections, records, args.output, identity)
        if args.prepare_only:
            print('ALL_SELECTIONS_FROZEN_NO_NEW_READOUTS', flush=True); return
    outcome = readout_phase(sections, originals, args.output, identity)
    manifest = {'started_utc': started, 'completed_utc': stamp(), 'elapsed_seconds': time.monotonic()-tic,
                'identity': identity, 'status': 'completed_conditional_chain_support_diagnostic_not_causal_attribution',
                'source_sections': len(sections), 'threads_maximum': 2, 'section_workers': 1,
                'numpy': np.__version__, 'pandas': pd.__version__, **outcome,
                'outputs': {p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name!='RUN_MANIFEST.json'}}
    js(args.output/'RUN_MANIFEST.json', manifest)
    print(json.dumps({k:manifest[k]for k in ('status','n_readout_attempt_rows','n_evaluable_readout_rows')}), flush=True)


if __name__ == '__main__':
    main()
