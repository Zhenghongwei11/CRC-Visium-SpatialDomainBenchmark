#!/usr/bin/env python3
"""Fixed external single-gene context in anatomical and existing domain masks.

--static-only validates locks, the entire candidate panel, and saved membership
identities without opening original count matrices or calculating new RNA values.
No fitting, composite program score, gene selection, or biological identity test.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time

import h5py
import numpy as np
import pandas as pd
import scipy
from scipy.io import mmread
from scipy.sparse import csc_matrix
from regional_profiles import exact_bands

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'config'
COVERAGE = ROOT / 'results/coverage_qualified_domains'
DEFAULT = ROOT / 'results/marker_context'
RULES = ('largest_pure', 'largest_coverage_qualified_pure')
ANATOMICAL = ('full_annotated_stroma', 'exacthex_stroma_near', 'exacthex_stroma_far')
SELECTED = ('selected_whole_domain', 'selected_annotated_stroma',
            'selected_stroma_near', 'selected_stroma_far')
NATIVE = ROOT / 'results/rctd'
LIMIT = 'same_RNA_marker_context_not_independent_or_prespecified_heldout_identity_validation'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def path_key(path):
    path=Path(path).resolve()
    try:return str(path.relative_to(ROOT))
    except ValueError:return str(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return pd.read_csv(path, sep='\t', float_precision='round_trip', low_memory=False)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def save(frame, path):
    frame.to_csv(path, sep='\t', index=False, na_rep='NA')


def load_plan():
    amendment_path = DOC / 'marker_context.json'
    amendment = json.loads(amendment_path.read_text())
    lock = json.loads((DOC / 'marker_context_lock.json').read_text())
    assert sha(amendment_path) == lock['sha256']
    stage_path = DOC / 'regional_profiles.json'
    assert sha(stage_path) == json.loads((DOC / 'PROTOCOL_LOCK_v1.json').read_text())['sha256']
    candidate_path = ROOT / amendment['source_candidate_file']
    assert sha(candidate_path) == amendment['source_sha256']
    source = read(candidate_path)
    core = source[source.candidate_tier.eq('core_candidate')].copy()
    ordered = [(program, gene) for program, genes in amendment['groups'].items() for gene in genes]
    assert len(ordered) == 53 and len({gene for _, gene in ordered}) == 53
    assert len(core) == 53 and set(map(tuple, core[['program', 'gene_symbol']].to_numpy())) == set(ordered)
    assert not core.preexisting_outcome_gene.astype(str).str.lower().eq('true').any()
    assert not set(core.gene_symbol) & {'TGFB1', 'CXCL12', 'ACTA2', 'TAGLN', 'MYH11', 'CNN1', 'DES'}
    panel = pd.DataFrame(ordered, columns=['program', 'gene_symbol']).merge(
        core, on=['program', 'gene_symbol'], validate='one_to_one', sort=False)
    panel.insert(0, 'panel_ordinal', np.arange(1, 54))
    panel['interpretation'] = LIMIT
    stage = json.loads(stage_path.read_text())
    manifest_path = ROOT / stage['source_manifest']
    sections = json.loads(manifest_path.read_text())['sections']
    assert len(sections) == 22 and len({s['sample_id'] for s in sections}) == 22
    cover_manifest = json.loads((COVERAGE / 'RUN_MANIFEST.json').read_text())
    coverage_path = COVERAGE / 'selection_coverage.tsv'
    assert sha(coverage_path) == cover_manifest['outputs']['selection_coverage.tsv']
    coverage = read(coverage_path)
    assert len(coverage) == 528 and set(coverage.selection_rule) == set(RULES)
    assert not coverage[['sample_id', 'setting_id', 'selection_rule']].duplicated().any()
    assert coverage.groupby(['sample_id', 'selection_rule']).size().eq(12).all()
    identity = {'script_sha256': sha(__file__), 'marker_amendment_sha256': sha(amendment_path),
                'stage2_protocol_sha256': sha(stage_path), 'candidate_panel_sha256': sha(candidate_path),
                'source_count_manifest_sha256': sha(manifest_path),
                'coverage_manifest_sha256': sha(COVERAGE / 'RUN_MANIFEST.json'),
                'coverage_table_sha256': sha(coverage_path),
                'exacthex_helper_sha256': sha(ROOT / 'scripts/regional_profiles.py')}
    return amendment, panel, sections, coverage, identity


def saved_inputs(section, coverage):
    sid = section['sample_id']
    source = ROOT / section['source_root'] / 'spot_inputs.tsv.gz'
    spots = read(source)
    assert not spots.barcode.duplicated().any()
    folder = COVERAGE / 'sections' / section['resource'] / sid
    done = json.loads((folder / 'DONE.json').read_text())
    assert sha(source) == done['source_spots_sha256']
    for name in ('selected_rules.npz', 'all_final_region_labels.npz', 'selection_coverage.tsv'):
        assert sha(folder / name) == done['files'][name]
    with np.load(folder / 'all_final_region_labels.npz', allow_pickle=False) as z:
        assert np.array_equal(z['barcodes'], spots.barcode.to_numpy(str))
    local = coverage[coverage.sample_id.eq(sid)].copy()
    assert len(local) == 24
    masks = {}
    with np.load(folder / 'selected_rules.npz', allow_pickle=False) as z:
        assert set(z.files) == set(local.selected_mask_key)
        for key in z.files:
            assert z[key].dtype == bool and z[key].shape == (len(spots),)
            masks[key] = z[key].copy()
    for row in local.itertuples(index=False):
        assert int(masks[row.selected_mask_key].sum()) == row.region_size
        if row.selection_rule == RULES[1]:
            assert bool(row.coverage_qualified) == bool(row.region_size)
    inputs = {path_key(p): sha(p) for p in
              (source, folder / 'selected_rules.npz', folder / 'all_final_region_labels.npz',
               folder / 'selection_coverage.tsv', folder / 'DONE.json')}
    return spots, local, masks, inputs


def load_original_counts(section, spots):
    """Keep every original feature; aggregate symbols only after full-library sums."""
    sid = section['sample_id']
    directory = Path(os.environ['CRC_COUNTS_ROOT']) / section['resource'] / sid / 'members'
    sources = []
    for expected in section['source_files']:
        path = directory / expected['name']
        assert path.stat().st_size == expected['size_bytes'] and sha(path) == expected['sha256']
        sources.append({'path': path_key(path), 'bytes': path.stat().st_size,
                        'sha256': expected['sha256']})
    path = directory / (sid + '_filtered_feature_bc_matrix.h5')
    if path.exists():
        with h5py.File(path, 'r') as handle:
            m = handle['matrix']
            counts = csc_matrix((m['data'][:], m['indices'][:], m['indptr'][:]),
                                shape=tuple(m['shape'][:])).T.tocsr()
            barcodes = [x.decode() for x in m['barcodes'][:]]
            symbols = np.asarray([x.decode() for x in m['features/name'][:]], dtype=str)
            ids = np.asarray([x.decode() for x in m['features/id'][:]], dtype=str)
    else:
        with gzip.open(directory / (sid + '_matrix.mtx.gz'), 'rb') as handle:
            counts = mmread(handle).T.tocsr()
        with gzip.open(directory / (sid + '_barcodes.tsv.gz'), 'rt') as handle:
            barcodes = [line.strip() for line in handle if line.strip()]
        feature = pd.read_csv(directory / (sid + '_features.tsv.gz'), sep='\t', header=None)
        ids = feature.iloc[:, 0].astype(str).to_numpy()
        symbols = feature.iloc[:, 1].astype(str).to_numpy()
    assert counts.shape == (section['matrix_barcode_count'], section['feature_count'])
    assert len(set(barcodes)) == len(barcodes) and counts.nnz > 0
    assert np.isfinite(counts.data).all() and (counts.data >= 0).all()
    assert np.equal(counts.data, np.round(counts.data)).all()
    assert len(spots) == len(barcodes) and set(spots.barcode) == set(barcodes)
    order = pd.Index(barcodes).get_indexer(spots.barcode)
    assert (order >= 0).all() and len(set(order)) == len(barcodes)
    counts = counts[order].tocsr()
    total = np.asarray(counts.sum(axis=1, dtype=np.int64)).ravel()
    assert (total > 0).all()
    return counts, total, symbols, ids, sources


def context_vectors(counts, total, symbols, ids, panel):
    values = np.full((counts.shape[0], len(panel)), np.nan)
    detected = np.zeros(values.shape, dtype=bool)
    present = np.zeros(len(panel), dtype=bool)
    rows = []
    for j, row in enumerate(panel.itertuples(index=False)):
        columns = np.flatnonzero(symbols == row.gene_symbol)
        present[j] = bool(len(columns))
        if len(columns):
            # Sum actual source columns before log normalization, including duplicate symbols.
            aggregate = np.asarray(counts[:, columns].sum(axis=1, dtype=np.int64)).ravel()
            values[:, j] = np.log1p(aggregate / total * 1e4)
            detected[:, j] = aggregate > 0
            assert np.allclose(np.expm1(values[:, j]) * total / 1e4, aggregate, rtol=1e-12, atol=1e-10)
        rows.append({'program': row.program, 'gene_symbol': row.gene_symbol,
                     'source_feature_present': bool(len(columns)), 'n_source_symbol_columns': len(columns),
                     'source_feature_indices': ','.join(map(str, columns)),
                     'source_feature_ids': '|'.join(ids[columns]),
                     'full_section_detected_spots': int(detected[:, j].sum()) if len(columns) else np.nan})
    return values, detected, present, pd.DataFrame(rows)


def fixed_populations(section, spots, local, masks):
    lattice = spots[['array_row', 'array_col']].to_numpy(float)
    assert np.isfinite(lattice).all() and np.equal(lattice, np.round(lattice)).all()
    stroma = spots.coarse_label.eq('stroma').to_numpy()
    near, far = exact_bands(lattice.astype(int), stroma, spots.coarse_label.eq('tumor').to_numpy())
    geo = ROOT / 'data/reference_inputs/geometry' / section['resource'] / section['sample_id'] / 'geometry_arrays.npz'
    with np.load(geo, allow_pickle=False) as z:
        assert np.array_equal(near, z['nominal_honeycomb_exact|baseline|near'])
        assert np.array_equal(far, z['nominal_honeycomb_exact|baseline|far'])
        blocks = z['historical_raw_6nn|baseline|blocks'].copy()
    base = {k: section[k] for k in ('resource', 'patient_id', 'sample_id')}
    populations = []
    for name, mask in zip(ANATOMICAL, (stroma, near, far)):
        populations.append(({**base, 'setting_id': 'anatomical', 'method': 'anatomical',
                             'requested_K': np.nan, 'merge_quantile': np.nan,
                             'selection_rule': 'anatomical', 'population': name,
                             'region_coverage_qualified': np.nan,
                             'selection_unavailable_reason': ''}, mask))
    for row in local.itertuples(index=False):
        whole = masks[row.selected_mask_key]
        assert int((whole & stroma).sum()) == row.annotated_stroma_count
        assert int((whole & near).sum()) == row.n_near and int((whole & far).sum()) == row.n_far
        if whole.any():
            assert (whole & stroma).sum() / whole.sum() >= .90
        metadata = {**base, **{k: getattr(row, k) for k in
                    ('setting_id', 'method', 'requested_K', 'merge_quantile', 'selection_rule')},
                    'region_coverage_qualified': bool(row.coverage_qualified),
                    'selection_unavailable_reason': '' if pd.isna(row.unavailable_reason) else row.unavailable_reason}
        for name, mask in zip(SELECTED, (whole, whole & stroma, whole & near, whole & far)):
            populations.append(({**metadata, 'population': name}, mask))
    assert len(populations) == 99
    return populations, blocks, stroma, {path_key(geo): sha(geo)}


def mask_summary(mask, values, detected, present):
    n = int(mask.sum())
    m = len(present)
    n_detected = np.full(m, np.nan)
    fraction = np.full(m, np.nan)
    q25 = np.full(m, np.nan)
    median = np.full(m, np.nan)
    q75 = np.full(m, np.nan)
    if n:
        n_detected[present] = detected[mask][:, present].sum(axis=0)
        fraction[present] = n_detected[present] / n
        q25[present], median[present], q75[present] = np.quantile(
            values[mask][:, present], [.25, .5, .75], axis=0, method='linear')
    else:
        n_detected[present] = 0
    return {'n_detected_spots': n_detected, 'detection_fraction': fraction,
            'q25_log1p_CP10K': q25, 'median_log1p_CP10K': median,
            'q75_log1p_CP10K': q75, 'IQR_log1p_CP10K': q75 - q25}


def process_section(section, panel, coverage, output, identity):
    folder = output / 'sections' / section['resource'] / section['sample_id']
    done_path = folder / 'DONE.json'
    if done_path.exists():
        done = json.loads(done_path.read_text())
        assert done['identity'] == identity
        for name, digest in done['outputs'].items():
            assert sha(folder / name) == digest
        return done
    folder.mkdir(parents=True, exist_ok=True)
    spots, local, masks, inputs = saved_inputs(section, coverage)
    populations, blocks, stroma, geometry_input = fixed_populations(section, spots, local, masks)
    inputs.update(geometry_input)
    counts, total, symbols, ids, sources = load_original_counts(section, spots)
    values, detected, present, feature_coverage = context_vectors(counts, total, symbols, ids, panel)
    full_feature_count = counts.shape[1]
    del counts
    base = {k: section[k] for k in ('resource', 'patient_id', 'sample_id')}
    feature_coverage = feature_coverage.assign(**base)
    save(feature_coverage, folder / 'GENE_FEATURE_COVERAGE.tsv')
    panel_rows = []
    for program, g in feature_coverage.groupby('program', sort=False):
        panel_rows.append({**base, 'program': program, 'candidate_gene_N': len(g),
                           'present_gene_N': int(g.source_feature_present.sum()),
                           'absent_gene_N': int((~g.source_feature_present).sum()),
                           'absent_gene_symbols': '|'.join(g.loc[~g.source_feature_present, 'gene_symbol'])})
    save(pd.DataFrame(panel_rows), folder / 'PANEL_FEATURE_COVERAGE.tsv')
    cache = {}
    support = []
    long = []
    for metadata, mask in populations:
        packed = np.packbits(mask, bitorder='little').tobytes()
        key = hashlib.sha256(len(mask).to_bytes(8, 'little') + packed).hexdigest()
        if key not in cache:
            cache[key] = (packed, mask_summary(mask, values, detected, present))
        assert cache[key][0] == packed  # collision check is local to this barcode universe
        n = int(mask.sum())
        meta = {**metadata, 'mask_sha256': key, 'n_spots': n,
                'n_blocks': len(np.unique(blocks[mask])), 'n_annotated_stroma': int((mask & stroma).sum()),
                'population_nonempty': n > 0}
        support.append(meta)
        summary = cache[key][1]
        for j, gene in enumerate(panel.itertuples(index=False)):
            status = ('gene_absent_and_population_empty' if not present[j] and not n else
                      'gene_absent' if not present[j] else 'population_empty' if not n else 'available')
            long.append({**meta, 'program': gene.program, 'gene_symbol': gene.gene_symbol,
                         'source_feature_present': bool(present[j]), 'status': status,
                         **{k: v[j] for k, v in summary.items()}})
    frame = pd.DataFrame(long)
    assert len(frame) == 99 * 53
    assert frame.groupby(['setting_id', 'selection_rule', 'population']).size().eq(53).all()
    assert frame.loc[~frame.source_feature_present, list(mask_summary(np.zeros(len(spots), bool), values, detected, present))].isna().all().all()
    assert frame.loc[frame.n_spots.eq(0), ['detection_fraction', 'q25_log1p_CP10K', 'median_log1p_CP10K', 'q75_log1p_CP10K', 'IQR_log1p_CP10K']].isna().all().all()
    save(frame, folder / 'GENE_CONTEXT.tsv.gz')
    save(pd.DataFrame(support), folder / 'POPULATION_SUPPORT.tsv')
    done = {'identity': identity, **base, 'completed_utc': stamp(), 'source_inputs': inputs,
            'raw_count_members': sources, 'registered_barcodes': len(spots),
            'all_original_source_feature_N': full_feature_count,
            'full_original_feature_library_UMIs': int(total.sum()),
            'normalized_denominator': 'all original source feature counts per spot before any symbol aggregation',
            'candidate_genes': len(panel), 'source_present_genes': int(present.sum()),
            'population_records': len(support), 'gene_context_records': len(frame),
            'unique_masks_computed': len(cache), 'mask_cache_hits': len(support) - len(cache),
            'quantile_method': 'numpy linear; includes zero-expression spots',
            'absent_gene_policy': 'NA, never zero imputation', 'interpretation': LIMIT,
            'outputs': {name: sha(folder / name) for name in ('GENE_FEATURE_COVERAGE.tsv',
                       'PANEL_FEATURE_COVERAGE.tsv', 'GENE_CONTEXT.tsv.gz', 'POPULATION_SUPPORT.tsv')}}
    write_json(done_path, done)
    print('MARKER_SECTION_DONE', section['sample_id'], 'genes', int(present.sum()),
          'unique_masks', len(cache), 'rows', len(frame), flush=True)
    return done


def rctd_overlap(panel, sections, output):
    """Snapshot actual native lists if present; absence never establishes holdout."""
    rows = []
    inputs = {}
    for s in sections:
        sid = s['sample_id']
        dirs = [NATIVE / mode / sid
                for mode in ('all', 'pilot')]
        found = False
        for folder in dirs:
            files = {'native_common_input': folder / 'COMMON_INPUT_GENES.tsv.gz',
                     'native_bulk': folder / 'NATIVE_BULK_GENES.tsv.gz',
                     'native_regression': folder / 'NATIVE_REGRESSION_GENES.tsv.gz'}
            if not any(path.exists() for path in files.values()):
                continue
            found = True
            lists = {}
            for name, path in files.items():
                if path.exists():
                    lists[name] = set(read(path).gene.astype(str))
                    inputs[path_key(path)] = sha(path)
            status_path = folder / 'RUN_STATUS.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {'status': 'not_recorded'}
            if status_path.exists():
                inputs[path_key(status_path)] = sha(status_path)
            for gene in panel.itertuples(index=False):
                rows.append({'resource': s['resource'], 'patient_id': s['patient_id'], 'sample_id': sid,
                             'rctd_mode': folder.parent.name, 'program': gene.program,
                             'gene_symbol': gene.gene_symbol,
                             **{key + '_included': gene.gene_symbol in lists[key] if key in lists else np.nan
                                for key in files},
                             'fit_status_snapshot': status.get('status', json.dumps(status, sort_keys=True)),
                             'interpretation': LIMIT})
        if not found:
            for gene in panel.itertuples(index=False):
                rows.append({'resource': s['resource'], 'patient_id': s['patient_id'], 'sample_id': sid,
                             'rctd_mode': 'native_lists_not_available', 'program': gene.program,
                             'gene_symbol': gene.gene_symbol,
                             'native_common_input_included': np.nan, 'native_bulk_included': np.nan,
                             'native_regression_included': np.nan, 'fit_status_snapshot': 'not_available',
                             'interpretation': LIMIT})
    save(pd.DataFrame(rows), output / 'RCTD_FIT_GENE_OVERLAP.tsv')
    return inputs


def actual_patient_numbers(frame):
    rows = []
    groupings = [('all_settings_descriptive_union', ['resource', 'selection_rule', 'population', 'program', 'gene_symbol']),
                 ('per_method_K_q', ['resource', 'method', 'requested_K', 'merge_quantile', 'selection_rule', 'population', 'program', 'gene_symbol'])]
    for scope, columns in groupings:
        f = frame if scope.startswith('all_') else frame[frame.selection_rule.ne('anatomical')]
        for key, g in f.groupby(columns, dropna=False, sort=True):
            valid = g.status.eq('available')
            qualified = g.region_coverage_qualified.eq(True) & valid
            rows.append({'scope': scope, **dict(zip(columns, key)),
                         'attempted_patient_N': g.patient_id.nunique(),
                         'attempted_section_N': g.sample_id.nunique(),
                         'population_record_N': len(g),
                         'nonempty_population_patient_N': g[g.population_nonempty].patient_id.nunique(),
                         'finite_gene_stat_patient_N': g[valid].patient_id.nunique(),
                         'finite_gene_stat_section_N': g[valid].sample_id.nunique(),
                         'gene_detected_patient_N': g[g.n_detected_spots.gt(0)].patient_id.nunique(),
                         'region_coverage_qualified_finite_patient_N': g[qualified].patient_id.nunique()
                             if g.selection_rule.ne('anatomical').all() else np.nan,
                         'counting_unit': 'unique patients; settings and sections do not add independent patients'})
    return pd.DataFrame(rows)


def assemble(panel, sections, output, records, identity, started, elapsed):
    frames = []
    for name in ('GENE_FEATURE_COVERAGE.tsv', 'PANEL_FEATURE_COVERAGE.tsv', 'POPULATION_SUPPORT.tsv', 'GENE_CONTEXT.tsv.gz'):
        full = pd.concat([read(output / 'sections' / s['resource'] / s['sample_id'] / name)
                          for s in sections], ignore_index=True)
        save(full, output / name)
        if name == 'GENE_CONTEXT.tsv.gz':
            frames = full
    assert len(frames) == len(sections) * 99 * 53
    support = read(output / 'POPULATION_SUPPORT.tsv')
    selected = support[support.selection_rule.ne('anatomical')]
    assert len(selected) == len(sections) * 12 * 2 * 4
    assert selected.groupby(['sample_id', 'setting_id', 'selection_rule']).size().eq(4).all()
    save(actual_patient_numbers(frames), output / 'ACTUAL_PATIENT_N.tsv.gz')
    save(panel, output / 'GENE_PANEL.tsv')
    overlap_inputs = rctd_overlap(panel, sections, output)
    manifest = {'started_utc': started, 'completed_utc': stamp(), 'elapsed_seconds': elapsed,
                'identity': identity, 'status': 'completed_single_gene_same_RNA_marker_context',
                'analysis_class': 'post_outcome_exploratory_fixed_external_marker_list',
                'n_sections': len(sections), 'n_external_candidate_genes': 53,
                'n_strategy_settings': len(sections) * 12, 'n_setting_rule_rows': len(sections) * 24,
                'n_population_records': len(support), 'n_gene_context_rows': len(frames),
                'no_new_composite_program_score': True, 'no_new_marker_result_selection': True,
                'same_RNA_context_not_independent_or_heldout_identity_validation': True,
                'rctd_native_list_snapshot_inputs': overlap_inputs,
                'threads_maximum': 2, 'section_workers': 1,
                'python': platform.python_version(), 'numpy': np.__version__,
                'pandas': pd.__version__, 'scipy': scipy.__version__, 'h5py': h5py.__version__,
                'sections': records,
                'outputs': {p.name: sha(p) for p in output.iterdir() if p.is_file() and
                            p.name not in ('RUN_MANIFEST.json', 'STARTED.json', 'STATIC_CHECK.json')}}
    write_json(output / 'RUN_MANIFEST.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('status', 'n_sections', 'n_strategy_settings',
                     'n_setting_rule_rows', 'n_population_records', 'n_gene_context_rows')}), flush=True)


def main():
    global COVERAGE, NATIVE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--counts', type=Path, required=True)
    parser.add_argument('--native', type=Path, default=NATIVE)
    parser.add_argument('--static-only', action='store_true', help='Metadata/membership check only; never open raw counts')
    parser.add_argument('--sample', action='append', help='Optional section smoke run; produces no full-cohort manifest')
    args = parser.parse_args()
    COVERAGE=args.coverage.resolve();NATIVE=args.native.resolve();os.environ['CRC_COUNTS_ROOT']=str(args.counts.resolve())
    started = stamp()
    tic = time.monotonic()
    amendment, panel, sections, coverage, identity = load_plan()
    if args.sample:
        assert set(args.sample) <= {s['sample_id'] for s in sections}
        sections = [s for s in sections if s['sample_id'] in args.sample]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.static_only:
        inputs = {}
        for section in sections:
            spots, local, masks, section_inputs = saved_inputs(section, coverage)
            populations, blocks, stroma, geometry = fixed_populations(section, spots, local, masks)
            inputs.update(section_inputs)
            inputs.update(geometry)
            assert len(populations) == 99
        write_json(args.output / 'STATIC_CHECK.json', {
            'checked_utc': stamp(), 'status': 'static_metadata_and_fixed_memberships_checked_no_new_RNA_values',
            'identity': identity, 'n_external_candidate_genes': len(panel), 'n_sections': len(sections),
            'n_setting_rule_rows': len(sections) * 24, 'original_count_matrices_opened': 0,
            'new_marker_values_computed': False, 'input_identities': inputs,
            'runtime_plan': 'one section worker, at most2 native threads; sparse full-feature matrix; dense53genes only; unique mask caching; no model fit or count-matrix copies'})
        print('STATIC_ONLY_COMPLETE', len(sections), 'sections', len(panel), 'genes; original matrices opened=0', flush=True)
        return
    write_json(args.output / 'STARTED.json', {'started_utc': started, 'identity': identity,
               'status': 'single_gene_context_running', 'interpretation': LIMIT})
    records = [process_section(section, panel, coverage, args.output, identity) for section in sections]
    if args.sample:
        write_json(args.output / 'SMOKE_CHECK.json', {'completed_utc': stamp(), 'identity': identity,
                   'status': 'section_smoke_complete_not_full_cohort', 'sections': records})
    else:
        assemble(panel, sections, args.output, records, identity, started, time.monotonic() - tic)


if __name__ == '__main__':
    main()
